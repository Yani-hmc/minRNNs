"""
Neural sequence model composed of stacked RNN-based blocks,
with optional causal convolution, normalization, and feedforward layers.
"""

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Module, ModuleList

from utils import exists

from minRNNs import sequence_modules
from baselines import plainGRU, plainLSTM, simpleTransformer, simpleMamba

sequence_modules.update({
    'gru': plainGRU,
    'lstm': plainLSTM,
    'transformer': simpleTransformer,
    'mamba': simpleMamba})


# Modules

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

class BatchNorm(Module):
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x):
        if len(x.shape) == 3:
            B, L, D = x.shape
            return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise ValueError

class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias=True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias

def FeedForward(dim, mult=4, dropout=0.1):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(dim_inner),
        nn.Linear(dim_inner, dim),
        nn.Dropout(dropout)
    )

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d


# Model

norms = {
    'BatchNorm': BatchNorm,
    'RMSNorm': RMSNorm
}

class Model(Module):
    def __init__(
        self,
        *,
        module,
        num_tokens,
        d_in,
        d_out,
        dim,
        depth,
        dropout,
        ff_mult,
        conv_kernel_size,
        enable_conv,
        enable_ff,
        norm_type,
        rnn_config
    ):
        super().__init__()
        if num_tokens is not None:
            self.token_emb = nn.Embedding(num_tokens, dim)
        else:
            self.token_emb = None
            self.layer_in = nn.Linear(d_in, dim)

        self.layers = ModuleList([])

        Norm = norms[norm_type]
        RNN_Module = sequence_modules[module]
        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(
                    dim, conv_kernel_size) if enable_conv else None,
                Norm(dim),
                RNN_Module(**rnn_config),
                Norm(dim),
                FeedForward(dim, mult=ff_mult,
                            dropout=dropout) if enable_ff else None,
            ]))

        self.norm = Norm(dim)
        self.to_out = nn.Linear(dim, d_out, bias=False)

    def forward(
        self,
        x,
        return_states=False,
        prev_states=None
    ):
        if exists(self.token_emb):
            x = self.token_emb(x)
        else:
            x = self.layer_in(x)

        next_prev_states = []

        if not exists(prev_states):
            prev_states = [None for _ in range(len(self.layers))]

        for (conv, norm, minlstm, ff_norm, ff), prev_state in zip(self.layers, prev_states):
            # conv
            if exists(conv):
                x = conv(x) + x
            
            # minRNN
            min_rnn_out, next_prev_state = minlstm(
                norm(x),
                prev_state=prev_state
            )
            x = min_rnn_out + x
            next_prev_states.append(next_prev_state)

            # feedforward
            if exists(ff):
                x = ff(ff_norm(x)) + x

        logits = self.to_out(self.norm(x))

        if not return_states:
            return logits

        return logits, next_prev_states
