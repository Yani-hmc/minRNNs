"""
Sequence modules used for benchmarking sequential architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard sequential GRU/LSTM

class plainGRU(nn.Module):
    """Sequential GRU using explicit time-step loop (non-parallel)."""
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        use_init_hidden_state: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim_inner = int(dim * expansion_factor)
        self.cell = nn.GRUCell(dim, self.dim_inner)

        self.use_init_hidden_state = use_init_hidden_state
        if self.use_init_hidden_state:
            self.h0 = nn.Parameter(torch.zeros(1, self.dim_inner))
        else:
            self.register_parameter("h0", None)

    def forward(self, x, prev_state=None):
        B, T, _ = x.shape

        if prev_state is not None:
            h = prev_state
        else:
            if self.h0 is None:
                h = x.new_zeros(B, self.dim_inner)
            else:
                h = self.h0.expand(B, -1).to(dtype=x.dtype, device=x.device)

        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), h


class plainLSTM(nn.Module):
    """Sequential LSTM using explicit time-step loop (non-parallel)."""
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        use_init_hidden_state: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim_inner = int(dim * expansion_factor)
        self.cell = nn.LSTMCell(dim, self.dim_inner)

        self.use_init_hidden_state = use_init_hidden_state
        if self.use_init_hidden_state:
            self.h0 = nn.Parameter(torch.zeros(1, self.dim_inner))
            self.c0 = nn.Parameter(torch.zeros(1, self.dim_inner))
        else:
            self.register_parameter("h0", None)
            self.register_parameter("c0", None)

    def forward(self, x, prev_state=None):
        B, T, _ = x.shape

        if prev_state is not None:
            h, c = prev_state
        else:
            if self.h0 is None:
                h = x.new_zeros(B, self.dim_inner)
                c = x.new_zeros(B, self.dim_inner)
            else:
                h = self.h0.expand(B, -1).to(dtype=x.dtype, device=x.device)
                c = self.c0.expand(B, -1).to(dtype=x.dtype, device=x.device)

        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), (h, c)



# Causal Transformer block

class simpleTransformer(nn.Module):
    """
    Minimal causal self-attention block.
    Compatible with rnn_config kwargs (ignored via **kwargs).
    """
    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        expansion_factor: float = 1.0,      # ignored, kept for compatibility
        use_init_hidden_state: bool = False, # ignored
        **kwargs,
    ):
        super().__init__()
        if dim % n_head != 0:
            # Avoid silent shape issues
            raise ValueError(f"dim={dim} must be divisible by n_head={n_head}")

        self.mha = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, prev_state=None):
        # x: (B, T, D)
        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn_out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        y = self.ln(x + attn_out)
        return y, None


# Simple Mamba-style block

class simpleMamba(nn.Module):
    """
    Lightweight block inspired by Mamba using depthwise causal conv + gating.
    Compatible with rnn_config kwargs (ignored via **kwargs).
    """
    def __init__(
        self,
        dim: int,
        conv_kernel_size: int = 4,
        kernel_size: int | None = None,     # allow repo-style naming
        expansion_factor: float = 1.0,      # ignored, kept for compatibility
        use_init_hidden_state: bool = False,# ignored
        **kwargs,
    ):
        super().__init__()

        k = kernel_size if kernel_size is not None else conv_kernel_size

        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=k,
            groups=dim,
            padding=k - 1,   # causal-ish then crop
        )
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, prev_state=None):
        B, T, D = x.shape

        xg = self.in_proj(x)
        x_branch, gate = xg.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)          # (B, D, T)
        x_branch = self.conv(x_branch)[:, :, :T]     # crop to length T
        x_branch = x_branch.transpose(1, 2)          # (B, T, D)

        y = F.silu(x_branch) * torch.sigmoid(gate)
        y = self.ln(x + self.out_proj(y))
        return y, None