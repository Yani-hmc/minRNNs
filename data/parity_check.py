"""
Creates sequences where the model must determine
whether the number of specific symbols is even or odd.
"""

import torch
from attrdict import AttrDict

VOCAB_SIZE = 3
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value


class ParityCheckSampler:
    def __init__(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def sample(self, batch_size, length, device="cpu"):
        batch = AttrDict()

        # Randomly generate tokens (skipping the SOS and EOS token values)
        tokens = torch.randint(
            low=0,
            high=2,
            size=(batch_size, length),
        )

        batch.x = torch.cat(
            (tokens, (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = tokens.sum(-1, keepdim=True) % 2

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
