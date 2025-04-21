"""Modified from <https://github.com/sculptor2022/sculptor/blob/main/utils.py>."""

import torch
from numpy.typing import ArrayLike


def to_tensor(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.tensor(x, dtype=dtype)
