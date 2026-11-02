"""Deterministically generated, held-out synthetic token retrieval tasks.

Both tasks require reading a non-CLS token. The fixed task asks for a
position-specific bit. The content task asks for the bit at the flagged token.
"""
import torch


def make_batch(size: int, length: int, task: str, seed: int,
               fixed_position: int = 3, device: str = "cpu",
               marker_start: int = 1, marker_end: int | None = None):
    if size < 1 or length < 5 or fixed_position < 1 or fixed_position >= length:
        raise ValueError("invalid batch, length, or fixed_position")
    if marker_end is None:
        marker_end = length
    if not (1 <= marker_start < marker_end <= length):
        raise ValueError("invalid marker position interval")
    if task not in {"fixed", "content"}:
        raise ValueError("task must be fixed or content")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (size, length - 1), generator=generator)
    x = torch.zeros(size, length, 5)
    x[:, 1:, 0] = (bits == 0).float()
    x[:, 1:, 1] = (bits == 1).float()
    x[:, 1:, 3] = torch.randn(size, length - 1, generator=generator) * 0.15
    x[:, 1:, 4] = torch.randn(size, length - 1, generator=generator) * 0.15
    x[:, 0, 4] = 1.0  # distinguish the query token
    if task == "fixed":
        index = torch.full((size,), fixed_position, dtype=torch.long)
    else:
        index = torch.randint(marker_start, marker_end, (size,), generator=generator)
        x[torch.arange(size), index, 2] = 1.0
    y = bits[torch.arange(size), index - 1]
    return x.to(device), y.to(device)
