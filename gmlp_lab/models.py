"""Small batch-first gMLP and attention baselines for controlled experiments."""
import torch
from torch import nn


class SpatialGatingUnit(nn.Module):
    """Learned, input-independent [position, position] mixing shared across channels.

    Input: [batch, tokens, channels], with an even channel dimension.
    The learned spatial matrix can be masked at application time.
    """
    def __init__(self, channels: int, max_length: int, init: str = "stable",
                 disable_mixing: bool = False):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("channels must be positive and even")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        if init not in {"stable", "random"}:
            raise ValueError("init must be stable or random")
        self.norm = nn.LayerNorm(channels // 2)
        self.weight = nn.Parameter(torch.empty(max_length, max_length))
        self.bias = nn.Parameter(torch.empty(max_length))
        with torch.no_grad():
            if init == "stable":
                nn.init.uniform_(self.weight, -0.01, 0.01)
                self.bias.fill_(1.0)
            else:
                nn.init.xavier_uniform_(self.weight)
                self.bias.zero_()
            if disable_mixing:
                self.weight.zero_()
                self.bias.fill_(1.0)
        if disable_mixing:
            self.weight.requires_grad_(False)
            self.bias.requires_grad_(False)
        self.max_length = max_length

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None,
                extra_gate: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != self.norm.normalized_shape[0] * 2:
            raise ValueError("expected [batch, tokens, even_channels]")
        n = z.shape[1]
        if n > self.max_length:
            raise ValueError("sequence exceeds max_length")
        carry, gate_input = z.chunk(2, dim=-1)
        gate_input = self.norm(gate_input)
        w = self.weight[:n, :n]
        if mask is not None:
            if mask.shape != (n, n):
                raise ValueError("mask must be [tokens, tokens]")
            w = w * mask.to(device=w.device, dtype=w.dtype)
        mixed = torch.einsum("ij,bjc->bic", w, gate_input)
        gate = mixed + self.bias[:n][None, :, None]
        if extra_gate is not None:
            if extra_gate.shape != gate.shape:
                raise ValueError("extra_gate must match half-channel gate shape")
            gate = gate + extra_gate
        return carry * gate


class TinyAttention(nn.Module):
    """Single-head, content-dependent branch added to the SGU gate (aMLP style)."""
    def __init__(self, width: int, output: int, head_dim: int = 16):
        super().__init__()
        self.qkv = nn.Linear(width, 3 * head_dim)
        self.out = nn.Linear(head_dim, output)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.to(dtype=torch.bool, device=x.device)[None, :, :], float("-inf"))
        probs = scores.softmax(-1)
        return self.out(torch.einsum("bij,bjd->bid", probs, v))


class GMLPBlock(nn.Module):
    def __init__(self, width: int, expansion: int, max_length: int,
                 init: str = "stable", disable_mixing: bool = False,
                 tiny_attention_dim: int = 0):
        super().__init__()
        if expansion % 2:
            raise ValueError("expansion must be even")
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, expansion)
        self.act = nn.GELU()
        self.gate = SpatialGatingUnit(expansion, max_length, init, disable_mixing)
        self.out_proj = nn.Linear(expansion // 2, width)
        self.tiny = TinyAttention(width, expansion // 2, tiny_attention_dim) if tiny_attention_dim else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.norm(x)
        z = self.act(self.in_proj(h))
        extra = self.tiny(h, mask) if self.tiny is not None else None
        return x + self.out_proj(self.gate(z, mask, extra))


class GMLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 5, length: int = 12, width: int = 48,
                 expansion: int = 96, layers: int = 2, init: str = "stable",
                 disable_mixing: bool = False, tiny_attention_dim: int = 0):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([GMLPBlock(width, expansion, length, init,
                                disable_mixing, tiny_attention_dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def features(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, mask)
        return self.norm(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.head(self.features(x, mask)[:, 0])


class AttentionClassifier(nn.Module):
    def __init__(self, input_dim: int = 5, length: int = 12, width: int = 48,
                 layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width*2,
            dropout=0.0, batch_first=True, norm_first=True,
            activation='gelu') for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor):
        h = self.embed(x) + self.position[:, :x.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)[:, 0])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
