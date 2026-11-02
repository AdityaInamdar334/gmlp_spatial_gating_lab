"""Print the SGU shape contract and a causal masking example."""
import torch
from .models import SpatialGatingUnit, GMLPBlock


def main():
    torch.manual_seed(7)
    z=torch.randn(2, 6, 16)
    sgu=SpatialGatingUnit(16,6)
    out=sgu(z)
    causal=torch.tril(torch.ones(6,6))
    layer=GMLPBlock(8,16,6)
    x=torch.randn(2,6,8)
    x2=x.clone()
    x2[:,4:]+=50
    a=layer(x, causal)
    b=layer(x2, causal)
    print("SGU input:", tuple(z.shape), "SGU output:", tuple(out.shape))
    print("Causal prefix absolute difference:", (a[:,:4]-b[:,:4]).abs().max().item())
    print("SGU init gate bias mean:", sgu.bias.mean().item())


if __name__ == "__main__":
    main()
