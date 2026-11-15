"""Linear layers under jvp_mode: one tangent matmul per primal matmul, correct tangents, unchanged gradients."""
import torch
from torch.autograd import forward_ad as fwAD
from torch.utils.flop_counter import FlopCounterMode

from osr.models import TorchLinear, jvp_mode


def test_jvp_linear():
    torch.manual_seed(0)
    lin = TorchLinear(64, 32, bias=True)
    x = torch.randn(4, 5, 64)
    dx = torch.randn_like(x)
    with FlopCounterMode(display=False) as fc:
        lin(x)
    f0 = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc, fwAD.dual_level(), jvp_mode():
        y, dy = fwAD.unpack_dual(lin(fwAD.make_dual(x, dx)))
    assert fc.get_total_flops() == 2 * f0
    assert (dy - torch.nn.functional.linear(dx, lin._flax_linear.weight)).abs().max() < 1e-5
    xg = x.clone().requires_grad_(True)
    with fwAD.dual_level(), jvp_mode():
        y, _ = fwAD.unpack_dual(lin(fwAD.make_dual(xg, dx)))
    (y ** 2).sum().backward()
    g1 = (lin._flax_linear.weight.grad.clone(), lin._flax_linear.bias.grad.clone(), xg.grad.clone())
    lin.zero_grad()
    xg.grad = None
    (lin(xg) ** 2).sum().backward()
    g2 = (lin._flax_linear.weight.grad, lin._flax_linear.bias.grad, xg.grad)
    for a, b in zip(g1, g2):
        assert (a - b).abs().max() < 1e-5
