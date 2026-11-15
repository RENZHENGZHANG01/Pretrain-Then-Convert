"""Function-preserving conversion: after converting an FM model, u(z, h, omega, [t_min, t_max]) equals the FM velocity
u(z, h=0, omega=1, [0, 1]) for every input, and the auxiliary head v (a copy of u) equals it too."""
import torch

from osr.models import build_model


def test_conversion_preserves_function():
    torch.manual_seed(0)
    fm = build_model("S", with_v_head=False)
    with torch.no_grad():
        for p in fm.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # as if trained: non-zero gates and final layers
    cv = build_model("S", with_v_head=False)
    cv.load_state_dict(fm.state_dict())
    cv.add_v_head_from_u()
    cv.freeze_new_condition_embedders()
    imf = build_model("S", with_v_head=True)
    imf.load_state_dict(cv.state_dict(), strict=True)
    B = 6
    z, y, t = torch.randn(B, 4, 32, 32), torch.randint(0, 1001, (B,)), torch.rand(B)
    zeros, ones = torch.zeros(B), torch.ones(B)
    with torch.no_grad():
        u_fm, _ = fm(z, t, zeros, ones, zeros, ones, y, heads=("u",))
        h, w = torch.rand(B), 1 + 7 * torch.rand(B)
        a, b = 0.5 * torch.rand(B), 0.5 + 0.5 * torch.rand(B)
        u_cv, v_cv = imf(z, t, h, w, a, b, y)
    assert (u_cv - u_fm).abs().max() < 1e-5
    assert (v_cv - u_fm).abs().max() < 1e-5
    # the new embedders output a constant; their last layer starts at zero and stays trainable
    assert float(imf.h_embedder.mlp[2]._flax_linear.weight.detach().norm()) == 0.0
    assert imf.h_embedder.mlp[2]._flax_linear.weight.requires_grad
