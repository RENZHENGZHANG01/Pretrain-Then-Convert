"""imfDiT: parameter counts per size; equality with the official PyTorch imfDiT on the same weights (only if the
official code is available, OSR_IMF_TORCH=<clone of github.com/Lyy-iiis/imeanflow, branch torch>); forward-mode AD
through trunk + u-head against finite differences, with unchanged reverse-mode gradients."""
import json
import os
import sys

import pytest
import torch
from torch.autograd import forward_ad as fwAD

from osr.models import build_model, count_params
from osr.paths import FLOPS_TABLE


def inputs(B=3):
    torch.manual_seed(1)
    x, t = torch.randn(B, 4, 32, 32), torch.rand(B)
    return x, t, torch.rand(B) * t, 1 + 7 * torch.rand(B), 0.5 * torch.rand(B), 0.5 + 0.5 * torch.rand(B), torch.randint(0, 1001, (B,))


def test_parameter_counts():
    for size, row in json.load(open(FLOPS_TABLE)).items():
        assert count_params(build_model(size)) == row["N_inference"], size


@pytest.mark.skipif(not os.environ.get("OSR_IMF_TORCH"), reason="set OSR_IMF_TORCH to the official imeanflow (torch) clone")
def test_equal_to_official():
    sys.path.insert(0, os.environ["OSR_IMF_TORCH"])
    from models import imfDiT as off
    ref = off.imfDiT_B_2(input_size=32, in_channels=4, num_classes=1000, eval_mode=False)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.05)  # the official init has zero gates / final layers
    ours = build_model("B")
    ours.load_state_dict(ref.state_dict(), strict=True)
    x, t, h, w, tmin, tmax, y = inputs()
    with torch.no_grad():
        u0, v0 = ref(x, t, h, w, tmin, tmax, y)
        u1, v1 = ours(x, t, h, w, tmin, tmax, y, attn="math")
        u2, _ = ours(x, t, h, w, tmin, tmax, y, attn="sdpa")
    assert (u0 - u1).abs().max() < 1e-4 and (u0 - u2).abs().max() < 1e-4 and (v0 - v1).abs().max() < 1e-4


def test_forward_ad_and_gradients():
    torch.manual_seed(0)
    m = build_model("S")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    x, t, h, w, tmin, tmax, y = inputs()
    tz = torch.randn_like(x)
    with fwAD.dual_level():
        seq = m.trunk(fwAD.make_dual(x, tz), fwAD.make_dual(h, torch.ones_like(h)), w, tmin, tmax, y, attn="math")
        u, dudt = fwAD.unpack_dual(m.head(seq, "u", attn="math"))
        seq_p = fwAD.unpack_dual(seq).primal
    v = m.head(seq_p, "v", attn="sdpa")
    assert u.requires_grad and v.requires_grad
    eps = 1e-3
    with torch.no_grad():
        up = m(x + eps * tz, t, h + eps, w, tmin, tmax, y, heads=("u",), attn="math")[0]
        um = m(x - eps * tz, t, h - eps, w, tmin, tmax, y, heads=("u",), attn="math")[0]
    fd = (up - um) / (2 * eps)
    assert float((fd - dudt).norm() / fd.norm()) < 1e-2
    ((u ** 2).mean() + (v ** 2).mean()).backward()
    g = m.shared_blocks[0].attn.q_proj._flax_linear.weight.grad.clone()
    m.zero_grad()
    u_r, v_r = m(x, t, h, w, tmin, tmax, y, attn="math")
    ((u_r ** 2).mean() + (v_r ** 2).mean()).backward()
    assert float((m.shared_blocks[0].attn.q_proj._flax_linear.weight.grad - g).norm() / g.norm()) < 1e-4
