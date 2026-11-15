"""The hand-written JVP (osr/jvp.py) equals forward-mode AD, and the iMF loss gives the same gradients with the
single-pass reference ('single') and the two-pass manual JVP ('manual'), in fp32 on CPU."""
import torch
from torch.autograd import forward_ad as fwAD

from osr.jvp import u_jvp
from osr.losses import imf_loss
from osr.models import build_model, jvp_mode


def rel(a, b):
    return float((a - b).norm() / b.norm())


def test_manual_jvp_and_gradients():
    torch.manual_seed(0)
    m = build_model("S")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    B = 4
    z = torch.randn(B, 4, 32, 32)
    dz = torch.randn_like(z)
    t = torch.rand(B)
    h, dh = torch.rand(B) * t, torch.ones(B)
    w, tmin, tmax = 1 + 7 * torch.rand(B), 0.5 * torch.rand(B), 0.5 + 0.5 * torch.rand(B)
    y = torch.randint(0, 1001, (B,))
    with fwAD.dual_level(), jvp_mode():
        seq = m.trunk(fwAD.make_dual(z, dz), fwAD.make_dual(h, dh), w, tmin, tmax, y, attn="math")
        u_ref, du_ref = fwAD.unpack_dual(m.head(seq, "u", attn="math"))
    u_man, du_man = u_jvp(m, z, dz, h, dh, w, tmin, tmax, y, dtype=torch.float32)
    assert rel(u_man, u_ref.detach()) < 1e-5
    assert rel(du_man, du_ref.detach()) < 1e-4
    x, yy = torch.randn(8, 4, 32, 32), torch.randint(0, 1000, (8,))
    grads = {}
    for impl in ["single", "manual"]:
        m.zero_grad()
        loss, _ = imf_loss(m, x, yy, {"jvp_impl": impl, "jvp_dtype": torch.float32}, torch.Generator().manual_seed(3))
        loss.backward()
        grads[impl] = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert rel(grads["manual"], grads["single"]) < 1e-4
