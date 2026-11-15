"""Shortcut objective (GPU): finite loss and gradients with a gradient-free target network; measured training FLOPs
relative to flow matching (~1.2 with fixed level counts, 4/3 padded); conversion from an FM model preserves the
network at every step size h; the fixed-level-count targets equal the padded ones."""
import copy

import pytest
import torch

from osr.experiments import SC_LOSS
from osr.flops import measure_step_flops
from osr.losses import fm_loss, shortcut_loss
from osr.models import build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
FM_CFG = dict(t_mean=-0.4, t_std=1.0, class_dropout=0.1)


def setup():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model("S", with_v_head=False).to(dev)
    tgt = copy.deepcopy(model).requires_grad_(False)
    cfg = dict(SC_LOSS)
    cfg.pop("ema_decay")
    cfg["_target"] = tgt
    return dev, model, tgt, cfg


def test_loss_and_flops():
    dev, model, tgt, cfg = setup()
    x, y = torch.randn(64, 4, 32, 32, device=dev), torch.randint(0, 1000, (64,), device=dev)
    for c in (cfg, dict(cfg, stratified=True)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = shortcut_loss(model, x, y, c, torch.Generator(device=dev).manual_seed(1))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in tgt.parameters())
        model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f_st = measure_step_flops(shortcut_loss, model, x, y, dict(cfg, stratified=True), seed=12345)
        f_pad = measure_step_flops(shortcut_loss, model, x, y, cfg, seed=12345)
        f_fm = measure_step_flops(fm_loss, model, x, y, FM_CFG, seed=12345)
    assert 1.15 < f_st / f_fm < 1.25
    assert abs(f_pad / f_fm - 4 / 3) < 0.02


def test_conversion_and_stratified_targets():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    fm = build_model("B", with_v_head=False).to(dev)
    with torch.no_grad():
        for p in fm.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    fm.eval()
    conv = copy.deepcopy(fm)
    conv.freeze_new_condition_embedders()
    conv.eval()
    z, yb = torch.randn(8, 4, 32, 32, device=dev), torch.randint(0, 1000, (8,), device=dev)
    zeros, ones = torch.zeros(8, device=dev), torch.ones(8, device=dev)
    with torch.no_grad():
        u_fm, _ = fm(z, zeros, zeros, ones, zeros, ones, yb, heads=("u",))
        for hv in (0.0, 0.25, 1.0):
            u_c, _ = conv(z, zeros, torch.full((8,), hv, device=dev), ones, zeros, ones, yb, heads=("u",))
            assert (u_c - u_fm).abs().max() < 1e-4
    # targets with fixed level counts (unconditional pass only for the smallest step) equal the padded ones
    n, nb, w = 16, 3, 1.5
    zz, ys = torch.randn(n, 4, 32, 32, device=dev), torch.randint(0, 1000, (n,), device=dev)
    h1 = torch.rand(n, device=dev)
    h1[:nb] = 0
    null = conv.num_classes
    with torch.no_grad():
        z2, o2 = torch.zeros(2 * n, device=dev), torch.ones(2 * n, device=dev)
        ua, _ = conv(torch.cat([zz, zz]), z2, torch.cat([h1, h1]), o2, z2, o2, torch.cat([ys, torch.full_like(ys, null)]), heads=("u",))
        base = torch.arange(n, device=dev) < nb
        pad = torch.where(base[:, None, None, None], ua[n:] + w * (ua[:n] - ua[n:]), ua[:n])
        zb, ob = torch.zeros(n + nb, device=dev), torch.ones(n + nb, device=dev)
        ub, _ = conv(torch.cat([zz, zz[:nb]]), zb, torch.cat([h1, h1[:nb]]), ob, zb, ob,
                     torch.cat([ys, torch.full_like(ys[:nb], null)]), heads=("u",))
        strat = torch.cat([ub[n:] + w * (ub[:nb] - ub[n:]), ub[nb:n]])
    assert (strat - pad).abs().max() < 1e-4
