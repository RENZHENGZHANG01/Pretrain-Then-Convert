"""FLOP accounting: forward-mode AD tangent matmuls are counted, per-sample training FLOPs match osr/flops_table.json,
and the measured forward pass matches the analytic estimate."""
import json

import torch
from torch.autograd import forward_ad as fwAD
from torch.utils.flop_counter import FlopCounterMode

from osr.flops import analytic_forward_flops, measure_step_flops
from osr.losses import fm_loss, imf_loss
from osr.models import build_model
from osr.paths import FLOPS_TABLE


def test_tangent_matmuls_are_counted():
    lin, x = torch.nn.Linear(64, 32), torch.randn(8, 64)
    with FlopCounterMode(display=False) as fc:
        lin(x)
    plain = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc, fwAD.dual_level():
        fwAD.unpack_dual(lin(fwAD.make_dual(x, torch.randn_like(x))))
    assert fc.get_total_flops() >= 2 * plain


def test_step_flops_match_table():
    table = json.load(open(FLOPS_TABLE))
    torch.manual_seed(0)
    B = 4
    x, y = torch.randn(B, 4, 32, 32), torch.randint(0, 1000, (B,))
    m_fm, m_imf = build_model("S", with_v_head=False), build_model("S", with_v_head=True)
    f_fm = measure_step_flops(fm_loss, m_fm, x, y, {}) / B
    f_imf = measure_step_flops(imf_loss, m_imf, x, y, {}) / B
    assert abs(f_fm / table["S"]["fm"] - 1) < 0.01
    assert abs(f_imf / table["S"]["imf"] - 1) < 0.01
    assert 2.7 < f_imf / f_fm < 3.0  # kappa = 13/6 + 8/D for depth D = 12
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        zeros, ones = torch.zeros(B), torch.ones(B)
        m_fm(x, zeros, zeros, ones, zeros, ones, y, heads=("u",))
    # the analytic estimate counts the transformer-block matmuls only (attention scores, embedders and the final layer
    # are approximated), so it agrees to ~10% for the smallest sizes
    assert abs(fc.get_total_flops() / B / analytic_forward_flops(m_fm) - 1) < 0.15
