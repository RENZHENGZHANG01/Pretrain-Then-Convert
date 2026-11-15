"""Measured training FLOPs per step (torch.utils.flop_counter.FlopCounterMode).

Counts matmul-class ops (mm/addmm/bmm/baddbmm/conv/SDPA), i.e. the usual convention
(elementwise ops are not counted). Forward-mode AD (JVP) tangent matmuls are dispatched
as ordinary aten ops and are therefore counted too (verified in tests/test_flops.py).
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def measure_step_flops(loss_fn, model, x, y, cfg, seed=0):
    """FLOPs of one full training step (loss forward incl. no-grad guidance passes and JVP,
    plus backward). Does not touch optimizer state; gradients are zeroed afterwards."""
    from . import models
    gen = torch.Generator(device=x.device).manual_seed(seed)
    cfg = dict(cfg, jvp_impl="single")  # algorithmic FLOPs: single-pass JVP (as in JAX)
    models._ATTN_OVERRIDE[0] = "math"
    try:
        with FlopCounterMode(display=False) as fc:
            loss, _ = loss_fn(model, x, y, cfg, gen)
            loss.backward()
    finally:
        models._ATTN_OVERRIDE[0] = None
    model.zero_grad(set_to_none=True)
    return int(fc.get_total_flops())


def measure_executed_flops(loss_fn, model, x, y, cfg, seed=0):
    """FLOPs actually executed by the configured implementation (reported, not used as C)."""
    from . import models
    gen = torch.Generator(device=x.device).manual_seed(seed)
    models._ATTN_OVERRIDE[0] = "math"
    try:
        with FlopCounterMode(display=False) as fc:
            loss, _ = loss_fn(model, x, y, cfg, gen)
            loss.backward()
    finally:
        models._ATTN_OVERRIDE[0] = None
    model.zero_grad(set_to_none=True)
    return int(fc.get_total_flops())


def analytic_forward_flops(model, n_tokens=None):
    """2 * (non-embedding matmul params) * T + attention (4 * T^2 * width per block),
    for the u-path (inference network). Used only as a cross-check of the counter."""
    T = n_tokens or (model.x_embedder.num_patches + model.prefix_tokens)
    blocks = len(model.shared_blocks) + len(model.u_heads)
    d = model.hidden_size
    per_block_params = sum(p.numel() for n, p in model.shared_blocks[0].named_parameters() if p.dim() == 2)
    return blocks * (2 * per_block_params * T + 4 * T * T * d)
