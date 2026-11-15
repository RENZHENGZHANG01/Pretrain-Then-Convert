"""Explicit forward-mode JVP of the imfDiT u-path (trunk + u-head), for the iMF objective.

Computes du/dt along the tangent (dz = v_c, dh = 1) with hand-written tangent rules, as
plain tensor code under no_grad, so it can be torch.compile'd (PyTorch forward-AD cannot,
and in eager it is slow and memory hungry). Primal and tangent matmuls share one GEMM by
stacking [x; dx] along the batch. Numerically checked against torch.autograd.forward_ad
in tests/test_jvp_manual.py.

FLOP accounting is unaffected: the training cost of iMF is measured on the single-pass
reference implementation in losses.py (primal + tangent + v-head + backward, as JAX
executes it); the extra primal forward of this two-pass implementation is a memory/speed
recomputation, reported separately, like activation checkpointing.
"""
import math

import torch
import torch.nn.functional as F

from .models import apply_rope


def _lin2(x, dx, lin, dtype):
    """(x W^T + b, dx W^T) with one GEMM."""
    w = lin._flax_linear.weight.to(dtype)
    b = lin._flax_linear.bias
    n = x.shape[0]
    y = F.linear(torch.cat([x, dx]).to(dtype), w)
    y, dy = y[:n], y[n:]
    if b is not None:
        y = y + b.to(dtype)
    return y, dy


def _rms2(x, dx, norm):
    """RMSNorm (fp32 math, as in models.RMSNorm) and its tangent."""
    x, dx = x.float(), dx.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    dr = -(r ** 3) * (x * dx).mean(-1, keepdim=True)
    w = norm.weight.float()
    return (x * r) * w, (dx * r + x * dr) * w


def _silu2(a, da):
    s = torch.sigmoid(a)
    return a * s, da * (s * (1 + a * (1 - s)))


def _attn2(att, x, dx, cos, sin, dtype):
    B, T, _ = x.shape
    H, Dh = att.num_heads, att.head_dim
    q, dq = _lin2(x, dx, att.q_proj, dtype)
    k, dk = _lin2(x, dx, att.k_proj, dtype)
    v, dv = _lin2(x, dx, att.v_proj, dtype)
    q, dq = _rms2(q.view(B, T, H, Dh), dq.view(B, T, H, Dh), att.q_norm)
    k, dk = _rms2(k.view(B, T, H, Dh), dk.view(B, T, H, Dh), att.k_norm)
    q, dq = apply_rope(q, cos, sin), apply_rope(dq, cos, sin)  # RoPE is linear
    k, dk = apply_rope(k, cos, sin), apply_rope(dk, cos, sin)
    sc = 1.0 / math.sqrt(Dh)
    tr = lambda a: a.transpose(1, 2).to(dtype)  # [B,H,T,Dh]
    q, dq, k, dk = tr(q) * sc, tr(dq) * sc, tr(k), tr(dk)
    v, dv = v.view(B, T, H, Dh).transpose(1, 2), dv.view(B, T, H, Dh).transpose(1, 2)
    s = torch.matmul(q, k.transpose(-1, -2)).float()
    p = torch.softmax(s, dim=-1)
    ds = (torch.matmul(dq, k.transpose(-1, -2)) + torch.matmul(q, dk.transpose(-1, -2))).float()
    dp = p * (ds - (p * ds).sum(-1, keepdim=True))
    pv = torch.cat([p, dp]).to(dtype)
    o2 = torch.matmul(pv, torch.cat([v, v]))  # [p v ; dp v]
    o, do = o2[:B], o2[B:] + torch.matmul(p.to(dtype), dv)
    o = o.transpose(1, 2).reshape(B, T, H * Dh)
    do = do.transpose(1, 2).reshape(B, T, H * Dh)
    return _lin2(o, do, att.out_proj, dtype)


def _mlp2(mlp, x, dx, dtype):
    a, da = _lin2(x, dx, mlp.w1, dtype)
    b, db = _lin2(x, dx, mlp.w3, dtype)
    sa, dsa = _silu2(a.float(), da.float())
    h = sa * b.float()
    dh = dsa * b.float() + sa * db.float()
    return _lin2(h, dh, mlp.w2, dtype)


def block_jvp(blk, x, dx, cos, sin, dtype=torch.bfloat16):
    """One TransformerBlock: residual stream x (fp32) and tangent dx (fp32)."""
    n, dn = _rms2(x, dx, blk.norm1)
    a, da = _attn2(blk.attn, n, dn, cos, sin, dtype)
    sa = blk.attn_scale.float()
    x, dx = x + a.float() * sa, dx + da.float() * sa
    n, dn = _rms2(x, dx, blk.norm2)
    m, dm = _mlp2(blk.mlp, n, dn, dtype)
    sm = blk.mlp_scale.float()
    return x + m.float() * sm, dx + dm.float() * sm


def _temb2(emb, h, dh):
    """TimestepEmbedder(h) and its tangent along dh."""
    half = emb.frequency_embedding_size // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=h.device) / half)
    args = h[:, None].float() * freqs[None]
    e = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    de = torch.cat([-torch.sin(args), torch.cos(args)], dim=-1) * torch.cat([freqs, freqs])[None] * dh[:, None].float()
    l1, l2 = emb.mlp[0]._flax_linear, emb.mlp[2]._flax_linear
    a = F.linear(e, l1.weight.float(), l1.bias.float())
    da = F.linear(de, l1.weight.float())
    s, ds = _silu2(a, da)
    return F.linear(s, l2.weight.float(), l2.bias.float()), F.linear(ds, l2.weight.float())


def build_sequence_jvp(m, z, dz, h, dh, w, t_min, t_max, y):
    B = z.shape[0]
    he, dhe = _temb2(m.h_embedder, h, dh)
    cls = m.class_tokens + m.y_embedder(y)[:, None]
    om = m.omega_tokens + m.omega_embedder(1 - 1 / w)[:, None]
    tmn = m.t_min_tokens + m.cfg_t_start_embedder(t_min)[:, None]
    tmx = m.t_max_tokens + m.cfg_t_end_embedder(t_max)[:, None]
    tim = m.time_tokens + he[:, None]
    proj = m.x_embedder.proj
    xe = F.conv2d(z.float(), proj.weight.float(), proj.bias.float(), stride=proj.stride)
    dxe = F.conv2d(dz.float(), proj.weight.float(), None, stride=proj.stride)
    flat = lambda a: a.permute(0, 2, 3, 1).reshape(B, -1, a.shape[1])
    seq = torch.cat([cls, om, tmn, tmx, tim, flat(xe)], dim=1).float()
    zero = lambda a: torch.zeros_like(a, dtype=torch.float32)
    dseq = torch.cat([zero(cls), zero(om), zero(tmn), zero(tmx),
                      (torch.zeros_like(m.time_tokens)[None] + dhe[:, None]).float(), flat(dxe)], dim=1)
    return seq, dseq


@torch.no_grad()
def u_jvp(model, z, dz, h, dh, w, t_min, t_max, y, dtype=torch.bfloat16, compiled=False):
    """Returns (u, du) where du = J_u(z, h) [dz, dh] (trunk + u-head)."""
    seq, dseq = build_sequence_jvp(model, z, dz, h, dh, w, t_min, t_max, y)
    fn = _compiled_block_jvp() if compiled else block_jvp
    for blk in list(model.shared_blocks) + list(model.u_heads):
        seq, dseq = fn(blk, seq, dseq, model.rope_cos, model.rope_sin, dtype)
    fl = model.u_final_layer
    n, dn = _rms2(seq[:, model.prefix_tokens:], dseq[:, model.prefix_tokens:], fl.norm)
    lw, lb = fl.linear._flax_linear.weight.float(), fl.linear._flax_linear.bias.float()
    u = model.unpatchify(F.linear(n, lw, lb))
    du = model.unpatchify(F.linear(dn, lw))
    return u, du


_CBJ = [None]


def _compiled_block_jvp():
    if _CBJ[0] is None:
        _CBJ[0] = torch.compile(block_jvp, dynamic=False)
    return _CBJ[0]
