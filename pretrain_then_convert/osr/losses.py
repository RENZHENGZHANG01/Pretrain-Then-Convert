"""Training objectives.

fm_loss  : plain flow matching on the same network (multi-step models and the pretraining phase).
           z_t = (1-t) x + t e, target v = e - x, h = 0, omega = 1, interval [0, 1].
imf_loss : improved MeanFlow (native one-step training and the conversion phase). A line-by-line
           port of the official JAX objective (github.com/Lyy-iiis/imeanflow, imf.py):
             - (t, r) ~ logit-normal(P_mean, P_std), t = max, r = min; the first
               `data_proportion` of the batch has r = t (flow-matching samples)
             - omega ~ power distribution on [1, 1 + s_max] (beta = cfg_beta)
             - CFG interval t_min ~ U(0, .5), t_max ~ U(.5, 1); FM samples use [0, 1]
             - guided target v_g = v_t + (1 - 1/w)(v_c - v_u) with the model's own v-head
             - class dropout applies to the *first* num_drop samples (official quirk)
             - JVP tangent = predicted v_c (iMF), compound V = u + (t - r) sg(du/dt)
             - adaptive weighting loss / sg(loss + eps)^p on both the u- and v-head losses
           Equivalences used for speed (exact, not approximations):
             - the three guidance forwards are one batched no-grad call, and the
               interval-adjusted v_c is recomputed only for the r != t half (for r = t
               samples the interval is [0, 1], so it equals the first v_c);
             - the JVP runs through trunk + u-head only; the v-head is applied to the
               trunk primal, as XLA would dead-code-eliminate in JAX.
           Two implementations of the JVP step (cfg["jvp_impl"]):
             - "single": one forward-mode-AD pass with grad tracking (torch.autograd.forward_ad).
               This is the reference; training FLOPs are always *measured on it* (see flops.py).
             - "manual" (default): explicit tangent rules under no_grad (osr/jvp.py, compiled)
               to get du/dt, then an ordinary compiled forward/backward of u and v with flash
               attention. Same math; the extra primal forward is a recomputation (not counted,
               like activation checkpointing). Equivalence is tested in tests/test_jvp_manual.py.
"""
import math

import torch
from torch.autograd import forward_ad as fwAD

from .jvp import u_jvp
from .models import jvp_mode


def logit_normal(n, m, s, gen, device):
    return torch.sigmoid(torch.randn(n, generator=gen, device=device) * s + m)


def sample_cfg_scale(n, beta, s_max, gen, device):
    u = torch.rand(n, generator=gen, device=device)
    if beta == 1.0:
        return torch.exp(u * math.log1p(s_max))
    log_base = (1.0 - beta) * math.log1p(s_max)
    return torch.exp(torch.log1p(u * math.expm1(log_base)) / (1.0 - beta))


def _b(v):  # [B] -> [B,1,1,1]
    return v.view(-1, 1, 1, 1)


def fm_loss(model, x, y, cfg, gen):
    """cfg keys: t_mean, t_std, class_dropout."""
    B, dev = x.shape[0], x.device
    t = logit_normal(B, cfg.get("t_mean", -0.4), cfg.get("t_std", 1.0), gen, dev)
    e = torch.randn(x.shape, generator=gen, device=dev)
    z = (1 - _b(t)) * x + _b(t) * e
    drop = torch.rand(B, generator=gen, device=dev) < cfg.get("class_dropout", 0.1)
    y = torch.where(drop, torch.full_like(y, model.num_classes), y)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    u, _ = model(z, t, zeros, ones, zeros, ones, y, heads=("u",), attn="sdpa")
    loss = ((u - (e - x)) ** 2).mean()
    return loss, {"loss": loss.detach(), "loss_fm": loss.detach()}


def imf_loss(model, x, y, cfg, gen):
    """cfg keys: P_mean, P_std, data_proportion, cfg_beta, s_max, class_dropout, norm_p, norm_eps."""
    B, dev = x.shape[0], x.device
    P_mean, P_std = cfg.get("P_mean", -0.4), cfg.get("P_std", 1.0)
    n_fm = int(B * cfg.get("data_proportion", 0.5))
    null = model.num_classes

    # ---- sample (t, r), noise, CFG scale and interval -------------------------------
    t1 = logit_normal(B, P_mean, P_std, gen, dev)
    t2 = logit_normal(B, P_mean, P_std, gen, dev)
    t, r = torch.maximum(t1, t2), torch.minimum(t1, t2)
    fm_mask = torch.arange(B, device=dev) < n_fm
    r = torch.where(fm_mask, t, r)
    e = torch.randn(x.shape, generator=gen, device=dev)
    z = (1 - _b(t)) * x + _b(t) * e
    v_t = e - x
    t_min = torch.rand(B, generator=gen, device=dev) * 0.5
    t_max = 0.5 + torch.rand(B, generator=gen, device=dev) * 0.5
    t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
    t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
    omega = sample_cfg_scale(B, cfg.get("cfg_beta", 1.0), cfg.get("s_max", 7.0), gen, dev)
    in_iv = (t >= t_min) & (t <= t_max)
    w_adj = torch.where(in_iv, omega, torch.ones_like(omega))

    # ---- guidance targets (no grad): [cond @omega | uncond @1 | cond @w_adj for r!=t] --
    mf = slice(n_fm, B)
    n_mf = B - n_fm
    with torch.no_grad():
        zz = torch.cat([z, z, z[mf]])
        tt = torch.cat([t, t, t[mf]])
        ww = torch.cat([omega, torch.ones_like(omega), w_adj[mf]])
        yy = torch.cat([y, torch.full_like(y, null), y[mf]])
        n_all = 2 * B + n_mf
        zeros, ones = torch.zeros(n_all, device=dev), torch.ones(n_all, device=dev)
        _, v_all = model(zz, tt, zeros, ww, zeros, ones, yy, heads=("v",), attn="sdpa")
        v_all = v_all.float()
        v_c, v_u, v_c_mf = v_all[:B], v_all[B:2 * B], v_all[2 * B:]
        v_g_fm = v_t + _b(1 - 1 / omega) * (v_c - v_u)
        v_c_adj = torch.cat([v_c[:n_fm], v_c_mf])
        v_g = v_t + _b(1 - 1 / w_adj) * (v_c_adj - v_u)
        v_g = torch.where(_b(fm_mask), v_g_fm, v_g)
        # class dropout: drop the first num_drop samples (as in the official code)
        num_drop = int((torch.rand(B, generator=gen, device=dev) < cfg.get("class_dropout", 0.1)).sum())
        drop = torch.arange(B, device=dev) < num_drop
        y_in = torch.where(drop, torch.full_like(y, null), y)
        v_g = torch.where(_b(drop), v_t, v_g)

    # ---- JVP through trunk + u-head; v-head on the trunk primal ----------------------
    h = t - r
    if cfg.get("jvp_impl", "manual") == "single":
        with fwAD.dual_level(), jvp_mode():
            z_d = fwAD.make_dual(z, v_c_adj)
            h_d = fwAD.make_dual(h, torch.ones_like(h))
            seq = model.trunk(z_d, h_d, omega, t_min, t_max, y_in, attn="math")
            u_d = model.head(seq, "u", attn="math")
            u, du_dt = fwAD.unpack_dual(u_d)
            seq_p = fwAD.unpack_dual(seq).primal
        v = model.head(seq_p, "v", attn="sdpa")
    else:
        _, du_dt = u_jvp(model, z, v_c_adj, h, torch.ones_like(h), omega, t_min, t_max, y_in,
                         dtype=cfg.get("jvp_dtype", torch.bfloat16), compiled=getattr(model, "_compiled", False))
        u, v = model(z, t, h, omega, t_min, t_max, y_in, heads=("u", "v"), attn="sdpa")
    du_dt = du_dt.detach().float()
    V = u.float() + _b(h) * du_dt

    p, eps = cfg.get("norm_p", 1.0), cfg.get("norm_eps", 0.01)

    def adp(l):
        return l / (l.detach() + eps) ** p

    l_u = ((V - v_g) ** 2).sum(dim=(1, 2, 3))
    l_v = ((v.float() - v_g) ** 2).sum(dim=(1, 2, 3))
    loss = (adp(l_u) + adp(l_v)).mean()
    with torch.no_grad():
        logs = {
            "loss": loss.detach(),
            "loss_u": ((V - v_g) ** 2).mean(),
            "loss_v": ((v.float() - v_g) ** 2).mean(),
            "loss_u_fm": ((V[:n_fm] - v_g[:n_fm]) ** 2).mean(),
            "loss_u_mf": ((V[n_fm:] - v_g[n_fm:]) ** 2).mean() if n_mf > 0 else torch.zeros((), device=dev),
        }
    return loss, logs


def shortcut_loss(model, x, y, cfg, gen):
    """Shortcut models (Frans et al., ICLR 2025, arXiv:2410.12557), the published recipe on our network: the u-head
    with interval length h is the shortcut s(x_t, t, d = h), i.e. the average velocity over [t - h, t] (t is
    inferred from z, as everywhere in this architecture). Guidance as in the paper: CFG at a fixed scale inside the
    bootstrap targets at the smallest step, none at sampling (omega input = 1, interval [0, 1]).
      - first (1 - k) of the batch: flow matching at h = 0 toward e - x (logit-normal t as our FM runs), class
        dropout on these samples
      - last k: self-consistency. d = 2^l / m with l ~ U{0, ..., log2(m) - 1} (d in [1/m, 1/2]), t = j d with
        j ~ U{2, ..., 1/d}; two steps of size d with the target network (EMA of the weights, cfg["_target"]),
        queried at h = d, or at h = 0 with CFG scale cfg_scale when d = 1/m; target (u1 + u2) / 2 for the
        prediction at h = 2d.
    cfg["stratified"] (the shortcut runs of the paper): fixed level counts in every batch (level of bootstrap sample i is
    i * log2(m) // n_sc, smallest step first), as the official implementation, and the unconditional bootstrap
    forwards only for the smallest-step samples, so the measured FLOPs are the algorithmic ones (~1.19x FM).
    Without it (GEN2 runs): levels drawn i.i.d. and, for fixed tensor shapes, the unconditional forwards run for
    every bootstrap sample and are used only at the smallest step; the measured FLOPs include them (4/3 x FM).
    cfg keys: t_mean, t_std, class_dropout, k (0.25), m (128), cfg_scale (1.5), stratified; "_target" (trainer)."""
    B, dev = x.shape[0], x.device
    null = model.num_classes
    tgt = cfg.get("_target") or model
    k, m, w = cfg.get("k", 0.25), int(cfg.get("m", 128)), cfg.get("cfg_scale", 1.5)
    n_sc = int(round(B * k))
    n_fm = B - n_sc
    n_lvl = int(round(math.log2(m)))
    t_fm = logit_normal(n_fm, cfg.get("t_mean", -0.4), cfg.get("t_std", 1.0), gen, dev)
    strat = bool(cfg.get("stratified", False))
    if strat:
        lvl = (torch.arange(n_sc, device=dev) * n_lvl) // n_sc
        nb = -(-n_sc // n_lvl)  # number of smallest-step samples (the first nb)
    else:
        lvl = torch.randint(0, n_lvl, (n_sc,), generator=gen, device=dev)
    d = torch.pow(2.0, lvl.float()) / m
    n_mult = torch.round(1.0 / d)
    j = 2 + torch.floor(torch.rand(n_sc, generator=gen, device=dev) * (n_mult - 1))
    t_sc = j * d
    t = torch.cat([t_fm, t_sc])
    e = torch.randn(x.shape, generator=gen, device=dev)
    z = (1 - _b(t)) * x + _b(t) * e
    drop = (torch.arange(B, device=dev) < n_fm) & (torch.rand(B, generator=gen, device=dev) < cfg.get("class_dropout", 0.1))
    y_in = torch.where(drop, torch.full_like(y, null), y)
    with torch.no_grad():  # bootstrap targets from the target network
        zs, ys = z[n_fm:], y[n_fm:]
        base = lvl == 0
        h1 = torch.where(base, torch.zeros_like(d), d)
        zeros, ones = torch.zeros(2 * n_sc, device=dev), torch.ones(2 * n_sc, device=dev)
        hh = torch.cat([h1, h1])
        yy = torch.cat([ys, torch.full_like(ys, null)])

        if strat:
            zeros, ones = torch.zeros(n_sc + nb, device=dev), torch.ones(n_sc + nb, device=dev)
            hh = torch.cat([h1, h1[:nb]])
            yy = torch.cat([ys, torch.full_like(ys[:nb], null)])

        def step_u(zz_):
            if strat:  # unconditional pass only for the smallest-step samples
                u_all, _ = tgt(torch.cat([zz_, zz_[:nb]]), zeros, hh, ones, zeros, ones, yy, heads=("u",), attn="sdpa")
                u_all = u_all.float()
                u_c, u_u = u_all[:n_sc], u_all[n_sc:]
                return torch.cat([u_u + w * (u_c[:nb] - u_u), u_c[nb:]])
            u_all, _ = tgt(torch.cat([zz_, zz_]), zeros, hh, ones, zeros, ones, yy, heads=("u",), attn="sdpa")
            u_all = u_all.float()
            u_c, u_u = u_all[:n_sc], u_all[n_sc:]
            return torch.where(_b(base), u_u + w * (u_c - u_u), u_c)

        u1 = step_u(zs)
        u2 = step_u(zs - _b(d) * u1)
        target_sc = (u1 + u2) / 2
    h = torch.cat([torch.zeros(n_fm, device=dev), 2 * d])
    ones_b, zeros_b = torch.ones(B, device=dev), torch.zeros(B, device=dev)
    u, _ = model(z, t, h, ones_b, zeros_b, ones_b, y_in, heads=("u",), attn="sdpa")
    u = u.float()
    l_fm = ((u[:n_fm] - (e - x)[:n_fm]) ** 2).mean(dim=(1, 2, 3))
    l_sc = ((u[n_fm:] - target_sc) ** 2).mean(dim=(1, 2, 3))
    loss = torch.cat([l_fm, l_sc]).mean()
    return loss, {"loss": loss.detach(), "loss_fm": l_fm.mean().detach(), "loss_sc": l_sc.mean().detach()}


LOSSES = {"fm": fm_loss, "imf": imf_loss, "sc": shortcut_loss}
