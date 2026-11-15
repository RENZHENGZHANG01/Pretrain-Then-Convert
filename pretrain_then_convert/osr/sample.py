"""Samplers.

meanflow_sample : k-NFE MeanFlow / iMF sampler (u-head, h = t - r, CFG-as-input omega and
                  interval). k = 1 is the one-step generator. Matches official generate().
fm_sample       : multi-step flow-matching ODE (Euler or Heun) with standard two-pass CFG
                  restricted to a guidance interval [t_lo, t_hi] (w = 1 outside), using the
                  u-head at h = 0, omega = 1 input (multi-step models).
"""
import torch


@torch.no_grad()
def meanflow_sample(model, z, y, nfe=1, omega=1.0, t_min=0.0, t_max=1.0):
    B, dev = z.shape[0], z.device
    ts = torch.linspace(1.0, 0.0, nfe + 1, device=dev)
    full = lambda v: torch.full((B,), float(v), device=dev)
    for i in range(nfe):
        t, r = ts[i], ts[i + 1]
        u, _ = model(z, full(t), full(t - r), full(omega), full(t_min), full(t_max), y, heads=("u",))
        z = z - (t - r) * u.float()
    return z


@torch.no_grad()
def fm_velocity(model, z, y, w, null):
    """two-pass CFG velocity; w is a python float (1.0 -> single conditional pass)."""
    B, dev = z.shape[0], z.device
    zeros, ones = torch.zeros(B, device=dev), torch.ones(B, device=dev)
    if w == 1.0:
        return model(z, zeros, zeros, ones, zeros, ones, y, heads=("u",))[0].float()
    zz, yy = torch.cat([z, z]), torch.cat([y, torch.full_like(y, null)])
    z2, o2 = torch.zeros(2 * B, device=dev), torch.ones(2 * B, device=dev)
    v = model(zz, z2, z2, o2, z2, o2, yy, heads=("u",))[0].float()
    vc, vu = v[:B], v[B:]
    return vu + w * (vc - vu)


@torch.no_grad()
def fm_sample(model, z, y, steps=50, solver="euler", omega=1.0, t_lo=0.0, t_hi=1.0, shift=1.0):
    """Integrate dz/dt = v from t = 1 (noise) to 0. `steps` network evaluations of the
    (CFG) velocity for Euler; Heun uses steps//2 intervals (same NFE). `shift` > 1 spends
    more steps at high noise (t' = s t / (1 + (s - 1) t))."""
    null = model.num_classes
    n_int = steps if solver == "euler" else max(1, steps // 2)
    ts = torch.linspace(1.0, 0.0, n_int + 1, device=z.device)
    if shift != 1.0:
        ts = shift * ts / (1 + (shift - 1) * ts)
    wt = lambda t: omega if (t_lo <= float(t) <= t_hi) else 1.0
    for i in range(n_int):
        t, tn = ts[i], ts[i + 1]
        v = fm_velocity(model, z, y, wt(t), null)
        if solver == "euler":
            z = z + (tn - t) * v
        else:  # Heun (2nd order)
            z_e = z + (tn - t) * v
            v2 = fm_velocity(model, z_e, y, wt(tn), null)
            z = z + (tn - t) * 0.5 * (v + v2)
    return z
