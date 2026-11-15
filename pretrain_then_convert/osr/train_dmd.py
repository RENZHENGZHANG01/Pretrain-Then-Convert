"""DMD2 distillation: frozen multi-step teacher (a cooled-down flow-matching model) -> one-step student, with an online fake score (teacher-sized, initialised from the teacher), a GAN
head on the fake score's trunk features, and TTUR (k fake/D updates per generator update).

  python -m osr.train_dmd --config cfg.yaml           (under torchrun)

Port of tianweiy/DMD2 (main/edm/edm_guidance.py) to flow matching / imfDiT:
  student      x = z - u_G(z, h=1, omega)  (same one-step parameterisation as iMF/convert),
               initialised from an FM milestone of the student size; omega is an input
               (CFG-as-input, like every other one-step model here); interval fixed to [0, 1]
  real score   teacher v with two-pass CFG at the student's omega:  v_u + omega (v_c - v_u)
  fake score   v_F(x_t; y, omega), FM loss on student samples (t ~ logit-normal(-0.4, 1),
               the teachers' training distribution)
  DMD grad     (x_hat_fake - x_hat_real) / mean|x - x_hat_real|,  x_hat = x_t - t v,
               t ~ U(t_min, t_max) = U(0.02, 0.98);   loss 0.5 ||x - sg(x - grad)||^2
  GAN          "diffusion GAN": D = small head on the fake score's trunk tokens of noisy samples
               (t ~ U(0, 1)); both branches use omega = 1 conditioning (otherwise D could read
               omega); D loss softplus(-D(real)) + softplus(D(fake)) (weight cls_w) in the fake
               update, generator loss softplus(-D(fake)) (weight gen_cls_w)
FLOPs: one generator step = k fake/D updates + 1 generator update, measured with FlopCounterMode
(math attention), all networks included (teacher CFG passes, fake score fwd/bwd, GAN, student).

License: this file adapts code from tianweiy/DMD2 (CC BY-NC-SA 4.0) and is distributed under the same license
(Attribution-NonCommercial-ShareAlike 4.0); see NOTICE.md. The rest of the repository is MIT-licensed.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as Fn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode

from .data import LatentData, Prefetcher
from .ema import PowerEMA
from .losses import _b, logit_normal
from .models import build_model, count_params

STOP = {"flag": False, "sig": None}


def _on_signal(sig, frame):
    STOP["flag"], STOP["sig"] = True, sig


class Disc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, seq, prefix):
        return self.net(seq[:, prefix:].float().mean(1)).squeeze(-1)


def _c(B, dev, v):
    return torch.full((B,), float(v), device=dev)


def gen_sample(G, z, y, w):
    B, dev = z.shape[0], z.device
    u, _ = G(z, _c(B, dev, 1), _c(B, dev, 1), w, _c(B, dev, 0), _c(B, dev, 1), y, heads=("u",))
    return z - u.float()


def teacher_v(T, xt, y, w, null):
    B, dev = xt.shape[0], xt.device
    zz, yy = torch.cat([xt, xt]), torch.cat([y, torch.full_like(y, null)])
    z2, o2 = _c(2 * B, dev, 0), _c(2 * B, dev, 1)
    v = T(zz, z2, z2, o2, z2, o2, yy, heads=("u",))[0].float()
    return v[B:] + _b(w) * (v[:B] - v[B:])


def fake_v(F_, xt, y, w):
    B, dev = xt.shape[0], xt.device
    return F_(xt, _c(B, dev, 0), _c(B, dev, 0), w, _c(B, dev, 0), _c(B, dev, 1), y, heads=("u",))[0].float()


def fake_trunk(F_, xt, y):
    B, dev = xt.shape[0], xt.device
    return F_.trunk(xt, _c(B, dev, 0), _c(B, dev, 1), _c(B, dev, 0), _c(B, dev, 1), y)


def sample_omega(n, lo, hi, gen, dev):  # log-uniform on [lo, hi]
    return torch.exp(math.log(lo) + torch.rand(n, generator=gen, device=dev) * (math.log(hi) - math.log(lo)))


class GenLoss(nn.Module):
    """DDP-wrapped generator objective. Teacher / fake score / D are passed in, not registered."""

    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, z, y, gen, ext):
        T, F_, D, cfg = ext
        B, dev = z.shape[0], z.device
        w = sample_omega(B, cfg["omega_lo"], cfg["omega_hi"], gen, dev)
        x = gen_sample(self.G, z, y, w)
        t = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * torch.rand(B, generator=gen, device=dev)
        e = torch.randn(x.shape, generator=gen, device=dev)
        xt = (1 - _b(t)) * x + _b(t) * e
        with torch.no_grad():
            xd = xt.detach()
            x_real = xd - _b(t) * teacher_v(T, xd, y, w, T.num_classes)
            x_fake = xd - _b(t) * fake_v(F_, xd, y, w)
            wgt = (x.detach() - x_real).abs().mean(dim=(1, 2, 3), keepdim=True)
            grad = torch.nan_to_num((x_fake - x_real) / wgt)
        l_dmd = 0.5 * Fn.mse_loss(x, (x - grad).detach())
        logs = {"loss_dmd": l_dmd.detach()}
        loss = l_dmd
        if cfg["gen_cls_w"] > 0:
            tg = torch.rand(B, generator=gen, device=dev)
            e2 = torch.randn(x.shape, generator=gen, device=dev)
            xg = (1 - _b(tg)) * x + _b(tg) * e2
            l_g = Fn.softplus(-D(fake_trunk(F_, xg, y), F_.prefix_tokens)).mean()
            loss = loss + cfg["gen_cls_w"] * l_g
            logs["loss_gen_cls"] = l_g.detach()
        return loss, logs


class FakeLoss(nn.Module):
    """DDP-wrapped fake-score (+ D) objective; the student G is passed in, not registered."""

    def __init__(self, F_, D):
        super().__init__()
        self.F_, self.D = F_, D

    def forward(self, z, y, x_real, y_real, gen, ext):
        G, cfg = ext
        B, dev = z.shape[0], z.device
        w = sample_omega(B, cfg["omega_lo"], cfg["omega_hi"], gen, dev)
        with torch.no_grad():
            x = gen_sample(G, z, y, w)
        t = logit_normal(B, cfg.get("t_mean", -0.4), cfg.get("t_std", 1.0), gen, dev)
        e = torch.randn(x.shape, generator=gen, device=dev)
        xt = (1 - _b(t)) * x + _b(t) * e
        l_fm = Fn.mse_loss(fake_v(self.F_, xt, y, w), e - x)
        loss, logs = l_fm, {"loss_fake": l_fm.detach()}
        if cfg["cls_w"] > 0:
            tr = torch.rand(B, generator=gen, device=dev); tf = torch.rand(B, generator=gen, device=dev)
            er = torch.randn(x.shape, generator=gen, device=dev); ef = torch.randn(x.shape, generator=gen, device=dev)
            sr = fake_trunk(self.F_, (1 - _b(tr)) * x_real + _b(tr) * er, y_real)
            sf = fake_trunk(self.F_, (1 - _b(tf)) * x + _b(tf) * ef, y)
            dr, df = self.D(sr, self.F_.prefix_tokens), self.D(sf, self.F_.prefix_tokens)
            l_d = (Fn.softplus(-dr) + Fn.softplus(df)).mean()
            loss = loss + cfg["cls_w"] * l_d
            logs.update(loss_cls=l_d.detach(), d_real=dr.mean().detach(), d_fake=df.mean().detach())
        return loss, logs


def load_weights(path, size):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(size, with_v_head=False)
    m.load_state_dict(st["model"], strict=True)
    return m


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_factor(k, n_steps, warmup, cd_frac):
    n_cd = int(round(cd_frac * n_steps))
    if k < warmup:
        return (k + 1) / warmup
    if n_cd > 0 and k >= n_steps - n_cd:
        return 1 - (k - (n_steps - n_cd) + 1) / n_cd
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dc = cfg["dmd"]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    torch.backends.cuda.matmul.allow_tf32 = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)
    run_dir = os.path.join(cfg["out_root"], cfg["run_id"])
    ck_dir = os.path.join(run_dir, "ckpt")
    if rank == 0:
        os.makedirs(ck_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
        except Exception:
            commit = "unknown"
        open(os.path.join(run_dir, "git_commit.txt"), "w").write(commit + "\n")
        if os.environ.get("SLURM_JOB_ID"):
            open(os.path.join(run_dir, "slurm_job_id.txt"), "a").write(os.environ["SLURM_JOB_ID"] + "\n")
        open(os.path.join(run_dir, "status.txt"), "w").write("running\n")
    dist.barrier()

    # ---- networks
    T = load_weights(dc["teacher"], dc["teacher_size"]).to(dev).eval().requires_grad_(False)
    G = load_weights(dc["student_init"], cfg["size"])
    G.freeze_new_condition_embedders()  # omega / h / interval inputs start as constants (function preserved)
    G = G.to(dev)
    F_ = load_weights(dc["teacher"], dc["teacher_size"])
    F_.freeze_new_condition_embedders()
    F_ = F_.to(dev)
    D = Disc(F_.hidden_size).to(dev)
    ema = PowerEMA(G, cfg["ema"]["sigma_rels"])
    step, flops = 0, 0.0
    latest = os.path.join(ck_dir, "latest.pt")
    resumed = os.path.exists(latest)
    opt_g_state = opt_f_state = None
    if resumed:
        st = torch.load(latest, map_location="cpu", weights_only=False)
        if st.get("done_marker"):
            st = torch.load(os.path.join(run_dir, "final.pt"), map_location="cpu", weights_only=False)
        G.load_state_dict(st["model"]); F_.load_state_dict(st["fake"]); D.load_state_dict(st["disc"])
        ema.load_state_dict(st["ema"]); opt_g_state, opt_f_state = st["opt"], st["opt_f"]
        step, flops = st["step"], st["flops"]
        del st
    if cfg.get("compile", True):
        for m in (G, F_, T):
            m.enable_compile()

    gb = cfg["data"]["global_batch"]
    lb = gb // world
    k_ttur = dc["ttur"]
    ext_cfg = dict(omega_lo=dc["omega_lo"], omega_hi=dc["omega_hi"], t_min=dc["t_min"], t_max=dc["t_max"],
                   gen_cls_w=dc["gen_cls_w"], cls_w=dc["cls_w"])

    # ---- measured FLOPs of one generator step (k fake/D updates + 1 generator update), per sample
    from . import models as M
    xx = torch.randn(4, 4, 32, 32, device=dev); yy = torch.randint(0, 1000, (4,), device=dev)
    gfl = torch.Generator(device=dev).manual_seed(1)
    gl, fl = GenLoss(G), FakeLoss(F_, D)
    M._ATTN_OVERRIDE[0] = "math"
    with torch.autocast("cuda", dtype=torch.bfloat16), FlopCounterMode(display=False) as fc:
        for _ in range(k_ttur):
            l, _ = fl(torch.randn_like(xx), yy, xx, yy, gfl, (G, ext_cfg)); l.backward()
        F_.requires_grad_(False); D.requires_grad_(False)
        l, _ = gl(torch.randn_like(xx), yy, gfl, (T, F_, D, ext_cfg)); l.backward()
        F_.requires_grad_(True); D.requires_grad_(True)
    M._ATTN_OVERRIDE[0] = None
    for m in (G, F_, D):
        m.zero_grad(set_to_none=True)
    flops_per_step = fc.get_total_flops() / 4 * gb
    flops_end = float(cfg["flops_end"])
    n_steps = max(1, math.ceil(flops_end / flops_per_step))

    opt_g = torch.optim.AdamW(G.parameters(), lr=dc["lr_g"], betas=tuple(dc["betas"]), weight_decay=dc["wd"], fused=True)
    opt_f = torch.optim.AdamW(list(F_.parameters()) + list(D.parameters()), lr=dc["lr_f"], betas=tuple(dc["betas"]),
                              weight_decay=dc["wd"], fused=True)
    if opt_g_state is not None:
        opt_g.load_state_dict(opt_g_state); opt_f.load_state_dict(opt_f_state)
    opt_g_state = opt_f_state = None
    ddp_g = DDP(gl, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
    ddp_f = DDP(fl, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True,
                find_unused_parameters=dc["cls_w"] <= 0)  # D unused when the GAN term is off
    cpu_group = dist.new_group(backend="gloo")
    data = LatentData(cfg["data"]["root"], gb, rank, world, cfg["seed"], dev)
    if rank == 0:
        info = dict(event="start", resumed=resumed, kind="distill", route="dmd", size=cfg["size"], teacher=dc["teacher"],
                    teacher_size=dc["teacher_size"], params_student=count_params(G), params_fake=count_params(F_),
                    flops_per_step=flops_per_step, flops_per_sample=flops_per_step / gb, ttur=k_ttur, world=world,
                    step=step, n_steps=n_steps, flops_end=flops_end, time=time.time())
        open(os.path.join(run_dir, "metrics.jsonl"), "a").write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)

    def state(full=True):
        st = {"model": G.state_dict(), "ema": ema.state_dict(), "step": step, "flops": flops, "step0": 0, "flops_start": 0.0,
              "ema_t0": 0, "cfg": cfg, "route": "dmd", "size": cfg["size"]}
        if full:
            st.update(fake=F_.state_dict(), disc=D.state_dict(), opt=opt_g.state_dict(), opt_f=opt_f.state_dict())
        return st

    pref = Prefetcher(data, step * (k_ttur + 1))
    last_ck, t_log, acc, n_acc = time.time(), time.time(), {}, 0
    while step < n_steps:
        fac = lr_factor(step, n_steps, dc.get("warmup", 500), dc.get("cooldown_frac", 0.15))
        for g in opt_g.param_groups:
            g["lr"] = dc["lr_g"] * fac
        for g in opt_f.param_groups:
            g["lr"] = dc["lr_f"] * fac
        gen = torch.Generator(device=dev).manual_seed((cfg["seed"] * 1_000_003 + step) * 4099 + rank)
        for _ in range(k_ttur):  # fake score + D updates
            _, mom, yr = pref.get()
            xr = data.to_latent(mom, gen); yr = yr.to(dev, non_blocking=True)
            z = torch.randn(xr.shape, generator=gen, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lf, logf = ddp_f(z, yr, xr, yr, gen, (G, ext_cfg))
            lf.backward()
            torch.nn.utils.clip_grad_norm_(list(F_.parameters()) + list(D.parameters()), dc.get("grad_clip", 10.0))
            opt_f.step(); opt_f.zero_grad(set_to_none=True)
        _, mom, yg = pref.get()
        yg = yg.to(dev, non_blocking=True)
        z = torch.randn((lb, 4, 32, 32), generator=gen, device=dev)
        F_.requires_grad_(False); D.requires_grad_(False)  # generator step: no grads into the fake score / D
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, logg = ddp_g(z, yg, gen, (T, F_, D, ext_cfg))
        lg.backward()
        F_.requires_grad_(True); D.requires_grad_(True)
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), dc.get("grad_clip", 10.0))
        opt_g.step(); opt_g.zero_grad(set_to_none=True)
        step += 1
        flops += flops_per_step
        ema.update(G, step)
        for kk, vv in {**logf, **logg}.items():
            acc[kk] = acc.get(kk, 0.0) + vv.float()
        n_acc += 1
        if step % cfg.get("log_every", 50) == 0 or step == n_steps:
            dt = time.time() - t_log
            if rank == 0:
                rec = {k: float(v) / n_acc for k, v in acc.items()}
                rec.update(step=step, flops=flops, lr_g=dc["lr_g"] * fac, gnorm_g=float(gn), img_s=n_acc * gb / dt,
                           tflops_gpu=n_acc * flops_per_step / dt / world / 1e12, time=time.time())
                open(os.path.join(run_dir, "metrics.jsonl"), "a").write(json.dumps(rec) + "\n")
            t_log, acc, n_acc = time.time(), {}, 0
        flags = torch.tensor([1.0 if (STOP["flag"] or time.time() - last_ck > 60 * cfg.get("ckpt_every_min", 20)) else 0.0,
                              1.0 if STOP["flag"] else 0.0])
        dist.all_reduce(flags, group=cpu_group)
        if flags[0].item() > 0 and step < n_steps:
            if rank == 0:
                atomic_save(state(), latest)
            last_ck = time.time()
            if flags[1].item() > 0:
                if rank == 0:
                    open(os.path.join(run_dir, "status.txt"), "w").write(f"preempted at step {step}; requeue\n")
                dist.barrier(); dist.destroy_process_group(); sys.exit(0)
        if args.max_steps and step >= args.max_steps:
            break
    pref.stop = True
    if rank == 0:
        done = step >= n_steps
        if done:
            atomic_save(state(full=False), os.path.join(run_dir, "final.pt"))
            atomic_save({"done_marker": True, "step": step, "flops": flops}, latest)
        else:
            atomic_save(state(), latest)
        open(os.path.join(run_dir, "status.txt"), "w").write(("done" if done else f"stopped at step {step}") + f" flops={flops:.4e}\n")
        open(os.path.join(run_dir, "metrics.jsonl"), "a").write(json.dumps(dict(event="end", step=step, flops=flops, done=done)) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
