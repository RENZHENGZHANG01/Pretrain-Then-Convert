"""Unified trainer for all routes.

  python -m osr.train --config cfg.yaml          (under torchrun)

Run kinds (cfg.kind):
  mainline : fresh init; warmup then constant LR (WSD stable phase) up to cfg.flops_end;
             saves milestones at absolute cumulative FLOPs (cooldown starts, convert points).
  cooldown : init from a milestone of a mainline with the SAME objective (model+EMA+Adam);
             LR decays to 0 over [flops_start, flops_end]. Produces the model at budget C.
  convert  : init from an FM milestone (raw weights), add the v-head as a copy of the
             u-head (iMF only), freeze the new condition embedders to constants (exact function
             preservation), fresh Adam + fresh EMA; one-step objective (iMF, or shortcut for
             route sc) with short warmup, constant LR and a final cooldown over [flops_start, flops_end].
Route sc (shortcut models) also keeps a target network, an EMA of the weights (decay
cfg.loss.ema_decay) that produces the bootstrap targets; it is saved with the checkpoints.
FLOPs are measured with FlopCounterMode for this exact (objective, size, local batch) and
accumulated per step; every trigger (milestones, schedule, end) is in FLOPs.
Preemption-safe: resumable state every cfg.ckpt_every_min minutes and on SIGTERM/SIGUSR1.
"""
import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import LatentData, Prefetcher, RandomLatents
from .ema import PowerEMA
from .flops import measure_step_flops
from .losses import LOSSES
from .models import build_model, count_params

STOP = {"flag": False, "sig": None}


def _on_signal(sig, frame):
    STOP["flag"], STOP["sig"] = True, sig


class LossWrapper(torch.nn.Module):
    """DDP needs the whole loss (incl. no-grad guidance passes and the JVP) inside forward."""

    def __init__(self, model, loss_fn, loss_cfg):
        super().__init__()
        self.model, self.loss_fn, self.loss_cfg = model, loss_fn, loss_cfg

    def forward(self, x, y, gen):
        return self.loss_fn(self.model, x, y, self.loss_cfg, gen)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_at(cfg, step, n_steps, step0):
    """LR for local step index (step - step0) of this run."""
    lr = cfg["optim"]["lr"]
    k = step - step0
    sch = cfg["schedule"]
    if cfg["kind"] == "mainline":
        w = sch.get("warmup_steps", 0)
        return lr * min(1.0, (step + 1) / max(1, w)) if w else lr
    if cfg["kind"] == "extend":  # continue a mainline's stable phase from one of its full milestones
        return lr
    if cfg["kind"] == "cooldown":
        prog = min(1.0, (k + 1) / n_steps)
        return lr * (1 - prog) if sch.get("cooldown_shape", "linear") == "linear" else lr * (1 - math.sqrt(prog))
    # convert: warmup (<= warmup_frac of the branch) -> constant -> cooldown (last cooldown_frac)
    w = max(1, min(sch.get("warmup_steps", 2000), int(sch.get("warmup_frac", 0.1) * n_steps)))
    n_cd = int(round(sch.get("cooldown_frac", 0.15) * n_steps))
    if k < w:
        return lr * (k + 1) / w
    if n_cd > 0 and k >= n_steps - n_cd:
        prog = (k - (n_steps - n_cd) + 1) / n_cd
        return lr * (1 - prog) if sch.get("cooldown_shape", "linear") == "linear" else lr * (1 - math.sqrt(prog))
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after this many steps")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    torch._dynamo.config.cache_size_limit = 64  # per-block compiled forwards x (shapes, grad modes)
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    run_dir = os.path.join(cfg["out_root"], cfg["run_id"])
    ck_dir, ms_dir = os.path.join(run_dir, "ckpt"), os.path.join(run_dir, "milestones")
    if rank == 0:
        os.makedirs(ck_dir, exist_ok=True)
        os.makedirs(ms_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        try:
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain", "osr", "scripts"], cwd=repo).decode().strip()
        except Exception:
            commit, dirty = "unknown", ""
        with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
            f.write(commit + ("\n(dirty)\n" + dirty if dirty else "\n"))
        if os.environ.get("SLURM_JOB_ID"):
            with open(os.path.join(run_dir, "slurm_job_id.txt"), "a") as f:
                f.write(os.environ["SLURM_JOB_ID"] + "\n")
        with open(os.path.join(run_dir, "status.txt"), "w") as f:
            f.write("running\n")
    dist.barrier()

    # ---------------- model / init ----------------
    route = cfg["route"]  # objective of this run: fm | imf | sc
    torch.manual_seed(cfg["seed"])  # same init on every rank (fresh runs)
    model = build_model(cfg["size"], with_v_head=(route == "imf")).to(dev)
    ema = None
    opt_state, step, flops = None, 0, 0.0
    latest = os.path.join(ck_dir, "latest.pt")
    resumed = os.path.exists(latest)
    if resumed:
        # load to CPU: a GPU copy of a full checkpoint (27 GB for H) stays resident and OOMs
        st = torch.load(latest, map_location="cpu", weights_only=False)
        if st.get("done_marker"):  # finished run being extended: final.pt holds the full state
            st = torch.load(os.path.join(run_dir, "final.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema = PowerEMA(model, cfg["ema"]["sigma_rels"])
        ema.load_state_dict(st["ema"])
        opt_state, step, flops = st["opt"], st["step"], st["flops"]
        ema_t0, step0, flops_start = st["ema_t0"], st["step0"], st["flops_start"]
        tgt_state = st.get("tgt")
        del st
    elif cfg["kind"] in ("cooldown", "convert", "extend"):
        st = torch.load(cfg["init"]["from"], map_location="cpu", weights_only=False)
        flops_start = float(st["flops"])
        tgt_state = None
        if cfg["kind"] in ("cooldown", "extend"):
            model.load_state_dict(st["model"])
            ema = PowerEMA(model, cfg["ema"]["sigma_rels"])
            ema.load_state_dict(st["ema"])
            opt_state, step = st["opt"], st["step"]
            ema_t0 = st["ema_t0"]  # EMA lifetime continues from the mainline
            tgt_state = st.get("tgt")
        elif cfg["init"].get("mode") == "naive":  # ablation: FM weights for trunk, u-head, class and patch embedders;
            # the four new conditioning embedders and the v-head keep their fresh (seeded) initialization
            new = ("h_embedder.", "omega_embedder.", "cfg_t_start_embedder.", "cfg_t_end_embedder.")
            sd = {k: v for k, v in st["model"].items() if not k.startswith(new)}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected and all(k.startswith(new + ("v_heads.", "v_final_layer.")) for k in missing), (missing, unexpected)
            ema = PowerEMA(model, cfg["ema"]["sigma_rels"])
            step, ema_t0 = 0, 0
        else:  # convert FM -> iMF / shortcut
            fm = build_model(cfg["size"], with_v_head=False).to(dev)
            fm.load_state_dict(st["model"])
            if route == "imf":
                fm.add_v_head_from_u()
            fm.freeze_new_condition_embedders()
            model.load_state_dict(fm.state_dict())
            del fm
            ema = PowerEMA(model, cfg["ema"]["sigma_rels"])
            step, ema_t0 = 0, 0
        step0, flops = step, flops_start
        del st
    else:
        ema = PowerEMA(model, cfg["ema"]["sigma_rels"])
        step0, ema_t0, flops_start = 0, 0, 0.0
        tgt_state = None
    flops_end = float(cfg["flops_end"])

    loss_fn = LOSSES[route]
    loss_cfg = dict(cfg.get("loss", {}))  # a copy: the target network below must not end up in cfg (saved, dumped)
    tgt, tgt_decay = None, None
    if route == "sc":  # created before any compiled forward (compiled block functions are per instance)
        tgt = copy.deepcopy(model).requires_grad_(False)
        if tgt_state is not None:
            tgt.load_state_dict(tgt_state)
        tgt_decay = float(loss_cfg.pop("ema_decay", 0.999))
        loss_cfg["_target"] = tgt
    tgt_state = None
    gb = cfg["data"]["global_batch"]
    lb = gb // world
    micro = min(cfg["data"].get("micro_batch") or lb, lb)
    n_accum = lb // micro
    assert lb % micro == 0

    # ---------------- measured FLOPs per global step ----------------
    # measured on 4 samples (the single-pass reference is memory hungry) and scaled: every
    # term is linear in the batch, and the r=t / r!=t split is exactly half for even batches
    assert micro % 4 == 0, micro
    # stratified shortcut batches are not linear in the batch size (fixed level counts): measure the micro-batch
    n_meas = micro if (route == "sc" and loss_cfg.get("stratified")) else 4
    xx = torch.randn(n_meas, 4, 32, 32, device=dev)
    yy = torch.randint(0, 1000, (n_meas,), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f_micro = measure_step_flops(loss_fn, model, xx, yy, loss_cfg, seed=12345) * (micro // n_meas)
    flops_per_step = float(f_micro) * n_accum * world
    n_steps = max(1, math.ceil((flops_end - flops_start) / flops_per_step))
    step_end = step0 + n_steps

    # milestones: absolute cumulative FLOPs -> the first step whose cumulative FLOPs reach it
    ms_list = []
    for m in cfg.get("milestones", []):
        mf = float(m["flops"])
        if flops_start < mf <= flops_end + 1e-6 * flops_end:
            ms_step = step0 + max(1, math.ceil((mf - flops_start) / flops_per_step))
            ms_list.append((ms_step, mf, m.get("kind", "full"), m.get("tag", f"{mf:.3e}")))

    if cfg.get("compile", False):
        model.enable_compile()
        if tgt is not None:
            tgt.enable_compile()

    data_cfg = cfg["data"]
    if not data_cfg.get("random", False) and data_cfg.get("warm_cache", True):
        t_w = time.time()
        if local_rank == 0:
            from .data import warm_page_cache
            warm_page_cache(os.path.join(data_cfg["root"], "moments.npy"))
        dist.barrier()
        if rank == 0:
            print(f"page cache warm-up {time.time() - t_w:.0f}s", flush=True)
    if data_cfg.get("random", False):
        data = RandomLatents(gb, rank, world, cfg["seed"], dev)
    else:
        data = LatentData(data_cfg["root"], gb, rank, world, data_cfg.get("seed", cfg["seed"]), dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], betas=tuple(cfg["optim"].get("betas", (0.9, 0.95))),
                            weight_decay=cfg["optim"].get("wd", 0.0), eps=cfg["optim"].get("eps", 1e-8), fused=True)
    if opt_state is not None and not (cfg["kind"] == "convert" and not resumed):
        opt.load_state_dict(opt_state)
    opt_state = None
    torch.cuda.empty_cache()
    ddp = DDP(LossWrapper(model, loss_fn, loss_cfg), device_ids=[local_rank], broadcast_buffers=False,
              gradient_as_bucket_view=True)
    # CPU-side group for the per-step checkpoint/stop consensus (no GPU->CPU sync in the loop)
    cpu_group = dist.new_group(backend="gloo")

    if rank == 0:
        info = dict(event="start", resumed=resumed, kind=cfg["kind"], route=route, size=cfg["size"],
                    params_inference=count_params(model), params_train=count_params(model, False),
                    flops_per_step=flops_per_step, flops_per_sample=flops_per_step / gb, world=world, micro=micro,
                    n_accum=n_accum, step0=step0, step=step, step_end=step_end, flops_start=flops_start,
                    flops_end=flops_end, milestones=[(s, f, k, t) for s, f, k, t in ms_list], time=time.time())
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)

    def state(full=True, ema_too=True):
        st = {"model": model.state_dict(), "step": step, "flops": flops, "step0": step0,
              "flops_start": flops_start, "ema_t0": ema_t0, "cfg": cfg, "route": route, "size": cfg["size"]}
        if ema_too:
            st["ema"] = ema.state_dict()
            if tgt is not None:
                st["tgt"] = tgt.state_dict()
        if full:
            st["opt"] = opt.state_dict()
        return st

    pref = Prefetcher(data, step)
    last_ck = time.time()
    t_log, logs_acc, n_acc = time.time(), {}, 0
    ck_every = 60 * cfg.get("ckpt_every_min", 20)
    log_every = cfg.get("log_every", 50)
    nonfinite_t = torch.zeros(1, device=dev)
    while step < step_end:
        lr = lr_at(cfg, step, n_steps, step0)
        for g in opt.param_groups:
            g["lr"] = lr
        s_data, mom, y = pref.get()
        assert s_data == step, (s_data, step)
        gen = torch.Generator(device=dev).manual_seed((cfg["seed"] * 1_000_003 + step) * 4099 + rank)
        x = data.to_latent(mom, gen)
        y = y.to(dev, non_blocking=True)
        for k in range(n_accum):
            sl = slice(k * micro, (k + 1) * micro)
            sync = (k == n_accum - 1)
            ctx = ddp.no_sync() if not sync else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, logs = ddp(x[sl], y[sl], gen)
                (loss / n_accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("grad_clip", 0) or float("inf"))
        # non-finite gradients: the fused Adam kernel skips the update on-device (no host sync)
        opt.found_inf = (~torch.isfinite(gnorm)).float()  # 0-dim, as GradScaler passes it
        nonfinite_t += opt.found_inf
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        flops += flops_per_step
        ema.update(model, step - ema_t0)
        if tgt is not None:  # target network: plain EMA of the weights
            with torch.no_grad():
                torch._foreach_lerp_(list(tgt.parameters()), list(model.parameters()), 1.0 - tgt_decay)

        for kk, vv in logs.items():
            logs_acc[kk] = logs_acc.get(kk, 0.0) + vv.detach().float()
        n_acc += 1
        if step % log_every == 0 or step == step_end:
            dt = time.time() - t_log
            if rank == 0:
                rec = {k: float(v) / n_acc for k, v in logs_acc.items()}
                rec.update(step=step, flops=flops, epochs=data.epochs(step), lr=lr, gnorm=float(gnorm),
                           img_s=n_acc * gb / dt, tflops_gpu=n_acc * flops_per_step / dt / world / 1e12,
                           nonfinite=int(nonfinite_t.item()), time=time.time())
                with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            t_log, logs_acc, n_acc = time.time(), {}, 0

        for ms_step, mf, kind, tag in ms_list:
            if step == ms_step and rank == 0:
                atomic_save(state(full=(kind == "full"), ema_too=(kind == "full")), os.path.join(ms_dir, f"{tag}.pt"))
                with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(dict(event="milestone", tag=tag, step=step, flops=flops, kind=kind)) + "\n")

        # periodic / signal-triggered checkpoint (decision broadcast so all ranks agree)
        flags = torch.tensor([1.0 if (STOP["flag"] or time.time() - last_ck > ck_every) else 0.0,
                              1.0 if STOP["flag"] else 0.0])
        dist.all_reduce(flags, group=cpu_group)
        if flags[0].item() > 0 and step < step_end:
            if rank == 0:
                atomic_save(state(), latest)
            last_ck = time.time()
            if flags[1].item() > 0:
                if rank == 0:
                    with open(os.path.join(run_dir, "status.txt"), "w") as f:
                        f.write(f"preempted at step {step} (signal {STOP['sig']}); requeue\n")
                    if os.environ.get("SLURM_JOB_ID") and STOP["sig"] == signal.SIGUSR1:
                        jid = os.environ["SLURM_JOB_ID"]  # inside an array, requeue only this task (the master id is the whole array)
                        if os.environ.get("SLURM_ARRAY_JOB_ID") and os.environ.get("SLURM_ARRAY_TASK_ID"):
                            jid = f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
                        subprocess.call(["scontrol", "requeue", jid])
                dist.barrier()
                dist.destroy_process_group()
                sys.exit(0)
        if args.max_steps and step - step0 >= args.max_steps:
            break

    pref.stop = True
    if rank == 0:
        done = step >= step_end
        if done:
            atomic_save(state(full=(cfg["kind"] == "mainline")), os.path.join(run_dir, "final.pt"))
            atomic_save({"done_marker": True, "step": step, "flops": flops}, latest)  # final.pt has the state
        else:
            atomic_save(state(), latest)
        with open(os.path.join(run_dir, "status.txt"), "w") as f:
            f.write(("done" if done else f"stopped at step {step} (max_steps)") + f" flops={flops:.4e}\n")
        with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(dict(event="end", step=step, flops=flops, done=done, time=time.time())) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
