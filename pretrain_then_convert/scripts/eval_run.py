"""Evaluate one finished run (cooldown / convert final.pt) with the unified protocol.

  torchrun --nproc_per_node G scripts/eval_run.py --run RUN_ID

Stage A  weights selection: {raw, ema0.050, ema0.100} at a few guidance points, 5k samples.
Stage B  guidance sweep for the selected weights, 10k samples (seed 1) -- selection set.
Stage C  50k samples (seed 0, independent noise) at the selected point; features saved.
One-step models (route imf): 1-NFE MeanFlow sampler with CFG-as-input (omega, interval);
  also 2-NFE at stage C point and the omega=1 (no guidance) 10k point for robustness.
Multi-step models (route fm): Euler 50-NFE ODE with two-pass CFG and guidance interval.
Writes eval/<run_id>/{results.jsonl, summary.json, inception_*.npy, samples_*.png, DONE}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import ADM_FEATS, EVAL, RUNS  # noqa: E402
from osr.evaluate import Evaluator, ema_tags, load_run_model  # noqa: E402


GRID = {
    "imf": dict(A=[dict(omega=w, t_min=0.3, t_max=0.7) for w in (2.0, 4.0, 7.0)],
                B=[dict(omega=w, t_min=a, t_max=b) for (a, b) in ((0.0, 1.0), (0.25, 0.75), (0.4, 0.65))
                   for w in (1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 8.5, 10.5)]),
    "dmd": dict(A=[dict(omega=w, t_min=0.0, t_max=1.0) for w in (1.5, 2.5, 4.0)],
                B=[dict(omega=w, t_min=0.0, t_max=1.0) for w in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)]),
    "fm": dict(A=[dict(omega=w, t_lo=0.2, t_hi=0.85) for w in (1.5, 2.5)],
               B=[dict(omega=w, t_lo=a, t_hi=b) for (a, b) in ((0.0, 1.0), (0.2, 0.85))
                  for w in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)]),
    # shortcut models: guidance is fixed inside training (CFG 1.5 at the smallest bootstrap step), none at sampling
    "sc": dict(A=[dict(omega=1.0, t_min=0.0, t_max=1.0)], B=[dict(omega=1.0, t_min=0.0, t_max=1.0)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--fm_nfe", type=int, default=50)
    ap.add_argument("--fm_solver", default="euler")
    ap.add_argument("--nA", type=int, default=5000)
    ap.add_argument("--nB", type=int, default=10000)
    ap.add_argument("--nC", type=int, default=50000)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = f"{EVAL}/{args.run}"
    os.makedirs(out_dir, exist_ok=True)
    ck = f"{RUNS}/{args.run}/{args.ckpt}"
    if not os.path.exists(ck) and args.ckpt == "final.pt":
        ck = f"{RUNS}/{args.run}/final_sel.pt"  # compacted checkpoint
    st = torch.load(ck, map_location="cpu", weights_only=False)
    route = st["route"]
    sampler, nfe = ("meanflow", 1) if route in ("imf", "dmd", "sc") else ("fm", args.fm_nfe)
    ev = Evaluator(dev)
    meta = dict(run_id=args.run, size=st["size"], route=route, kind=st["cfg"]["kind"], step=st["step"], flops=st["flops"])

    def log(rec):
        if rank == 0:
            rec = dict(meta, **rec)
            print(json.dumps(rec), flush=True)
            with open(f"{out_dir}/results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def run_point(model, weights, stage, point, n, seed, nfe_=None, keep_png=0):
        res, feats, pngs = ev.evaluate(model, sampler, nfe_ or nfe, point, n, seed, solver=args.fm_solver, keep_png=keep_png)
        fid = None
        if rank == 0:
            log(dict(res, weights=weights, stage=stage))
            fid = res["fid"]
        obj = [fid]
        dist.broadcast_object_list(obj, src=0)
        return obj[0], feats, pngs

    # ---- stage A: weights selection
    best = (1e9, None)
    for w in ["raw"] + ema_tags(st):
        model = load_run_model(st, w, dev)
        for p in GRID[route if route in ("fm", "dmd", "sc") else "imf"]["A"]:
            fid, _, _ = run_point(model, w, "A", p, args.nA, seed=1)
            best = min(best, (fid, w), key=lambda x: x[0])
        del model
    weights = best[1]
    model = load_run_model(st, weights, dev)
    # ---- stage B: guidance sweep
    bestB = (1e9, None)
    for p in GRID[route if route in ("fm", "dmd", "sc") else "imf"]["B"]:
        fid, _, _ = run_point(model, weights, "B", p, args.nB, seed=1)
        bestB = min(bestB, (fid, p), key=lambda x: x[0])
    point = bestB[1]
    # ---- stage C: 50k at the selected point (independent seed); features for bootstrap
    from osr.extra_metrics import ADM_REF, DINO_REF, DinoV2, frechet, precision_recall, stats
    adm_feats = ADM_FEATS
    have_refs = os.path.exists(DINO_REF) and os.path.exists(adm_feats)
    if have_refs:  # same samples, plus DINOv2 features -> FD-DINOv2, ADM-FID, P/R (extra.json)
        feats, logits, pngs, dfeats = ev.generate_scores(model, sampler, nfe, point, args.nC, seed=0, solver=args.fm_solver,
                                                         keep_png=100, extra=DinoV2(dev))
        fid = None
        if rank == 0:
            m = ev.metrics(feats, logits)
            m.update(sampler=sampler, nfe=nfe, n=args.nC, seed=0, **point)
            log(dict(m, weights=weights, stage="C"))
            fid = m["fid"]
            mu_g, s_g = stats(feats); adm = np.load(ADM_REF); dref = np.load(DINO_REF); mu_d, s_d = stats(dfeats)
            extra = dict(run_id=args.run, weights=weights, point=point, n=args.nC, fid_jit_recheck=fid, fid50k_stageC=fid,
                         fid_adm=frechet(mu_g, s_g, adm["mu"].astype(np.float64), adm["sigma"].astype(np.float64)),
                         fd_dinov2=frechet(mu_d, s_d, dref["mu"].astype(np.float64), dref["sigma"].astype(np.float64)),
                         inception_score=m["inception_score"],
                         **precision_recall(torch.from_numpy(np.load(adm_feats)), feats.float(), k=3, dev=dev))
            json.dump(extra, open(f"{out_dir}/extra.json", "w"), indent=1)
            np.save(f"{out_dir}/dinov2_{weights}_n{args.nC}.npy", dfeats.half().numpy())
        obj = [fid]; dist.broadcast_object_list(obj, src=0); fid = obj[0]
    else:
        fid, feats, pngs = run_point(model, weights, "C", point, args.nC, seed=0, keep_png=100)
    if rank == 0:
        key = "_".join(f"{k}{v:g}" for k, v in point.items())
        np.save(f"{out_dir}/inception_{weights}_{sampler}{nfe}_{key}_n{args.nC}.npy", feats.half().numpy())
        if pngs:
            from torchvision.utils import save_image
            save_image(torch.cat(pngs)[:100].float() / 255, f"{out_dir}/samples_{weights}_{key}.png", nrow=10)
    extra = {}
    if route in ("imf", "sc"):  # robustness: 2-NFE at the same guidance, and no guidance
        f2, _, _ = run_point(model, weights, "C2nfe", point, args.nB, seed=1, nfe_=2)
        f1, _, _ = run_point(model, weights, "Bw1", dict(omega=1.0, t_min=0.0, t_max=1.0), args.nB, seed=1)
        extra = dict(fid_2nfe_10k=f2, fid_noguid_10k=f1)
    if rank == 0:
        summ = dict(meta, weights=weights, point=point, fid50k=fid, fid10k_sel=bestB[0], **extra)
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summ, f, indent=1)
        open(f"{out_dir}/DONE", "w").close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
