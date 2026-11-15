"""FD-DINOv2 with its own guidance selection. The main protocol picks the guidance point by FID (stage B, 10k,
seed 1) and reports FD-DINOv2 at that point, which can bias FD-DINOv2 comparisons (it usually prefers
different guidance). Here: sweep the same stage-B grid on 10k samples (seed 1) recording FD-DINOv2 and FID at
every point, pick the FD-DINOv2-optimal point, and report FD-DINOv2-50k and FID-50k there (seed 0).
Same selected weights as the main evaluation.

  torchrun --nproc_per_node G scripts/eval_fdsel.py --run RUN_ID [--extend]
Writes eval/<run>/fdsel.json. For multi-step models the guidance ladder is extended (omega 10 ... 24 on the
best interval) while the FD-optimal omega sits on the largest swept value; --extend applies only that
step to an existing fdsel.json (its sweep is reused; the previous file is kept as fdsel_prev.json).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import EVAL, RUNS  # noqa: E402
from osr.evaluate import Evaluator, load_run_model  # noqa: E402
from osr.extra_metrics import DINO_REF, DinoV2, frechet, stats  # noqa: E402
from scripts.eval_run import GRID  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nB", type=int, default=10000)
    ap.add_argument("--nC", type=int, default=50000)
    ap.add_argument("--out", default="fdsel.json", help="output file name in eval/<run>/ (tests use another name)")
    ap.add_argument("--extend", action="store_true", help="only extend the omega ladder of an existing fdsel.json")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out_dir = f"{EVAL}/{args.run}"
    summ = json.load(open(f"{out_dir}/summary.json"))
    ck = f"{RUNS}/{args.run}/final.pt"
    if not os.path.exists(ck):
        ck = f"{RUNS}/{args.run}/final_sel.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    model = load_run_model(st, summ["weights"], dev)
    route = st["route"]
    sampler, nfe = ("meanflow", 1) if route in ("imf", "dmd") else ("fm", 50)
    ev = Evaluator(dev)
    dino = DinoV2(dev)
    dref = np.load(DINO_REF)
    mu_r, s_r = dref["mu"].astype(np.float64), dref["sigma"].astype(np.float64)

    def score(point, n, seed):
        feats, logits, _, dfeats = ev.generate_scores(model, sampler, nfe, point, n, seed=seed, extra=dino)
        res = None
        if rank == 0:
            mu, s = stats(dfeats)
            res = dict(fd_dinov2=frechet(mu, s, mu_r, s_r), fid=ev.metrics(feats, logits)["fid"])
        obj = [res]; dist.broadcast_object_list(obj, src=0)
        return obj[0]

    prev = json.load(open(f"{out_dir}/fdsel.json")) if args.extend else None
    sweep = list(prev["sweep_10k"]) if prev else []
    if not prev:
        grid = list(GRID[route if route in ("fm", "dmd") else "imf"]["B"])
        if route == "fm":  # FD-DINOv2 prefers strong guidance: the FID grid (omega <= 4) would put the FD optimum on its edge
            grid += [dict(omega=w, t_lo=a, t_hi=b) for (a, b) in ((0.0, 1.0), (0.2, 0.85)) for w in (5.0, 6.0, 8.0)]
        for p in grid:
            r = score(p, args.nB, 1)
            sweep.append(dict(point=p, **r))
            if rank == 0:
                print(json.dumps(sweep[-1]), flush=True)
    best = min(sweep, key=lambda r: r["fd_dinov2"])
    ladder = (10.0, 12.0, 16.0, 20.0, 24.0)
    while route == "fm":  # extend while the optimum is on the largest omega swept at its interval
        iv = {k: v for k, v in best["point"].items() if k != "omega"}
        same = [r["point"]["omega"] for r in sweep if {k: v for k, v in r["point"].items() if k != "omega"} == iv]
        nxt = [w for w in ladder if w > max(same)]
        if best["point"]["omega"] < max(same) or not nxt:
            break
        for w in nxt[:2]:
            p = dict(iv, omega=w)
            r = score(p, args.nB, 1)
            sweep.append(dict(point=p, **r))
            if rank == 0:
                print(json.dumps(sweep[-1]), flush=True)
        best = min(sweep, key=lambda r: r["fd_dinov2"])
    if prev and best["point"] == prev["fd_selected_point"]:
        final = dict(fd_dinov2=prev["fd_dinov2_50k_fdsel"], fid=prev["fid50k_at_fdsel"])
    else:
        final = score(best["point"], args.nC, 0)
    if rank == 0:
        if prev:
            json.dump(prev, open(f"{out_dir}/fdsel_prev.json", "w"), indent=1)
        rec = dict(run_id=args.run, weights=summ["weights"], fid_selected_point=summ["point"], fd_selected_point=best["point"],
                   fd_dinov2_50k_fdsel=final["fd_dinov2"], fid50k_at_fdsel=final["fid"], fid50k_fidsel=summ["fid50k"],
                   sweep_10k=sweep, nB=args.nB, nC=args.nC, grid_version=3 if route == "fm" else 2)
        ex = f"{out_dir}/extra.json"
        if os.path.exists(ex):
            rec["fd_dinov2_50k_fidsel"] = json.load(open(ex))["fd_dinov2"]
        json.dump(rec, open(f"{out_dir}/{args.out}", "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "sweep_10k"}), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
