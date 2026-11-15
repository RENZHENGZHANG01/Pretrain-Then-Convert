"""Multi-step models with fewer sampling steps: FID against the number of network evaluations (generation cost).

For a finished multi-step FM model (selected weights from the main evaluation), every step count in --steps is
sampled with the Euler and (for even counts) the Heun solver; for each the guidance is selected on 10k samples
(seed 1) over omega in --omegas with the full interval and with the interval selected at 50 steps, and FID-50k
(seed 0, the main protocol's noise and labels) is computed at the best (solver, guidance) of every step count.
Network evaluations are counted exactly: a guided step makes two passes (conditional and unconditional).

  torchrun --nproc_per_node G scripts/eval_steps.py --run r1_XL_3e20_cd_s0   ->  $OSR_EVAL/<run>/steps.json
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.evaluate import Evaluator, load_run_model  # noqa: E402
from osr.paths import EVAL, RUNS  # noqa: E402


def passes(steps, solver, omega, t_lo, t_hi):
    """Network evaluations of osr.sample.fm_sample (two per guided evaluation, one where omega is 1)."""
    n_int = steps if solver == "euler" else max(1, steps // 2)
    ts = torch.linspace(1.0, 0.0, n_int + 1)
    w = lambda t: omega if (t_lo <= float(t) <= t_hi) else 1.0  # noqa: E731
    evals = [ts[i] for i in range(n_int)] if solver == "euler" else [t for i in range(n_int) for t in (ts[i], ts[i + 1])]
    return sum(2 if w(t) != 1.0 else 1 for t in evals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", default="1,2,4,8,16,32")
    ap.add_argument("--omegas", default="1,1.5,2,3,4,5")
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

    summ = json.load(open(f"{EVAL}/{args.run}/summary.json"))
    assert summ["route"] == "fm", "multi-step (FM) models only"
    ck = f"{RUNS}/{args.run}/final_sel.pt"
    st = torch.load(ck if os.path.exists(ck) else f"{RUNS}/{args.run}/final.pt", map_location="cpu", weights_only=False)
    model = load_run_model(st, "selected" if "selected_weights" in st else summ["weights"], dev)
    ev = Evaluator(dev)
    out_path = f"{EVAL}/{args.run}/steps.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else dict(run_id=args.run, size=st["size"], select={}, final={})
    p50 = summ["point"]
    intervals = [(0.0, 1.0), (p50["t_lo"], p50["t_hi"])]
    omegas = [float(x) for x in args.omegas.split(",")]
    points = [dict(omega=1.0, t_lo=0.0, t_hi=1.0)] + [dict(omega=w, t_lo=a, t_hi=b) for w in omegas if w != 1.0 for a, b in intervals]

    def save():
        if rank == 0:
            json.dump(res, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)

    for steps in [int(s) for s in args.steps.split(",")]:
        for solver in (["euler", "heun"] if steps % 2 == 0 else ["euler"]):
            for p in points:
                key = f"{steps}|{solver}|{p['omega']:g}|{p['t_lo']:g}|{p['t_hi']:g}"
                if key in res["select"]:
                    continue
                r, _, _ = ev.evaluate(model, "fm", steps, p, args.nB, 1, solver=solver)
                if rank == 0:
                    res["select"][key] = dict(steps=steps, solver=solver, **p, nfe=passes(steps, solver, **p), fid10k=r["fid"])
                    print(key, round(r["fid"], 3), flush=True)
                save()
        if str(steps) in res["final"]:
            continue
        cand = [v for v in res["select"].values() if v["steps"] == steps] if rank == 0 else []
        obj = [min(cand, key=lambda v: v["fid10k"]) if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        b = obj[0]
        p = dict(omega=b["omega"], t_lo=b["t_lo"], t_hi=b["t_hi"])
        r, _, _ = ev.evaluate(model, "fm", steps, p, args.nC, 0, solver=b["solver"])
        if rank == 0:
            res["final"][str(steps)] = dict(b, fid50k=r["fid"], inception_score=r["inception_score"])
            print("final", steps, b["solver"], p, "nfe", b["nfe"], "FID-50k", round(r["fid"], 3), flush=True)
        save()
    if rank == 0:
        res["final"]["50"] = dict(steps=50, solver="euler", **p50, nfe=passes(50, "euler", **p50), fid50k=summ["fid50k"],
                                  source="summary.json (main evaluation)")
        save()
    dist.barrier()


if __name__ == "__main__":
    main()
