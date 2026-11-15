"""Precision and recall along the guidance sweep. The main protocol reports P/R at the FID-selected guidance,
but guidance trades precision for recall, so a comparison of routes at one selected point can confuse a
coverage loss with a different operating point. Here every point of the stage-B grid (for multi-step
models also the extended omega <= 8 points of eval_fdsel.py) is scored on 10k samples (seed 1): FID,
improved precision and recall (k=3, Inception pool features, ADM reference images). Same selected
weights as the main evaluation.

  torchrun --nproc_per_node G scripts/eval_prcurve.py --run RUN_ID
Writes eval/<run>/prcurve.json.
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
from osr.evaluate import Evaluator, load_run_model  # noqa: E402
from osr.extra_metrics import precision_recall  # noqa: E402
from scripts.eval_run import GRID  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="prcurve.json", help="output file name in eval/<run>/ (tests use another name)")
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
    ref = torch.from_numpy(np.load(ADM_FEATS)) if rank == 0 else None
    grid = list(GRID[route if route in ("fm", "dmd") else "imf"]["B"])
    if route == "fm":
        grid += [dict(omega=w, t_lo=a, t_hi=b) for (a, b) in ((0.0, 1.0), (0.2, 0.85)) for w in (5.0, 6.0, 8.0)]
    curve = []
    for p in grid:
        feats, logits, _ = ev.generate_scores(model, sampler, nfe, p, args.n, seed=1)
        if rank == 0:
            rec = dict(point=p, fid=ev.metrics(feats, logits)["fid"], **precision_recall(ref, feats.float(), k=3, dev=dev))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
        dist.barrier()
    if rank == 0:
        json.dump(dict(run_id=args.run, weights=summ["weights"], fid_selected_point=summ["point"], n=args.n, curve=curve),
                  open(f"{out_dir}/{args.out}", "w"), indent=1)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
