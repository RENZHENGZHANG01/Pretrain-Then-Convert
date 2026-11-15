"""Sampling noise of FID-50k: how much FID moves with the 50k generated samples alone (fixed model, fixed reference
statistics), from the stage-C Inception features that eval_run.py stores. Split halves: for K random 25k/25k splits,
Var(FID_25k) = mean((FID_A - FID_B)^2) / 2 and Var(FID_50k) ~ Var(FID_25k) / 2 (variance ~ 1/n); reported as an SD of
FID-50k and of log FID-50k. (Resampling with replacement duplicates samples and biases FID upward, so it is not used.)

  python scripts/fid_sample_noise.py RUN_ID [RUN_ID ...]  ->  $OSR_EVAL/fid_sample_noise.json (or --out)
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import EVAL, FID_REF  # noqa: E402


def fid_from(mu, cov, mu_r, s_half, tr_r):
    """FID via tr sqrt(S_ref^1/2 S S_ref^1/2) (eigenvalues), identical to the sqrtm(S S_ref) form."""
    m = s_half @ cov @ s_half
    ev = np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)
    d = mu - mu_r
    return float(d @ d + np.trace(cov) + tr_r - 2 * np.sqrt(ev).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{EVAL}/fid_sample_noise.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ref = np.load(FID_REF)
    mu_r, s_r = ref["mu"].astype(np.float64), ref["sigma"].astype(np.float64)
    s_half = linalg.sqrtm(s_r).real
    s_half = (s_half + s_half.T) / 2
    tr_r = float(np.trace(s_r))
    out = {}
    for run in args.runs:
        summ = json.load(open(f"{EVAL}/{run}/summary.json"))
        key = "_".join(f"{k}{v:g}" for k, v in summ["point"].items())  # features of the selected point
        fs = glob.glob(f"{EVAL}/{run}/inception_{summ['weights']}_*_{key}_n50000.npy")
        assert fs, f"no stage-C features of {run} (eval_run.py writes them)"
        x = np.load(fs[0]).astype(np.float64)
        full = fid_from(x.mean(0), np.cov(x, rowvar=False), mu_r, s_half, tr_r)
        d2 = []
        for _ in range(args.splits):
            perm = rng.permutation(len(x))
            a, b = x[perm[: len(x) // 2]], x[perm[len(x) // 2:]]
            fa = fid_from(a.mean(0), np.cov(a, rowvar=False), mu_r, s_half, tr_r)
            fb = fid_from(b.mean(0), np.cov(b, rowvar=False), mu_r, s_half, tr_r)
            d2.append((fa - fb) ** 2)
        sd50 = math.sqrt(np.mean(d2) / 2 / 2)
        out[run] = dict(fid_reported=summ["fid50k"], fid_recomputed=full, split_sd_fid50k=sd50, sd_logfid=sd50 / full)
        print(run, json.dumps({k: round(v, 4) for k, v in out[run].items()}), flush=True)
    res = dict(splits=args.splits, runs=out, mean_sd_logfid=float(np.mean([v["sd_logfid"] for v in out.values()])))
    json.dump(res, open(args.out, "w"), indent=1)
    print("mean sampling SD of log FID-50k:", round(res["mean_sd_logfid"], 4))


if __name__ == "__main__":
    main()
