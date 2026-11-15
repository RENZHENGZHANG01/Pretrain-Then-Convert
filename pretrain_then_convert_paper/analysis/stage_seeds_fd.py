"""Two robustness checks on the grid.

Stage seeds at 3e20 (group `stage_seeds`) -- the cooldown / conversion stage re-run with seed 1 from the same seed-0 FM/iMF
milestone. Their spread is the noise of a within-mainline comparison (all f-branches of one mainline share
the FM run), as opposed to the full between-seed noise (different mainlines).

FD-DINOv2: the main comparisons (frontiers, native/convert, one-step tax, f*) recomputed with FD-DINOv2
instead of FID, and every place where the two metrics order two models differently.

  python analysis/stage_seeds_fd.py  ->  analysis/out/stage_seeds_fd.json (+ summary on stdout)
"""
import csv
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from common import ROOT
from common import EVAL
BUD = ("1e19", "3e19", "1e20", "3e20")


def fid_of(run):
    p = f"{EVAL}/{run}/summary.json"
    return json.load(open(p))["fid50k"] if os.path.exists(p) else None


def extra_of(run):
    p = f"{EVAL}/{run}/extra.json"
    return json.load(open(p)) if os.path.exists(p) else None


def seed2():
    pairs = {}
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        if r["run_id"].endswith("_s0st1"):
            base = r["run_id"][:-3]
            a, b = fid_of(base), float(r["fid50k"])
            if a:
                pairs[base] = dict(seed0=a, stage_seed1=b, rel_diff=b / a - 1)
    d = np.array([math.log(v["stage_seed1"] / v["seed0"]) for v in pairs.values()])
    return dict(pairs=pairs, n=len(d), stage_sd_logfid=float(np.sqrt(np.mean(d ** 2) / 2)))


def fd_tables():
    cells = defaultdict(list)
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        ex = extra_of(r["run_id"])
        if not ex:
            continue
        route = {"r1": "R1", "r2": "NAT", "r3": "CONV"}[m.group(1)]
        f = float(m.group(4)[1:]) if route == "CONV" else (0.0 if route == "NAT" else None)
        cells[(route, m.group(2), m.group(3), f)].append((float(r["fid50k"]), ex["fd_dinov2"]))
    mean = {k: (float(np.exp(np.mean(np.log([v[0] for v in vs])))), float(np.exp(np.mean(np.log([v[1] for v in vs]))))) for k, vs in cells.items()}
    out = dict(frontier={}, best_cell={})
    for route in ("R1", "CONV", "NAT"):
        for b in BUD:
            ks = [k for k in mean if k[0] == route and k[2] == b]
            if not ks:
                continue
            kf = min(ks, key=lambda k: mean[k][0]); kd = min(ks, key=lambda k: mean[k][1])
            out["frontier"][f"{route}_{b}"] = dict(fid=mean[kf][0], fid_cell=list(map(str, kf[1:])), fd=mean[kd][1], fd_cell=list(map(str, kd[1:])))
    fr = out["frontier"]
    out["tax_fid"] = {b: fr[f"CONV_{b}"]["fid"] / fr[f"R1_{b}"]["fid"] for b in BUD}
    out["tax_fd"] = {b: fr[f"CONV_{b}"]["fd"] / fr[f"R1_{b}"]["fd"] for b in BUD}
    out["nat_over_conv_fid"] = {b: fr[f"NAT_{b}"]["fid"] / fr[f"CONV_{b}"]["fid"] for b in BUD}
    out["nat_over_conv_fd"] = {b: fr[f"NAT_{b}"]["fd"] / fr[f"CONV_{b}"]["fd"] for b in BUD}
    # f* by FD-DINOv2: best f at the FID-best convert size of each budget
    fstar = {}
    for b in BUD:
        size = fr[f"CONV_{b}"]["fid_cell"][0]
        pts = sorted((k[3], mean[k][1], mean[k][0]) for k in mean if k[0] in ("CONV", "NAT") and k[1] == size and k[2] == b)
        fstar[b] = dict(size=size, f_best_fd=min(pts, key=lambda p: p[1])[0], f_best_fid=min(pts, key=lambda p: p[2])[0],
                        curve=[dict(f=p[0], fd=p[1], fid=p[2]) for p in pts])
    out["fstar"] = fstar
    # rank disagreements between FID and FD-DINOv2 within each (route, budget) size sweep
    dis = []
    for route in ("R1", "CONV", "NAT"):
        for b in BUD:
            by_size = {}
            for k, (q, fd) in mean.items():
                if k[0] == route and k[2] == b:
                    if k[1] not in by_size or q < by_size[k[1]][0]:
                        by_size[k[1]] = (q, fd)
            sizes = sorted(by_size)
            for i in range(len(sizes)):
                for j in range(i + 1, len(sizes)):
                    a, c = by_size[sizes[i]], by_size[sizes[j]]
                    if (a[0] - c[0]) * (a[1] - c[1]) < 0:
                        dis.append(dict(route=route, budget=b, pair=[sizes[i], sizes[j]], fid=[a[0], c[0]], fd=[a[1], c[1]]))
    out["fid_fd_disagreements"] = dis
    allv = np.array(list(mean.values()))
    from scipy.stats import spearmanr
    out["spearman_fid_fd_all_cells"] = float(spearmanr(allv[:, 0], allv[:, 1])[0])
    return out


def main():
    res = dict(seed2=seed2(), fd=fd_tables())
    json.dump(res, open(f"{ROOT}/analysis/out/stage_seeds_fd.json", "w"), indent=1)
    s = res["seed2"]
    print(f"stage seeds: {s['n']} pairs, stage-seed SD of log FID = {s['stage_sd_logfid']:.4f}")
    for k, v in s["pairs"].items():
        print(f"  {k:24s} {v['seed0']:.3f} -> {v['stage_seed1']:.3f} ({v['rel_diff']:+.1%})")
    f = res["fd"]
    print("FD-DINOv2 frontier (route_budget: FID cell / FD cell):")
    for k, v in f["frontier"].items():
        print(f"  {k:10s} FID {v['fid']:.2f} {v['fid_cell']}  FD {v['fd']:.1f} {v['fd_cell']}")
    print("tax FID", {b: round(x, 3) for b, x in f["tax_fid"].items()}, "tax FD", {b: round(x, 3) for b, x in f["tax_fd"].items()})
    print("nat/conv FID", {b: round(x, 3) for b, x in f["nat_over_conv_fid"].items()}, "FD", {b: round(x, 3) for b, x in f["nat_over_conv_fd"].items()})
    print("f*:", {b: (v["size"], v["f_best_fid"], v["f_best_fd"]) for b, v in f["fstar"].items()})
    print("spearman FID vs FD over cells:", round(f["spearman_fid_fd_all_cells"], 3), "| size-order disagreements:", len(f["fid_fd_disagreements"]))
    for d in f["fid_fd_disagreements"]:
        print("  ", d["route"], d["budget"], d["pair"], "FID", [round(x, 2) for x in d["fid"]], "FD", [round(x, 1) for x in d["fd"]])


if __name__ == "__main__":
    main()
