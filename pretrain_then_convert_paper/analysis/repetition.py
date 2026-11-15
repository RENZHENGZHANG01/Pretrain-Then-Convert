"""Data repetition: epochs of every evaluated grid cell (conversions: FM epochs up to the branch point plus conversion
epochs), and, per route, the correlation of the residuals of the (N, C) surface (analysis/surface.py) with log epochs.

  python analysis/repetition.py  ->  analysis/out/repetition.json
"""
import csv
import json
import math
import re
from collections import defaultdict

import numpy as np

from common import BUD, DATA, NPAR, ROOT
from surface import fit as surface_fit, model as surface_model

N_IMG = 1281167
SIZES = ["XS", "S", "B", "M", "L", "XL", "H"]
RUN = json.load(open(f"{DATA}/run_epochs.json"))  # per run: FLOPs per sample and the epochs at the end of training


def cells_with_epochs():
    fm_fps = {s: RUN.get(f"r1_{s}_s0", {}).get("flops_per_sample") for s in SIZES}
    rows = []
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        route, size, b, kind, seed = m.groups()
        ep = RUN.get(r["run_id"], {}).get("epochs")
        if ep is None:
            continue
        if route == "r3":  # FM epochs up to the branch point
            ep += float(kind[1:]) * BUD[b] / fm_fps[size] / N_IMG
        rows.append(dict(route={"r1": "R1", "r2": "NAT", "r3": "CONV"}[route], size=size, budget=b,
                         f=float(kind[1:]) if route == "r3" else (0.0 if route == "r2" else None), seed=int(seed),
                         epochs=float(ep), logfid=math.log(float(r["fid50k"]))))
    return rows


def repetition(rows):
    res = {}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["route"], r["size"], r["budget"], r["f"])].append(r)
    mean = {k: (float(np.mean([x["logfid"] for x in v])), float(np.mean([x["epochs"] for x in v]))) for k, v in cells.items()}
    for route in ("R1", "NAT", "CONV"):  # residuals of the per-route (N, C) surface vs log epochs
        pts = {}
        for (r, s, b, f), (q, ep) in mean.items():
            if r == route and ((s, b) not in pts or q < pts[(s, b)][0]):
                pts[(s, b)] = (q, ep)
        p, _ = surface_fit({k: v[0] for k, v in pts.items()})
        resid = np.array([v[0] - float(surface_model(p, NPAR[s], BUD[b])) for (s, b), v in pts.items()])
        lep = np.log(np.array([v[1] for v in pts.values()]))
        res[route] = dict(n_cells=len(pts), epochs_range=[float(np.exp(lep.min())), float(np.exp(lep.max()))],
                          corr_resid_logepochs=float(np.corrcoef(lep, resid)[0, 1]),
                          slope_resid_per_log_epoch=float(np.polyfit(lep, resid, 1)[0]))
    res["max_epochs_per_route_budget"] = {f"{r}_{b}": float(max(ep for (rr, s, bb, f), (q, ep) in mean.items() if rr == r and bb == b))
                                          for r in ("R1", "NAT", "CONV") for b in BUD}
    # epochs of the frontier models (best cell of each route and budget)
    best = {}
    for (r, s, b, f), (q, ep) in mean.items():
        if (r, b) not in best or q < best[(r, b)][0]:
            best[(r, b)] = (q, ep, s, f)
    res["frontier_epochs"] = {f"{r}_{b}": dict(size=v[2], f=v[3], epochs=v[1]) for (r, b), v in sorted(best.items())}
    return res


def main():
    rows = cells_with_epochs()
    out = dict(summary=repetition(rows), cells=rows)
    json.dump(out, open(f"{ROOT}/analysis/out/repetition.json", "w"), indent=1)
    for route in ("R1", "NAT", "CONV"):
        print(route, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["summary"][route].items()})
    print("largest epochs of a frontier model:", round(max(v["epochs"] for v in out["summary"]["frontier_epochs"].values())))


if __name__ == "__main__":
    main()
