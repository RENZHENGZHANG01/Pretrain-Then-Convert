"""Conversion learning rate: a sweep of the conversion-phase rate across the f-curve of M@3e19, the short conversions
of B@1e19 and three splits of XL@3e20 (group `conversion_lr` of the code). Every run branches from the same FM
milestone and uses stage seed 0, so it is compared with the seed-0 run at the default rate (1e-4) of the same cell
(paired comparison; the between-seed spread of short conversions is large, e.g. 8.77 vs 12.78 for M@3e19 f=0.99).

  python analysis/conversion_lr.py  ->  analysis/out/conversion_lr.json, paper/tables/ta_ctune.tex
"""
import json
import math
import os

import numpy as np

from common import ROOT
from common import EVAL
LRS = (5e-5, 1e-4, 2e-4, 4e-4)
CELLS = {("M", "3e19"): (0.27, 0.5, 0.65, 0.8, 0.95, 0.99), ("B", "1e19"): (0.25, 0.5, 0.65, 0.8, 0.95, 0.99),
         ("XL", "3e20"): (0.27, 0.5, 0.65, 0.8, 0.95, 0.99)}
# swept (cell, f, lr) besides the default rate (groups conversion_lr and lr of osr/experiments.py)
PLANNED = {("M", f, lr) for f in (0.5, 0.65, 0.8, 0.95, 0.99) for lr in (5e-5, 2e-4, 4e-4)} | \
          {("B", 0.8, 5e-5), ("B", 0.8, 2e-4)} | {("B", f, lr) for f in (0.95, 0.99) for lr in (2e-4, 4e-4)} | \
          {("XL", f, 5e-5) for f in (0.5, 0.65, 0.8)}


def fid(run_id):
    p = f"{EVAL}/{run_id}/summary.json"
    return json.load(open(p))["fid50k"] if os.path.exists(p) else None


def run_id(size, b, f, lr):
    return f"r3_{size}_{b}_f{f:.2f}" + ("" if lr == 1e-4 else f"_lr{lr:g}") + "_s0"


def parabola_fstar(fs, qs):
    """f* by a parabola in log(1-f) through the best point and its neighbours (as frontier.local_parabola_f)."""
    u = np.log(1 - np.array(fs)); q = np.log(np.array(qs)); o = np.argsort(u); u, q = u[o], q[o]
    i = int(np.argmin(q)); j = min(max(i - 1, 0), len(u) - 3)
    cf = np.polyfit(u[j:j + 3], q[j:j + 3], 2)
    if cf[0] <= 0 or not 0 < i < len(u) - 1:
        return float(1 - math.exp(u[i]))
    return float(1 - math.exp(min(max(-cf[1] / (2 * cf[0]), u[j]), u[j + 2])))


def main():
    out = {}
    L = ["\\begin{tabular}{llccccc}", "\\toprule",
         "Cell & $f$ & $5{\\times}10^{-5}$ & $10^{-4}$ (default) & $2{\\times}10^{-4}$ & $4{\\times}10^{-4}$ & best $\\div$ default \\\\", "\\midrule"]
    for (size, b), fs in CELLS.items():
        rows = {}
        for f in fs:
            v = {lr: fid(run_id(size, b, f, lr)) for lr in LRS}
            if v[1e-4] is None:
                continue
            done = {lr: x for lr, x in v.items() if x is not None}
            best_lr = min(done, key=done.get)
            rows[f] = dict(fid=v, best_lr=best_lr, best=done[best_lr], gain=done[best_lr] / v[1e-4] - 1,
                           complete=all(v[lr] is not None for lr in LRS if (size, f, lr) in PLANNED))
            cells = " & ".join(f"{v[lr]:.2f}" if v[lr] is not None else ("\\dots" if (size, f, lr) in PLANNED else "--") for lr in LRS)
            L.append(f"{size}@{b} & {f:.2f} & {cells} & {done[best_lr] / v[1e-4]:.3f} \\\\")
        L.append("\\bottomrule" if size == "XL" else "\\midrule")
        fsd = [f for f in rows]
        res = dict(rows={f"{f:.2f}": r for f, r in rows.items()})
        if len(fsd) >= 3:
            res["fstar_default"] = parabola_fstar(fsd, [rows[f]["fid"][1e-4] for f in fsd])
            res["fstar_tuned"] = parabola_fstar(fsd, [rows[f]["best"] for f in fsd])
            best_default = min(rows[f]["fid"][1e-4] for f in fsd); best_tuned = min(rows[f]["best"] for f in fsd)
            res["penalty_default"] = {f"{f:.2f}": rows[f]["fid"][1e-4] / best_default for f in fsd}
            res["penalty_tuned"] = {f"{f:.2f}": rows[f]["best"] / best_tuned for f in fsd}
        out[f"{size}_{b}"] = res
    L.append("\\end{tabular}")
    open(f"{ROOT}/paper/tables/ta_ctune.tex", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{ROOT}/analysis/out/conversion_lr.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {f: (round(r["best"], 2), r["best_lr"], f"{r['gain']:+.1%}", "complete" if r["complete"] else "partial") for f, r in v["rows"].items()})
        for key in ("fstar_default", "fstar_tuned", "penalty_default", "penalty_tuned"):
            if key in v:
                print("  ", key, v[key] if not isinstance(v[key], dict) else {f: round(x, 2) for f, x in v[key].items()})


if __name__ == "__main__":
    main()
