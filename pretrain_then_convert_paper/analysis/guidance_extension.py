"""Effect of the extended guidance search (scripts/eval_gridext.py of the code): per route, how many models moved and by
how much; per budget, the frontiers and the one-step tax under the main protocol's grid (grid1) and with the extended
search. Reads eval/*/summary.json ("grid1" holds the main protocol's values) and eval/*/gridext.json.

  python analysis/guidance_extension.py  ->  analysis/out/guidance_extension.json
"""
import csv
import glob
import json
import math
import re

import numpy as np

from common import ROOT
from common import EVAL
BUD = ("1e19", "3e19", "1e20", "3e20")


def main():
    per = {"fm": [], "imf": []}
    for f in glob.glob(f"{EVAL}/*/gridext.json"):
        d = json.load(open(f))
        s = json.load(open(f"{EVAL}/{d['run_id']}/summary.json"))
        g1 = (s.get("grid1") or s)["fid50k"]
        per[s["route"]].append(dict(run_id=d["run_id"], grid1=g1, ext=s["fid50k"], point=s["point"],
                                    point_grid1=(s.get("grid1") or s)["point"]))
    out = {"models": {}}
    for route, v in per.items():
        r = np.array([x["ext"] / x["grid1"] - 1 for x in v])
        out["models"][route] = dict(n=len(v), n_changed=int((np.abs(r) > 1e-9).sum()), median=float(np.median(r)),
                                    min=float(r.min()), max=float(r.max()))
    # frontiers per budget (seed-mean log FID of a cell, as in the fits), grid1 vs extended
    cells = {}
    for row in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", row["run_id"])
        if not m or "_lr" in row["run_id"]:
            continue
        route = {"r1": "R1", "r2": "NAT", "r3": "CONV"}[m.group(1)]
        c = cells.setdefault((route, m.group(2), m.group(3), m.group(4)), {"ext": [], "grid1": []})
        c["ext"].append(math.log(float(row["fid50k"])))
        c["grid1"].append(math.log(float(row["fid50k_grid1"])))

    def front(route, b, which):
        v = [(math.exp(np.mean(c[which])), k[1]) for k, c in cells.items() if k[0] == route and k[2] == b]
        return min(v)
    out["frontier"] = {}
    for b in BUD:
        out["frontier"][b] = {w: {r: front(r, b, w) for r in ("R1", "CONV", "NAT")} for w in ("grid1", "ext")}
    # the 1e21 runs
    r21 = {}
    for rid in ("v1_XL_1e21_cd_s0", "v2_XL_1e21_f0.60_s0", "v2_H_1e21_f0.60_s0", "v2_H_1e21_cd_s0"):
        s = json.load(open(f"{EVAL}/{rid}/summary.json"))
        r21[rid] = dict(grid1=(s.get("grid1") or s)["fid50k"], ext=s["fid50k"])
    out["1e21"] = r21
    json.dump(out, open(f"{ROOT}/analysis/out/guidance_extension.json", "w"), indent=1)
    print(json.dumps(out["models"], indent=1))
    for b in BUD:
        print(b, {w: {r: (round(v[0], 3), v[1]) for r, v in d.items()} for w, d in out["frontier"][b].items()})
    print(r21)


if __name__ == "__main__":
    main()
