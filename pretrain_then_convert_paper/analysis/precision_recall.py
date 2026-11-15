"""Where do the one-step tax and the native deficit show up? Improved precision / recall (k = 3, Inception pool
features, ADM reference images) of the frontier models (best FID-50k per route and budget, seed-averaged where
replicated), at each model's FID-selected guidance; plus agreement of FID against the JiT statistics and against the
ADM reference batch over all evaluated models.

  python analysis/precision_recall.py  ->  analysis/out/precision_recall.json, paper/figures/fig_precision_recall
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import figures as F

from common import EVAL, ROOT
BUD = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}


def load():
    cells = defaultdict(list)
    allrows = []
    EV = EVAL
    for d in sorted(os.listdir(EV)):
        if not re.match(r"(r[123]_|v[12]_)", d) or "_ext" in d or not os.path.exists(f"{EV}/{d}/extra.json"):
            continue
        ex = json.load(open(f"{EV}/{d}/extra.json"))
        if "fid_adm" not in ex:
            continue
        r = dict(run_id=d, fid50k_jit=json.load(open(f"{EV}/{d}/summary.json"))["fid50k"], fid_adm=ex["fid_adm"],
                 precision=ex["precision"], recall=ex["recall"])
        allrows.append(r)
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        route = {"r1": "R1", "r2": "NAT", "r3": "CONV"}[m.group(1)]
        cells[(route, m.group(2), m.group(3), m.group(4))].append({k: float(r[k]) for k in ("fid50k_jit", "fid_adm", "precision", "recall")})
    mean = {k: {q: float(np.mean([x[q] for x in v])) for q in v[0]} | {"n_seeds": len(v)} for k, v in cells.items()}
    return mean, allrows


def main():
    mean, allrows = load()
    front = {}
    for route in F.ROUTES:
        for b in BUD:
            k = min((k for k in mean if k[0] == route and k[2] == b), key=lambda k: mean[k]["fid50k_jit"])
            front[(route, b)] = dict(mean[k], size=k[1], kind=k[3])
    out = dict(frontier={f"{r}_{b}": v for (r, b), v in front.items()})
    out["tax_conv_minus_r1"] = {b: dict(dP=front[("CONV", b)]["precision"] - front[("R1", b)]["precision"],
                                        dR=front[("CONV", b)]["recall"] - front[("R1", b)]["recall"]) for b in BUD}
    out["native_minus_conv"] = {b: dict(dP=front[("NAT", b)]["precision"] - front[("CONV", b)]["precision"],
                                        dR=front[("NAT", b)]["recall"] - front[("CONV", b)]["recall"]) for b in BUD}
    fj = np.array([float(r["fid50k_jit"]) for r in allrows]); fa = np.array([float(r["fid_adm"]) for r in allrows])
    out["fid_jit_vs_adm"] = dict(n=len(fj), spearman=float(spearmanr(fj, fa)[0]),
                                 median_rel_diff=float(np.median((fa - fj) / fj)), max_abs_diff=float(np.abs(fa - fj).max()))
    json.dump(out, open(f"{ROOT}/analysis/out/precision_recall.json", "w"), indent=1)
    # figure: precision and recall of the frontier models vs compute (two panels, one y-axis each)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = np.array(list(BUD.values()))
    fig, axs = plt.subplots(1, 2, figsize=(F.FULL, 2.3), layout="constrained")
    for ax, q, title in ((axs[0], "precision", "(a) precision (fidelity)"), (axs[1], "recall", "(b) recall (coverage)")):
        ax.set_xscale("log")
        for route in F.ROUTES:
            ax.plot(cs, [front[(route, b)][q] for b in BUD], "-o", color=F.ROUTE_C[route], mec=F.SURF, mew=1.2, label=F.ROUTE_N[route])
        ax.set_title(title, color=F.INK); ax.set_xlabel("training compute $C$ (FLOPs)")
        ax.set_xlim(7e18, 4.6e20); ax.minorticks_off()
        ax.set_xticks(cs); ax.set_xticklabels([F.BTEX[b] for b in BUD])
    axs[0].set_ylabel("value at the FID-selected guidance")
    fig.legend(handles=[plt.Line2D([], [], color=F.ROUTE_C[r], marker="o", lw=1.2, label=F.ROUTE_N[r]) for r in F.ROUTES],
               loc="outside upper center", ncol=3)
    F.save(fig, "fig_precision_recall")
    print(json.dumps({k: out[k] for k in ("tax_conv_minus_r1", "native_minus_conv", "fid_jit_vs_adm")}, indent=1))


if __name__ == "__main__":
    main()
