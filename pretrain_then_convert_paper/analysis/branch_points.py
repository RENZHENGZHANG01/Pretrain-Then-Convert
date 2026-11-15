"""What the FM model hands to the conversion at each branch point: ODE straightness S and the relative error G of a
one-step jump of the FM model (computed from the FM checkpoints by scripts/branch_diagnostics.py of the code, stored
in data/frozen/branch_diagnostics.json; they do not depend on guidance), next to the FID of the conversion from that
point. Three panels with one y-axis each, against the branch fraction f; colour = budget, marker = size.

  python analysis/branch_points.py  ->  paper/figures/fig_branch_points, analysis/out/branch_points.json
"""
import csv
import json
from collections import defaultdict

import numpy as np

import figures as F

from common import DATA, ROOT

COL = {"1e19": "#5598e7", "3e19": "#0d366b"}  # budget: FM blues (light, dark)
MK = {"B": "o", "M": "s"}  # size: marker
DX = {("B", "1e19"): -0.03, ("B", "3e19"): -0.01, ("M", "1e19"): 0.01, ("M", "3e19"): 0.03}  # dodge the error bars


def load():
    """Branch points with the FID-50k of the conversion from each (seed 0), from the current evaluation."""
    rows = json.load(open(f"{DATA}/frozen/branch_diagnostics.json"))
    fids = {r["run_id"]: float(r["fid50k"]) for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv"))}
    for r in rows:
        r["conv_fid50k"] = fids.get(f"r3_{r['size']}_{r['budget']}_f{r['f']:.2f}_s0", r["conv_fid50k"])
    return rows


def main():
    rows = load()
    grp = defaultdict(list)
    for r in rows:
        grp[(r["size"], r["budget"])].append(r)
    summ = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(F.FULL, 2.2), layout="constrained")
    for (size, b), rs in sorted(grp.items()):
        rs = sorted(rs, key=lambda r: r["f"])
        f = np.array([r["f"] for r in rs]); x = -np.log10(1 - f) + DX[(size, b)]  # spreads 0.95 / 0.99 apart
        S = np.array([r["S"] for r in rs]); G = np.array([r["G"] for r in rs])
        q = np.array([r["conv_fid50k"] for r in rs]); qn = q / q.min()
        kw = dict(color=COL[b], marker=MK[size], mec=F.SURF, mew=0.6, ms=4, lw=1.1)
        axs[0].errorbar(x, S, yerr=[r["S_se"] for r in rs], capsize=1.5, elinewidth=0.8, **kw)
        axs[1].errorbar(x, G, yerr=[r["G_se"] for r in rs], capsize=1.5, elinewidth=0.8, **kw)
        axs[2].plot(x, qn, **kw)
        summ[f"{size}_{b}"] = dict(f=f.tolist(), S=S.tolist(), G=G.tolist(), conv_fid50k=q.tolist(),
                                   S_change_pct=float((S[-1] / S[0] - 1) * 100), G_change_pct=float((G[-1] / G[0] - 1) * 100),
                                   fid_change_pct=float((q[-1] / q[0] - 1) * 100))
    titles = ("(a) straightness of the ODE path", "(b) error of a one-step jump", "(c) FID of the conversion")
    ylabels = ("straightness $S$ (0: straight)", "relative jump error $G$", "converted FID / best of the curve")
    for ax, t, yl in zip(axs, titles, ylabels):
        ax.set_title(t, color=F.INK); ax.set_xlabel("branch fraction $f$ (axis in $-\\log(1-f)$)"); ax.set_ylabel(yl)
        ax.set_xticks(-np.log10(1 - np.array([0.5, 0.8, 0.95, 0.99]))); ax.set_xticklabels(["0.5", "0.8", "0.95", "0.99"])
    for ax in axs[:2]:
        lo, hi = ax.get_ylim(); mid = (lo + hi) / 2
        ax.set_ylim(min(lo, mid * 0.8), max(hi, mid * 1.2))  # same relative scale as the FID panel's story
    axs[2].legend(handles=[Line2D([], [], color=COL[b], marker=MK[sz], ms=4, lw=1.1, label=f"{sz}, {F.BTEX[b]}")
                           for sz in ("B", "M") for b in ("1e19", "3e19")], loc="upper left", fontsize=6.4)
    F.save(fig, "fig_branch_points")
    json.dump(summ, open(f"{ROOT}/analysis/out/branch_points.json", "w"), indent=1)
    for k, v in summ.items():
        print(k, "S %+.1f%%  G %+.1f%%  convFID %+.0f%%  (f %.2f -> %.2f)" % (v["S_change_pct"], v["G_change_pct"], v["fid_change_pct"], v["f"][0], v["f"][-1]))


if __name__ == "__main__":
    main()
