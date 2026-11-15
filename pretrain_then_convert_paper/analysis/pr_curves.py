"""Precision/recall along the guidance sweep (eval/<run>/prcurve.json, scripts/eval_prcurve.py of the code) for the
FID frontier model of each route and budget (analysis/pr_curve_runs.txt). Guidance trades precision for recall, so
routes are compared at matched precision: for each model the attainable set is the upper envelope
R(p) = max recall over sweep points with precision >= p; the gap at a given precision p is R_route(p) - R_ref(p),
evaluated on the overlap of the two precision ranges.

  python analysis/pr_curves.py  ->  analysis/out/pr_curves.json, paper/figures/fig_prcurve
All values at 10k samples (seed 1) against the 10k ADM reference images, so they are not comparable in level with the
50k numbers of Table 1, only with each other.
"""
import json
import os
import re

import numpy as np

import figures as F

from common import ROOT
from common import EVAL
BUD = ("1e19", "3e19", "1e20", "3e20")


def load():
    out = {}
    for rid in open(f"{ROOT}/analysis/pr_curve_runs.txt").read().split():
        p = f"{EVAL}/{rid}/prcurve.json"
        if not os.path.exists(p):
            continue
        m = re.match(r"(r[123])_\w+?_(1e19|3e19|1e20|3e20)_", rid)
        route = {"r1": "R1", "r2": "NAT", "r3": "CONV"}[m.group(1)]
        d = json.load(open(p))
        pts = [(c["precision"], c["recall"], c["fid"], c["point"]) for c in d["curve"]]
        sel = next((c for c in d["curve"] if c["point"] == d["fid_selected_point"]), None)
        out[(route, m.group(2))] = dict(run_id=rid, pts=pts, sel=sel)
    return out


def envelope(pts, ps):
    """max recall among points with precision >= p (nan if none)."""
    P = np.array([q[0] for q in pts]); R = np.array([q[1] for q in pts])
    return np.array([R[P >= p].max() if (P >= p).any() else np.nan for p in ps])


def gap(a, b):
    """recall gap a - b on the overlap of the precision ranges (21 levels)."""
    lo = max(min(q[0] for q in a["pts"]), min(q[0] for q in b["pts"]))
    hi = min(max(q[0] for q in a["pts"]), max(q[0] for q in b["pts"]))
    if hi <= lo:
        return None
    ps = np.linspace(lo, hi, 21)
    g = envelope(a["pts"], ps) - envelope(b["pts"], ps)
    psel = b["sel"]["precision"] if b["sel"] else float(np.median(ps))
    gs = float(envelope(a["pts"], [psel])[0] - envelope(b["pts"], [psel])[0]) if lo <= psel <= hi else None
    return dict(p_range=[float(lo), float(hi)], gap_min=float(np.nanmin(g)), gap_max=float(np.nanmax(g)),
                gap_mean=float(np.nanmean(g)), gap_at_ref_selected_precision=gs)


def main():
    d = load()
    res = dict(models={f"{k[0]}_{k[1]}": v["run_id"] for k, v in d.items()}, native_vs_convert={}, convert_vs_multistep={})
    for b in BUD:
        if ("NAT", b) in d and ("CONV", b) in d:
            res["native_vs_convert"][b] = gap(d[("NAT", b)], d[("CONV", b)])
        if ("CONV", b) in d and ("R1", b) in d:
            res["convert_vs_multistep"][b] = gap(d[("CONV", b)], d[("R1", b)])
    json.dump(res, open(f"{ROOT}/analysis/out/pr_curves.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("native_vs_convert", "convert_vs_multistep")}, indent=1))
    # figure: attainable precision-recall pairs per budget (one panel per budget, one y-axis each)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 4, figsize=(F.FULL, 2.15), sharey=True, sharex=True, layout="constrained")
    for ax, b in zip(axs, BUD):
        for route in F.ROUTES:
            if (route, b) not in d:
                continue
            m = d[(route, b)]
            P = np.array([q[0] for q in m["pts"]]); R = np.array([q[1] for q in m["pts"]])
            ax.plot(P, R, "o", color=F.ROUTE_C[route], ms=2.2, alpha=0.35, mew=0)
            ps = np.linspace(P.min(), P.max(), 60)
            ax.plot(ps, envelope(m["pts"], ps), color=F.ROUTE_C[route], lw=1.2)
            if m["sel"]:
                ax.plot(m["sel"]["precision"], m["sel"]["recall"], "o", color=F.ROUTE_C[route], mec=F.INK, mew=0.7, ms=4.5, zorder=4)
        ax.set_title(f"$C=${F.BTEX[b]}", color=F.INK); ax.set_xlabel("precision")
    axs[0].set_ylabel("recall (10k samples)")
    fig.legend(handles=[Line2D([], [], color=F.ROUTE_C[r], lw=1.2, label=F.ROUTE_N[r]) for r in F.ROUTES]
               + [Line2D([], [], color=F.INK2, marker="o", ls="none", ms=2.2, alpha=0.35, mew=0, label="one guidance setting"),
                  Line2D([], [], color=F.INK2, marker="o", mec=F.INK, ls="none", ms=4.5, label="FID-selected guidance")],
               loc="outside lower center", ncol=5)
    F.save(fig, "fig_prcurve")


if __name__ == "__main__":
    main()
