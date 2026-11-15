"""Generality: the pretraining-fraction curve with shortcut models on B@1e19, M@1e19 and M@3e19, three seeds (group
`shortcut` of the code). The shortcut objective uses fixed level counts and unconditional passes only at the smallest
step, so its measured FLOPs are the algorithmic ones (1.198x FM). Seed k of every curve branches from one FM mainline
(seed 0: the mainlines of the iMF study, with the f = 0.25 points from the FM re-runs at 1e19; seeds 1-2: fresh FM
mainlines carrying every branch point) and has its own native shortcut mainline. Next to it: the iMF curve of the same
cells (seed 0, and seed 1 where replicated).

Statistics: per cell, the curve of the seed-mean log FID; f* (parabola in log(1-f) through the best split and its
neighbours) per seed and of the mean curve; native / best convert per seed and of the mean curve, with a 95%
interval from the seed-to-seed spread (t with n-1 df on the log ratio). 1 NFE vs 2 NFE on the same 10k samples
(seed 1, selected point).

  python analysis/shortcut.py  ->  analysis/out/shortcut.json, paper/figures/fig_shortcut, paper/tables/ta_shortcut.tex
"""
import json
import math
import os
import re

import numpy as np
from scipy import stats as sps

import figures as F

from common import EVAL, ROOT

CELLS = [("B", "1e19"), ("M", "1e19"), ("M", "3e19")]
SEEDS = (0, 1, 2)


def summ(rid):
    p = f"{EVAL}/{rid}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fval(s, key):
    """Current value (after the guidance-grid extension where it applied, as in the main analysis)."""
    return None if s is None else s.get(key)


def curve(prefix_n, prefix_c, size, b, seed, key="fid50k"):
    out = {}
    v = fval(summ(f"{prefix_n}_{size}_{b}_cd_s{seed}"), key)
    if v is not None:
        out[0.0] = v
    for d in os.listdir(EVAL):  # exact names only: no learning-rate (_lr, _flr) or initialization (_naive) variants
        m = re.fullmatch(rf"{prefix_c}_{size}_{b}_f([0-9.]+)_s{seed}", d)
        if m:
            v = fval(summ(d), key)
            if v is not None:
                out[round(float(m.group(1)), 2)] = v
    return dict(sorted(out.items()))


def parabola_fstar(c):
    fs = [f for f in c if f < 1]
    if len(fs) < 3:
        return None
    u = np.log(1 - np.array(fs) + 1e-12); q = np.log(np.array([c[f] for f in fs])); o = np.argsort(u); u, q = u[o], q[o]
    i = int(np.argmin(q)); j = min(max(i - 1, 0), len(u) - 3)
    cf = np.polyfit(u[j:j + 3], q[j:j + 3], 2)
    if cf[0] <= 0 or not 0 < i < len(u) - 1:
        return float(1 - math.exp(u[i]))
    return float(1 - math.exp(min(max(-cf[1] / (2 * cf[0]), u[j]), u[j + 2])))


def nat_ratio(c):
    conv = [v for f, v in c.items() if f > 0]
    return c[0.0] / min(conv) if 0.0 in c and conv else None


def ci_log(vals):
    v = np.log(np.array(vals))
    if len(v) < 2:
        return None
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return [float(math.exp(v.mean() - h)), float(math.exp(v.mean() + h))]


def cell_sc(size, b):
    per = {k: curve("t2", "t3", size, b, k) for k in SEEDS}
    per = {k: c for k, c in per.items() if c}
    fs = sorted(set.intersection(*[set(c) for c in per.values()])) if per else []
    mean = {f: float(math.exp(np.mean([math.log(per[k][f]) for k in per]))) for f in fs}
    sd = {f: float(np.std([math.log(per[k][f]) for k in per], ddof=1)) for f in fs} if len(per) > 1 else {}
    r_seed = {k: nat_ratio(c) for k, c in per.items()}
    rs = [r for r in r_seed.values() if r]
    f_seed = {k: parabola_fstar(c) for k, c in per.items()}
    best_f = min(mean, key=mean.get) if mean else None
    two10 = {k: curve("t2", "t3", size, b, k, "fid_2nfe_10k") for k in per}
    def gmean(curves):
        fs_ = sorted(set.intersection(*[set(c) for c in curves.values()])) if curves else []
        return {f: float(math.exp(np.mean([math.log(c[f]) for c in curves.values()]))) for f in fs_}
    m1, m2 = gmean(one10 := {k: curve("t2", "t3", size, b, k, "fid10k_sel") for k in per}), \
        gmean({k: curve("t2", "t3", size, b, k, "fid_2nfe_10k") for k in per})
    rel = lambda c: {f: v / min(c.values()) for f, v in c.items()} if c else {}  # noqa: E731
    return dict(seeds=sorted(per), per_seed=per, mean_curve=mean, sd_log=sd, best_f=best_f,
                mean_10k_1nfe=m1, mean_10k_2nfe=m2, rel_10k_1nfe=rel(m1), rel_10k_2nfe=rel(m2),
                penalty_f08=mean[0.8] / mean[best_f] if 0.8 in mean else None,
                fstar_mean_curve=parabola_fstar(mean) if len(mean) >= 3 else None, fstar_per_seed=f_seed,
                native_over_best_convert=dict(mean_curve=nat_ratio(mean), per_seed=r_seed,
                                              # point estimate consistent with the interval over seeds
                                              geomean_seeds=float(math.exp(np.mean(np.log(rs)))) if rs else None,
                                              ci95_over_seeds=ci_log(rs) if len(rs) >= 2 else None),
                penalty_f095=mean[0.95] / mean[best_f] if 0.95 in mean else None,
                penalty_f099=mean[0.99] / mean[best_f] if 0.99 in mean else None,
                nfe1_10k=one10, nfe2_10k=two10)


def cell_imf(size, b):
    per = {k: curve("r2", "r3", size, b, k) for k in (0, 1)}
    per = {k: c for k, c in per.items() if c}
    return dict(per_seed=per, fstar_per_seed={k: parabola_fstar(c) for k, c in per.items()},
                native_over_best_convert={k: nat_ratio(c) for k, c in per.items()})


def main():
    out = {}
    for size, b in CELLS:
        out[f"{size}_{b}"] = dict(sc=cell_sc(size, b), imf=cell_imf(size, b))
    json.dump(out, open(f"{ROOT}/analysis/out/shortcut.json", "w"), indent=1, default=float)
    for k, v in out.items():
        s = v["sc"]
        print(k, "seeds", s["seeds"], "| mean", {f: round(x, 1) for f, x in s["mean_curve"].items()},
              "| f* mean", s["fstar_mean_curve"] and round(s["fstar_mean_curve"], 2),
              "per seed", {kk: vv and round(vv, 2) for kk, vv in s["fstar_per_seed"].items()},
              "| native/best", s["native_over_best_convert"])
        print("   imf", {kk: {f: round(x, 2) for f, x in c.items()} for kk, c in v["imf"]["per_seed"].items()})
    table(out)
    figure(out)


def table(out):
    cols = [(0.0,), (0.25, 0.27), (0.5,), (0.65,), (0.8,), (0.95,), (0.99,)]
    head = ["native", "$f{\\approx}0.25^a$"] + [f"$f{{=}}{c[0]:.2f}$" for c in cols[2:]]
    L = ["\\begin{tabular}{ll" + "c" * len(cols) + "cc}", "\\toprule",
         "Cell & Objective & " + " & ".join(head) + " & $\\fstar$ & native $\\div$ best convert \\\\", "\\midrule"]

    def val(c, cc, fmt="{:.1f}"):
        for f in cc:
            for g in c:
                if abs(g - f) < 0.006:
                    return fmt.format(c[g])
        return "--"
    for size, b in CELLS:
        v = out[f"{size}_{b}"]
        for k, c in v["imf"]["per_seed"].items():
            fsr = v["imf"]["fstar_per_seed"][k]
            nb = v["imf"]["native_over_best_convert"][k]
            L.append(f"{size}@{b} & iMF, seed {k} & " + " & ".join(val(c, cc) for cc in cols) +
                     f" & {fsr:.2f} & {nb:.2f} \\\\" if fsr is not None and nb else
                     f"{size}@{b} & iMF, seed {k} & " + " & ".join(val(c, cc) for cc in cols) + " & -- & -- \\\\")
        s = v["sc"]
        if s["mean_curve"]:
            nb = s["native_over_best_convert"]
            ci = nb["ci95_over_seeds"]
            fsr = s["fstar_mean_curve"]
            fs_txt = f"{fsr:.2f}" if fsr is not None else "--"
            nb_txt = (f"{nb['geomean_seeds']:.2f}" if nb["geomean_seeds"] else "--") + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            L.append(f"{size}@{b} & Shortcut, {len(s['seeds'])}-seed mean & " + " & ".join(val(s["mean_curve"], cc) for cc in cols) +
                     f" & {fs_txt} & {nb_txt} \\\\")
            if s["sd_log"]:
                L.append(" & \\quad SD of $\\log$ FID & " + " & ".join(val(s["sd_log"], cc, "{:.3f}") for cc in cols) + " & & \\\\")
        L.append("\\midrule" if (size, b) != CELLS[-1] else "\\bottomrule")
    L.append("\\end{tabular}")
    open(f"{ROOT}/paper/tables/ta_shortcut.tex", "w").write("\n".join(L) + "\n")


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    col = {"imf": F.ROUTE_C["CONV"], "sc": F.INK}
    fig, axs = plt.subplots(1, len(CELLS), figsize=(F.FULL, 2.3), sharey=True, layout="constrained")
    for ax, (size, b) in zip(axs, CELLS):
        v = out[f"{size}_{b}"]
        c0 = v["imf"]["per_seed"].get(0)
        if c0:
            best = min(c0.values())
            ax.plot(list(c0), np.array(list(c0.values())) / best, "-o", color=col["imf"], mec=F.SURF, mew=0.7, ms=4)
        s = v["sc"]
        if s["mean_curve"]:
            best = min(s["mean_curve"].values())
            for k, c in s["per_seed"].items():  # individual seeds, thin
                ax.plot(list(c), np.array(list(c.values())) / best, "-", color=col["sc"], lw=0.6, alpha=0.45)
            ax.plot(list(s["mean_curve"]), np.array(list(s["mean_curve"].values())) / best, "-s", color=col["sc"],
                    mec=F.SURF, mew=0.7, ms=4)
            for fv, qv in s["mean_curve"].items():  # off-scale points (late conversions): value at the top edge
                if qv / best > 3.0:
                    ax.annotate(f"{qv / best:.1f}↑", (fv, 3.0), xytext=(-10, -2), textcoords="offset points", ha="right",
                                va="top", fontsize=6.3, color=F.INK, zorder=6,
                                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        cb = {"1e19": "$10^{19}$", "3e19": "$3{\\times}10^{19}$"}[b]
        ax.set_title(f"{size} at $C=${cb}", color=F.INK); ax.set_xlabel("pretraining fraction $f$")
        ax.set_xlim(-0.03, 1.03); ax.set_yscale("log"); ax.set_ylim(0.9, 3.0)
        ax.set_yticks([1, 1.25, 1.5, 2, 3]); ax.set_yticklabels(["1", "1.25", "1.5", "2", "3"])
        ax.minorticks_off()
    axs[0].set_ylabel("FID / best FID of the curve")
    fig.legend(handles=[Line2D([], [], color=col["imf"], marker="o", mec=F.SURF, label="iMF (seed 0)"),
                        Line2D([], [], color=col["sc"], marker="s", mec=F.SURF, label="shortcut models (3-seed mean; thin: seeds)")],
               loc="outside lower center", ncol=2)
    F.save(fig, "fig_shortcut")


if __name__ == "__main__":
    main()
