"""Why conversion wins: native vs converted one-step models at equal training compute and at an equal number of
training samples (optimizer steps x batch 256). For every cell with an f-curve, the best conversion is compared
with the native model of the same size, (a) at the same FLOPs and (b) at the same number of samples, read off the
native model's FID-vs-samples curve (log-log interpolation between its budgets; outside them, the nearest two
budgets are extended and the cell is marked). A converted model sees f*C/c_FM + (1-f)*C/c_iMF samples, where c is
the training cost per sample (data/flops_table.json); a native one C/c_iMF. Intervals: parametric bootstrap
with the seed-noise model of analysis/noise.py (route-specific SDs, noise shared within a mainline).

  python analysis/matched_steps.py  ->  analysis/out/matched_steps.json, paper/figures/fig_matched, paper/tables/ta_matched.tex
"""
import json
import math

import numpy as np

import figures as F
import noise as nz

from common import EVAL as EV, FT, ROOT

BUD = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
SIZES = ("XS", "S", "B", "M", "L", "XL", "H")

# the native XL run at 1e21 is a point of the native XL curve (so XL@3e20 is interpolated, not extrapolated)
NAT21_XL = math.log(json.load(open(f"{EV}/v1_XL_1e21_cd_s0/summary.json"))["fid50k"])


def native_curves(cells, nat21=NAT21_XL):
    curves = {}
    for size in SIZES:
        nat = sorted((BUD[b], v) for (r, s, b, f), v in cells.items() if r == "NAT" and s == size)
        if size == "XL" and nat21 is not None:
            nat.append((1e21, nat21))
        if len(nat) >= 2:
            curves[size] = (np.log([c / FT[size]["imf"] for c, _ in nat]), np.array([v for _, v in nat]))
    return curves


def native_at(curve, x):
    """log FID of the native model at log-samples x: log-log interpolation; outside, the nearest two points extended."""
    ls, lq = curve
    if ls[0] <= x <= ls[-1]:
        return float(np.interp(x, ls, lq)), False
    i = (len(ls) - 2) if x > ls[-1] else 0
    return float(lq[i] + (lq[i + 1] - lq[i]) / (ls[i + 1] - ls[i]) * (x - ls[i])), True


def native_at_quadratic(curve, x):
    """Sensitivity variant of native_at: a quadratic in log-log through the three budgets
    nearest to x (the native curves are convex, so straight lines between budgets overstate the native FID); sizes with
    two native points fall back to the straight line."""
    ls, lq = curve
    if len(ls) < 3:
        return native_at(curve, x)
    idx = np.sort(np.argsort(np.abs(ls - x))[:3])
    return float(np.polyval(np.polyfit(ls[idx], lq[idx], 2), x)), not ls[0] <= x <= ls[-1]


def analyse(cells, nat21=NAT21_XL, at=None):
    at = at or native_at
    out = []
    curves = native_curves(cells, nat21)
    for size, curve in curves.items():
        for b, C in BUD.items():
            conv = [(v, f) for (r, s, bb, f), v in cells.items() if r == "CONV" and s == size and bb == b]
            nat_c = [v for (r, s, bb, f), v in cells.items() if r == "NAT" and s == size and bb == b]
            if not conv or not nat_c:
                continue
            q, f = min(conv)
            samples = f * C / FT[size]["fm"] + (1 - f) * C / FT[size]["imf"]
            qn, extra = at(curve, math.log(samples))
            out.append(dict(size=size, budget=b, f=f, conv=math.exp(q), native_same_flops=math.exp(nat_c[0]),
                            native_same_samples=math.exp(qn), samples_ratio=samples / (C / FT[size]["imf"]),
                            ratio_flops=math.exp(nat_c[0] - q), ratio_samples=math.exp(qn - q), extrapolated=extra))
    return out


def exchange(cells, nat21=NAT21_XL, at=None):
    """The equal-samples ratio (native FID at the converted model's sample count / converted FID) for every split of
    every cell with an f-curve: the value of FM samples relative to one-step samples as a function of f."""
    at = at or native_at
    out = []
    curves = native_curves(cells, nat21)
    for size, curve in curves.items():
        for b, C in BUD.items():
            for (r, s, bb, f), q in cells.items():
                if r != "CONV" or s != size or bb != b:
                    continue
                samples = f * C / FT[size]["fm"] + (1 - f) * C / FT[size]["imf"]
                qn, extra = at(curve, math.log(samples))
                out.append(dict(size=size, budget=b, f=f, ratio=math.exp(qn - q), extrapolated=extra))
    return out


F_BINS = ((0.0, 0.2, "0.08-0.10"), (0.2, 0.3, "0.24-0.27"), (0.45, 0.55, "0.50"), (0.6, 0.7, "0.65"), (0.75, 0.85, "0.80"),
          (0.9, 0.97, "0.95"), (0.97, 1.0, "0.99"))


def exchange_summary(rows):
    out = {}
    for lo, hi, lab in F_BINS:
        v = [r["ratio"] for r in rows if lo <= r["f"] < hi and not r["extrapolated"]]
        if v:
            out[lab] = dict(n=len(v), median=float(np.median(v)), lo=float(min(v)), hi=float(max(v)))
    return out


def main():
    raw = nz.load_runs()
    base = nz.base_of(raw)
    noise = nz.estimate(raw)
    est = analyse(base)
    rng = np.random.default_rng(20261022)
    boots, xboots = [], []
    for _ in range(2000):
        cells = nz.perturb(raw, base, noise, rng)
        n21 = NAT21_XL + rng.normal(0, noise["sd"]["NAT"])
        boots.append(analyse(cells, n21))
        xboots.append(exchange_summary(exchange(cells, n21)))
    xs = exchange_summary(exchange(base))
    for lab, e in xs.items():
        v = [b[lab]["median"] for b in xboots if lab in b]
        e["median_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    xq = exchange_summary(exchange(base, at=native_at_quadratic))
    json.dump(dict(by_f=xs, by_f_quadratic=xq, cells=exchange(base)), open(f"{ROOT}/analysis/out/exchange_rate.json", "w"), indent=1)
    print("exchange rate by f:", {k: (round(v["median"], 3), [round(x, 3) for x in v["median_ci95"]], v["n"]) for k, v in xs.items()})
    for i, e in enumerate(est):
        rf = [b[i]["ratio_flops"] for b in boots]; rs = [b[i]["ratio_samples"] for b in boots]
        e["ratio_flops_ci95"] = [float(np.percentile(rf, 2.5)), float(np.percentile(rf, 97.5))]
        e["ratio_samples_ci95"] = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
    med = [float(np.median([x["ratio_samples"] for x in b])) for b in boots]
    med_in = [float(np.median([x["ratio_samples"] for x in b if not x["extrapolated"]])) for b in boots]
    estq = analyse(base, at=native_at_quadratic)
    gm = [float(np.exp(np.mean(np.log([x["ratio_samples"] for x in b])))) for b in boots]
    summ = dict(n=len(est), ratio_flops_range=[min(e["ratio_flops"] for e in est), max(e["ratio_flops"] for e in est)],
                ratio_samples_range=[min(e["ratio_samples"] for e in est), max(e["ratio_samples"] for e in est)],
                ratio_samples_median=float(np.median([e["ratio_samples"] for e in est])),
                ratio_samples_median_ci95=[float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                ratio_samples_geomean=float(np.exp(np.mean(np.log([e["ratio_samples"] for e in est])))),
                ratio_samples_geomean_ci95=[float(np.percentile(gm, 2.5)), float(np.percentile(gm, 97.5))],
                samples_ratio_range=[min(e["samples_ratio"] for e in est), max(e["samples_ratio"] for e in est)],
                n_interpolated=sum(not e["extrapolated"] for e in est),
                ratio_samples_median_interp=float(np.median([e["ratio_samples"] for e in est if not e["extrapolated"]])),
                ratio_samples_median_interp_ci95=[float(np.percentile(med_in, 2.5)), float(np.percentile(med_in, 97.5))],
                ratio_samples_range_interp=[min(e["ratio_samples"] for e in est if not e["extrapolated"]),
                                            max(e["ratio_samples"] for e in est if not e["extrapolated"])],
                ratio_samples_median_interp_quadratic=float(np.median([e["ratio_samples"] for e in estq if not e["extrapolated"]])))
    json.dump(dict(cells=est, summary=summ), open(f"{ROOT}/analysis/out/matched_steps.json", "w"), indent=1)
    for e in est:
        print(f"{e['size']}@{e['budget']} f={e['f']:.2f} conv {e['conv']:.2f} | native same FLOPs {e['native_same_flops']:.2f} "
              f"(x{e['ratio_flops']:.2f}) | same samples {e['native_same_samples']:.2f} (x{e['ratio_samples']:.2f} "
              f"[{e['ratio_samples_ci95'][0]:.2f}, {e['ratio_samples_ci95'][1]:.2f}]){' extrapolated' if e['extrapolated'] else ''} | samples x{e['samples_ratio']:.2f}")
    print(json.dumps(summ, indent=1))
    # table
    L = ["\\begin{tabular}{lccccc}", "\\toprule",
         "Cell & best $f$ & samples (convert $\\div$ native) & native $\\div$ convert, equal FLOPs & native $\\div$ convert, equal samples \\\\",
         "\\midrule"]
    for e in est:
        mark = "$^\\dagger$" if e["extrapolated"] else ""
        L.append(f"{e['size']}@{e['budget']} & {e['f']:.2f} & {e['samples_ratio']:.2f} & {e['ratio_flops']:.2f} "
                 f"[{e['ratio_flops_ci95'][0]:.2f}, {e['ratio_flops_ci95'][1]:.2f}] & {e['ratio_samples']:.2f}{mark} "
                 f"[{e['ratio_samples_ci95'][0]:.2f}, {e['ratio_samples_ci95'][1]:.2f}] \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/ta_matched.tex", "w").write("\n".join(L) + "\n")
    # figure: per cell, native/convert at equal FLOPs and at equal samples
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(3.4, 2.8), layout="constrained")
    est_f = sorted(est, key=lambda e: (list(F.BUD).index(e["budget"]), F.SIZES.index(e["size"])))
    y = np.arange(len(est_f))[::-1]
    for yi, e in zip(y, est_f):
        ax.plot([e["ratio_samples"], e["ratio_flops"]], [yi, yi], color=F.GRID, lw=2.0, zorder=1)
        ax.errorbar(e["ratio_flops"], yi, xerr=[[e["ratio_flops"] - e["ratio_flops_ci95"][0]], [e["ratio_flops_ci95"][1] - e["ratio_flops"]]],
                    fmt="o", color=F.ROUTE_C["CONV"], mec=F.SURF, mew=0.7, ms=4, lw=0.8, capsize=0, zorder=3)
        ax.errorbar(e["ratio_samples"], yi, xerr=[[e["ratio_samples"] - e["ratio_samples_ci95"][0]], [e["ratio_samples_ci95"][1] - e["ratio_samples"]]],
                    fmt="s", color=F.INK, mfc=F.SURF if e["extrapolated"] else F.INK, mec=F.INK, mew=0.8, ms=3.8, lw=0.8, capsize=0, zorder=3)
    ax.axvline(1.0, color=F.INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([f"{e['size']}, {F.BTEX[e['budget']]}" for e in est_f], fontsize=6.5)
    ax.set_xscale("log"); ax.set_xticks([0.8, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.8", "1", "1.5", "2", "3"]); ax.minorticks_off()
    ax.set_xlabel("FID of native / FID of best conversion")
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C["CONV"], marker="o", ls="none", ms=4, label="equal training FLOPs"),
                       Line2D([], [], color=F.INK, marker="s", ls="none", ms=3.8, label="equal training samples"),
                       Line2D([], [], color=F.INK, marker="s", mfc=F.SURF, ls="none", ms=3.8, label="beyond the largest native budget"),
                       Line2D([], [], color=F.GRID, lw=2.0, label="the two ratios of a cell")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.9, markerscale=1, columnspacing=0.8,
              handletextpad=0.4, borderaxespad=0.1)
    F.save(fig, "fig_matched")


if __name__ == "__main__":
    main()
