"""Figure style and helpers shared by the figure scripts, the IsoFLOP figure, the f* figure of the appendix, and the
native compute-optimal size prediction.

  python analysis/figures.py  ->  paper/figures/fig_isoflop, fig_fstar_stab; analysis/out/nstar_pred.json

Colour (dataviz method, validated with validate_palette.js):
  routes  categorical slots 1-3  multi-step #2a78d6 / convert #eb6834 / native #1baf7a  (all-pairs PASS;
          native is below 3:1 on the surface -> every route curve carries a direct label, plus the table view)
  budgets ordinal blue ramp #86b6ef #3987e5 #1c5cab #0d366b (ordinal PASS)
Text in ink colours, recessive grid, one y-axis per panel, legend + direct labels on every multi-series panel.
"""
import csv
import io
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

from common import FIGS as OUT, FT, ROOT  # noqa: E402

NPAR = {k: v["N_inference"] for k, v in FT.items()}
BUD = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
SIZES = ["XS", "S", "B", "M", "L", "XL", "H"]
ROUTES = ("R1", "CONV", "NAT")
ROUTE_C = {"R1": "#2a78d6", "CONV": "#eb6834", "NAT": "#1baf7a"}
ROUTE_N = {"R1": "multi-step (50 steps)", "CONV": "pretrain → convert (1 NFE)", "NAT": "native one-step (1 NFE)"}
ROUTE_S = {"R1": "multi-step", "CONV": "pretrain → convert", "NAT": "native one-step"}
BUD_C = {"1e19": "#b1b0aa", "3e19": "#83827d", "1e20": "#56554f", "3e20": "#1f1f1d"}  # ordinal gray (light end 2.2:1)
BTEX = {"1e19": "$10^{19}$", "3e19": "$3{\\times}10^{19}$", "1e20": "$10^{20}$", "3e20": "$3{\\times}10^{20}$"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e4", "#ffffff"


def out_json(name):
    """An output of an earlier analysis step (analysis/out/<name>.json)."""
    return json.load(open(f"{ROOT}/analysis/out/{name}.json"))


# Figures are drawn at their printed size (ICML: column 3.25 in, text 6.75 in) and saved without cropping, so point
# sizes below are the printed sizes and every figure shares one type scale. STIX matches the Times body text; fonts are
# embedded as TrueType (no Type 3 fonts in the PDF).
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7.0, "axes.titlesize": 7.4, "axes.labelsize": 7.0, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "legend.title_fontsize": 6.3,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.45, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 4.0,
                     "axes.titlelocation": "left", "axes.titlepad": 4.0, "legend.handlelength": 1.6,
                     "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35, "legend.columnspacing": 1.0,
                     "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
                     "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04})
HALF, FULL = 3.25, 6.75
plt.rcParams.update({"image.composite_image": False, "image.interpolation": "none", "legend.handlelength": 2.2})

# Super- and subscripts at 80% instead of mathtext's 70%, so tick exponents and index labels print at >= 5 pt.
import matplotlib._mathtext as _mathtext  # noqa: E402
_mathtext.SHRINK_FACTOR = 0.8

# Legend handles: a line with a white-edged marker prints like a dashed line at small sizes ("-o-"); give legend markers
# their own colour as edge (hollow markers keep their coloured edge and white face).
from matplotlib.colors import same_color  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
_legend_init = Legend.__init__


def _legend_init_solid(self, *a, **k):
    _legend_init(self, *a, **k)
    for h in self.legend_handles:
        if isinstance(h, Line2D) and h.get_marker() not in (None, "None", "", " "):
            try:
                if same_color(h.get_markeredgecolor(), SURF) or same_color(h.get_markeredgecolor(), "white"):
                    h.set_markeredgecolor(h.get_color())
            except ValueError:
                pass


Legend.__init__ = _legend_init_solid


def load():
    """(route, size, budget, f) -> FID-50k, geometric mean over seeds."""
    raw = defaultdict(list)
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        route, size, b, kind, seed = m.groups()
        key = {"r1": "R1", "r2": "NAT"}.get(route, "CONV")
        f = 0.0 if route == "r2" else (float(kind[1:]) if route == "r3" else None)
        raw[(key, size, b, f)].append(float(r["fid50k"]))
    return {k: float(np.exp(np.mean(np.log(v)))) for k, v in raw.items()}


def best(cells, route, b=None, size=None):
    v = [q for (r, s, bb, f), q in cells.items() if r == route and (b is None or bb == b) and (size is None or s == size)]
    return min(v) if v else None


def logaxis(ax, which, ticks, labels=None):
    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FixedFormatter(labels or [f"{t:g}" for t in ticks]))
    axis.set_minor_formatter(NullFormatter())


def inline_label(ax, fn, x0, text, off=3.5, **kw):
    """Text along curve y = fn(x) on log-log axes, centred at x0, rotated to the local display slope, just above it."""
    p1 = ax.transData.transform((x0 / 1.08, fn(x0 / 1.08)))
    p2 = ax.transData.transform((x0 * 1.08, fn(x0 * 1.08)))
    th = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    ax.annotate(text, (x0, fn(x0)), xytext=(-math.sin(th) * off, math.cos(th) * off), textcoords="offset points",
                rotation=math.degrees(th), rotation_mode="anchor", ha="center", va="bottom", color=INK, **kw)


def end_labels(ax, items, gap=8.5, dx=4, **kw):
    """Direct labels at line ends, pushed apart vertically to at least `gap` points. items: [(x, y, text)]."""
    fig = ax.figure
    fig.canvas.draw()  # final (auto)limits before converting to display coordinates
    pts = sorted(((ax.transData.transform((x, y))[1] * 72 / fig.dpi, x, y, t) for x, y, t in items))
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(p[0] for p in pts) - sum(ys)) / len(ys)  # re-centre the block on the data
    for (y0, x, y, t), yl in zip(pts, ys):
        ax.annotate(t, (x, y), xytext=(dx, yl + shift - y0), textcoords="offset points", va="center", color=INK,
                    annotation_clip=False, **kw)


def save(fig, name, exact=None):
    """Paper figures are saved at their exact printed size (constrained layout, or exact=True for hand-placed
    illustrations), so LaTeX includes them at 100% and all figures share one type scale; figures without a layout
    engine (not in the paper) keep the tight crop."""
    exact = fig.get_layout_engine() is not None if exact is None else exact
    kw = {} if exact else dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(io.BytesIO(), format="png", dpi=220, **kw)  # a first draw settles the constrained layout
    fig.savefig(f"{OUT}/{name}.pdf", **kw)
    plt.close(fig)


def law(route):
    """The current fitted frontier law of the route (Table 2)."""
    E, K, g = (out_json("t2_fits")[route][p]["point"] for p in ("E", "K", "gamma"))  # the fits of Table 2
    return lambda c: E + K * (np.asarray(c) / 1e20) ** (-g)


def size_axis(ax, sizes=SIZES, label=True):
    """x axis of model sizes: log parameters, ticks at the seven sizes labelled by name."""
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([NPAR[s] for s in sizes]); ax.set_xticklabels(sizes)
    ax.set_xlim(NPAR[sizes[0]] / 1.35, NPAR[sizes[-1]] * 1.35)
    if label:
        ax.set_xlabel("model size")


def fig_isoflop(cells):
    """IsoFLOP curves, one panel per budget: best 1-NFE FID per size for pretrain->convert (best split) and native
    training, the gap between them shaded, the 50-step multi-step curve as context, and each route's optimum ringed."""
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.95), layout="constrained")
    yt = {"1e19": [4, 6, 10, 20, 30], "3e19": [2.5, 4, 6, 10, 16], "1e20": [1.8, 2.5, 4, 6, 10], "3e20": [1.25, 1.5, 2, 3]}
    for ax, b in zip(axs, BUD):
        curves = {}
        for route in ROUTES:
            pts = [(NPAR[s], best(cells, route, b, s)) for s in SIZES if best(cells, route, b, s) is not None]
            curves[route] = pts
            c = ROUTE_C[route]
            em = route != "R1"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=c, lw=1.5 if em else 1.2, alpha=1 if em else 0.9,
                    zorder=3 if em else 2)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=c, mec=SURF, mew=0.7, ms=3.6 if em else 3.0,
                    alpha=1 if em else 0.9, zorder=4)
            n0, q0 = min(pts, key=lambda p: p[1])
            ax.plot([n0], [q0], "o", ms=7.5, mfc="none", mec=c, mew=1.0, zorder=5)
        # the conversion advantage: native over convert on their common size range (log-log interpolation)
        nn = np.array([p[0] for p in curves["NAT"]]); qn = np.array([p[1] for p in curves["NAT"]])
        nc = np.array([p[0] for p in curves["CONV"]]); qc = np.array([p[1] for p in curves["CONV"]])
        lo, hi = max(nn.min(), nc.min()), min(nn.max(), nc.max())
        xg = np.exp(np.linspace(np.log(lo), np.log(hi), 80))
        yn = np.exp(np.interp(np.log(xg), np.log(nn), np.log(qn)))
        yc = np.exp(np.interp(np.log(xg), np.log(nc), np.log(qc)))
        ax.fill_between(xg, yc, yn, color=ROUTE_C["CONV"], alpha=0.13, lw=0, zorder=1)
        ratio = best(cells, "NAT", b) / best(cells, "CONV", b)
        ax.set_title(f"native / convert: {ratio:.2f}×", loc="right", fontsize=6.2, color=INK2)
        size_axis(ax, label=False)
        logaxis(ax, "y", yt[b])
        ax.set_title({"1e19": "$C=10^{19}$", "3e19": "$C=3{\\times}10^{19}$", "1e20": "$C=10^{20}$",
                      "3e20": "$C=3{\\times}10^{20}$"}[b])
    axs[0].set_ylabel("FID-50k")
    fig.supxlabel("model size (XS 21M to H 1.36B parameters)", fontsize=7.0, color=INK)
    handles = [Line2D([], [], color=ROUTE_C["CONV"], marker="o", mec=SURF, ms=3.6, lw=1.5, label="pretrain → convert (1 NFE, best split)"),
               Line2D([], [], color=ROUTE_C["NAT"], marker="o", mec=SURF, ms=3.6, lw=1.5, label="native one-step (1 NFE)"),
               Line2D([], [], color=ROUTE_C["R1"], marker="o", mec=SURF, ms=3.0, lw=1.2, alpha=0.9, label="multi-step FM (50 steps)"),
               plt.Rectangle((0, 0), 1, 1, color=ROUTE_C["CONV"], alpha=0.13, lw=0, label="conversion advantage"),
               Line2D([], [], color=INK2, marker="o", mfc="none", ms=6.5, ls="none", mew=1.0, label="compute-optimal size")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=6.3, handlelength=1.6, columnspacing=1.2)
    save(fig, "fig_isoflop")


def fig_fstar_stab(cells):
    """(a) f* per budget under three estimators, with all seeds (filled) and seed 0 only (hollow), and the 95% bootstrap
    interval of the three-point estimator; band: mean over budgets. (b) f* of every size against its size relative to
    the compute-optimal conversion size."""
    fsb = out_json("noise_model")["bootstrap"]["fstar"]
    E = json.load(open(f"{ROOT}/analysis/out/noise_model.json"))["fstar_estimators"]
    est = E["by_cell"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.05), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.15]))
    m, (lo, hi) = fsb["mean"], fsb["mean_ci95"]
    a.axvspan(lo, hi, color=ROUTE_C["CONV"], alpha=0.12, lw=0, zorder=0)
    a.axvline(m, color=ROUTE_C["CONV"], lw=0.9, zorder=1)
    marks = {"three_point": ("o", "three-point parabola (main)"), "quad_interior": ("s", "quadratic, all splits"),
             "grid": ("D", "best grid point")}
    budgets = list(BUD)
    for i, bb in enumerate(budgets):
        y = len(budgets) - 1 - i
        v = fsb["by_budget"][bb]
        a.plot(v["ci95"], [y + 0.17, y + 0.17], color=INK2, lw=0.9, zorder=2, solid_capstyle="butt")
        for (k, (mk, _)), dy in zip(marks.items(), (0.12, 0.0, -0.12)):  # offsets so coinciding estimates stay visible
            a.plot(E["all_seeds"][k][bb]["f"], y + 0.17 + dy, mk, ms=4.0, color=BUD_C[bb], mec=SURF, mew=0.6, zorder=4)
            a.plot(E["seed0"][k][bb]["f"], y - 0.22 + dy, mk, ms=3.8, mfc=SURF, mec=BUD_C[bb], mew=0.9, zorder=4)
    a.set_yticks(range(len(budgets)))
    a.set_yticklabels([f"{BTEX[bb]} ({fsb['by_budget'][bb]['size']})" for bb in budgets[::-1]], fontsize=6.5)
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, len(budgets) - 0.1); a.set_xlim(0.25, 1.0)
    a.grid(axis="y", visible=False); a.spines["left"].set_visible(False)
    a.set_xlabel("optimal pretraining fraction $f^*$")
    a.text(m + 0.012, len(budgets) - 0.15, f"mean {m:.2f}", fontsize=6.3, color=INK, va="top")
    hs = [Line2D([], [], marker=mk, color=INK2, mec=SURF, ls="none", ms=4.2, label=lab) for mk, lab in marks.values()]
    hs += [Line2D([], [], marker="o", mfc=SURF, mec=INK2, ls="none", ms=4.0, label="seed 0 only (hollow)"),
           Line2D([], [], color=INK2, lw=1.0, label="95% interval (main)")]
    a.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.3, handletextpad=0.3,
             borderaxespad=0.1, labelspacing=0.25, columnspacing=0.7)
    a.set_title("(a) $f^*$ under three estimators", color=INK)
    nst = {bb: nstar(cells, "CONV", bb)[0] for bb in BUD}
    for bb in BUD:
        pts = sorted((NPAR[k.split("@")[0]] / nst[bb], v["f_three_point"]) for k, v in est.items() if k.split("@")[1] == bb)
        b.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=BUD_C[bb], mec=SURF, mew=0.7, ms=4.0, lw=1.1, label=BTEX[bb])
    b.axvline(1.0, color=INK2, lw=0.7)
    b.text(1.05, 0.405, "optimal size", fontsize=6.0, color=INK2, va="bottom", ha="left")
    b.text(0.27, 0.88, "larger models prefer\nmore pretraining", fontsize=6.2, color=INK, ha="left", va="top")
    b.set_xscale("log"); b.set_ylim(0.4, 0.9)
    b.set_xticks([0.25, 0.5, 1, 2, 4, 8]); b.set_xticklabels(["¼", "½", "1", "2", "4", "8"]); b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("model size / compute-optimal size (conversion)"); b.set_ylabel("$f^*$ (three-point)")
    b.legend(title="budget $C$", loc="lower right", ncol=2, fontsize=6.3, title_fontsize=6.3, borderaxespad=0.2)
    b.set_title("(b) $f^*$ of every size", color=INK)
    save(fig, "fig_fstar_stab")


def nstar(cells, route, b):
    """N* by a parabola in log N through the best size and its neighbours; (N*, interior?)."""
    pts = [(math.log(NPAR[s]), math.log(best(cells, route, b, s))) for s in SIZES if best(cells, route, b, s) is not None]
    u = np.array([p[0] for p in pts]); q = np.array([p[1] for p in pts])
    i = int(np.argmin(q)); j = min(max(i - 1, 0), len(u) - 3)
    cf = np.polyfit(u[j:j + 3], q[j:j + 3], 2)
    if cf[0] <= 0 or not 0 < i < len(u) - 1:
        return math.exp(u[i]), False
    return math.exp(min(max(-cf[1] / (2 * cf[0]), u[j]), u[j + 2])), True


def nstar_prediction(cells):
    """Compute-optimal sizes (parabola in log N) of every route and budget, and the native optimum predicted from the
    multi-step size law and the step cost: N*_nat(C) = N*_FM(C / kappa), kappa at the predicted size
    (analysis/out/nstar_pred.json)."""
    cs = np.array(list(BUD.values()))
    a1, c1 = np.polyfit(np.log(cs), np.log([nstar(cells, "R1", b)[0] for b in BUD]), 1)
    ln = np.log([NPAR[s] for s in SIZES]); kap = np.array([FT[s]["imf"] / FT[s]["fm"] for s in SIZES])
    pred = []
    for c in cs:
        n = math.exp(c1 + a1 * math.log(c))
        for _ in range(20):
            k = float(np.interp(math.log(n), ln, kap))
            n = math.exp(c1 + a1 * math.log(c / k))
        pred.append(n)
    self_pred = dict(zip(BUD, pred))
    json.dump(dict(a_fm=float(a1), native_pred_M={b: v / 1e6 for b, v in self_pred.items()},
                   native_meas_M={b: nstar(cells, "NAT", b)[0] / 1e6 for b in BUD}, fm_M={b: nstar(cells, "R1", b)[0] / 1e6 for b in BUD},
                   conv_M={b: nstar(cells, "CONV", b)[0] / 1e6 for b in BUD}),
              open(f"{ROOT}/analysis/out/nstar_pred.json", "w"), indent=1)


def main():
    cells = load()
    fig_isoflop(cells); nstar_prediction(cells); fig_fstar_stab(cells)
    print("figures ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
