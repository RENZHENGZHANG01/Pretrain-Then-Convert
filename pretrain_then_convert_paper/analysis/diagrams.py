"""Illustrations of the paper: overview (budget split and headline curve), related-work map, training protocol (WSD
branching) and the function-preserving conversion with the training cost per sample. Same palette and type as
analysis/figures.py: multi-step FM blue, pretrain->convert orange, native aqua, distillation violet; text in ink.

  python analysis/diagrams.py  ->  paper/figures/fig_overview, fig_related, fig_method (.pdf)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import figures as F  # noqa: E402
from common import DATA, ROOT  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = F.ROUTE_C["R1"], F.ROUTE_C["CONV"], F.ROUTE_C["NAT"], "#4a3aa7"
INK, INK2, GRID, SURF = F.INK, F.INK2, F.GRID, F.SURF
LIGHT = {"R1": "#d6e6f8", "CONV": "#fbe0d4", "NAT": "#d1efe4", "VIO": "#dcd8f0", "GREY": "#efeeea"}


def box(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.012, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z,
                       transform=ax.transAxes, mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0, z=3, style="-|>", ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 zorder=z, transform=ax.transAxes, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=7, color=INK, ha="center", va="center", weight="normal", z=4, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z, transform=ax.transAxes,
            fontstyle=style)


# ------------------------------------------------------------------------------------------------- overview
SAMPLES = f"{DATA}/samples/samples"


def sample_tile(run, cls=387, j=3):
    """One 256x256 sample of a 1e21 model (the same noise and class for every model; scripts/sample_grid.py of the code)."""
    from PIL import Image
    meta = json.load(open(f"{SAMPLES}/meta.json"))
    g = np.array(Image.open(f"{SAMPLES}/{run}.png"))
    i = meta["classes"].index(cls)
    return g[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256], meta["runs"][run]["fid50k"]


def rbox(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.04, z=2, ls="solid"):
    """Rounded box in data coordinates (the overview axes are in inches)."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p)
    return p


def overview():
    """Fig. 1. Left: one budget C spent three ways, each route ending in its sampler and a real sample of its 1e21
    model (same noise and class). Right: 1-NFE FID over the best FID against the pretraining fraction at the
    compute-optimal size of each budget, with the basin (all splits within 9% of the best) and the two ends."""
    W, H = F.FULL, 2.2
    fig = plt.figure(figsize=(W, H))
    wl = 4.05
    ax = fig.add_axes([0, 0, wl / W, 1]); ax.set_xlim(0, wl); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.02, H - 0.13, "(a) one training budget $C$, three ways to spend it", fontsize=7.4, color=INK, va="center")
    fig.text(4.22 / W, (H - 0.13) / H, "(b) sensitivity to the split $f$", fontsize=7.4, color=INK, va="center")
    xb0, xb1, hb = 0.74, 2.62, 0.26
    rng = np.random.default_rng(3)
    noise = np.kron(np.clip(0.5 + 0.22 * rng.standard_normal((24, 24, 3)), 0, 1), np.ones((8, 8, 1)))  # crisp blocks
    rows = [(1.56, "Native\none-step", [(0.0, 1.0, AQUA, "one-step objective (iMF) for all of $C$")], "v1_XL_1e21_cd_s0", 1),
            (0.98, "Pretrain\n→ convert", [(0.0, 0.6, BLUE, "FM pretraining, $fC$"), (0.6, 1.0, ORANGE, "one-step, $(1{-}f)C$")],
             "v2_XL_1e21_f0.60_s0", 1),
            (0.40, "Multi-step\nFM", [(0.0, 1.0, BLUE, "flow matching for all of $C$")], "v2_H_1e21_cd_s0", 50)]
    ts = 0.44  # tile size (in)
    xn, xo = 2.74, 3.55  # noise tile and output tile (left edges)
    for y, name, segs, run, nfe in rows:
        ax.text(0.02, y, name, fontsize=7.0, color=INK, va="center", linespacing=1.1)
        for a, b, c, lab in segs:
            xa, xb_ = xb0 + (xb1 - xb0) * a, xb0 + (xb1 - xb0) * b
            rbox(ax, xa + (0.015 if a > 0 else 0), y - hb / 2, xb_ - xa - (0.015 if a > 0 else 0), hb, c, r=0.05)
            ax.text((xa + xb_) / 2, y, lab, fontsize=6.3, color="white", ha="center", va="center", weight="bold")
        ax.imshow(noise, extent=(xn, xn + ts, y - ts / 2, y + ts / 2), aspect="auto", interpolation="none", zorder=3)
        tile, fid = sample_tile(run)
        ax.imshow(tile, extent=(xo, xo + ts, y - ts / 2, y + ts / 2), aspect="auto", interpolation="none", zorder=3)
        for x0 in (xn, xo):
            rbox(ax, x0, y - ts / 2, ts, ts, "none", ec="#d4d3ce", lw=0.5, r=0.0, z=4)
        if nfe == 1:  # one jump
            ax.annotate("", (xo - 0.03, y), (xn + ts + 0.03, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                        mutation_scale=7, shrinkA=0, shrinkB=0), zorder=4)
            ax.text((xn + ts + xo) / 2, y + 0.06, "1 step", fontsize=6.1, color=INK, ha="center", va="bottom")
        else:  # an ODE path in many small steps
            tt = np.linspace(0, 1, 120)
            px = xn + ts + 0.035 + (xo - xn - ts - 0.075) * tt
            py = y - 0.045 * np.sin(np.pi * tt)
            ax.plot(px[:-8], py[:-8], "-", color=INK2, lw=0.7, zorder=4)
            k = np.linspace(10, 100, 6).astype(int)
            ax.plot(px[k], py[k], "o", ms=1.6, color=INK, zorder=5)
            ax.annotate("", (xo - 0.03, py[-1]), (px[-14], py[-14]), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                        mutation_scale=6, shrinkA=0, shrinkB=0), zorder=5)
            ax.text((xn + ts + xo) / 2, y + 0.06, "50 steps", fontsize=6.1, color=INK, ha="center", va="bottom")
        ax.text(xo + ts / 2, y - ts / 2 - 0.035, f"FID {fid:.2f}", fontsize=6.1, color=INK, ha="center", va="top")
    xf = xb0 + (xb1 - xb0) * 0.6
    ax.plot([xf, xf], [0.98 - 0.19, 0.98 + 0.19], color=INK, lw=0.9, zorder=5)
    ax.text(xf, 0.98 + 0.21, "split $f$ (best ≈ 0.6)", fontsize=6.2, color=INK, ha="center", va="bottom")
    ax.text(xn + ts / 2, 1.56 + ts / 2 + 0.05, "noise", fontsize=6.1, color=INK2, ha="center", va="bottom")
    ax.text(xo + ts / 2, 1.56 + ts / 2 + 0.05, "sample ($10^{21}$ FLOPs)", fontsize=6.3, color=INK2, ha="center", va="bottom")
    ax.text(xb0, 0.015, "training cost per sample: FM step 1×, iMF step 2.3–2.9×", fontsize=6.2, color=INK2, va="bottom")
    # right: the f-curve at the compute-optimal size of each budget
    axr = fig.add_axes([4.62 / W, 0.20, 2.03 / W, 0.53])
    cells = F.load()
    nm = json.load(open(f"{ROOT}/analysis/out/noise_model.json"))["bootstrap"]["fstar"]
    axr.axvspan(0.25, 0.8, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    axr.axvspan(nm["mean_ci95"][0], nm["mean_ci95"][1], color=ORANGE, alpha=0.16, lw=0, zorder=0)
    axr.axvline(nm["mean"], color=ORANGE, lw=1.0, zorder=1)
    for b, v in nm["by_budget"].items():
        pts = sorted((f, q) for (r, sz, bb, f), q in cells.items() if r in ("CONV", "NAT") and sz == v["size"] and bb == b)
        f = np.array([p_[0] for p_ in pts]); q = np.array([p_[1] for p_ in pts]) / min(p_[1] for p_ in pts)
        axr.plot(f, q, "-o", color=F.BUD_C[b], mec=SURF, mew=0.6, ms=3.0, lw=1.1, label=f"{F.BTEX[b]} ({v['size']})", zorder=3)
    axr.set_yscale("log"); axr.set_yticks([1, 1.25, 1.5, 2, 2.5]); axr.set_yticklabels(["1", "1.25", "1.5", "2", "2.5"])
    axr.minorticks_off(); axr.set_xlim(-0.04, 1.03); axr.set_ylim(0.96, 2.75)
    axr.set_xticks([0, 0.25, 0.5, 0.75, 1]); axr.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axr.set_xlabel("pretraining fraction $f$"); axr.set_ylabel("1-NFE FID / best FID")
    axr.text(0.02, 2.62, "native\n1.3–1.7×", fontsize=6.0, color=INK, ha="left", va="top", linespacing=1.05)
    axr.text(0.90, 2.62, "late\n1.8–2.4×", fontsize=6.0, color=INK, ha="right", va="top", linespacing=1.05)
    axr.text(0.39, 2.62, "within 9%\nof the best", fontsize=6.0, color=INK, ha="center", va="top", linespacing=1.05)
    axr.text(nm["mean"] + 0.02, 1.62, f"$f^*\\!\\approx${nm['mean']:.2f}", fontsize=6.2, color=INK, ha="left", va="center")
    axr.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3,
               title_fontsize=6.3, handlelength=1.6, columnspacing=0.8, borderaxespad=0.1)
    F.save(fig, "fig_overview", exact=True)


# ------------------------------------------------------------------------------------------------- related work
def related():
    """Fig. 2: routes to a one-step generator. Columns: one-step objectives from scratch, one-step training from a
    pretrained model, distillation; rows: generative objectives and losses in a pretrained feature space. This work
    is the axis between the first two columns at a fixed total budget; distillation is compared at equal compute."""
    W, H = F.FULL, 1.78
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    cols = [(0.80, 2.72, "Train from scratch", "native one-step objectives", LIGHT["NAT"], AQUA),
            (2.92, 4.84, "Start from a pretrained model", "conversion, fine-tuning", LIGHT["CONV"], ORANGE),
            (5.04, 6.73, "Distill from a teacher", "teacher cost rarely counted", LIGHT["VIO"], VIOLET)]
    items = {(0, 0): ["MeanFlow, iMF", "AlphaFlow, pixel MeanFlow", "consistency models (iCT, sCM)", "shortcut models, IMM"],
             (1, 0): ["ECT, CMT", "Decoupled MeanFlow", "MeanFlow-Transfer", "MeanFlow on RAE"],
             (2, 0): ["DMD2", "flow-map distillation", "FACM, PVD"],
             (0, 1): ["Drifting models", "W-Flow, UOT-GF"], (1, 1): ["Priming models"],
             (2, 1): ["ADD (pretrained-feature", "discriminator)"]}
    rows = [(0.86, 1.38, "generative\nobjectives"), (0.08, 0.42, "losses in a\npretrained\nfeature space")]
    for ci, (x0, x1, title, sub, fill, edge) in enumerate(cols):
        ax.text((x0 + x1) / 2, H - 0.10, title, fontsize=7.3, weight="bold", color=INK, ha="center", va="center")
        ax.text((x0 + x1) / 2, H - 0.25, sub, fontsize=6.3, color=INK2, ha="center", va="center", style="italic")
        for ri, (y0, y1, _) in enumerate(rows):
            names = items[(ci, ri)]
            rbox(ax, x0, y0, x1 - x0, y1 - y0, fill if names else "#f3f3f1", r=0.015, z=1)
            rbox(ax, x0, y0, 0.035, y1 - y0, edge if names else "#d9d8d3", r=0.0, z=2)  # accent strip
            for k, nm_ in enumerate(names):
                ax.text((x0 + x1) / 2 + 0.02, (y0 + y1) / 2 + ((len(names) - 1) / 2 - k) * 0.112, nm_, fontsize=6.4,
                        color=INK, ha="center", va="center")
    for y0, y1, lab in rows:
        ax.text(0.04, (y0 + y1) / 2, lab, fontsize=6.4, color=INK2, va="center", linespacing=1.1)
    # this work: the axis f between training from scratch and starting from a pretrained model, at a fixed budget C
    ys, xa, xb = 0.635, cols[0][0] + 0.10, cols[1][1] - 0.10
    n = 200
    grad = np.linspace(0, 1, n)[None, :]
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("f", [AQUA, "#7fc7b4", "#f4b393", ORANGE])
    ax.imshow(grad, extent=(xa, xb, ys - 0.035, ys + 0.035), aspect="auto", cmap=cm, zorder=3)
    for xv, lab in ((xa, "$f=0$"), (xb, "$f\\to1$")):
        ax.text(xv, ys - 0.065, lab, fontsize=6.2, color=INK, ha="center", va="top")
    xs_ = xa + 0.6 * (xb - xa)
    ax.plot(xs_, ys, "v", ms=5.5, color=INK, mec="white", mew=0.6, zorder=5, transform=ax.transData)
    ax.text(xs_, ys + 0.055, "best $f\\approx0.6$", fontsize=6.2, color=INK, ha="center", va="bottom")
    ax.text((xa + xb) / 2 - 0.55, ys + 0.055, "this work: split a fixed budget $C$", fontsize=6.5, color=INK, ha="center",
            va="bottom", weight="bold")
    xd = cols[2][0] + 0.12
    ax.annotate("", (xb + 0.005, ys), (xd + 1.45, ys), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.0, mutation_scale=7,
                shrinkA=0, shrinkB=0))
    ax.text((cols[2][0] + cols[2][1]) / 2, ys + 0.055, "compared at equal total compute", fontsize=6.2, color=INK,
            ha="center", va="bottom")
    F.save(fig, "fig_related", exact=True)


# ------------------------------------------------------------------------------------------------- protocol
def _curve(ax, p0, p1, color, lw, z=2, bend=0.5):
    """Smooth branch from p0 to p1 (cubic Bezier, horizontal tangents)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    (x0, y0), (x1, y1) = p0, p1
    dx = (x1 - x0) * bend
    path = Path([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, zorder=z, capstyle="round"))


def draw_protocol(ax):
    ax.set_xlim(-0.03, 1.42); ax.set_ylim(-0.17, 1.0); ax.axis("off")
    yF, yN, C = 0.62, 0.13, 1.0
    # mainlines (constant learning rate; they continue for larger budgets)
    for y, c, lab in ((yF, BLUE, "FM mainline (constant LR)"), (yN, AQUA, "native one-step mainline (constant LR)")):
        ax.plot([0, C], [y, y], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([C, 1.1], [y, y], color=c, lw=1.2, ls=(0, (1.5, 1.5)), zorder=2)
        ax.text(0.0, y + 0.075, lab, fontsize=6.4, color=INK, va="bottom")
    # conversions branch from the FM mainline at fC and train the one-step objective until C
    for f in (0.25, 0.5, 0.65, 0.8, 0.95):
        _curve(ax, (f, yF), (C, 0.42), ORANGE, 1.0, z=2, bend=0.45)
        ax.plot(f, yF, "o", ms=3.4, color=ORANGE, mec="white", mew=0.6, zorder=4)
    ax.plot(C, 0.42, "o", ms=4.2, color=ORANGE, mec="white", mew=0.7, zorder=4)
    ax.text(C + 0.04, 0.42, "converted\nmodels", fontsize=6.2, color=INK, va="center", linespacing=1.05)
    ax.text(0.30, 0.45, "branch at $fC$, train the\none-step objective to $C$", fontsize=6.1, color=INK2, ha="center",
            va="center", linespacing=1.05)
    # cooldowns: branch at 0.85 C and decay the learning rate to zero
    for y, c, lab, dy in ((yF, BLUE, "multi-step\nmodel", 0.17), (yN, AQUA, "native\nmodel", 0.10)):
        _curve(ax, (0.85, y), (C, y + dy), c, 1.3, z=3, bend=0.5)
        ax.plot(0.85, y, "o", ms=3.4, color=c, mec="white", mew=0.6, zorder=4)
        ax.plot(C, y + dy, "o", ms=4.2, color=c, mec="white", mew=0.7, zorder=4)
        ax.text(C + 0.04, y + dy, lab, fontsize=6.2, color=INK, va="center", linespacing=1.05)
    # compute axis
    ax.annotate("", (1.12, -0.06), (0, -0.06), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=6))
    for x, lab in ((0, "0"), (0.85, "$0.85C$"), (C, "$C$")):
        ax.plot([x, x], [-0.085, -0.035], color=INK2, lw=0.7)
        ax.text(x, -0.11, lab, fontsize=6.0, color=INK2, ha="center", va="top")
    ax.text(0.45, -0.11, "training compute", fontsize=6.2, color=INK2, ha="center", va="top")


def draw_convert(ax):
    """Function-preserving conversion: trunk, u-head, class and patch embedders from the FM model; the new embedders
    of omega, the interval and h start as constants at the values FM was trained with; the v-head starts as a copy."""
    ax.axis("off")
    NEW = "#fff4ee"
    x0, x1 = 0.02, 0.66
    toks = [("$y$", BLUE), ("$\\omega$", None), ("$t_{\\min}$", None), ("$t_{\\max}$", None), ("$h$", None)]
    wt, gap = 0.078, 0.008
    xs = [x0 + k * (wt + gap) for k in range(5)]
    yt, ht = 0.08, 0.11
    for (lab, c), xx in zip(toks, xs):
        if c is None:
            box(ax, xx, yt, wt, ht, NEW, ec=ORANGE, lw=0.9, r=0.012)
        else:
            box(ax, xx, yt, wt, ht, LIGHT["R1"], ec=BLUE, lw=0.7, r=0.012)
        text(ax, xx + wt / 2, yt + ht / 2, lab, size=5.9 if "t_" in lab else 6.6)
    xz = xs[-1] + wt + gap
    box(ax, xz, yt, x1 - xz, ht, LIGHT["R1"], ec=BLUE, lw=0.7, r=0.012)
    text(ax, (xz + x1) / 2, yt + ht / 2, "$z_t$ patches", size=6.4)
    yb, hb = 0.30, 0.15
    box(ax, x0, yb, x1 - x0, hb, LIGHT["R1"], ec=BLUE, lw=0.7, r=0.015)
    text(ax, (x0 + x1) / 2, yb + hb / 2, "shared trunk: $D-8$ blocks", size=6.4)
    ybus = yt + ht + 0.035  # every input token feeds the trunk
    ax.plot([x0 + 0.01, x1 - 0.01], [ybus, ybus], color=INK2, lw=0.7, transform=ax.transAxes, zorder=3)
    for xx in [x + wt / 2 for x in xs] + [(xz + x1) / 2]:
        ax.plot([xx, xx], [yt + ht, ybus], color=INK2, lw=0.7, transform=ax.transAxes, zorder=3)
    arrow(ax, (x0 + x1) / 2, ybus, (x0 + x1) / 2, yb - 0.005, color=INK2, lw=0.7, ms=5)
    yh, hh = 0.55, 0.15
    xm = (x0 + x1) / 2
    box(ax, x0, yh, xm - x0 - 0.012, hh, LIGHT["R1"], ec=BLUE, lw=0.7, r=0.015)
    text(ax, (x0 + xm) / 2, yh + hh / 2, "$u$-head: 8 blocks", size=6.2)
    pv = box(ax, xm + 0.012, yh, x1 - xm - 0.012, hh, LIGHT["GREY"], ec=INK2, lw=0.8, r=0.015)
    pv.set_linestyle((0, (3, 2)))
    text(ax, (xm + x1) / 2, yh + hh / 2, "$v$-head: copy", size=6.2)
    for xc in ((x0 + xm) / 2, (xm + x1) / 2):
        arrow(ax, xc, yb + hb + 0.005, xc, yh - 0.005, color=INK2, lw=0.7, ms=5)
        arrow(ax, xc, yh + hh + 0.005, xc, 0.80, color=INK2, lw=0.7, ms=5)
    text(ax, (x0 + xm) / 2, 0.845, "$u$: 1-NFE jump", size=6.1)
    text(ax, (xm + x1) / 2, 0.845, "$v$: velocity", size=6.1)
    xl = 0.70
    items = [(LIGHT["R1"], BLUE, "solid", "from the FM\nmodel"), (NEW, ORANGE, "solid", "new, starts\nconstant"),
             (LIGHT["GREY"], INK2, (0, (3, 2)), "copy of $u$-head\n(training only)")]
    for k, (fc, ec, ls, lab) in enumerate(items):
        yy = 0.70 - k * 0.165
        pb = box(ax, xl, yy - 0.03, 0.055, 0.06, fc, ec=ec, lw=0.8, r=0.008)
        pb.set_linestyle(ls)
        text(ax, xl + 0.075, yy, lab, size=6.0, ha="left")
    text(ax, 0.835, 0.13, "at step 0:\n$u_\\theta=v_\\theta=v_{\\mathrm{FM}}$", size=6.5)


def draw_cost(ax):
    """Training cost per sample in FM steps: the u-path forward and backward (one FM step), guidance passes over 2.5x
    the batch without gradient, the JVP tangent, and the 8-block auxiliary head (24 F / D): kappa = 13/6 + 8/D;
    measured totals (FLOP counter) as ticks; shortcut models for comparison."""
    ft = json.load(open(f"{ROOT}/data/flops_table.json"))
    A1, A2, A3 = "#1baf7a", "#0e7c56", "#9edcc3"  # extra one-step work: guidance, JVP, auxiliary head (aqua ramp)
    rows = [("FM step", None, None), ("iMF, $D=12$", 12, ["XS", "S", "B"]), ("iMF, $D=24$", 24, ["M"]),
            ("iMF, $D=32$", 32, ["L"]), ("iMF, $D=48$", 48, ["XL", "H"]), ("shortcut", "sc", None)]
    for i, (lab, D, sz) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 1.0, 0.62, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
        if D == "sc":
            ax.barh(y, 0.198, 0.62, left=1.0, color="#c9c8c3", edgecolor="white", lw=0.6, zorder=3)
            meas = 1.198
        elif D is not None:
            x = 1.0
            for v, c in ((2.5 / 3, A1), (1 / 3, A2), (8 / D, A3)):
                ax.barh(y, v, 0.62, left=x, color=c, edgecolor="white", lw=0.6, zorder=3)
                x += v
            meas = float(np.mean([ft[k]["imf_over_fm"] for k in sz]))
        else:
            meas = 1.0
        ax.plot([meas, meas], [y - 0.4, y + 0.4], color=INK, lw=1.0, zorder=4)
        ax.text(meas + 0.06, y, f"{meas:.2f}", fontsize=6.0, color=INK, va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.2)
    ax.set_xlim(0, 3.55); ax.set_xticks([0, 1, 2, 3]); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("cost per sample (FM steps)", fontsize=6.6)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="$u$-path (= FM step)"),
                       plt.Rectangle((0, 0), 1, 1, color=A1, label="guidance passes"),
                       plt.Rectangle((0, 0), 1, 1, color=A2, label="JVP"),
                       plt.Rectangle((0, 0), 1, 1, color=A3, label="aux. $v$-head"),
                       plt.Rectangle((0, 0), 1, 1, color="#c9c8c3", label="shortcut bootstrap")],
              loc="lower center", bbox_to_anchor=(0.30, 1.0), ncol=2, fontsize=6.0, handlelength=1.0, columnspacing=0.6,
              handletextpad=0.35, borderaxespad=0.15)


def method():
    W, H = F.FULL, 1.95
    fig = plt.figure(figsize=(W, H))
    draw_protocol(fig.add_axes([0.0, 0.02, 2.55 / W, 0.88]))
    draw_convert(fig.add_axes([2.62 / W, 0.03, 2.28 / W, 0.86]))
    draw_cost(fig.add_axes([5.55 / W, 0.20, 1.12 / W, 0.48]))
    for x, t in ((0.004, "(a) training protocol"), (2.62 / W, "(b) function-preserving conversion"), (4.98 / W, "(c) cost of a training step")):
        fig.text(x, 0.975, t, fontsize=7.4, color=INK, va="top")
    F.save(fig, "fig_method", exact=True)


if __name__ == "__main__":
    overview()
    related()
    method()
    print("diagrams written")
