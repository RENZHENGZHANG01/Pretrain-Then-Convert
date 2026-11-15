"""Results figures of the paper: why conversion wins (fig_mechanism: where the conversions land, the value of an FM
sample, compute-optimal sizes, shortcut models), the compute laws (fig_laws: one law for both one-step routes, the
conversion advantage, predicted vs measured frontiers) and the FID landscape over model size and split (fig_landscape).

  python analysis/paper_figs.py  ->  paper/figures/fig_mechanism, fig_laws, fig_landscape (.pdf)
"""
import json
import math

import numpy as np
from matplotlib.lines import Line2D

import figures as F
from figures import plt

ROOT = F.ROOT


def panel_nstar(ax, cells):
    cs = np.array(list(F.BUD.values()))
    edge = False
    for route in F.ROUTES:
        ns = [F.nstar(cells, route, b) for b in F.BUD]
        edge |= any(not ins for _, ins in ns)
        dx = {"R1": 0.965, "CONV": 1.0, "NAT": 1.035}[route]
        c = F.ROUTE_C[route]
        ax.plot(cs * dx, [n for n, _ in ns], "-", color=c, lw=1.3)
        for x, (n, ins) in zip(cs * dx, ns):
            ax.plot(x, n, "o", color=c, mfc=c if ins else F.SURF, mec=F.SURF if ins else c, mew=1.0 if ins else 1.1, ms=4.0, zorder=3)
    pr = json.load(open(f"{ROOT}/analysis/out/nstar_pred.json"))["native_pred_M"]
    ax.plot(cs * 1.035, [pr[b] * 1e6 for b in F.BUD], ls=(0, (2.5, 1.8)), color=F.ROUTE_C["NAT"], lw=1.1, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_yticks([4e7, 1e8, 2e8, 4e8]); ax.set_yticklabels(["40M", "100M", "200M", "400M"])
    ax.set_xticks([1e19, 1e20]); ax.set_xticklabels(["$10^{19}$", "$10^{20}$"])
    ax.set_xlim(7e18, 4.6e20); ax.set_ylim(2.6e7, 1.3e9)
    ax.set_xlabel("training compute $C$ (FLOPs)"); ax.set_ylabel("compute-optimal size $N^*$")
    short = {"R1": "multi-step", "CONV": "convert", "NAT": "native"}
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C[r], marker="o", mec=F.SURF, ms=4, label=short[r]) for r in F.ROUTES]
              + [Line2D([], [], color=F.ROUTE_C["NAT"], ls=(0, (2.5, 1.8)), label="native, predicted")]
              + ([Line2D([], [], color=F.INK2, marker="o", mfc=F.SURF, ls="none", ms=4, label="grid edge")] if edge else []),
              loc="upper left", fontsize=6.0, handlelength=1.4, ncol=2, columnspacing=1.0, borderaxespad=0.3)
    ax.set_title("(b) compute-optimal size", color=F.INK)


def panel_shortcut(ax):
    g = json.load(open(f"{ROOT}/analysis/out/shortcut.json"))
    cells = [("B_1e19", "B, $10^{19}$"), ("M_1e19", "M, $10^{19}$"), ("M_3e19", "M, $3{\\times}10^{19}$")]
    x = np.arange(len(cells)); w = 0.36
    for i, (k, _) in enumerate(cells):
        v = g[k]
        pi = list(v["imf"]["native_over_best_convert"].values())  # iMF: two seeds (geometric mean, range as whisker)
        pen_imf = math.exp(np.mean(np.log(pi))); fs_imf = float(np.mean(list(v["imf"]["fstar_per_seed"].values())))
        nb = v["sc"]["native_over_best_convert"]
        pen_sc, ci = nb.get("geomean_seeds") or nb["mean_curve"], nb.get("ci95_over_seeds")
        fs_sc = v["sc"]["fstar_mean_curve"]
        ax.bar(x[i] - w / 2 - 0.01, pen_imf - 1, w, bottom=1, color=F.ROUTE_C["CONV"], zorder=3, label="iMF (2.3–2.9 FM steps)" if i == 0 else None)
        ax.bar(x[i] + w / 2 + 0.01, pen_sc - 1, w, bottom=1, color=F.INK2, zorder=3, label="shortcut (1.2 FM steps)" if i == 0 else None)
        ax.errorbar(x[i] - w / 2 - 0.01, pen_imf, yerr=[[pen_imf - min(pi)], [max(pi) - pen_imf]], fmt="none", ecolor=F.INK, lw=0.8, zorder=4)
        if ci:
            ax.errorbar(x[i] + w / 2 + 0.01, pen_sc, yerr=[[pen_sc - ci[0]], [ci[1] - pen_sc]], fmt="none", ecolor=F.INK, lw=0.8, zorder=4)
        ax.text(x[i] - w / 2 - 0.01, max(pi) + 0.07, f"{fs_imf:.2f}", ha="center", va="bottom", fontsize=6.0, color=F.INK)
        ax.text(x[i] + w / 2 + 0.01, (ci[1] if ci else pen_sc) + 0.07, f"{fs_sc:.2f}", ha="center", va="bottom", fontsize=6.0, color=F.INK)
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in cells], fontsize=6.5)
    ax.set_ylim(1, 3.85); ax.grid(axis="x", visible=False)
    ax.set_ylabel("FID$_\\mathrm{native}$ / FID$_\\mathrm{best\\ split}$")
    ax.legend(loc="upper center", fontsize=6.3, title="numbers: optimal split $f^*$", title_fontsize=6.3, ncol=2, columnspacing=1.2,
              borderaxespad=0.3)
    ax.set_title("(c) a cheaper one-step objective", color=F.INK)


def panel_curve(ax, cells, size="M"):
    """Native FID against training samples for one size (line) and every conversion of two budgets at its own sample
    count: moving along a budget's conversions, f grows to the right; the best split sits on the native curve, late
    conversions above it. Dashed: the two-component model along f (analysis/two_component.py)."""
    ft = F.FT[size]
    tc = json.load(open(f"{ROOT}/analysis/out/two_component.json"))["forms"]["c4"]["curves"]
    nat = sorted((F.BUD[b] / ft["imf"], q) for (r, s, b, f), q in cells.items() if r == "NAT" and s == size)
    ax.plot([x for x, _ in nat], [q for _, q in nat], "-o", color=F.ROUTE_C["NAT"], mec=F.SURF, mew=0.7, ms=3.8, lw=1.3, zorder=2)
    for b, ofs in (("3e19", (4, 3)), ("1e20", (4, 3))):
        C = F.BUD[b]
        pts = sorted(((f * C / ft["fm"] + (1 - f) * C / ft["imf"]), q, f) for (r, s, bb, f), q in cells.items()
                     if r == "CONV" and s == size and bb == b)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=F.ROUTE_C["CONV"], lw=0.8, alpha=0.8, zorder=2)
        if f"{size}@{b}" in tc:
            mc = tc[f"{size}@{b}"]
            keep = [i for i, f in enumerate(mc["f"]) if pts[0][2] <= f <= pts[-1][2]]  # the measured range of splits
            ax.plot([mc["samples"][i] for i in keep], [mc["fid"][i] for i in keep], "--", color=F.INK2, lw=0.7, zorder=1)
        best = min(pts, key=lambda p: p[1])
        for x, q, f in pts:
            ax.plot([x], [q], "D" if (x, q, f) == best else "o", color=F.ROUTE_C["CONV"], mec=F.SURF, mew=0.6,
                    ms=4.6 if (x, q, f) == best else 3.2, zorder=3)
        for x, q, f in pts:
            if f in (best[2], pts[-1][2]):  # the optimum and the latest conversion of each budget
                below = f == best[2]  # optimum labels below-left of the diamond, clear of the next split's marker
                first = f == pts[0][2]
                ax.annotate(f"{f:.3g}", (x, q), xytext=(-2, -8) if below else ((-3, -7) if first else ofs),
                            textcoords="offset points", fontsize=6.0, color=F.INK2, ha="right" if (below or first) else "left")
        nq = [q for (r, s_, bb, f), q in cells.items() if r == "NAT" and s_ == size and bb == b][0]
        ax.annotate(f"$C=${F.BTEX[b]}", (C / ft["imf"], nq), xytext=(-3, -12), textcoords="offset points", fontsize=6.3,
                    color=F.INK2, ha="right")
        if b == "3e19":  # the exchange: same compute, m(f) times the samples, landing on the native curve
            xb, qb, fb = best
            ax.annotate("", (xb, nq), (C / ft["imf"], nq), arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=0.8,
                        mutation_scale=6, shrinkA=2.5, shrinkB=0.5), zorder=4)
            ax.plot([xb, xb], [qb * 1.06, nq], ":", color=F.INK, lw=0.7, zorder=4)
            ax.text(math.sqrt(xb * C / ft["imf"]), nq * 1.09, f"same compute,\n{xb / (C / ft['imf']):.1f}× the samples",
                    fontsize=6.0, color=F.INK, ha="center", va="bottom", linespacing=1.05, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_yticks([3, 5, 8, 12, 20]); ax.set_yticklabels(["3", "5", "8", "12", "20"]); ax.set_ylim(1.8, 75)
    ax.set_xticks([1e7, 3e7, 1e8, 3e8]); ax.set_xticklabels(["$10^{7}$", "$3{\\times}10^{7}$", "$10^{8}$", "$3{\\times}10^{8}$"])
    ax.set_xlabel("training samples"); ax.set_ylabel("FID-50k")
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C["NAT"], marker="o", mec=F.SURF, ms=3.8, label="native, by budget"),
                       Line2D([], [], color=F.ROUTE_C["CONV"], marker="o", mec=F.SURF, ms=3.2, lw=0.8, label="converted (numbers: $f$)"),
                       Line2D([], [], color=F.ROUTE_C["CONV"], marker="D", mec=F.SURF, ms=4.6, ls="none", label="best split"),
                       Line2D([], [], color=F.INK2, ls="--", lw=0.7, label="two-component model")],
              loc="upper right", fontsize=6.0, handlelength=1.6, borderaxespad=0.3, ncol=2, columnspacing=1.0)
    ax.set_title(f"(a) size {size}: where the conversions land", color=F.INK)


def panel_exchange(ax):
    """Native FID at the converted model's sample count / converted FID, for every split of every cell (thin) and the
    median by split with its 95% bootstrap interval (analysis/matched_steps.py exchange)."""
    d = json.load(open(f"{ROOT}/analysis/out/exchange_rate.json"))
    by = {}
    for c in d["cells"]:
        if not c["extrapolated"]:
            by.setdefault((c["size"], c["budget"]), []).append((c["f"], c["ratio"]))
    for k, v in by.items():
        v = sorted(v)
        if len(v) >= 3:
            ax.plot([p[0] for p in v], [p[1] for p in v], "-", color=F.INK2, lw=0.5, alpha=0.35, zorder=1)
    xs = {"0.08-0.10": 0.09, "0.24-0.27": 0.25, "0.50": 0.5, "0.65": 0.65, "0.80": 0.8, "0.95": 0.95, "0.99": 0.99}
    pts = [(xs[k], v["median"], v["median_ci95"]) for k, v in d["by_f"].items() if k in xs]
    pts.sort()
    ax.fill_between([p[0] for p in pts], [p[2][0] for p in pts], [p[2][1] for p in pts], color=F.ROUTE_C["CONV"], alpha=0.18, lw=0, zorder=2)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=F.ROUTE_C["CONV"], mec=F.SURF, mew=0.7, ms=3.8, lw=1.4, zorder=3)
    q = sorted((xs[k], v["median"]) for k, v in d["by_f_quadratic"].items() if k in xs)
    ax.plot([p[0] for p in q], [p[1] for p in q], "--", color=F.ROUTE_C["CONV"], lw=0.9, zorder=3)
    ax.axhline(1.0, color=F.INK2, lw=0.7)
    ax.text(0.98, 1.20, "an FM sample is worth more ↑", fontsize=6.0, color=F.INK2, va="bottom", ha="right")
    ax.text(0.62, 0.62, "worth less ↓", fontsize=6.0, color=F.INK2, va="center", ha="center")
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C["CONV"], marker="o", mec=F.SURF, ms=3.6, lw=1.4, label="median (95% interval)"),
                       Line2D([], [], color=F.ROUTE_C["CONV"], ls="--", lw=0.9, label="median, quadratic native curve"),
                       Line2D([], [], color=F.INK2, lw=0.5, alpha=0.5, label="single cells")],
              loc="lower left", fontsize=6.0, handlelength=1.6)
    ax.set_xlim(0, 1.0); ax.set_ylim(0.15, 1.35)
    ax.set_xlabel("pretraining fraction $f$"); ax.set_ylabel("FID$_\\mathrm{native}$ / FID$_\\mathrm{convert}$\nat equal samples")
    ax.set_title("(b) value of an FM sample relative to a one-step sample", color=F.INK)


def panel_frontier_joint(ax, cells):
    """Frontiers with the joint one-step law (analysis/joint_law.py, model B: native L(C), converted L(m_hat C)) and the
    multi-step law (Table 2); at 1e21 the predictions (hollow, 95% interval) next to the runs (filled); published iMF."""
    J = json.load(open(f"{ROOT}/analysis/out/joint_law.json"))
    B = J["B"]
    Lj = lambda c: B["E"] + B["K"] * (np.asarray(c) / 1e20) ** (-B["gamma"])  # noqa: E731
    curves = {"NAT": lambda c: Lj(c), "CONV": lambda c: Lj(B["m_hat"] * np.asarray(c)), "R1": F.law("R1")}
    # 95% predictive intervals of a single run: law uncertainty plus run-to-run noise; for the multi-step
    # law the run noise (SD 0.019 of log FID) is added in quadrature to its bootstrap interval
    sd1 = json.load(open(f"{ROOT}/analysis/out/robustness.json"))["seed_sd_by_route"]["R1"]
    nm = F.out_json("noise_model")["bootstrap"]
    p1, c1 = nm["R1_F21"]["point"], nm["R1_F21"]["ci95"]
    w = lambda a: math.sqrt(math.log(a) ** 2 + (1.96 * sd1) ** 2)  # noqa: E731
    pred21 = {"NAT": (B["nat21"], B["nat21_pred95"]), "CONV": (B["conv21"], B["conv21_pred95"]),
              "R1": (p1, [p1 * math.exp(-w(p1 / c1[0])), p1 * math.exp(w(c1[1] / p1))])}
    run21 = {"R1": J["runs_1e21"]["ms"], "CONV": J["runs_1e21"]["conv"], "NAT": J["runs_1e21"]["nat"]}
    dodge = {"NAT": 0.84, "CONV": 1.0, "R1": 1.0}  # NAT and CONV are close in FID; the run sits just right of its prediction
    ax.set_xscale("log"); F.logaxis(ax, "y", [1.5, 2, 3, 5, 8, 12])
    ax.set_xlim(8e18, 3.2e21); ax.set_ylim(1.1, 17)
    cs = np.array(list(F.BUD.values())); grid = np.logspace(19, 21.2, 160)
    for route in F.ROUTES:
        c, fn = F.ROUTE_C[route], curves[route]
        ax.plot(grid[grid <= 3e20], fn(grid[grid <= 3e20]), color=c, lw=1.3)
        ax.plot(grid[grid >= 3e20], fn(grid[grid >= 3e20]), color=c, lw=1.3, ls=(0, (3, 2)))
        ax.plot(cs, [F.best(cells, route, b) for b in F.BUD], "o", color=c, mec=F.SURF, mew=0.9, ms=4.2, zorder=3)
        pt, ci = pred21[route]
        ax.errorbar([1e21 * dodge[route]], [pt], yerr=[[pt - ci[0]], [ci[1] - pt]], fmt="s", mfc=F.SURF, mec=c, mew=1.0,
                    color=c, capsize=1.8, lw=0.9, ms=3.9, zorder=3)
        ax.plot([1e21 * dodge[route] * 1.09], [run21[route]], "s", color=c, mec=F.SURF, mew=0.7, ms=4.2, zorder=4)
    pub = [(3.49e20, 3.39), (6.16e20, 2.27), (1.39e21, 1.86), (2.52e21, 1.72)]
    ax.plot([p[0] for p in pub], [p[1] for p in pub], "D", color=F.INK2, mfc="none", ms=3.3, mew=0.9, zorder=2)
    # the compute multiplier: at equal FID the converted frontier needs m_hat times less compute than the native one
    gf = grid[grid <= 3e20]
    ax.fill_between(gf, curves["CONV"](gf), curves["NAT"](gf), color=F.ROUTE_C["CONV"], alpha=0.10, lw=0, zorder=1)
    yv = 4.2
    cn = 1e20 * ((yv - B["E"]) / B["K"]) ** (-1 / B["gamma"])
    ax.annotate("", (cn / B["m_hat"], yv), (cn, yv), arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=0.9,
                mutation_scale=6.5, shrinkA=1.0, shrinkB=1.0), zorder=5)
    ax.annotate(f"same FID with\n{B['m_hat']:.2f}× less compute", (cn / math.sqrt(B["m_hat"]), yv), xytext=(1.25e20, 6.4),
                fontsize=6.2, color=F.INK, ha="left", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.5, shrinkA=1, shrinkB=2), zorder=5)
    ax.set_xlabel("training compute $C$ (FLOPs)"); ax.set_ylabel("best FID-50k at budget $C$")
    ax.set_title("(a) one law for both one-step routes", color=F.INK)


def panel_advantage_joint(ax):
    """Native over converted FID at the frontier: measured (95% intervals; 1e21 = single runs) and the joint law's
    L(C) / L(m_hat C) with its 95% band."""
    J = json.load(open(f"{ROOT}/analysis/out/joint_law.json"))
    B = J["B"]
    C = np.array(B["curve_C"])
    ax.fill_between(C, B["curve_ci95"][0], B["curve_ci95"][1], color=F.INK, alpha=0.09, lw=0)
    ax.plot(C, B["curve"], "-", color=F.INK, lw=1.2, zorder=2)
    xs = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20, "1e21": 1e21}
    for b, e in J["measured_ratio"].items():
        grid = b != "1e21"
        ax.errorbar(xs[b], e["point"], yerr=[[e["point"] - e["ci95"][0]], [e["ci95"][1] - e["point"]]],
                    fmt="o" if grid else "s", color=F.ROUTE_C["CONV"], mfc=F.ROUTE_C["CONV"],
                    mec=F.SURF, mew=0.7, ms=4.4 if grid else 4.2, lw=0.9, capsize=0, zorder=3)
    ax.axhline(1.0, color=F.INK2, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([1e19, 1e20, 1e21, 1e22]); ax.set_xticklabels(["$10^{19}$", "$10^{20}$", "$10^{21}$", "$10^{22}$"])
    ax.set_xlim(7e18, 1.2e22); ax.set_ylim(0.9, 1.8)
    ax.set_xlabel("training compute $C$ (FLOPs)"); ax.set_ylabel("FID$_\\mathrm{native}$ / FID$_\\mathrm{convert}$")
    ax.text(1.2e21, 1.43, f"$\\hat m$ = {B['m_hat']:.2f} [{B['m_hat_ci95'][0]:.2f}, {B['m_hat_ci95'][1]:.2f}]", fontsize=6.3, color=F.INK)
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C["CONV"], marker="o", mec=F.SURF, ls="none", label="measured (grid)"),
                       Line2D([], [], color=F.ROUTE_C["CONV"], marker="s", mec=F.SURF, ls="none", label="measured ($10^{21}$ runs)"),
                       Line2D([], [], color=F.INK, lw=1.2, label="joint law: $L(C)/L(\\hat m C)$")], loc="upper right", fontsize=6.0)
    ax.set_title("(b) the conversion advantage", color=F.INK)


def panel_pred_meas(ax, cells):
    """Predicted against measured frontier FID: the joint one-step law fitted on the grid (filled), refitted without the
    budget it predicts (hollow), and at 1e21 with the 95% predictive interval of a single run (large); multi-step:
    its own law (Table 2)."""
    J = json.load(open(f"{ROOT}/analysis/out/joint_law.json"))
    B = J["B"]
    Lj = lambda c: B["E"] + B["K"] * (np.asarray(c) / 1e20) ** (-B["gamma"])  # noqa: E731
    lr1 = F.law("R1")
    ax.fill_between([1, 20], [1 / 1.05, 20 / 1.05], [1.05, 20 * 1.05], color=F.INK, alpha=0.14, lw=0, zorder=0)
    ax.plot([1, 20], [1, 20], color=F.INK2, lw=0.7, zorder=1)
    for b, C in F.BUD.items():
        for route, pred in (("NAT", Lj(C)), ("CONV", Lj(B["m_hat"] * C)), ("R1", lr1(C))):
            meas = F.best(cells, route, b)
            ax.plot(meas, pred, "^" if route == "R1" else "o", color=F.ROUTE_C[route], mec=F.SURF, mew=0.6,
                    ms=4.4 if route == "R1" else 4.0, zorder=5 if route == "R1" else 3)
            if route != "R1":
                e = J["held_out"]["B"]["lobo"][b]["native" if route == "NAT" else "convert"]
                ax.plot(meas, meas * (1 + e), "o", mfc="none", mec=F.ROUTE_C[route], mew=0.8, ms=6.4, zorder=4)
    sd1 = json.load(open(f"{ROOT}/analysis/out/robustness.json"))["seed_sd_by_route"]["R1"]
    nm = F.out_json("noise_model")["bootstrap"]
    p1, c1 = nm["R1_F21"]["point"], nm["R1_F21"]["ci95"]
    w = lambda a: math.sqrt(math.log(a) ** 2 + (1.96 * sd1) ** 2)  # noqa: E731
    for route, pt, ci, meas in (("NAT", B["nat21"], B["nat21_pred95"], J["runs_1e21"]["nat"]),
                                ("CONV", B["conv21"], B["conv21_pred95"], J["runs_1e21"]["conv"]),
                                ("R1", p1, [p1 * math.exp(-w(p1 / c1[0])), p1 * math.exp(w(c1[1] / p1))], J["runs_1e21"]["ms"])):
        ax.errorbar(meas, pt, yerr=[[pt - ci[0]], [ci[1] - pt]], fmt="s", color=F.ROUTE_C[route], mec=F.SURF, mew=0.6,
                    ms=4.6, lw=0.9, capsize=1.6, zorder=4)
    F.logaxis(ax, "x", [1.5, 2, 3, 5, 8, 12]); F.logaxis(ax, "y", [1.5, 2, 3, 5, 8, 12])
    ax.set_xlim(1.1, 15); ax.set_ylim(1.1, 15)
    ax.set_xlabel("measured frontier FID"); ax.set_ylabel("predicted by the law")
    ax.text(0.04, 0.96, "band: ±5%", transform=ax.transAxes, fontsize=6.0, color=F.INK2, va="top")
    ax.legend(handles=[Line2D([], [], color=F.INK2, marker="o", mec=F.SURF, ls="none", ms=4, label="fitted"),
                       Line2D([], [], color=F.INK2, marker="o", mfc="none", ls="none", ms=6.4, mew=0.8, label="budget held out"),
                       Line2D([], [], color=F.INK2, marker="^", mec=F.SURF, ls="none", ms=4.4, label="multi-step law"),
                       Line2D([], [], color=F.INK2, marker="s", mec=F.SURF, ls="none", ms=4.6, label="$10^{21}$ (95% PI)")],
              loc="lower right", fontsize=6.0, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c) predictions", color=F.INK)


LANDSCAPE_RAMP = ["#ebe7f8", "#d3ccef", "#b3a8e3", "#8574d1", "#5a48b3", "#3a2c86", "#241a57"]  # violet, L 0.93 -> 0.27


def fig_landscape(cells):
    """The FID landscape over model size and pretraining fraction, one tile map per budget: 1-NFE FID over the best
    1-NFE FID of the budget (native at f=0). Outlined: within 9% of the best; star: the best cell; circles: the best split
    of each size; gray dots: not trained."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("vio", LANDSCAPE_RAMP)
    norm = LogNorm(1.0, 3.0)
    cols = [(0.0, (0.0,)), (0.1, (0.083, 0.095)), (0.25, (0.24, 0.25, 0.27)), (0.5, (0.5,)), (0.65, (0.65,)),
            (0.8, (0.8,)), (0.95, (0.95,)), (0.99, (0.99,))]
    xs = [0.0] + [1.35 + k for k in range(len(cols) - 1)]  # a gap after the native column
    fig, axs = plt.subplots(2, 2, figsize=(F.HALF, 2.62), layout="constrained")
    for ax, b in zip(axs.flat, F.BUD):
        val = {}
        for (r, sz, bb, f), q in cells.items():
            if bb != b or r not in ("NAT", "CONV"):
                continue
            for k, (_, fs) in enumerate(cols):
                if any(abs(f - x) < 1e-6 for x in fs):
                    val[(sz, k)] = q
        best = min(val.values())
        sizes = [sz for sz in F.SIZES if any(k[0] == sz for k in val)]
        near = {(sz, k) for (sz, k), v in val.items() if v / best < 1.095}  # within 9% after rounding, as in the text
        for i, sz in enumerate(sizes):
            y = len(sizes) - 1 - i
            row = {k: v for (s_, k), v in val.items() if s_ == sz}
            for k, x in enumerate(xs):
                if k in row:
                    ax.add_patch(Rectangle((x - 0.46, y - 0.44), 0.92, 0.88, fc=cmap(norm(min(row[k] / best, 3.0))), ec="none", zorder=1))
                else:  # not trained: blank, with a faint dot
                    ax.plot(x, y, "o", ms=1.0, color="#c9c8c3", zorder=1)
                if (sz, k) in near:  # outline of the region within 10% of the best: only edges facing outside it
                    nb = {"l": (sz, k - 1), "r": (sz, k + 1), "u": (sizes[i - 1] if i > 0 else None, k),
                          "d": (sizes[i + 1] if i + 1 < len(sizes) else None, k)}
                    x0, x1 = x - (0.5 if k > 1 else 0.46), x + (0.5 if 0 < k < len(xs) - 1 else 0.46)
                    for side, key in nb.items():
                        if key not in near:
                            seg = {"l": ([x0, x0], [y - 0.5, y + 0.5]), "r": ([x1, x1], [y - 0.5, y + 0.5]),
                                   "u": ([x0, x1], [y + 0.5, y + 0.5]), "d": ([x0, x1], [y - 0.5, y - 0.5])}[side]
                            ax.plot(*seg, color=F.ROUTE_C["CONV"], lw=1.2, zorder=3, solid_capstyle="projecting")
            conv = {k: v for k, v in row.items() if k > 0}
            if len(conv) >= 2:
                kb = min(conv, key=conv.get)
                star = row[kb] == best
                ax.plot(xs[kb], y + (0.08 if star else 0), "*" if star else "o", ms=6.2 if star else 2.6, color=F.INK if star else "white",
                        mec="white" if star else F.INK, mew=0.6 if star else 0.5, zorder=4)
        ax.set_xlim(-0.6, xs[-1] + 0.6); ax.set_ylim(-0.6, len(sizes) - 0.2)
        ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes[::-1])
        ax.set_xticks(xs); ax.set_xticklabels(["0", ".1", ".25", ".5", ".65", ".8", ".95", ".99"], fontsize=6.0)
        ax.tick_params(length=0, pad=1.5); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title({"1e19": "$C=10^{19}$", "3e19": "$C=3{\\times}10^{19}$", "1e20": "$C=10^{20}$", "3e20": "$C=3{\\times}10^{20}$"}[b],
                     fontsize=7.0)
    for ax in axs[1]:
        ax.set_xlabel("pretraining fraction $f$ (0: native)", fontsize=6.6, labelpad=2)
    for ax in axs[:, 0]:
        ax.set_ylabel("model size", fontsize=6.6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    handles = [Line2D([], [], marker="*", ms=7, color=F.INK, mec=F.INK, ls="none", label="best"),
               Line2D([], [], marker="o", ms=2.8, mfc="white", mec=F.INK, mew=0.5, ls="none", label="best split of a size"),
               Line2D([], [], marker="o", ms=1.4, color="#c9c8c3", ls="none", label="not trained"),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=F.ROUTE_C["CONV"], lw=1.2, label="within 9% of best")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.0, handlelength=1.3, columnspacing=0.8,
               handletextpad=0.4)
    cb = fig.colorbar(sm, ax=axs, location="bottom", shrink=0.62, aspect=28, pad=0.02)
    cb.set_ticks([1, 1.25, 1.5, 2, 3]); cb.set_ticklabels(["1", "1.25", "1.5", "2", "≥3"]); cb.minorticks_off()
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.0)
    cb.set_label("1-NFE FID / best FID of the budget", fontsize=6.4, labelpad=1.5)
    F.save(fig, "fig_landscape")


def mechanism_figure(cells):
    """Why conversion wins: samples vs split, the value of an FM sample, optimal sizes, shortcut models."""
    fig, axs = plt.subplots(2, 2, figsize=(F.FULL, 3.05), layout="constrained")
    panel_curve(axs[0, 0], cells); panel_exchange(axs[0, 1]); panel_nstar(axs[1, 0], cells); panel_shortcut(axs[1, 1])
    axs[1, 0].set_title("(c) compute-optimal model size", color=F.INK); axs[1, 1].set_title("(d) a cheaper one-step objective", color=F.INK)
    F.save(fig, "fig_mechanism")


def laws_figure(cells):
    fig, axs = plt.subplots(1, 3, figsize=(F.FULL, 2.3), layout="constrained", gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.82]))
    panel_frontier_joint(axs[0], cells); panel_advantage_joint(axs[1]); panel_pred_meas(axs[2], cells)
    handles = [Line2D([], [], color=F.ROUTE_C["NAT"], lw=1.3, label="native one-step"),
               Line2D([], [], color=F.ROUTE_C["CONV"], lw=1.3, label="pretrain → convert"),
               Line2D([], [], color=F.ROUTE_C["R1"], lw=1.3, label="multi-step FM (50 steps)"),
               Line2D([], [], color=F.INK2, marker="o", mec=F.SURF, ls="none", ms=4, label="frontier (measured)"),
               Line2D([], [], color=F.INK2, marker="s", mfc=F.SURF, ls="none", ms=3.9, label="law at $10^{21}$ (95% PI)"),
               Line2D([], [], color=F.INK2, marker="s", mec=F.SURF, ls="none", ms=4.2, label="run at $10^{21}$"),
               Line2D([], [], color=F.INK2, marker="D", mfc="none", ms=3.3, ls="none", label="published iMF")]
    fig.legend(handles=handles, loc="outside upper center", ncol=7, fontsize=6.3, handlelength=1.5, columnspacing=0.9,
               handletextpad=0.4)
    F.save(fig, "fig_laws")


def main():
    cells = F.load()
    mechanism_figure(cells)
    laws_figure(cells)
    fig_landscape(cells)
    print("fig_mechanism, fig_laws, fig_landscape written")


if __name__ == "__main__":
    main()
