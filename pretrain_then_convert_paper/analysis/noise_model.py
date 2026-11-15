"""Parametric bootstrap of the paper's statistics under the seed-noise model of analysis/noise.py (route-specific seed
SDs, noise shared by all runs of one mainline), and the optimal split under several estimators.

Outputs (analysis/out/noise_model.json, paper/tables/ta_noise.tex, ta_fstar_nc.tex):
  - the noise model (seed SDs, mainline correlations);
  - 95% intervals of the frontier laws, N* exponents, f* (mean, slope, per budget, smallest detectable slope), the
    native/convert crossover and the 1e21 ratios;
  - f* under three estimators, with all seeds and with seed 0 only, f* for every (size, budget) cell, and f* of the
    two seeds of the replicated cells.

  python analysis/noise_model.py
"""
import json
import math
import time

import numpy as np
from scipy.optimize import brentq

from common import ROOT
from surface import NPAR, fit as surface_fit, model as surface_model
from frontier import BUD, analyse, local_parabola_f
from figures import nstar
import noise as nz

N_BOOT = 2000
ROUTES = ("R1", "CONV", "NAT")
SIZES = ("XS", "S", "B", "M", "L", "XL", "H")


def law(p, c):
    E, K, g = p
    return E + K * (c / 1e20) ** (-g)


def crossover(est):
    pn = (est["NAT"]["E"], est["NAT"]["K"], est["NAT"]["gamma"])
    pc = (est["CONV"]["E"], est["CONV"]["K"], est["CONV"]["gamma"])
    g = lambda lc: law(pn, 10 ** lc) - law(pc, 10 ** lc)  # noqa: E731
    if g(25.0) > 0:
        return math.inf
    return 10 ** brentq(g, 20.0, 25.0) if g(20.0) > 0 else 1e20


def size_penalties(base):
    surf = {}
    for route in ROUTES:
        pts = {}
        for (r, s, b, f), q in base.items():
            if r == route:
                pts[(s, b)] = min(pts.get((s, b), 1e9), q)
        p, _ = surface_fit(pts)
        q21 = {s: float(surface_model(p, NPAR[s], 1e21)) for s in NPAR}
        qmin = min(float(surface_model(p, n, 1e21)) for n in np.exp(np.linspace(np.log(3e7), np.log(3e9), 400)))
        surf[route] = {s: math.exp(q21[s] - qmin) for s in NPAR}
    return surf


def stats(cells):
    est = analyse(cells)
    cs = np.log(np.array(list(BUD.values())))
    cexp = {k: math.exp(v) for k, v in cells.items()}
    for route in ROUTES:
        est[route]["a"] = float(np.polyfit(cs, np.log([nstar(cexp, route, b)[0] for b in BUD]), 1)[0])
    est["cross"] = crossover(est)
    return est


def pct(a, q=(2.5, 97.5)):
    return [float(np.percentile(a, x)) for x in q]


def summarize(est, boots, surf):
    out = {}
    for route in ROUTES:
        for p in ("E", "K", "gamma", "a", "F21"):
            v = np.array([b[route][p] for b in boots])
            out[f"{route}_{p}"] = dict(point=float(est[route][p]), ci95=pct(v), median=float(np.median(v)))
        out[f"{route}_F21"]["median_shift"] = out[f"{route}_F21"]["median"] / est[route]["F21"] - 1
    fm = np.array([np.mean([v["f"] for v in b["fstar"]["by_budget"].values()]) for b in boots])
    sl = np.array([b["fstar"]["slope_per_decade"] for b in boots])
    out["fstar"] = dict(mean=float(np.mean([v["f"] for v in est["fstar"]["by_budget"].values()])), mean_ci95=pct(fm),
                        slope=float(est["fstar"]["slope_per_decade"]), slope_ci95=pct(sl), slope_sd=float(sl.std()),
                        min_detectable_slope_80pc=float((1.96 + 0.8416) * sl.std()),
                        by_budget={b: dict(f=float(est["fstar"]["by_budget"][b]["f"]), size=est["fstar"]["by_budget"][b]["size"],
                                           ci95=pct([x["fstar"]["by_budget"][b]["f"] for x in boots]),
                                           size_freq={s: float(np.mean([x["fstar"]["by_budget"][b]["size"] == s for x in boots]))
                                                      for s in SIZES if any(x["fstar"]["by_budget"][b]["size"] == s for x in boots)})
                                   for b in est["fstar"]["by_budget"]})
    xs = np.array([b["cross"] for b in boots])
    out["crossover"] = dict(point=float(est["cross"]), median=float(np.median(xs)),
                            frac_before={f"{c:.0e}": float(np.mean(xs < c)) for c in (1e21, 3e21, 1e22, 1e23)},
                            frac_never_by_1e25=float(np.mean(np.isinf(xs))))
    rat = np.array([b["NAT"]["F21"] / b["CONV"]["F21"] for b in boots])
    same = rat * surf["NAT"]["XL"] / surf["CONV"]["XL"]
    tax = np.array([b["CONV"]["F21"] / b["R1"]["F21"] for b in boots])
    out["native_over_convert_frontier_1e21"] = dict(point=est["NAT"]["F21"] / est["CONV"]["F21"], ci95=pct(rat))
    out["native_over_convert_same_size_XL_1e21"] = dict(point=est["NAT"]["F21"] * surf["NAT"]["XL"] / (est["CONV"]["F21"] * surf["CONV"]["XL"]),
                                                        ci95=pct(same))
    out["tax_frontier_1e21"] = dict(point=est["CONV"]["F21"] / est["R1"]["F21"], ci95=pct(tax))
    return out


def run(raw, base, noise, seed):
    rng = np.random.default_rng(seed)
    return [stats(nz.perturb(raw, base, noise, rng)) for _ in range(N_BOOT)]


# ---- f* estimators
def fcurve(cells, b, size):
    return sorted((f, q) for (r, s, bb, f), q in cells.items() if r in ("CONV", "NAT") and s == size and bb == b and f is not None)


def f_est(cells, b, size, how):
    pts = fcurve(cells, b, size)
    if len(pts) < 3:
        return None, None
    if how == "three_point":
        return local_parabola_f(cells, b, size)
    fs = np.array([p[0] for p in pts]); q = np.array([p[1] for p in pts])
    if how == "grid":
        i = int(np.argmin(q))
        return float(fs[i]), float(q[i])
    sel = (fs > 0) & (fs < 0.95)  # quadratic in log(1 - f) over all interior grid points
    if sel.sum() < 3:
        return local_parabola_f(cells, b, size)
    u = np.log(1 - fs[sel]); cf = np.polyfit(u, q[sel], 2)
    if cf[0] <= 0:
        i = int(np.argmin(q[sel]))
        return float(fs[sel][i]), float(q[sel][i])
    um = min(max(-cf[1] / (2 * cf[0]), u.min()), u.max())
    return 1 - math.exp(um), float(np.polyval(cf, um))


def fstar_estimators(raw):
    base = nz.base_of(raw)
    base0 = {k: x for k, v in raw.items() for s, x, _ in v if s == 0}
    out = {}
    for lab, cells in (("all_seeds", base), ("seed0", base0)):
        out[lab] = {}
        for how in ("three_point", "quad_interior", "grid"):
            res = {}
            for b in BUD:
                best = None
                for s in SIZES:
                    f, q = f_est(cells, b, s, how)
                    if f is not None and (best is None or q < best[1]):
                        best = (f, q, s)
                res[b] = dict(f=float(best[0]), size=best[2])
            out[lab][how] = res
    # f* of every (size, budget) cell with an f-curve (all seeds), and whether that size is the budget's best
    best_size = {b: out["all_seeds"]["three_point"][b]["size"] for b in BUD}
    out["by_cell"] = {f"{s}@{b}": dict(f_three_point=f_est(base, b, s, "three_point")[0], f_grid=f_est(base, b, s, "grid")[0],
                                       npar=NPAR[s], best_size=best_size[b])
                      for b in BUD for s in SIZES if f_est(base, b, s, "grid")[0] is not None}
    return out


def fstar_by_seed(raw):
    """f* of seed 0 and of seed 1 (independent FM, native and conversion runs) on their common grid
    f in {0, 0.5, 0.8, 0.95, 0.99}, three-point parabola, for every cell replicated at every f."""
    grid_f = (0.0, 0.5, 0.8, 0.95, 0.99)
    out = {}
    for size, b in sorted({(k[1], k[2]) for k in raw if k[0] in ("CONV", "NAT")}):
        per = {}
        for seed in (0, 1):
            c = {k: x for k, v in raw.items() for s, x, _ in v
                 if s == seed and k[0] in ("CONV", "NAT") and k[1] == size and k[2] == b and k[3] in grid_f}
            if len({k[3] for k in c}) == len(grid_f):
                per[seed] = dict(f=float(local_parabola_f(c, b, size)[0]), grid_best=min(c, key=c.get)[3])
        if len(per) == 2:
            out[f"{size}@{b}"] = per
    d = np.array([v[0]["f"] - v[1]["f"] for v in out.values()])
    return dict(cells=out, n=len(d), max_abs_diff=float(np.abs(d).max()), sd_one_run=float(np.sqrt(np.mean(d ** 2) / 2)))


def main():
    t0 = time.time()
    raw = nz.load_runs()
    base = nz.base_of(raw)
    noise = nz.estimate(raw)
    res = dict(noise=noise)
    res["bootstrap"] = summarize(stats(base), run(raw, base, noise, 20261023), size_penalties(base))
    res["fstar_estimators"] = fstar_estimators(raw)
    res["fstar_by_seed"] = fstar_by_seed(raw)
    json.dump(res, open(f"{ROOT}/analysis/out/noise_model.json", "w"), indent=1, default=float)
    tables(res)
    print(f"done {time.time() - t0:.0f}s")


def tables(res):
    nm = res["noise"]
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         " & Multi-step & Native & Convert ($f<0.95$) & Convert ($f\\ge0.95$) \\\\", "\\midrule",
         "Seed SD of $\\log$ FID (one run) & " + " & ".join(f"{nm['sd'][g]:.3f}" for g in nz.GROUPS) + " \\\\",
         "Replicated cells & " + " & ".join(str(nm["n_cells"][g]) for g in nz.GROUPS) + " \\\\",
         "Correlation of two branches of one mainline & " + " & ".join(
             f"{nm['rho_within'][g][0]:.2f} ({nm['rho_within'][g][1]})" for g in ("R1", "NAT", "CONV")) +
         f" & {nm['conv_short']['observed']:.2f}$^\\dagger$ \\\\",
         "\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/ta_noise.tex", "w").write("\n".join(L) + "\n")
    # f* of every size and budget (bold: the compute-optimal conversion size)
    F = res["fstar_estimators"]
    L = ["\\begin{tabular}{l" + "c" * len(BUD) + "}", "\\toprule", "Size & " + " & ".join(BUD) + " \\\\", "\\midrule"]
    for s in SIZES:
        row = []
        for b in BUD:
            v = F["by_cell"].get(f"{s}@{b}")
            if v is None:
                row.append("--")
            else:
                t = f"{v['f_three_point']:.2f}"
                row.append(f"\\textbf{{{t}}}" if v["best_size"] == s else t)
        L.append(f"{s} & " + " & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{ROOT}/paper/tables/ta_fstar_nc.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
