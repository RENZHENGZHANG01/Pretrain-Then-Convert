"""Robustness checks of the frontier laws and ratios: backtest on 3e20 (floored and plain power laws), profile of the
floor E, the native/convert crossover and the asymptotic one-step tax that the floors imply, seed-noise intervals for
the FD-DINOv2 ratios, native vs convert at matched size (= matched 1-NFE inference cost), the selection bias of the
converted frontier, and DMD2 self-initialized students vs conversion from the same FM run.

  python analysis/robustness.py  ->  analysis/out/robustness.json (+ summary on stdout)
"""
import csv
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import brentq, least_squares

from common import ROOT
from common import EVAL
from frontier import BUD, analyse, fit_floor, frontier, load
import noise as nz

ROUTES = ("R1", "CONV", "NAT")


def law(p, c):
    E, K, g = p
    return E + K * (c / 1e20) ** (-g)


def backtest(base):
    """Fit on 1e19-1e20, predict the 3e20 frontier (floored law and a plain power law)."""
    out = {}
    for route in ROUTES:
        fr = frontier(base, route)
        fr3 = {b: fr[b] for b in ("1e19", "3e19", "1e20")}
        p = fit_floor(fr3)
        lc = np.log([BUD[b] for b in fr3]); ly = np.array(list(fr3.values()))
        sl, ic = np.polyfit(lc, ly, 1)
        meas = math.exp(fr["3e20"])
        out[route] = dict(measured=meas, floored=float(law(p, 3e20)), power=float(math.exp(ic + sl * math.log(3e20))),
                          floored_fit=[float(x) for x in p])
    return out


def profile_E(base, sds):
    """chi^2(E) with K, gamma refitted, noise = the route's single-run seed SD of log FID (analysis/noise.py);
    95% set of E (delta chi^2 <= 3.84)."""
    out = {}
    for route in ROUTES:
        sd = sds[route]
        fr = frontier(base, route)
        c = np.array([BUD[b] for b in fr]) / 1e20; y = np.array(list(fr.values()))
        grid = np.linspace(0.0, math.exp(y.min()) * 0.999, 200)
        chi = []
        for E in grid:
            def resid(p):
                return (np.log(E + p[0] * c ** (-p[1])) - y) / sd
            best = min((least_squares(resid, [k0, g0], bounds=([1e-6, 0.01], [1e3, 3.0])) for k0 in (0.5, 2.0) for g0 in (0.3, 0.8)),
                       key=lambda r: r.cost)
            chi.append(2 * best.cost)
        chi = np.array(chi); ok = grid[chi - chi.min() <= 3.84]
        out[route] = dict(E_best=float(grid[chi.argmin()]), E_95=[float(ok.min()), float(ok.max())], chi2_min=float(chi.min()))
    return out


def crossover(est):
    """Compute where the floored native and convert frontiers cross (inf if not before 1e25)."""
    pn = (est["NAT"]["E"], est["NAT"]["K"], est["NAT"]["gamma"]); pc = (est["CONV"]["E"], est["CONV"]["K"], est["CONV"]["gamma"])
    g = lambda lc: law(pn, 10 ** lc) - law(pc, 10 ** lc)
    if g(25.0) > 0:
        return math.inf
    return 10 ** brentq(g, 20.0, 25.0) if g(20.0) > 0 else 1e20


def fd_seed_noise(by_route=False):
    """Pooled between-seed SD of log FD-DINOv2 (FID-selected guidance) over replicated grid cells, excluding short
    conversions (f >= 0.95), as for the FID seed SD of frontier.load (they never enter a frontier).
    by_route: one SD per route (r1 multi-step, r2 native, r3 convert)."""
    groups = defaultdict(list)
    for rid in os.listdir(EVAL):
        m = re.match(r"(r[123]_\w+?_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+))_s(\d)$", rid)
        p = f"{EVAL}/{rid}/extra.json"
        if m and m.group(3).startswith("f") and float(m.group(3)[1:]) >= 0.95:
            continue
        if m and "_lr" not in rid and os.path.exists(p):
            groups[m.group(1)].append(math.log(json.load(open(p))["fd_dinov2"]))
    if by_route:
        return {r: float(math.sqrt(np.mean([np.var(g, ddof=1) for k, g in groups.items() if k.startswith(r) and len(g) >= 2])))
                for r in ("r1", "r2", "r3")}
    v = [np.var(g, ddof=1) for g in groups.values() if len(g) >= 2]
    return float(math.sqrt(np.mean(v))), len(v)


def selection_bias():
    """Selection bias of the converted frontier: in the cells with two seeds, choose the split on seed 0
    and report seed 1, against the minimum over splits on seed 1 (splits shared by both seeds)."""
    fid = {}
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        m = re.match(r"(r[23])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if m and "_lr" not in r["run_id"]:
            rt, size, b, k, sd = m.groups()
            fid[(rt, size, b, k, int(sd))] = float(r["fid50k"])
    cells = []
    for (rt, size, b, k, sd) in sorted(fid):
        if rt != "r2" or sd != 1:
            continue
        fs = sorted({kk for (r_, s_, b_, kk, d_) in fid if r_ == "r3" and s_ == size and b_ == b and d_ == 1
                     and ("r3", size, b, kk, 0) in fid})
        if len(fs) < 3:
            continue
        f0 = min(fs, key=lambda kk: fid[("r3", size, b, kk, 0)])
        f1 = min(fs, key=lambda kk: fid[("r3", size, b, kk, 1)])
        n1 = fid[("r2", size, b, "cd", 1)]
        cells.append(dict(cell=f"{size}@{b}", f_seed0=float(f0[1:]), f_seed1=float(f1[1:]),
                          ratio_selected_on_seed0=n1 / fid[("r3", size, b, f0, 1)], ratio_min_on_seed1=n1 / fid[("r3", size, b, f1, 1)]))
    g = lambda k: float(np.exp(np.mean([math.log(c[k]) for c in cells])))  # noqa: E731
    return dict(cells=cells, n=len(cells), same_split=sum(c["f_seed0"] == c["f_seed1"] for c in cells),
                geomean_selected_on_seed0=g("ratio_selected_on_seed0"), geomean_min_on_seed1=g("ratio_min_on_seed1"),
                inflation=g("ratio_min_on_seed1") / g("ratio_selected_on_seed0") - 1)


def suite():
    d = defaultdict(list)
    for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv")):
        d[r["run_id"].rsplit("_s", 1)[0]].append(float(r["fid50k"]))
    return {k: float(np.exp(np.mean(np.log(v)))) for k, v in d.items()}


def main():
    raw, sd = load()
    base = {k: float(np.mean(v)) for k, v in raw.items()}
    est = analyse(base)
    rawn = nz.load_runs()
    nm = nz.estimate(rawn)
    res = dict(seed_sd_logfid=sd, seed_sd_by_route=nm["sd"])
    res["backtest_3e20"] = backtest(base)
    res["profile_E"] = profile_E(base, nm["sd"])
    # crossover and asymptotic tax: parametric bootstrap under the seed-noise model (the replicates of analysis/noise_model.py)
    rng_b = np.random.default_rng(20261023)
    boots = [analyse(nz.perturb(rawn, nz.base_of(rawn), nm, rng_b)) for _ in range(2000)]
    xs = np.array([crossover(b) for b in boots]); x0 = crossover(est)
    res["native_convert_crossover"] = dict(point=x0, median=float(np.median(xs)),
                                           frac_before={f"{c:.0e}": float(np.mean(xs < c)) for c in (1e21, 3e21, 1e22, 1e23)},
                                           frac_never_by_1e25=float(np.mean(np.isinf(xs))))
    at = np.array([b["CONV"]["E"] / b["R1"]["E"] for b in boots])
    res["asymptotic_tax_floors"] = dict(point=est["CONV"]["E"] / est["R1"]["E"], ci95=[float(np.percentile(at, 2.5)), float(np.percentile(at, 97.5))])
    # FD-DINOv2 ratios with seed-noise intervals (two independent single-seed models per ratio)
    sfd, ncell = fd_seed_noise()
    sfr = fd_seed_noise(by_route=True)
    fs = json.load(open(f"{ROOT}/analysis/out/fd_dinov2.json"))
    hws = {"tax_fd_fdsel": 1.96 * math.sqrt(sfr["r3"] ** 2 + sfr["r1"] ** 2),  # convert vs multi-step, two runs
           "nat_over_conv_fd_fdsel": 1.96 * math.sqrt(sfr["r2"] ** 2 + sfr["r3"] ** 2)}
    res["fd_seed_sd_log"] = dict(sd=sfd, n_cells=ncell, by_route=sfr)
    res["fd_ratio_ci95"] = {key: {b: [v * math.exp(-hws[key]), v, v * math.exp(hws[key])] for b, v in fs[key].items()}
                            for key in ("tax_fd_fdsel", "nat_over_conv_fd_fdsel")}
    # matched size (equal 1-NFE inference cost): native / best convert at equal training compute
    cells = defaultdict(dict)
    for (route, size, b, f), q in base.items():
        if route in ("NAT", "CONV"):
            cells[(size, b)][route] = min(cells[(size, b)].get(route, 1e9), math.exp(q))
    pairs = {f"{s}_{b}": v["NAT"] / v["CONV"] for (s, b), v in cells.items() if "NAT" in v and "CONV" in v}
    # 95% intervals of each pair under the seed-noise model (the best split is re-chosen per replicate)
    rng_p = np.random.default_rng(20261026)
    reps = defaultdict(list)
    for _ in range(2000):
        pc = defaultdict(dict)
        for (route, size, b, f), q in nz.perturb(rawn, nz.base_of(rawn), nm, rng_p).items():
            if route in ("NAT", "CONV"):
                pc[(size, b)][route] = min(pc[(size, b)].get(route, 1e9), math.exp(q))
        for key in pairs:
            s_, b_ = key.split("_")
            reps[key].append(pc[(s_, b_)]["NAT"] / pc[(s_, b_)]["CONV"])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()}
    res["matched_size_native_over_convert"] = dict(pairs=pairs, ci95=ci, n=len(pairs), min=min(pairs.values()), max=max(pairs.values()),
                                                   n_excluding_parity=sum(1 for v in ci.values() if v[0] > 1))
    res["selection_bias"] = selection_bias()
    # DMD2 self-init vs conversion from the same FM run (the FM model is free for both)
    s = suite()
    r4 = {(r["student"], r["teacher"], r["teacher_budget"]): r for r in json.load(open(f"{ROOT}/analysis/out/distillation.json"))
          if r["variant"] == "selfinit"}
    res["dmd_vs_convert_same_fm"] = {
        "XL": dict(dmd=dict(fm=3e20, extra=3e19, fid=r4[("XL", "XL", "3e20")]["fid50k"]),
                   convert=dict(fm=2.85e20, extra=1.5e19, fid=s["r3_XL_3e20_f0.95"])),
        "L": dict(dmd=dict(fm=1e20, extra=3e19, fid=r4[("L", "L", "1e20")]["fid50k"]),
                  convert_f080=dict(fm=8e19, extra=2e19, fid=s["r3_L_1e20_f0.80"]),
                  convert_f095=dict(fm=9.5e19, extra=5e18, fid=s["r3_L_1e20_f0.95"]))}
    json.dump(res, open(f"{ROOT}/analysis/out/robustness.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4))[:6000])


if __name__ == "__main__":
    main()
