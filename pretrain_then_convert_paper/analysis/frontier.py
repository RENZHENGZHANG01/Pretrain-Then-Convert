"""Frontiers and floored compute laws: the best cell per route and budget, FID*(C) = E + K (C / 1e20)^-gamma fitted on
log FID with E >= 0, and the optimal split f* (vertex of a parabola in log(1 - f) through the best split and its two
neighbours, at the best size of each budget). Cells: (route, size, budget, f) -> log FID-50k, routes R1 (multi-step FM),
NAT (native iMF, f = 0) and CONV (pretrain -> convert)."""
import csv
import math
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from common import BUD, NPAR, OUT  # noqa: F401  (BUD is used by importers)


def load():
    """Seed-wise log FID-50k of every grid cell, and the pooled between-seed SD (f < 0.95)."""
    raw = defaultdict(list)
    for r in csv.DictReader(open(f"{OUT}/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        route, size, b, kind, seed = m.groups()
        key = {"r1": "R1", "r2": "NAT"}.get(route, "CONV")
        f = 0.0 if route == "r2" else (float(kind[1:]) if route == "r3" else None)
        raw[(key, size, b, f)].append(math.log(float(r["fid50k"])))
    sd = float(np.sqrt(np.mean([np.var(v, ddof=1) for k, v in raw.items() if len(v) >= 2 and (k[3] is None or k[3] < 0.95)])))
    return raw, sd


def frontier(cells, route):
    out = {}
    for b in BUD:
        v = [q for (r, s, bb, f), q in cells.items() if r == route and bb == b]
        if v:
            out[b] = min(v)
    return out


def fit_floor(fr):
    c = np.array([BUD[b] for b in fr]) / 1e20
    y = np.exp(np.array([fr[b] for b in fr]))
    def resid(p):
        E, K, g = p
        return np.log(E + K * c ** (-g)) - np.log(y)
    best = None
    for E0 in (0.3, 0.8, 1.2, 1.6):
        for g0 in (0.3, 0.6, 1.0):
            e0 = min(E0, 0.95 * y.min())
            r = least_squares(resid, [e0, max(y.min() - e0, 0.1), g0], bounds=([0, 1e-6, 0.01], [y.min(), 1e3, 3.0]))
            if best is None or r.cost < best.cost:
                best = r
    return best.x


def local_parabola_f(cells, b, size):
    pts = sorted((math.log(1 - f + 1e-12), q) for (r, s, bb, f), q in cells.items()
                 if r in ("CONV", "NAT") and s == size and bb == b and f is not None)
    if len(pts) < 3:
        return None, None
    u = np.array([p[0] for p in pts]); q = np.array([p[1] for p in pts])
    i = int(np.argmin(q)); j = min(max(i - 1, 0), len(u) - 3)
    cf = np.polyfit(u[j:j + 3], q[j:j + 3], 2)
    if cf[0] <= 0 or not (0 < i < len(u) - 1):
        return 1 - math.exp(u[i]), q[i]
    um = min(max(-cf[1] / (2 * cf[0]), u[j]), u[j + 2])
    return 1 - math.exp(um), np.polyval(cf, um)


def fstar_by_budget(cells):
    res = {}
    for b in BUD:
        best = None
        for s in NPAR:
            f, q = local_parabola_f(cells, b, s)
            if f is not None and (best is None or q < best[1]):
                best = (f, q, s)
        if best:
            res[b] = best
    return res


def analyse(cells):
    out = {}
    for route in ("NAT", "CONV", "R1"):
        fr = frontier(cells, route)
        E, K, g = fit_floor(fr)
        out[route] = dict(E=E, K=K, gamma=g, F21=E + K * (10.0) ** (-g))
    fs = fstar_by_budget(cells)
    lc = np.log10([BUD[b] for b in fs]); fv = np.array([fs[b][0] for b in fs])
    slope, icpt = np.polyfit(lc, fv, 1)
    out["fstar"] = dict(by_budget={b: dict(f=float(fs[b][0]), size=fs[b][2]) for b in fs}, slope_per_decade=float(slope),
                        linear_1e21=float(icpt + slope * 21), const=float(fv.mean()))
    return out
