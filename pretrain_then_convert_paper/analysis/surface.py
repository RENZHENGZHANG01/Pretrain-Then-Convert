"""Per-route (N, C) surface fitted in log space, FID(N, C) = E + A (N/1e8)^-alpha + B ((C/N)/1e11)^-beta, on the cells
of analysis/out/suite_metrics.csv (CONV: the best split per size and budget)."""
import csv
import math
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from common import BUD, NPAR, OUT


def load_cells():
    raw = defaultdict(list)
    for r in csv.DictReader(open(f"{OUT}/suite_metrics.csv")):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        route, size, b, kind, seed = m.groups()
        key = {"r1": "R1", "r2": "NAT"}.get(route, "CONVf")
        f = float(kind[1:]) if route == "r3" else None
        raw[(key, size, b, f)].append(math.log(float(r["fid50k"])))
    cells = {"R1": {}, "NAT": {}, "CONV": {}}
    for (key, size, b, f), v in raw.items():
        q = float(np.mean(v))
        if key == "CONVf":
            cur = cells["CONV"].get((size, b))
            if cur is None or q < cur[0]:
                cells["CONV"][(size, b)] = (q, len(v), f)
        else:
            cells[key][(size, b)] = (q, len(v), None)
    seed_sd = float(np.sqrt(np.mean([np.var(v, ddof=1) for k, v in raw.items() if len(v) >= 2 and (k[3] is None or k[3] < 0.95)])))
    return cells, seed_sd


def model(p, n, c):
    E, A, B, a, b = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), np.exp(p[3]), np.exp(p[4])
    return np.log(E + A * (n / 1e8) ** (-a) + B * ((c / n) / 1e11) ** (-b))


def fit(points):
    n = np.array([NPAR[s] for s, b in points]); c = np.array([BUD[b] for s, b in points])
    y = np.array([points[k] for k in points])
    best = None
    for E0 in (0.5, 1.0, 1.5):
        for a0 in (0.3, 0.7):
            for b0 in (0.3, 0.7):
                p0 = np.log([E0, 2.0, 2.0, a0, b0])
                res = minimize(lambda p: np.sum((model(p, n, c) - y) ** 2), p0, method="Nelder-Mead",
                               options=dict(maxiter=20000, xatol=1e-9, fatol=1e-12))
                if best is None or res.fun < best.fun:
                    best = res
    return best.x, float(np.sqrt(best.fun / max(1, len(y) - 5)))


def nstar(p, c):
    E, A, B, a, b = np.exp(p)
    # d/dN [A (N/1e8)^-a + B (C/N/1e11)^-b] = 0
    # a A (N/1e8)^-a = b B (C/(N 1e11))^-b  ->  solve numerically on a log grid
    ns = np.exp(np.linspace(np.log(1e7), np.log(1e10), 4000))
    q = model(p, ns, c)
    return float(ns[int(np.argmin(q))])
