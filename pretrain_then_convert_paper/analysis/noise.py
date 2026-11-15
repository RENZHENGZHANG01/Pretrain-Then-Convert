"""Seed-noise model for the parametric bootstraps.

Runs that branch from the same mainline share most of their noise (a worse FM mainline makes all of its conversions
and cooldowns worse), and the seed SD differs by route:

    log FID(run) = mu(cell) + sd_g * (lam_g * z[root mainline] + sqrt(1 - lam_g^2) * eps(run))

with z ~ N(0, 1) shared by every run whose parent chain ends in the same mainline (FM mainlines carry both
the multi-step cooldowns and the conversions; native mainlines carry the native cooldowns) and eps ~ N(0, 1)
per run. Groups g: R1 (multi-step), NAT (native), CONV (conversions with f < 0.95) and SHORT (f >= 0.95,
much noisier; they share the CONV mainline component and add independent noise). sd_g and lam_g are moment
estimates from the seed-0/seed-1 replicate pairs: sd_g^2 = mean(d^2) / 2 over the group's replicated cells,
lam_g^2 = pooled uncentred correlation of d between distinct replicated cells of the group that share both
mainlines; SHORT cells load on the mainline component so as to reproduce their observed correlation with the
CONV cells of the same mainline. A cell's value is the mean over its seeds, as in the fits.
"""
import csv
import math
import re
from collections import defaultdict

import numpy as np

from common import ROOT

GROUPS = ("R1", "NAT", "CONV", "SHORT")


def group(key):
    route, _, _, f = key
    return "SHORT" if route == "CONV" and f >= 0.95 else route


def _roots():
    par = {r["run_id"]: (r["parent"].split(":")[0] if r["parent"] else None) for r in csv.DictReader(open(f"{ROOT}/data/runs.csv"))}

    def root(rid):
        seen = set()
        while par.get(rid) and rid not in seen:
            seen.add(rid)
            rid = par[rid]
        return rid
    return root


def load_runs(path=f"{ROOT}/analysis/out/suite_metrics.csv"):
    """cell (route, size, budget, f) -> [(seed, log FID-50k, root mainline)], the cells of frontier.load."""
    root = _roots()
    raw = defaultdict(list)
    for r in csv.DictReader(open(path)):
        m = re.match(r"(r[123])_(\w+?)_(1e19|3e19|1e20|3e20)_(cd|f[0-9.]+)_s(\d)$", r["run_id"])
        if not m or "_lr" in r["run_id"]:
            continue
        rt, size, b, kind, seed = m.groups()
        key = {"r1": "R1", "r2": "NAT"}.get(rt, "CONV")
        f = 0.0 if rt == "r2" else (float(kind[1:]) if rt == "r3" else None)
        raw[(key, size, b, f)].append((int(seed), math.log(float(r["fid50k"])), root(r["run_id"])))
    return dict(raw)


def base_of(raw):
    return {k: float(np.mean([x for _, x, _ in v])) for k, v in raw.items()}


def estimate(raw):
    seeds = {k: {s: x for s, x, _ in v} for k, v in raw.items()}
    roots = {k: tuple(sorted({rt for _, _, rt in v})) for k, v in raw.items()}
    d = {k: v[0] - v[1] for k, v in seeds.items() if 0 in v and 1 in v}
    sd = {g: math.sqrt(np.mean([d[k] ** 2 for k in d if group(k) == g]) / 2) for g in GROUPS}
    n = {g: sum(1 for k in d if group(k) == g) for g in GROUPS}

    def corr(g1, g2):
        num = s1 = s2 = 0.0
        npair = 0
        ks = sorted(d, key=str)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                if roots[a] != roots[b]:
                    continue
                ga, gb = group(a), group(b)
                if (ga, gb) == (g1, g2):
                    x, y = d[a], d[b]
                elif (gb, ga) == (g1, g2):
                    x, y = d[b], d[a]
                else:
                    continue
                num += x * y
                s1 += x * x
                s2 += y * y
                npair += 1
        return (num / math.sqrt(s1 * s2) if npair else float("nan")), npair

    rho = {g: corr(g, g) for g in ("R1", "NAT", "CONV")}
    lam = {g: math.sqrt(min(max(rho[g][0], 0.0), 1.0)) for g in rho}
    cross, cs = corr("R1", "CONV"), corr("CONV", "SHORT")
    shared = {g: sd[g] * lam[g] for g in lam}  # absolute amplitude of the mainline component (log FID)
    shared["SHORT"] = min(cs[0] * sd["SHORT"] / lam["CONV"], sd["SHORT"])
    return dict(sd=sd, n_cells=n, rho_within=rho, lam=lam, shared=shared,
                cross_r1_conv=dict(observed=cross[0], n_pairs=cross[1], implied=lam["R1"] * lam["CONV"]),
                conv_short=dict(observed=cs[0], n_pairs=cs[1]))


def perturb(raw, base, noise, rng):
    """One parametric-bootstrap replicate of the cell means (log FID)."""
    z = {}
    out = {}
    sd = noise["sd"]
    for k, runs in raw.items():
        g = group(k)
        shared = noise["shared"][g]
        ind = math.sqrt(max(sd[g] ** 2 - shared ** 2, 0.0))
        dev = 0.0
        for _, _, rt in runs:
            if shared > 0 and rt not in z:
                z[rt] = rng.normal()
            dev += (shared * z[rt] if shared > 0 else 0.0) + ind * rng.normal()
        out[k] = base[k] + dev / len(runs)
    return out
