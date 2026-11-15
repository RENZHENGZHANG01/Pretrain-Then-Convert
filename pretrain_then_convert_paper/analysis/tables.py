"""Tables of the paper from analysis/out/suite_metrics.csv and the evaluation results: fitted frontier laws (t2_fits),
pipeline checks (t4_repro), FLOPs per sample (ta_flops), learning rates (ta_lr), DMD2 students (t6_distillation).

  python analysis/tables.py  ->  paper/tables/{t2_fits,t4_repro,ta_flops,ta_lr,t6_distillation}.tex, analysis/out/t2_fits.json

Uncertainty: parametric bootstrap with the seed-noise model of analysis/noise.py (route-specific SDs, noise shared
by the runs of one mainline); 2000 replicates, fixed seed.
"""
import json
import math
import os

import numpy as np

from common import EVAL, ROOT
from frontier import BUD, fit_floor, fstar_by_budget
from figures import nstar
import noise as nz

OUT = f"{ROOT}/paper/tables"
os.makedirs(OUT, exist_ok=True)
ROUTE_TEX = {"R1": "Multi-step (50 steps)", "CONV": "Pretrain$\\to$convert (1 NFE)", "NAT": "Native one-step (1 NFE)"}


def best_cell(cells, route, b):
    c = [(q, k) for k, q in cells.items() if k[0] == route and k[2] == b]
    return min(c) if c else (None, None)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def t2(rawn, nm):
    base = nz.base_of(rawn)

    def stats_of(cells):
        out = {}
        cs = np.log(np.array(list(BUD.values())))
        for route in ("R1", "CONV", "NAT"):
            fr = {b: best_cell(cells, route, b)[0] for b in BUD}
            E, K, g = fit_floor(fr)
            ns = np.log([nstar({kk: math.exp(v) for kk, v in cells.items()}, route, b)[0] for b in BUD])
            out[route] = dict(E=E, K=K, gamma=g, a=float(np.polyfit(cs, ns, 1)[0]))
        fs = fstar_by_budget(cells)
        out["fstar_mean"] = float(np.mean([v[0] for v in fs.values()]))
        out["fstar_slope"] = float(np.polyfit(np.log10([BUD[b] for b in fs]), [fs[b][0] for b in fs], 1)[0])
        return out

    est = stats_of(base)
    rng_t2 = np.random.default_rng(20261023)  # the replicates of analysis/noise_model.py
    boots = [stats_of(nz.perturb(rawn, base, nm, rng_t2)) for _ in range(2000)]
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Route & $E$ (floor) & $K$ & $\\gamma$ & $a$ in $N^*\\propto C^{a}$ \\\\", "\\midrule"]
    res = {}
    for route in ("R1", "CONV", "NAT"):
        row = [ROUTE_TEX[route]]
        res[route] = {}
        for p in ("E", "K", "gamma", "a"):
            lo, hi = ci([b[route][p] for b in boots])
            row.append(f"{est[route][p]:.2f} [{lo:.2f}, {hi:.2f}]")
            res[route][p] = dict(point=float(est[route][p]), ci95=[lo, hi])
        L.append(" & ".join(row) + " \\\\")
    lo, hi = ci([b["fstar_mean"] for b in boots]); slo, shi = ci([b["fstar_slope"] for b in boots])
    L += ["\\midrule",
          f"\\multicolumn{{5}}{{l}}{{$f^*$ averaged over budgets: {est['fstar_mean']:.2f} [{lo:.2f}, {hi:.2f}];"
          f" change per decade of $C$: {est['fstar_slope']:+.3f} [{slo:+.3f}, {shi:+.3f}]}} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    res["fstar"] = dict(mean=est["fstar_mean"], mean_ci95=[lo, hi], slope_per_decade=est["fstar_slope"], slope_ci95=[slo, shi])
    open(f"{OUT}/t2_fits.tex", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{ROOT}/analysis/out/t2_fits.json", "w"), indent=1)


def cur(run_id):
    return json.load(open(f"{EVAL}/{run_id}/summary.json"))["fid50k"]


def t4():
    L = ["\\begin{tabular}{lccc}", "\\toprule", "Check & Ours & Official code & Paper \\\\", "\\midrule",
         "iMF-B/2 official weights, 1 NFE, FID-50k & 3.33 & 3.32 & 3.39 \\\\",
         "iMF-XL/2 official weights, 1 NFE, FID-50k & 1.73 & 1.72 & 1.72 \\\\",
         "\\midrule",
         f"Our native B, $C=1\\text{{e}}20$ (184 ep.) vs.\\ published iMF-B/2, 240 ep.\\ ($1.3\\text{{e}}20$) & {cur('r2_B_1e20_cd_s0'):.2f} & -- & 3.82 \\\\",
         f"Our native B, $C=3\\text{{e}}20$ (551 ep.) vs.\\ published iMF-B/2, 640 ep.\\ ($3.5\\text{{e}}20$) & {cur('r2_B_3e20_cd_s0'):.2f} & -- & 3.39 \\\\",
         "\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/t4_repro.tex", "w").write("\n".join(L) + "\n")


def t_a_flops():
    ft = json.load(open(f"{ROOT}/data/flops_table.json"))
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Size & $N$ (inference) & FM / sample & iMF / sample & iMF $\\div$ FM & FM $\\div$ $6N\\cdot256$ \\\\", "\\midrule"]
    for s_ in ("XS", "S", "B", "M", "L", "XL", "H"):
        v = ft[s_]
        L.append(f"{s_} & {v['N_inference'] / 1e6:.0f}M & {v['fm'] / 1e9:.0f} G & {v['imf'] / 1e9:.0f} G & {v['imf_over_fm']:.2f} & "
                 f"{v['fm'] / (6 * v['N_inference'] * 256):.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_flops.tex", "w").write("\n".join(L) + "\n")


def t_a_lr():
    """Learning rate (group `lr`): FID-50k of cooldown models (seed 0) of B and L at 1e19/3e19 (both objectives) and of
    native M at 1e20/3e20 (the native optimum there); at the frontier sizes of 1e19/3e19, native S and conversions whose
    FM pretraining and conversion both use the rate (S@1e19 f=0.5, M@3e19 f=0.65, the converted optima)."""
    rates = (5e-5, 1e-4, 2e-4, 4e-4, 8e-4)
    btex = {"1e19": "$10^{19}$", "3e19": "$3{\\times}10^{19}$", "1e20": "$10^{20}$", "3e20": "$3{\\times}10^{20}$"}
    def get(run):
        sp = f"{EVAL}/{run}/summary.json"
        return json.load(open(sp))["fid50k"] if os.path.exists(sp) else None
    tag = lambda lr: "" if lr == 1e-4 else f"_lr{lr:g}"  # noqa: E731
    rows = []
    for route, pre in (("FM (50 steps)", "r1"), ("native (1 NFE)", "r2")):
        for size, b in (("B", "1e19"), ("B", "3e19"), ("L", "1e19"), ("L", "3e19"), ("M", "1e20"), ("M", "3e20")):
            vals = {lr: v for lr in rates if (v := get(f"{pre}_{size}{tag(lr)}_{b}_cd_s0")) is not None}
            if len(vals) >= 2:
                rows.append((route, size, b, vals))
    lrs = (1e-4, 2e-4, 4e-4, 8e-4)
    for size, b in (("S", "1e19"), ("S", "3e19")):
        vals = {lr: v for lr in lrs if (v := get(f"r2_S{tag(lr)}_{b}_cd_s0")) is not None}
        if len(vals) == len(lrs):
            rows.append(("native (1 NFE)", size, b, vals))
    for size, b, f in (("S", "1e19", 0.5), ("M", "3e19", 0.65)):
        vals = {lr: v for lr in lrs if (v := get(f"r3_{size}_{b}_f{f:.2f}" + ("" if lr == 1e-4 else f"_flr{lr:g}") + "_s0")) is not None}
        if len(vals) == len(lrs):
            rows.append((f"convert, $f={f:g}$", size, b, vals))
    L = ["\\begin{tabular}{lllccccc}", "\\toprule",
         "Route & Size & $C$ & $5{\\times}10^{-5}$ & $10^{-4}$ (used) & $2{\\times}10^{-4}$ & $4{\\times}10^{-4}$ & $8{\\times}10^{-4}$ \\\\",
         "\\midrule"]
    border, sorder = list(btex), ["XS", "S", "B", "M", "L", "XL", "H"]
    groups = [sorted([r for r in rows if r[0].startswith(g)], key=lambda r: (border.index(r[2]), sorder.index(r[1])))
              for g in ("FM", "native", "convert")]
    for gi, grp in enumerate(g for g in groups if g):
        if gi:
            L.append("\\midrule")
        for route, size, b, v in grp:
            best = min(v.values())
            cell = [("\\textbf{%.2f}" % v[lr] if v[lr] == best else f"{v[lr]:.2f}") if lr in v else "--" for lr in rates]
            L.append(f"{route} & {size} & {btex[b]} & " + " & ".join(cell) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_lr.tex", "w").write("\n".join(L) + "\n")
    out = {f"{route}|{size}|{b}": {f"{lr:g}": x for lr, x in v.items()} for route, size, b, v in rows}
    json.dump(out, open(f"{ROOT}/analysis/out/lr_sweep.json", "w"), indent=1)


def t6_distillation():
    """DMD2 students from analysis/out/distillation.json (run analysis/distillation.py first)."""
    path = f"{ROOT}/analysis/out/distillation.json"
    if not os.path.exists(path):
        return
    rs = json.load(open(path))
    L = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Student $\\leftarrow$ teacher & Distill $C$ & FID-50k & \\multicolumn{3}{c}{FID $\\div$ convert law at equal compute} \\\\",
         " & & (1 NFE) & teacher counted & amortized ($k{=}16$) & teacher sunk \\\\", "\\midrule"]
    key = lambda r: ({"main": 0, "init3e19": 1, "selfinit": 2}[r["variant"]], r["student"], r["teacher"], r["teacher_budget"], r["C_distill"])
    prev = None
    for r in sorted(rs, key=key):
        grp = (r["variant"], r["student"], r["teacher"], r["teacher_budget"])
        if prev and grp != prev:
            L.append("\\addlinespace[2pt]")
        prev = grp
        tag = {"main": "", "init3e19": " (init $2.55{\\times}10^{19}$)", "selfinit": " (init = teacher)"}[r["variant"]]
        L.append(f"{r['student']} $\\leftarrow$ {r['teacher']}@{r['teacher_budget']}{tag} & {r['distill']} & {r['fid50k']:.2f} & "
                 f"{r['ratio_vs_convert_counted']:.2f} & {r['ratio_vs_convert_amort16']:.2f} & {r['ratio_vs_convert_sunk']:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/t6_distillation.tex", "w").write("\n".join(L) + "\n")


def main():
    rawn = nz.load_runs()
    nm = nz.estimate(rawn)
    t2(rawn, nm); t4(); t_a_flops(); t_a_lr(); t6_distillation()
    print("tables ->", OUT, sorted(os.listdir(OUT)), "seed SDs of log FID", {g: round(v, 4) for g, v in nm["sd"].items()})


if __name__ == "__main__":
    main()
