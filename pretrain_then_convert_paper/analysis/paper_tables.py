"""Table 1 and appendix tables: every grid model (FID-50k), the joint law and held-out budgets, the compute-optimal
sizes with the step-cost prediction for native training, the value of FM samples, kappa with an efficient iMF
implementation, sunk FM models (DMD2 vs conversion) and the conversion initialization.

  python analysis/paper_tables.py  ->  paper/tables/t1_col.tex, ta_all, ta_joint, ta_heldout, ta_nstar, ta_efficient, ta_exchange,
                                       ta_sunk, ta_init (.tex)
"""
import json
import math
import os

import numpy as np

import figures as F
import frontier
from common import EVAL as EV, ROOT, TAB

SIZES = ["XS", "S", "B", "M", "L", "XL", "H"]
FCOLS = [(0.25, (0.24, 0.25, 0.27)), (0.5, (0.5,)), (0.65, (0.65,)), (0.8, (0.8,)), (0.95, (0.95,)), (0.99, (0.99,))]
BTEX = {"1e19": "$10^{19}$", "3e19": "$3{\\times}10^{19}$", "1e20": "$10^{20}$", "3e20": "$3{\\times}10^{20}$", "1e21": "$10^{21}$"}


def ta_all():
    cells = F.load()
    L = ["\\begin{tabular}{llcccccccc}", "\\toprule",
         " & & 50 steps & \\multicolumn{7}{c}{1 NFE: pretraining fraction $f$ ($f=0$: native)} \\\\",
         "\\cmidrule(lr){3-3}\\cmidrule(lr){4-10}",
         "$C$ & Size & multi-step & 0 & 0.25$^a$ & 0.5 & 0.65 & 0.8 & 0.95 & 0.99 \\\\", "\\midrule"]
    for b in F.BUD:
        first = True
        for s in SIZES:
            r1 = cells.get(("R1", s, b, None)); nat = cells.get(("NAT", s, b, 0.0))
            conv = []
            for _, fs in FCOLS:
                v = [cells[("CONV", s, b, f)] for f in fs if ("CONV", s, b, f) in cells]
                conv.append(v[0] if v else None)
            if r1 is None and nat is None and not any(conv):
                continue
            one = [nat] + conv
            best = min(v for v in one if v is not None) if any(v is not None for v in conv) else None  # bold only next to conversions
            fmt = lambda v: "--" if v is None else (f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}")  # noqa: E731
            L.append(f"{BTEX[b] if first else ''} & {s} & {'--' if r1 is None else f'{r1:.2f}'} & " + " & ".join(fmt(v) for v in one) + " \\\\")
            first = False
        L.append("\\midrule" if b != "3e20" else "\\bottomrule")
    L.append("\\end{tabular}")
    open(f"{TAB}/ta_all.tex", "w").write("\n".join(L) + "\n")


def ta_nstar():
    d = json.load(open(f"{ROOT}/analysis/out/nstar_pred.json"))
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "$C$ & multi-step & pretrain$\\to$convert & native, measured & native, predicted $\\kappa^{-a}N^*_\\text{FM}$ \\\\", "\\midrule"]
    for b in F.BUD:
        L.append(f"{BTEX[b]} & {d['fm_M'][b]:.0f}M & {d['conv_M'][b]:.0f}M & {d['native_meas_M'][b]:.0f}M & {d['native_pred_M'][b]:.0f}M \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_nstar.tex", "w").write("\n".join(L) + "\n")


def t1_col():
    """Table 1: the best model of each route and budget (cell means over seeds) with FD-DINOv2, precision and recall
    averaged over the same seeds."""
    raw, _ = frontier.load()
    cells = {k: float(np.mean(v)) for k, v in raw.items()}
    short = {"R1": "multi-step", "CONV": "convert", "NAT": "native"}

    def extra_mean(key):
        route, size, b, f = key
        stem = {"R1": f"r1_{size}_{b}_cd", "NAT": f"r2_{size}_{b}_cd"}.get(route) or f"r3_{size}_{b}_f{f:.2f}"
        ids = [f"{stem}_s{s}" for s in range(5)]
        ex = [json.load(open(f"{EV}/{r}/extra.json")) for r in ids if os.path.exists(f"{EV}/{r}/extra.json")]
        return {q: float(np.mean([e[q] for e in ex])) for q in ("fd_dinov2", "precision", "recall")} if ex else None
    L = ["\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
         "$C$ & Route & Size & $f$ & FID & FD$_\\text{D}$ & P & R \\\\", "\\midrule"]
    for i, b in enumerate(F.BUD):
        if i:
            L.append("\\midrule")
        for j, route in enumerate(("R1", "CONV", "NAT")):
            q, k = min((q, k) for k, q in cells.items() if k[0] == route and k[2] == b)
            ex = extra_mean(k)
            fd = f"{ex['fd_dinov2']:.1f}" if ex else "--"
            pr = (f"{ex['precision']:.2f}", f"{ex['recall']:.2f}") if ex else ("--", "--")
            f = "--" if route == "R1" else ("0" if route == "NAT" else f"{k[3]:.2f}")
            L.append(f"{BTEX[b] if j == 0 else ''} & {short[route]} & {k[1]} & {f} & {math.exp(q):.2f} & {fd} & {pr[0]} & {pr[1]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/t1_col.tex", "w").write("\n".join(L) + "\n")


def ta_joint():
    """Separate floored laws vs one law for both one-step routes (analysis/joint_law.py)."""
    J = json.load(open(f"{ROOT}/analysis/out/joint_law.json"))
    t2 = json.load(open(f"{ROOT}/analysis/out/t2_fits.json"))
    ci = lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"  # noqa: E731
    L = ["\\begin{tabular}{lccccccccc}", "\\toprule",
         "Model & parameters & $E$ & $\\gamma$ & $\\hat m$ & rms & AIC & AICc & native $10^{21}$ & convert $10^{21}$ \\\\", "\\midrule"]
    sp = J["separate"]
    L.append(f"Separate laws (Table~\\ref{{tab:fits}}) & 6 & {t2['NAT']['E']['point']:.2f} / {t2['CONV']['E']['point']:.2f} & "
             f"{t2['NAT']['gamma']['point']:.2f} / {t2['CONV']['gamma']['point']:.2f} & -- & {100 * sp['rms']:.1f}\\% & ${sp['aic']:.1f}$ & "
             f"${sp['aicc']:.1f}$ & {sp['nat21']:.2f} & {sp['conv21']:.2f} \\\\")
    a = J["A"]
    L.append(f"One law, $m$ from step costs & 3 & {a['E']:.2f} {ci(a['E_ci95'])} & {a['gamma']:.2f} & 1.67--1.99 & {100 * a['rms']:.1f}\\% & "
             f"${a['aic']:.1f}$ & ${a['aicc']:.1f}$ & {a['nat21']:.2f} {ci(a['nat21_pred95'])} & {a['conv21']:.2f} {ci(a['conv21_pred95'])} \\\\")
    b = J["B"]
    L.append(f"One law, free $\\hat m$ & 4 & {b['E']:.2f} {ci(b['E_ci95'])} & {b['gamma']:.2f} & {b['m_hat']:.2f} {ci(b['m_hat_ci95'])} & "
             f"{100 * b['rms']:.1f}\\% & ${b['aic']:.1f}$ & ${b['aicc']:.1f}$ & {b['nat21']:.2f} {ci(b['nat21_pred95'])} & "
             f"{b['conv21']:.2f} {ci(b['conv21_pred95'])} \\\\")
    r = J["runs_1e21"]
    L += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{$10^{{21}}$ runs}} & {r['nat']:.2f} & {r['conv']:.2f} \\\\", "\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_joint.tex", "w").write("\n".join(L) + "\n")
    # held-out budgets: each law refitted without one budget and evaluated there
    names = (("separate", "Separate laws"), ("A", "One law, $m$ from step costs"), ("B", "One law, free $\\hat m$"))
    L = ["\\begin{tabular}{llcccc}", "\\toprule", "Law & frontier & " + " & ".join(BTEX[b] for b in ("1e19", "3e19", "1e20", "3e20")) + " \\\\", "\\midrule"]
    for k, name in names:
        for fr, lab in (("native", "native"), ("convert", "convert")):
            L.append(f"{name if fr == 'native' else ''} & {lab} & " +
                     " & ".join(f"${100 * J['held_out'][k]['lobo'][b][fr]:+.0f}$\\%" for b in ("1e19", "3e19", "1e20", "3e20")) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_heldout.tex", "w").write("\n".join(L) + "\n")


def ta_exchange():
    x = json.load(open(f"{ROOT}/analysis/out/exchange_rate.json"))
    d, q = x["by_f"], x["by_f_quadratic"]
    L = ["\\begin{tabular}{lccccc}", "\\toprule", "$f$ & cells & median & 95\\% interval & range & quadratic \\\\", "\\midrule"]
    for k, v in d.items():
        L.append(f"{k.replace('-', '--')} & {v['n']} & {v['median']:.2f} & [{v['median_ci95'][0]:.2f}, {v['median_ci95'][1]:.2f}] & {v['lo']:.2f}--{v['hi']:.2f} & "
                 f"{q[k]['median']:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_exchange.tex", "w").write("\n".join(L) + "\n")


def ta_efficient():
    """kappa and the frontier advantage with an iMF implementation that skips the JVP for the r = t half and the
    interval-adjusted guidance pass where t lies inside the interval (analysis/joint_law.py efficient)."""
    e = json.load(open(f"{ROOT}/analysis/out/joint_law.json"))["efficient"]
    sizes = [s for s in ["XS", "S", "B", "M", "L", "XL", "H"] if s in e["kappa"]]
    L = ["\\begin{tabular}{l" + "c" * len(sizes) + "}", "\\toprule", "Size & " + " & ".join(sizes) + " \\\\", "\\midrule",
         "$\\kappa$, measured & " + " & ".join(f"{e['kappa'][s]['measured']:.2f}" for s in sizes) + " \\\\",
         "$\\kappa$, efficient implementation & " + " & ".join(f"{e['kappa'][s]['efficient']:.2f}" for s in sizes) + " \\\\",
         "\\bottomrule", "\\end{tabular}", "", "\\vspace{4pt}", "",
         "\\begin{tabular}{l" + "c" * len(e["ratio"]) + "}", "\\toprule",
         "Budget & " + " & ".join(BTEX[b] for b in e["ratio"]) + " \\\\", "\\midrule",
         "native $\\div$ convert, measured FLOPs & " + " & ".join(f"{v['measured']:.2f}" for v in e["ratio"].values()) + " \\\\",
         "native $\\div$ convert, efficient FLOPs & " + " & ".join(f"{v['efficient']:.2f}" for v in e["ratio"].values()) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_efficient.tex", "w").write("\n".join(L) + "\n")


def fid(run):
    p = f"{EV}/{run}/summary.json"
    return json.load(open(p))["fid50k"] if os.path.exists(p) else None


def ta_sunk():
    """An FM model already exists (group `sunk`). Same-size DMD2 student initialized from it vs converting it, each with
    3e19 FLOPs of new compute; the 50-step FID of the FM model for reference."""
    rows = [("XL", "r1_XL_3e20_cd_s0", "r4_XL_XL3e20_d3e19_selfinit_s0", "r3_XL_sunk3e20_d3e19_s0", "$3{\\times}10^{20}$"),
            ("L", "r1_L_1e20_cd_s0", "r4_L_L1e20_d3e19_selfinit_s0", "r3_L_sunk1e20_d3e19_s0", "$10^{20}$")]
    f = lambda v: "--" if v is None else f"{v:.2f}"  # noqa: E731
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "FM model (training compute) & FM, 50 steps & DMD2 student, 1 NFE & converted, 1 NFE \\\\", "\\midrule"]
    for size, fm, dmd, conv, c in rows:
        L.append(f"{size} ({c}) & {f(fid(fm))} & {f(fid(dmd))} & {f(fid(conv))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_sunk.tex", "w").write("\n".join(L) + "\n")


def ta_init():
    """Function-preserving vs naive conversion initialization (M, f = 0.65; group `ablations`)."""
    rows = [("$3{\\times}10^{19}$", "r3_M_3e19_f0.65_s0", "r3_M_3e19_f0.65_naive_s0"),
            ("$10^{20}$", "r3_M_1e20_f0.65_s0", "r3_M_1e20_f0.65_naive_s0")]
    f = lambda v: "--" if v is None else f"{v:.2f}"  # noqa: E731
    L = ["\\begin{tabular}{lccc}", "\\toprule", "$C$ (size M, $f=0.65$) & function-preserving & naive & naive $\\div$ preserving \\\\",
         "\\midrule"]
    for c, a, b in rows:
        fa, fb = fid(a), fid(b)
        L.append(f"{c} & {f(fa)} & {f(fb)} & {'--' if fa is None or fb is None else f'{fb / fa:.2f}'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_init.tex", "w").write("\n".join(L) + "\n")


def main():
    t1_col(); ta_all(); ta_nstar(); ta_joint(); ta_efficient(); ta_exchange(); ta_sunk(); ta_init()
    print("tables written")


if __name__ == "__main__":
    main()
