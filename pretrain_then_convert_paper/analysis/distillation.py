"""DMD2 distillation with the teacher counted (group `distillation`): each distilled student's 1-NFE FID-50k against total training
compute under three accountings, compared with the pretrain->convert frontier law at the same compute.

  C_counted   = C_teacher + C_student_init + C_distill
  C_sunk      = C_student_init + C_distill                (teacher already exists)
  C_amort(k)  = C_teacher / k + C_student_init + C_distill  (one teacher serves k students; k = 4, 16)

C_teacher = budget of the R1 cooldown teacher (1e20 or 3e20); C_student_init = 8.5e18 (student-size FM milestone
cd_1e19; 0 for self-init runs, whose init is the teacher). The convert frontier is the converted branch of the joint
one-step law (analysis/joint_law.py, refitted here on the current grid), so "beats convert" means below it at that compute.

  python analysis/distillation.py  ->  analysis/out/distillation.json, paper/figures/fig_distillation
"""
import csv
import json
import os
import re

import numpy as np

import figures as F

from common import ROOT
from common import EVAL
BVAL = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
INIT = 8.5e18


def runs():
    out = []
    for d in sorted(os.listdir(EVAL)):
        m = re.match(r"r4_(\w+?)_(\w+?)(1e20|3e20)_d(1e19|3e19|1e20)(_selfinit|_init3e19)?_s(\d)$", d)
        if not m or not os.path.exists(f"{EVAL}/{d}/summary.json"):
            continue
        ss, ts, tb, db, var, seed = m.groups()
        init = 0.0 if var == "_selfinit" else (2.55e19 if var == "_init3e19" else INIT)
        s = json.load(open(f"{EVAL}/{d}/summary.json"))
        ct, cd = BVAL[tb], BVAL[db]
        out.append(dict(run_id=d, student=ss, teacher=ts, teacher_budget=tb, distill=db, variant=(var or "").strip("_") or "main",
                        fid50k=s["fid50k"], C_teacher=ct, C_init=init, C_distill=cd,
                        C_counted=ct + init + cd, C_sunk=init + cd, C_amort4=ct / 4 + init + cd, C_amort16=ct / 16 + init + cd))
    return out


def main():
    # the converted frontier of the reported joint law (analysis/joint_law.py, free multiplier m_hat)
    import joint_law as jl
    import noise as nz
    pj, _, _ = jl.fit_joint(nz.base_of(nz.load_runs()), True)
    law = lambda c: jl.law(pj[:3], pj[3] * np.asarray(c) / 1e20)  # noqa: E731
    rs = runs()
    for r in rs:
        for acc in ("counted", "sunk", "amort4", "amort16"):
            r[f"convert_law_at_{acc}"] = float(law(r[f"C_{acc}"]))
            r[f"ratio_vs_convert_{acc}"] = r["fid50k"] / r[f"convert_law_at_{acc}"]
    json.dump(rs, open(f"{ROOT}/analysis/out/distillation.json", "w"), indent=1)
    print(f"{len(rs)} evaluated distillation runs")
    for r in sorted(rs, key=lambda r: (r["teacher"], r["teacher_budget"], r["student"], r["C_distill"])):
        print(f"{r['run_id']:34s} FID {r['fid50k']:6.2f} | /convert: counted {r['ratio_vs_convert_counted']:.2f} "
              f"sunk {r['ratio_vs_convert_sunk']:.2f} amort16 {r['ratio_vs_convert_amort16']:.2f}")
    if not rs:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    DIST = "#4a3aa7"  # distillation route colour (categorical slot 7; all-pairs PASS with the three route colours)
    SCOL = {"B": "#9d93e2", "M": DIST}  # student size: lighter = smaller (validated with the convert colour; shapes differ too)
    cs_fr = np.array([1e19, 3e19, 1e20, 3e20])
    fr_conv = [min(float(r["fid50k"]) for r in csv.DictReader(open(f"{ROOT}/analysis/out/suite_metrics.csv"))
                   if re.match(rf"r3_\w+?_{b}_f[0-9.]+_s0$", r["run_id"])) for b in ("1e19", "3e19", "1e20", "3e20")]
    fig, axs = plt.subplots(1, 2, figsize=(F.FULL, 2.7), sharey=True, layout="constrained")
    grid = np.logspace(18.7, 21, 160)
    for ax, acc, title in zip(axs, ("counted", "sunk"), ("teacher compute counted", "teacher compute sunk (already paid)")):
        ax.set_xscale("log"); F.logaxis(ax, "y", [2, 3, 5, 8, 12], ["2", "3", "5", "8", "12"])
        ax.plot(grid[grid <= 3e20], law(grid[grid <= 3e20]), color=F.ROUTE_C["CONV"], lw=1.3)
        ax.plot(grid[grid >= 3e20], law(grid[grid >= 3e20]), color=F.ROUTE_C["CONV"], lw=1.3, ls=(0, (3, 2)))
        ax.plot(cs_fr, fr_conv, "o", color=F.ROUTE_C["CONV"], mec=F.SURF, mew=1.0, zorder=3)
        for ts, tb in sorted({(r["teacher"], r["teacher_budget"]) for r in rs}):
            for ss, mk in (("B", "o"), ("M", "s")):
                pts = sorted((r[f"C_{acc}"], r["fid50k"]) for r in rs if r["teacher"] == ts and r["teacher_budget"] == tb
                             and r["student"] == ss and r["variant"] == "main")
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SCOL[ss], lw=0.7, alpha=0.6)
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=SCOL[ss], mec=F.SURF, mew=0.7, ms=4, zorder=3)
        for r in rs:
            if r["variant"] == "selfinit":
                ax.plot(r[f"C_{acc}"], r["fid50k"], "*", color=DIST, mec=F.SURF, mew=0.6, ms=8.5, zorder=4)
                tb_ = {"1e20": "$10^{20}$", "3e20": "$3{\\times}10^{20}$"}.get(r["teacher_budget"], r["teacher_budget"])
                ax.annotate(f"{r['student']} from {r['teacher']} ({tb_})", (r[f"C_{acc}"], r["fid50k"]),
                            textcoords="offset points", color=F.INK, fontsize=6.3,  # placed clear of the law curve and the other star
                            **({"xytext": (5, -5), "ha": "left", "va": "top"} if r["student"] != "L" else
                               {"xytext": (0, 6), "ha": "center", "va": "bottom"} if acc == "counted" else
                               {"xytext": (-7, 0), "ha": "right", "va": "center"}))
        ax.set_title({"counted": "(a) teacher compute counted", "sunk": "(b) teacher compute sunk (already paid)"}[acc], color=F.INK)
        ax.set_xlabel("training compute (FLOPs)"); ax.set_xlim(6e18, 1.2e21); ax.set_ylim(1.7, 16)
    axs[0].set_ylabel("1-NFE FID-50k")
    fig.legend(handles=[Line2D([], [], color=F.ROUTE_C["CONV"], marker="o", mec=F.ROUTE_C["CONV"], label="pretrain → convert (law, dashed: extrapolated; points: frontier)"),
                           Line2D([], [], color=SCOL["B"], marker="o", ls="none", mec=F.SURF, label="DMD2 student B (4 teachers × 3 budgets)"),
                           Line2D([], [], color=SCOL["M"], marker="s", ls="none", mec=F.SURF, label="DMD2 student M (4 teachers × 3 budgets)"),
                           Line2D([], [], color=DIST, marker="*", ls="none", ms=8, mec=F.SURF, label="DMD2 same-size student, initialized from its teacher")],
                  loc="outside lower center", ncol=2)
    F.save(fig, "fig_distillation")


if __name__ == "__main__":
    main()
