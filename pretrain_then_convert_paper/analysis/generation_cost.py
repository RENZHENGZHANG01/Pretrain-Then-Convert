"""What one-step generation buys: FID against the compute of generating an image, and against the total compute of
training a model and generating K images with it.

Generation compute of one image = network evaluations x forward FLOPs of the network (the u-path; c_FM / 3 per sample,
data/flops_table.json) + the SD-VAE decoder (620 GFLOPs for a 32x32x4 latent, FlopCounterMode). One-step models make
one evaluation; multi-step models are evaluated at 1-32 sampling steps (Euler or Heun, guidance selected per step
count; scripts/eval_steps.py of the code, data/eval/<run>/steps.json) and at the 50 Euler steps of the main protocol,
with two evaluations per guided step. Models: the best model of each route at every budget (seed 0) and the 1e21 runs.

  python analysis/generation_cost.py  ->  analysis/out/generation_cost.json, paper/figures/fig_gencost, paper/tables/ta_steps.tex
"""
import json
import math

from matplotlib.lines import Line2D

import figures as F
from common import EVAL, FT, ROOT, TAB
from figures import plt

VAE_DEC = 620.04e9                       # FLOPs of the SD-VAE decoder per 256x256 image
BUDGETS = {**F.BUD, "1e21": 1e21}
RUNS_1E21 = {"R1": ("H", "v2_H_1e21_cd_s0"), "CONV": ("XL", "v2_XL_1e21_f0.60_s0"), "NAT": ("XL", "v1_XL_1e21_cd_s0")}
K_PANELS = (1e7, 1e9)                    # generated images for the total-compute panels


def fwd(size):
    return FT[size]["fm"] / 3


def gen_cost(size, nfe):
    return nfe * fwd(size) + VAE_DEC


def frontier_runs(cells):
    """route -> budget -> (size, run id of seed 0) of the best cell (geometric mean over seeds)."""
    out = {r: {} for r in F.ROUTES}
    for route in F.ROUTES:
        for b in F.BUD:
            (r, s, bb, f), _ = min(((k, q) for k, q in cells.items() if k[0] == route and k[2] == b), key=lambda kv: kv[1])
            rid = {"R1": f"r1_{s}_{b}_cd_s0", "NAT": f"r2_{s}_{b}_cd_s0"}.get(route) or f"r3_{s}_{b}_f{f:.2f}_s0"
            out[route][b] = (s, rid)
        out[route]["1e21"] = RUNS_1E21[route]
    return out


def collect(cells):
    fr = frontier_runs(cells)
    pts = []  # one entry per (route, budget, sampler setting)
    for route in F.ROUTES:
        for b, (size, rid) in fr[route].items():
            if route == "R1":
                st = json.load(open(f"{EVAL}/{rid}/steps.json"))["final"]
                assert "50" in st, f"{rid}: eval_steps.py has not finished"
                for k, v in sorted(st.items(), key=lambda kv: int(kv[0])):
                    pts.append(dict(route=route, budget=b, C=BUDGETS[b], size=size, run=rid, steps=int(k), solver=v["solver"],
                                    nfe=v["nfe"], fid=v["fid50k"], gen=gen_cost(size, v["nfe"])))
            else:
                fid = json.load(open(f"{EVAL}/{rid}/summary.json"))["fid50k"]
                pts.append(dict(route=route, budget=b, C=BUDGETS[b], size=size, run=rid, steps=1, solver="one-step",
                                nfe=1, fid=fid, gen=gen_cost(size, 1)))
    return pts


def pareto(items):
    """Lower-left envelope of (cost, fid) pairs, sorted by cost."""
    out, best = [], math.inf
    for c, q in sorted(items):
        if q < best:
            out.append((c, q))
            best = q
    return out


def summarize(pts):
    get = lambda route, b, steps=None: next(p for p in pts if p["route"] == route and p["budget"] == b  # noqa: E731
                                           and (steps is None or p["steps"] == steps))
    h50, xl = get("R1", "1e21", 50), get("CONV", "1e21")
    res = dict(vae_decoder_gflops=VAE_DEC / 1e9,
               per_image_1e21=dict(multistep_50=dict(nfe=h50["nfe"], tflops=h50["gen"] / 1e12, fid=h50["fid"]),
                                   convert=dict(tflops=xl["gen"] / 1e12, fid=xl["fid"]),
                                   ratio_with_vae=h50["gen"] / xl["gen"],
                                   ratio_network_only=(h50["gen"] - VAE_DEC) / (xl["gen"] - VAE_DEC)))
    # the cheapest multi-step setting that matches or beats the converted model of the same budget
    match = {}
    for b in BUDGETS:
        conv = get("CONV", b)
        ok = [p for p in pts if p["route"] == "R1" and p["budget"] == b and p["fid"] <= conv["fid"]]
        m = min(ok, key=lambda p: p["gen"]) if ok else None
        match[b] = dict(convert_fid=conv["fid"], convert_tflops=conv["gen"] / 1e12,
                        multistep=None if m is None else dict(steps=m["steps"], nfe=m["nfe"], fid=m["fid"], tflops=m["gen"] / 1e12,
                                                              cost_ratio=m["gen"] / conv["gen"]))
    res["cheapest_multistep_matching_convert"] = match
    # ... and over all budgets, the cheapest multi-step image as good as the best converted model
    m = min((p for p in pts if p["route"] == "R1" and p["fid"] <= xl["fid"]), key=lambda p: p["gen"])
    res["cheapest_multistep_matching_best_convert"] = dict(budget=m["budget"], size=m["size"], steps=m["steps"], nfe=m["nfe"],
                                                          fid=m["fid"], tflops=m["gen"] / 1e12, cost_ratio=m["gen"] / xl["gen"],
                                                          network_ratio=(m["gen"] - VAE_DEC) / (xl["gen"] - VAE_DEC))
    # images after which a model has spent as much compute on generation as on its training
    res["images_until_generation_equals_training_1e21"] = {
        ("one-step convert" if p["route"] == "CONV" else f"multi-step {p['steps']} steps"): p["C"] / p["gen"]
        for p in pts if p["budget"] == "1e21" and p["route"] in ("R1", "CONV")}
    # images above which the converted model of each budget is the cheapest way, in training + generation compute, to
    # reach its FID (every multi-step setting that is at least as good is then more expensive)
    cross = {}
    for b in BUDGETS:
        conv = get("CONV", b)
        ks = [(conv["C"] - p["C"]) / (p["gen"] - conv["gen"]) for p in pts if p["route"] == "R1" and p["fid"] <= conv["fid"]]
        cross[b] = dict(convert_fid=conv["fid"], images=max([0.0] + ks))
    res["images_above_which_convert_is_cheapest"] = cross
    # total-compute frontiers
    res["total_frontier"] = {f"{K:.0e}": {r: pareto([(p["C"] + K * p["gen"], p["fid"]) for p in pts if p["route"] == r])
                                          for r in F.ROUTES} for K in (1e6, 1e7, 1e8, 1e9, 1e10)}
    return res


def figure(pts):
    bud_blue = {"1e19": "#9cc3f0", "3e19": "#5d9be3", "1e20": "#2a78d6", "3e20": "#1b54a0", "1e21": "#0d2f5e"}
    fig, axs = plt.subplots(1, 1 + len(K_PANELS), figsize=(F.FULL, 2.0), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1.35] + [1] * len(K_PANELS)))
    ax = axs[0]
    for b in BUDGETS:
        ms = sorted((p for p in pts if p["route"] == "R1" and p["budget"] == b), key=lambda p: p["gen"])
        ax.plot([p["gen"] / 1e12 for p in ms], [p["fid"] for p in ms], "-o", color=bud_blue[b], ms=2.6, lw=1.0, mec=F.SURF, mew=0.4)
        if b == "1e21":
            for p in ms:
                if p["steps"] in (1, 4, 16, 50):
                    ax.annotate(f"{p['steps']} step" + "s" * (p["steps"] > 1), (p["gen"] / 1e12, p["fid"]), xytext=(4, 1),
                                textcoords="offset points", fontsize=6.0, color=F.INK2)
    for route, mk in (("CONV", "D"), ("NAT", "o")):
        ps = sorted((p for p in pts if p["route"] == route), key=lambda p: p["C"])
        ax.scatter([p["gen"] / 1e12 for p in ps], [p["fid"] for p in ps], marker=mk, s=11, color=F.ROUTE_C[route],
                   edgecolor=F.SURF, linewidth=0.5, zorder=4)
    xl = next(p for p in pts if p["route"] == "CONV" and p["budget"] == "1e21")
    m = min((p for p in pts if p["route"] == "R1" and p["fid"] <= xl["fid"]), key=lambda p: p["gen"])
    ax.annotate("", (m["gen"] / 1e12, 1.19), (xl["gen"] / 1e12, 1.19),
                arrowprops=dict(arrowstyle="<->", color=F.ROUTE_C["CONV"], lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(xl["gen"] / 1e12 * 1.15, 1.22, f"{m['gen'] / xl['gen']:.0f}$\\times$ per image at FID$\\,\\leq\\,${xl['fid']:.2f}",
            fontsize=6.0, color=F.ROUTE_C["CONV"], ha="left", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
    ax.axvline(VAE_DEC / 1e12, color=F.INK2, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.annotate("VAE decoder", (VAE_DEC / 1e12, 30), xytext=(-2, 0), textcoords="offset points", fontsize=6.0,
                color=F.INK2, rotation=90, va="center", ha="right")
    F.logaxis(ax, "y", [1.5, 2, 3, 5, 10, 30, 100, 300], ["1.5", "2", "3", "5", "10", "30", "100", "300"])
    ax.set_ylim(1.08, 330)
    F.logaxis(ax, "x", [0.5, 1, 3, 10, 30, 100], ["0.5", "1", "3", "10", "30", "100"])
    ax.set_xlim(0.45, 140)
    ax.set_xlabel("generation compute per image (TFLOPs)"); ax.set_ylabel("FID-50k")
    ax.set_title("(a) quality against the cost of an image", color=F.INK)
    leg = ax.legend(handles=[Line2D([], [], color=bud_blue[b], marker="o", ms=2.6, lw=1.0, label=F.BTEX.get(b, "$10^{21}$"))
                             for b in BUDGETS], title="multi-step, budget", title_fontsize=6.0, loc="upper right", fontsize=6.0,
                    handlelength=1.6)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], color=F.ROUTE_C["CONV"], marker="D", ls="none", ms=3.2, mec=F.SURF, label="convert, 1 step"),
                       Line2D([], [], color=F.ROUTE_C["NAT"], marker="o", ls="none", ms=3.2, mec=F.SURF, label="native, 1 step")],
              loc="center right", bbox_to_anchor=(1.0, 0.30), fontsize=6.0, handletextpad=0.3)
    for ax, K in zip(axs[1:], K_PANELS):
        for route in F.ROUTES:
            env = pareto([(p["C"] + K * p["gen"], p["fid"]) for p in pts if p["route"] == route])
            xs = [c for c, _ in env]; ys = [q for _, q in env]
            ax.step(xs, ys, where="post", color=F.ROUTE_C[route], lw=1.2)
            ax.plot(xs, ys, {"CONV": "D"}.get(route, "o"), color=F.ROUTE_C[route], ms=2.6, mec=F.SURF, mew=0.4, ls="none")
        F.logaxis(ax, "y", [1.5, 2, 3, 5, 10, 20], ["1.5", "2", "3", "5", "10", "20"])
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_ylim(1.2, 22)
        allx = [p["C"] + K * p["gen"] for p in pts]
        ax.set_xlim(min(allx) / 1.25, max(allx) * 1.25)
        ax.set_xlabel("training + generation compute (FLOPs)")
        e = int(round(math.log10(K)))
        ax.set_title(f"({'bc'[K_PANELS.index(K)]}) total cost, $10^{{{e}}}$ images", color=F.INK)
    axs[1].set_ylabel("best FID-50k")
    axs[1].legend(handles=[Line2D([], [], color=F.ROUTE_C["R1"], lw=1.2, label="multi-step (1–50 steps)"),
                           Line2D([], [], color=F.ROUTE_C["CONV"], marker="D", lw=1.2, ms=3, label="pretrain → convert, 1 step"),
                           Line2D([], [], color=F.ROUTE_C["NAT"], marker="o", lw=1.2, ms=3, label="native, 1 step")],
                  loc="upper right", fontsize=6.0, handlelength=1.6)
    F.save(fig, "fig_gencost")


def table(pts):
    """FID-50k (network evaluations) of every multi-step frontier model against the number of sampling steps."""
    steps = sorted({p["steps"] for p in pts if p["route"] == "R1"})
    L = ["\\begin{tabular}{l" + "c" * len(steps) + "c}", "\\toprule",
         "Model & " + " & ".join(f"{k} step" + "s" * (k > 1) for k in steps) + " & convert, 1 step \\\\", "\\midrule"]
    for b in BUDGETS:
        ms = {p["steps"]: p for p in pts if p["route"] == "R1" and p["budget"] == b}
        conv = next(p for p in pts if p["route"] == "CONV" and p["budget"] == b)
        L.append(f"{ms[1]['size']} @ {F.BTEX.get(b, '$10^{21}$')} & "
                 + " & ".join(f"{ms[k]['fid']:.2f} ({ms[k]['nfe']})" for k in steps) + f" & {conv['fid']:.2f} (1) \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/ta_steps.tex", "w").write("\n".join(L) + "\n")


def main():
    cells = F.load()
    pts = collect(cells)
    res = summarize(pts)
    res["points"] = pts
    json.dump(res, open(f"{ROOT}/analysis/out/generation_cost.json", "w"), indent=1)
    p = res["per_image_1e21"]
    print(f"1e21 per image: multi-step 50 steps {p['multistep_50']['tflops']:.1f} TFLOPs (FID {p['multistep_50']['fid']:.2f}), "
          f"converted XL {p['convert']['tflops']:.2f} (FID {p['convert']['fid']:.2f}); ratio {p['ratio_with_vae']:.0f}x with the VAE, "
          f"{p['ratio_network_only']:.0f}x network only")
    for b, m in res["cheapest_multistep_matching_convert"].items():
        print(b, m)
    print("images above which convert is cheapest:", {b: f"{v['images']:.2e}" for b, v in res["images_above_which_convert_is_cheapest"].items()})
    print("images until generation = training compute (1e21):", {k: f"{v:.2e}" for k, v in res["images_until_generation_equals_training_1e21"].items()})
    figure(pts)
    table(pts)


if __name__ == "__main__":
    main()
