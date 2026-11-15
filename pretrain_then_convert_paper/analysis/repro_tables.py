"""Reproducibility tables of the appendix: architecture, training hyper-parameters, evaluation protocol and the compute
used. Values are read from the code repository (osr.models.SIZES, osr.experiments, scripts/eval_run.GRID; set PTC_CODE
if it is not next to this repository), data/flops_table.json, data/throughput.json and data/frozen/gpu_hours.json
(SLURM accounting of all jobs of the project).

  python analysis/repro_tables.py  ->  paper/tables/{ta_arch,ta_hparams,ta_eval,ta_compute}.tex
"""
import json
import os
import sys

from common import DATA, ROOT, TAB

CODE = os.environ.get("PTC_CODE", os.path.join(os.path.dirname(ROOT), "pretrain_then_convert"))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))
from osr import experiments as grid  # noqa: E402
from osr.models import SIZES  # noqa: E402
from eval_run import GRID  # noqa: E402

OUT = TAB
FT = json.load(open(f"{DATA}/flops_table.json"))
TP = json.load(open(f"{DATA}/throughput.json"))


def pc(x):
    return f"{100 * x:.0f}\\%"


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"10^{{{int(e)}}}" if m == "1" else f"{m}{{\\times}}10^{{{int(e)}}}"


def fmt_list(xs):
    return ", ".join(f"{x:g}" for x in xs)


def arch():
    L = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Size & Depth & Width & Heads & Parameters (inference) & GPUs per run & Micro-batch (FM / iMF) \\\\", "\\midrule"]
    for s, c in SIZES.items():
        n = FT[s]["N_inference"]
        L.append(f"{s} & {c['depth']} & {c['hidden_size']} & {c['num_heads']} & "
                 f"{f'{n / 1e9:.2f}B' if n >= 1e9 else f'{n / 1e6:.0f}M'} & "
                 f"{'8' if grid.GPUS[s] == 8 else f'{grid.GPUS[s]}--8'} & {grid.MICRO['fm'][s]} / {grid.MICRO['imf'][s]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_arch.tex", "w").write("\n".join(L) + "\n")


def hparams():
    d, sc = grid.DMD["hp"], grid.SC_LOSS
    rows = [
        ("\\emph{All routes}", ""),
        ("Global batch, optimizer", f"{grid.GLOBAL_BATCH}; AdamW, $\\beta=(0.9, 0.95)$, no weight decay, no gradient clipping"),
        ("Learning rate", f"${sci(grid.LR)}$ constant (WSD stable phase); linear decay to 0 over the last {pc(grid.COOLDOWN_FRAC)} of each budget"),
        ("Mainline warmup", f"linear, $\\min({grid.WARMUP_STEPS}$ steps, 10\\% of the steps to the first cooldown start$)$"),
        ("Precision", "bf16 autocast, fp32 master weights; \\texttt{torch.compile}"),
        ("EMA", "power-function EMA, $\\sigma_\\text{rel}\\in\\{0.05, 0.10\\}$; raw or EMA weights selected on 5k samples"),
        ("Data", "ImageNet-1k train (1.28M), ADM centre crop 256, SD-VAE (ft-MSE) latents of the image and its flip; "
                 "posterior sample per step, random flip, per-channel normalization"),
        ("\\emph{Multi-step FM}", ""),
        ("Time sampling, class dropout", "logit-normal $t$ ($\\mu=-0.4$, $\\sigma=1.0$); class dropout 0.1"),
        ("\\emph{iMF (native training and conversion)}", ""),
        ("$(t, r)$", "max/min of two logit-normal draws ($\\mu=-0.4$, $\\sigma=1.0$); $r=t$ for 50\\% of each batch"),
        ("Guidance in training", "$\\omega\\in[1, 8]$, power-law density with $\\beta_\\text{cfg}$ = 1 (S, B) or 2 (M--H); "
                                 "interval $t_\\text{min}\\sim U[0, 0.5]$, $t_\\text{max}\\sim U[0.5, 1]$"),
        ("Loss weighting", "adaptive: loss $/\\,\\mathrm{sg}(\\text{loss}+0.01)$, on the average- and instantaneous-velocity heads"),
        ("\\emph{Conversion}", ""),
        ("Initialization", "FM weights; new conditioning embedders start as constants at their FM values ($h=0$, $\\omega=1$, "
                           "interval $[0, 1]$), auxiliary head copied from the FM head; fresh optimizer and EMA"),
        ("Schedule", f"warmup $\\min(2000$ steps, 10\\%$)$, constant, linear decay over the last {pc(grid.COOLDOWN_FRAC)}"),
        ("\\emph{Shortcut models}", ""),
        ("Objective", f"{pc(1 - sc['k'])} FM targets, {pc(sc['k'])} self-consistency over $\\log_2 {sc['m']}=7$ levels; "
                      f"EMA target network (decay {sc['ema_decay']}); CFG {sc['cfg_scale']} at the smallest level; weight decay {grid.SC_WD}"),
        ("\\emph{DMD2}", ""),
        ("Optimization", f"batch {grid.DMD['batch']}; {d['ttur']} fake-score updates per generator update; learning rate "
                         f"${sci(d['lr_g'])}$ (both); AdamW $\\beta=(0.9, 0.999)$, weight decay {d['wd']}; warmup {d['warmup']} steps, "
                         f"linear decay over the last {pc(d['cooldown_frac'])}"),
        ("Distribution matching", f"real-score CFG $\\omega\\in[{d['omega_lo']:g}, {d['omega_hi']:g}]$ (also a student input); "
                                  f"$t\\in[{d['t_min']}, {d['t_max']}]$; GAN weights {d['gen_cls_w']:g} (generator), {d['cls_w']:g} (classifier)"),
    ]
    L = ["\\begin{tabular}{p{0.22\\linewidth}p{0.70\\linewidth}}", "\\toprule", "Setting & Value \\\\", "\\midrule"]
    for k, v in rows:
        L.append(f"\\multicolumn{{2}}{{l}}{{{k}}} \\\\" if not v else f"{k} & {v} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_hparams.tex", "w").write("\n".join(L) + "\n")


def evaluation():
    imf_w = sorted({p["omega"] for p in GRID["imf"]["B"]})
    imf_iv = sorted({(p["t_min"], p["t_max"]) for p in GRID["imf"]["B"]})
    fm_w = sorted({p["omega"] for p in GRID["fm"]["B"]})
    fm_iv = sorted({(p["t_lo"], p["t_hi"]) for p in GRID["fm"]["B"]})
    dmd_w = sorted({p["omega"] for p in GRID["dmd"]["B"]})
    iv = lambda xs: ", ".join(f"$[{a:g}, {b:g}]$" for a, b in xs)  # noqa: E731
    rows = [
        ("Stage A (weights)", "raw, EMA 0.05, EMA 0.10, each at "
         f"{len(GRID['imf']['A'])} (iMF), {len(GRID['fm']['A'])} (FM) or {len(GRID['dmd']['A'])} (DMD2) guidance points; 5k samples, seed 1"),
        ("Stage B (guidance), iMF", f"$\\omega\\in\\{{{fmt_list(imf_w)}\\}}$ $\\times$ intervals {iv(imf_iv)}; 10k samples, seed 1"),
        ("Stage B, multi-step FM", f"$\\omega\\in\\{{{fmt_list(fm_w)}\\}}$ $\\times$ intervals {iv(fm_iv)}; 50 Euler steps, two-pass CFG; 10k, seed 1"),
        ("Stage B, DMD2", f"$\\omega\\in\\{{{fmt_list(dmd_w)}\\}}$ (student input), full interval; 10k, seed 1"),
        ("Stage B, shortcut", "none (one step at $h=1$, no guidance)"),
        ("Extended search", "where the stage-B optimum lay on a grid edge (all multi-step models; one-step models on the largest "
                            "scale or narrowest interval): coordinate search over the interval, three scales per interval, on 5k "
                            "samples, best two re-scored on 10k; multi-step models then scored on 8 shared candidate points "
                            "(Appendix~\\ref{app:eval})"),
        ("Stage C (reported)", "50k samples, seed 0, 50 per class, at the selected weights and guidance; the same noise and "
                               "labels for every model (common random numbers)"),
        ("FID reference", "ImageNet-256 training-set statistics of the JiT/iMF evaluation (\\texttt{jit\\_in256\\_stats.npz})"),
        ("ADM-FID, precision/recall", "ADM reference batch (\\texttt{VIRTUAL\\_imagenet256\\_labeled.npz}); P/R with $k=3$ on Inception pool features"),
        ("FD-DINOv2", "ViT-L/14, 224 px bicubic, CLS token; reference statistics of 100k ImageNet training images (dgm-eval protocol)"),
    ]
    L = ["\\begin{tabular}{p{0.22\\linewidth}p{0.70\\linewidth}}", "\\toprule", "Step & Setting \\\\", "\\midrule"]
    L += [f"{k} & {v} \\\\" for k, v in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_eval.tex", "w").write("\n".join(L) + "\n")


def compute():
    h = json.load(open(f"{DATA}/frozen/gpu_hours.json"))
    rows = [("Hardware", "NVIDIA H100 80GB (the last runs on H200 141GB), 8 per node; every run on one node (1--8 GPUs, DDP)"),
            ("Software", "PyTorch 2.14.1 (CUDA 13), \\texttt{torch.compile}, flash attention for training"),
            ("Throughput (algorithmic)", f"FM {min(TP['fm'].values()) / 1e12:.0f}--{max(TP['fm'].values()) / 1e12:.0f}, "
                                         f"iMF {min(TP['imf'].values()) / 1e12:.0f}--{max(TP['imf'].values()) / 1e12:.0f} TFLOP/s per GPU"),
            ("GPU-hours used", f"{h['train'] / 1e3:.1f}k training, {h['eval'] / 1e3:.1f}k evaluation, {h['other']:.0f} for tests "
                               "(all jobs, including failed, preempted and repeated ones)")]
    L = ["\\begin{tabular}{p{0.22\\linewidth}p{0.70\\linewidth}}", "\\toprule", "Item & Value \\\\", "\\midrule"]
    L += [f"{k} & {v} \\\\" for k, v in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{OUT}/ta_compute.tex", "w").write("\n".join(L) + "\n")
    return h


if __name__ == "__main__":
    arch()
    hparams()
    evaluation()
    print(compute())
