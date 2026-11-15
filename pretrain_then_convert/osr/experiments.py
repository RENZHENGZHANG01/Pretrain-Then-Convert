"""The training runs of the paper.

Every run is a dict with
  run_id      see "Run ids" below
  kind        mainline | cooldown | convert | extend | distill
  route       objective of the run: fm (flow matching) | imf (improved MeanFlow) | sc (shortcut) | dmd (DMD2)
  size        XS, S, B, M, L, XL, H (osr.models.CONFIGS)
  seed, lr    training seed and peak learning rate (None for DMD2, whose rates are in `hp`)
  flops_end   training compute C of the model in FLOPs (cumulative, from the start of its mainline)
  parent      (run_id, milestone tag) the run starts from, or None; tags FINAL / FINAL_SEL denote the parent's
              final state / its compacted selected weights
  milestones  mainlines and extensions: checkpoints at cumulative FLOPs, kind "full" (weights, EMAs, optimizer: a
              cooldown or a further extension starts there) or "weights" (raw weights: a conversion starts there)
  group       the part of the paper the run belongs to (below)
and, depending on the kind, budget, f (pretraining fraction), src_flops (extensions), warmup_ref, teacher and
student_init (distillation), hp (DMD2 overrides), init_mode ("naive" initialization ablation), gpus (the number of
GPUs the run used).

Training protocol (warmup-stable-decay): every route and size has one constant-learning-rate mainline; the model of
budget C is a cooldown branch from the mainline at 0.85 C whose learning rate decays to zero at C, and a conversion
with split f starts from the raw weights of the FM mainline at f C and trains the one-step objective until C (warmup,
constant rate, decay over the last 15%).

Run ids
  r1_<size>_s<seed>, r2_<size>_s<seed>     FM / native iMF mainline
  r1_<size>_<C>_cd_s<seed>                 multi-step FM model of budget C (cooldown)
  r2_<size>_<C>_cd_s<seed>                 native one-step model of budget C (cooldown)
  r3_<size>_<C>_f<f>_s<seed>               pretrain -> convert model of budget C with split f
  r1_<size>_ext<C>_s0, ..._rerun*_s0       FM stable phase continued / re-run to provide extra branch points
  ..._lr<rate>..., ..._flr<rate>...        learning-rate variants (flr: FM pretraining and conversion at the rate)
  ..._s0st1                                pretraining seed 0, last stage re-run with seed 1
  r4_<student>_<teacher><C_T>_d<C_D>_s0    DMD2 student; r4s_*: hyper-parameter scan
  t2_*, t3_*, g3fm_*                       native / converted shortcut models, FM mainlines of the shortcut seeds 1, 2
  v1_*, v2_*                               the 1e21-FLOP runs (v1: native XL; v2: conversion and multi-step H)

Groups
  grid           IsoFLOP grid: FM and native mainlines with cooldowns at 1e19 ... 3e20, conversions with
                 f in {0.5, 0.8, 0.95, 0.99}; a second seed at 1e19 and 3e19
  split          extra splits f ~ 0.25 and f ~ 0.65 at the compute-optimal sizes
  edges          sizes beyond the grid edges (XS at 1e19 and 3e19, native S at 3e19)
  stage_seeds    the last stage (cooldown or conversion) re-run with a new seed at 3e20
  lr             pretraining learning rates (FM and native iMF)
  conversion_lr  learning rate of the conversion phase
  ablations      f ~ 0.1, and the naive initialization of the conversion
  runs_1e21      native XL, converted XL and H (f = 0.6), multi-step H at 1e21 FLOPs
  distillation   DMD2 students of FM teachers, same-size students, the hyper-parameter scan
  sunk           conversion of an existing (cooled-down) FM model with new compute
  shortcut       the f-curves with shortcut models (three seeds)

  python -m osr.experiments [group ...]      (summary of the runs and their FLOPs)
"""
import json
import os

from .paths import LATENTS, RUNS, FLOPS_TABLE

BUDGETS = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
SIZES = ["S", "B", "M", "L", "XL", "H"]
COOLDOWN_FRAC = 0.15

# IsoFLOP windows: sizes per budget and route (multi-step FM, native iMF, conversion)
WIN_R1 = {"1e19": ["S", "B", "M", "L"], "3e19": ["B", "M", "L", "XL"],
          "1e20": ["B", "M", "L", "XL", "H"], "3e20": ["M", "L", "XL", "H"]}
WIN_R2 = {"1e19": ["S", "B", "M", "L"], "3e19": ["B", "M", "L", "XL"],
          "1e20": ["B", "M", "L", "XL", "H"], "3e20": ["B", "M", "L", "XL", "H"]}
WIN_R12 = {"fm": WIN_R1, "imf": WIN_R2}
WIN_R3 = {"1e19": ["S", "B", "M", "L"], "3e19": ["B", "M", "L", "XL"],
          "1e20": ["B", "M", "L", "XL"], "3e20": ["L", "XL", "H"]}
F_GRID = [0.5, 0.8, 0.95, 0.99]
SEED1_BUDGETS = ["1e19", "3e19"]  # full second seed at the small budgets

LR = 1e-4  # the iMF learning rate, used for every run except the learning-rate groups
LR_SWEEP = [5e-5, 2e-4]
GLOBAL_BATCH = 256
CFG_BETA = {"XS": 1.0, "S": 1.0, "B": 1.0, "M": 2.0, "L": 2.0, "XL": 2.0, "H": 2.0}  # iMF guidance-scale sampling
WARMUP_STEPS = 5000  # mainlines; at most 10% of the steps to the first cooldown start

GPUS = {"XS": 1, "S": 1, "B": 1, "M": 2, "L": 4, "XL": 8, "H": 8}  # minimum per size (memory)
MICRO = {"fm": {"XS": 256, "S": 256, "B": 128, "M": 64, "L": 64, "XL": 32, "H": 32},
         "imf": {"XS": 128, "S": 128, "B": 64, "M": 64, "L": 32, "XL": 32, "H": 16}}
MICRO["sc"] = dict(MICRO["fm"])

_ft = json.load(open(FLOPS_TABLE))
FLOPS_PER_SAMPLE = {"fm": {k: v["fm"] for k, v in _ft.items()}, "imf": {k: v["imf"] for k, v in _ft.items()}}
FLOPS_PER_SAMPLE["sc"] = {k: v["fm"] * 4 / 3 for k, v in _ft.items()}  # warmup length only; C is measured in training

# Shortcut models (Frans et al. 2025): lr 1e-4, weight decay 0.1, EMA 0.999 for the bootstrap targets, 1/4 bootstrap
# samples, m = 128, guidance 1.5 at the smallest step (class dropout 0.1); fixed level counts per batch.
SC_LOSS = dict(t_mean=-0.4, t_std=1.0, class_dropout=0.1, k=0.25, m=128, cfg_scale=1.5, ema_decay=0.999)
SC_WD = 0.1

# DMD2: the ImageNet defaults of DMD2 (TTUR 5, GAN weights 1e-2 / 3e-3, AdamW(0.9, 0.999), wd 0.01, t in [0.02, 0.98])
# with learning rate 3e-5, the best of a one-factor scan (group "distillation", runs r4s_*).
DMD = dict(batch=128, hp=dict(ttur=5, lr_g=3e-5, lr_f=3e-5, betas=[0.9, 0.999], wd=0.01, omega_lo=1.0, omega_hi=4.0,
                              t_min=0.02, t_max=0.98, gen_cls_w=3e-3, cls_w=1e-2, warmup=500, cooldown_frac=0.15))

GROUPS = ("grid", "split", "edges", "stage_seeds", "lr", "conversion_lr", "ablations", "runs_1e21", "distillation",
          "sunk", "shortcut")


def ftag(f):
    return f"f{f:.2f}"


def mainline_id(route, size, seed, lr=LR):
    r = "r1" if route == "fm" else "r2"
    return f"{r}_{size}_s{seed}" if lr == LR else f"{r}_{size}_lr{lr:g}_s{seed}"


def init_path(parent):
    """Checkpoint a run starts from: a milestone of the parent run, or (tags FINAL / FINAL_SEL) the parent's final state /
    its compacted selected weights."""
    pid, tag = parent
    if tag == "FINAL":
        return f"{RUNS}/{pid}/final.pt"
    if tag == "FINAL_SEL":
        return f"{RUNS}/{pid}/final_sel.pt"
    return f"{RUNS}/{pid}/milestones/{tag}.pt"


def _mainline_extent(route, size, seed):
    """Largest milestone this size's grid mainline must reach."""
    fl = []
    for b, C in BUDGETS.items():
        if seed == 1 and b not in SEED1_BUDGETS:
            continue
        if size in WIN_R12[route][b]:
            fl.append((1 - COOLDOWN_FRAC) * C)
        if route == "fm" and size in WIN_R3[b]:
            fl += [f * C for f in F_GRID]
    return max(fl) if fl else 0.0


def _milestones(route, size, seed):
    ms = []
    for b, C in BUDGETS.items():
        if seed == 1 and b not in SEED1_BUDGETS:
            continue
        if size in WIN_R12[route][b]:
            ms.append(dict(flops=(1 - COOLDOWN_FRAC) * C, kind="full", tag=f"cd_{b}"))
        if route == "fm" and size in WIN_R3[b]:
            ms += [dict(flops=f * C, kind="weights", tag=f"{ftag(f)}_{b}") for f in F_GRID]
    return sorted(ms, key=lambda m: m["flops"])


# ------------------------------------------------------------------------------------------------------------- groups
def _grid(add):
    for route in ["fm", "imf"]:
        r = "r1" if route == "fm" else "r2"
        for size in SIZES:
            for seed in (0, 1):
                ext = _mainline_extent(route, size, seed)
                if ext == 0:
                    continue
                mid = mainline_id(route, size, seed)
                add(run_id=mid, kind="mainline", route=route, size=size, seed=seed, lr=LR, flops_end=ext,
                    milestones=_milestones(route, size, seed), parent=None)
                for b, C in BUDGETS.items():
                    if size in WIN_R12[route][b] and (seed == 0 or b in SEED1_BUDGETS):
                        add(run_id=f"{r}_{size}_{b}_cd_s{seed}", kind="cooldown", route=route, size=size, seed=seed,
                            lr=LR, flops_end=C, parent=(mid, f"cd_{b}"), budget=b)
    for size in SIZES:
        for seed in (0, 1):
            for b, C in BUDGETS.items():
                if size not in WIN_R3[b] or (seed == 1 and b not in SEED1_BUDGETS):
                    continue
                for f in F_GRID:
                    add(run_id=f"r3_{size}_{b}_{ftag(f)}_s{seed}", kind="convert", route="imf", size=size, seed=seed,
                        lr=LR, flops_end=C, parent=(mainline_id("fm", size, seed), f"{ftag(f)}_{b}"), budget=b, f=f)


def _split(add):
    """f ~ 0.25 and f ~ 0.65 at the compute-optimal sizes: f = 0.65 by continuing the FM stable phase from the nearest
    full milestone (or, at 1e19, re-running the FM mainline), f ~ 0.25 from existing branch points."""
    def conv(size, b, src, flops_src, tag):
        f = flops_src / BUDGETS[b]
        add(run_id=f"r3_{size}_{b}_{ftag(f)}_s0", kind="convert", route="imf", size=size, seed=0, lr=LR,
            flops_end=BUDGETS[b], parent=(src, tag), budget=b, f=f)
    for size in ["S", "B"]:  # 1e19: re-run the FM mainline (same config and seed) to 6.5e18
        mid = f"r1_{size}_rerun_s0"
        add(run_id=mid, kind="mainline", route="fm", size=size, seed=0, lr=LR, flops_end=6.5e18,
            milestones=[dict(flops=2.5e18, kind="weights", tag="f0.25_1e19"), dict(flops=6.5e18, kind="weights", tag="f0.65_1e19")],
            parent=None, warmup_ref=0.85e19)
        conv(size, "1e19", mid, 2.5e18, "f0.25_1e19")
        conv(size, "1e19", mid, 6.5e18, "f0.65_1e19")
    for size, b, src_tag, src_flops in [("B", "3e19", "cd_1e19", 0.85e19), ("M", "3e19", "cd_1e19", 0.85e19),
                                        ("M", "1e20", "cd_3e19", 0.85 * 3e19), ("L", "3e20", "cd_1e20", 0.85e20),
                                        ("XL", "3e20", "cd_1e20", 0.85e20)]:
        target = 0.65 * BUDGETS[b]
        eid = f"r1_{size}_ext{b}_s0"
        add(run_id=eid, kind="extend", route="fm", size=size, seed=0, lr=LR, flops_end=target,
            milestones=[dict(flops=target, kind="weights", tag=f"f0.65_{b}")], parent=(mainline_id("fm", size, 0), src_tag),
            src_flops=src_flops)
        conv(size, b, eid, target, f"f0.65_{b}")
    for size, b, tag, fl in [("B", "3e19", "f0.80_1e19", 0.8e19), ("M", "3e19", "f0.80_1e19", 0.8e19),
                             ("M", "1e20", "f0.80_3e19", 0.8 * 3e19), ("L", "1e20", "f0.80_3e19", 0.8 * 3e19),
                             ("L", "3e20", "f0.80_1e20", 0.8e20), ("XL", "3e20", "f0.80_1e20", 0.8e20)]:
        conv(size, b, mainline_id("fm", size, 0), fl, tag)


def _edges(add):
    """Native S at 3e19 (native mainline extended), and XS at 1e19 (all three routes) and 3e19 (native)."""
    add(run_id="r2_S_ext3e19_s0", kind="extend", route="imf", size="S", seed=0, lr=LR, flops_end=0.85 * 3e19,
        milestones=[dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")], parent=("r2_S_s0", "FINAL"), src_flops=0.85e19, gpus=4)
    add(run_id="r2_S_3e19_cd_s0", kind="cooldown", route="imf", size="S", seed=0, lr=LR, flops_end=3e19,
        parent=("r2_S_ext3e19_s0", "cd_3e19"), budget="3e19", gpus=2)
    add(run_id="r2_XS_s0", kind="mainline", route="imf", size="XS", seed=0, lr=LR, flops_end=0.85e19,
        milestones=[dict(flops=0.85e19, kind="full", tag="cd_1e19")], parent=None, gpus=2)
    add(run_id="r2_XS_1e19_cd_s0", kind="cooldown", route="imf", size="XS", seed=0, lr=LR, flops_end=1e19,
        parent=("r2_XS_s0", "cd_1e19"), budget="1e19", gpus=1)
    fx = (0.25, 0.5, 0.65, 0.8)
    ms = [dict(flops=f * 1e19, kind="weights", tag=f"{ftag(f)}_1e19") for f in fx] + [dict(flops=0.85e19, kind="full", tag="cd_1e19")]
    add(run_id="r1_XS_s0", kind="mainline", route="fm", size="XS", seed=0, lr=LR, flops_end=0.85e19,
        milestones=sorted(ms, key=lambda m: m["flops"]), parent=None, gpus=2)
    add(run_id="r1_XS_1e19_cd_s0", kind="cooldown", route="fm", size="XS", seed=0, lr=LR, flops_end=1e19,
        parent=("r1_XS_s0", "cd_1e19"), budget="1e19", gpus=1)
    for f in fx:
        add(run_id=f"r3_XS_1e19_{ftag(f)}_s0", kind="convert", route="imf", size="XS", seed=0, lr=LR, flops_end=1e19,
            parent=("r1_XS_s0", f"{ftag(f)}_1e19"), budget="1e19", f=f, gpus=1)
    add(run_id="r2_XS_ext3e19_s0", kind="extend", route="imf", size="XS", seed=0, lr=LR, flops_end=0.85 * 3e19,
        milestones=[dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")], parent=("r2_XS_s0", "FINAL"), src_flops=0.85e19, gpus=4)
    add(run_id="r2_XS_3e19_cd_s0", kind="cooldown", route="imf", size="XS", seed=0, lr=LR, flops_end=3e19,
        parent=("r2_XS_ext3e19_s0", "cd_3e19"), budget="3e19", gpus=2)
    add(run_id="r2_S_ext3e19_s1", kind="extend", route="imf", size="S", seed=1, lr=LR, flops_end=0.85 * 3e19,
        milestones=[dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")], parent=("r2_S_s1", "FINAL"), src_flops=0.85e19, gpus=4)
    add(run_id="r2_S_3e19_cd_s1", kind="cooldown", route="imf", size="S", seed=1, lr=LR, flops_end=3e19,
        parent=("r2_S_ext3e19_s1", "cd_3e19"), budget="3e19", gpus=2)


def _stage_seeds(add):
    """Same FM / iMF milestone, new seed (data order and noise) for the last stage only."""
    for size in ("L", "XL"):
        add(run_id=f"r3_{size}_3e20_f0.50_s0st1", kind="convert", route="imf", size=size, seed=1, lr=LR, flops_end=3e20,
            parent=(mainline_id("fm", size, 0), "f0.50_3e20"), budget="3e20", f=0.5)
        add(run_id=f"r1_{size}_3e20_cd_s0st1", kind="cooldown", route="fm", size=size, seed=1, lr=LR, flops_end=3e20,
            parent=(mainline_id("fm", size, 0), "cd_3e20"), budget="3e20")
    for size in ("M", "L"):
        add(run_id=f"r2_{size}_3e20_cd_s0st1", kind="cooldown", route="imf", size=size, seed=1, lr=LR, flops_end=3e20,
            parent=(mainline_id("imf", size, 0), "cd_3e20"), budget="3e20")


def _lr(add):
    """Pretraining learning rates: B and L at 1e19 (both objectives), B at 3e19, native M at 1e20 and 3e20, native S and
    FM pretraining + conversion at the converted optima (S at 1e19, M at 3e19) at 1e19 and 3e19."""
    for route in ["fm", "imf"]:
        r = "r1" if route == "fm" else "r2"
        for size in ["B", "L"]:
            for lr in LR_SWEEP + [4e-4]:
                mid = mainline_id(route, size, 0, lr)
                add(run_id=mid, kind="mainline", route=route, size=size, seed=0, lr=lr, flops_end=0.85e19,
                    milestones=[dict(flops=0.85e19, kind="full", tag="cd_1e19")], parent=None)
                add(run_id=f"{r}_{size}_lr{lr:g}_1e19_cd_s0", kind="cooldown", route=route, size=size, seed=0, lr=lr,
                    flops_end=1e19, parent=(mid, "cd_1e19"), budget="1e19")
        lr = 2e-4
        mid = f"{r}_B_lr{lr:g}_3e19_s0"
        add(run_id=mid, kind="mainline", route=route, size="B", seed=0, lr=lr, flops_end=0.85 * 3e19,
            milestones=[dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")], parent=None)
        add(run_id=f"{r}_B_lr{lr:g}_3e19_cd_s0", kind="cooldown", route=route, size="B", seed=0, lr=lr,
            flops_end=3e19, parent=(mid, "cd_3e19"), budget="3e19")
    for lr in (2e-4, 4e-4):
        mid = f"r2_M_lr{lr:g}_big_s0"
        add(run_id=mid, kind="mainline", route="imf", size="M", seed=0, lr=lr, flops_end=0.85 * 3e20,
            milestones=[dict(flops=0.85e20, kind="full", tag="cd_1e20"), dict(flops=0.85 * 3e20, kind="full", tag="cd_3e20")],
            parent=None, warmup_ref=0.85e19, gpus=8)
        for b, C in (("1e20", 1e20), ("3e20", 3e20)):
            add(run_id=f"r2_M_lr{lr:g}_{b}_cd_s0", kind="cooldown", route="imf", size="M", seed=0, lr=lr, flops_end=C,
                parent=(mid, f"cd_{b}"), budget=b, gpus=8 if b == "3e20" else 4)
    for lr in (2e-4, 4e-4, 8e-4):
        mid = f"r2_S_lr{lr:g}_small_s0"
        add(run_id=mid, kind="mainline", route="imf", size="S", seed=0, lr=lr, flops_end=0.85 * 3e19,
            milestones=[dict(flops=0.85e19, kind="full", tag="cd_1e19"), dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")],
            parent=None, warmup_ref=0.85e19, gpus=4)
        for b, C in (("1e19", 1e19), ("3e19", 3e19)):
            add(run_id=f"r2_S_lr{lr:g}_{b}_cd_s0", kind="cooldown", route="imf", size="S", seed=0, lr=lr, flops_end=C,
                parent=(mid, f"cd_{b}"), budget=b, gpus=2)
        for size, b, C, f, g in (("S", "1e19", 1e19, 0.5, 2), ("M", "3e19", 3e19, 0.65, 4)):
            fm = f"r1_{size}_lr{lr:g}_small_s0"
            add(run_id=fm, kind="mainline", route="fm", size=size, seed=0, lr=lr, flops_end=f * C,
                milestones=[dict(flops=f * C, kind="weights", tag=f"{ftag(f)}_{b}")], parent=None, warmup_ref=0.85e19, gpus=g)
            add(run_id=f"r3_{size}_{b}_{ftag(f)}_flr{lr:g}_s0", kind="convert", route="imf", size=size, seed=0, lr=lr,
                flops_end=C, parent=(fm, f"{ftag(f)}_{b}"), budget=b, f=f, gpus=g)


def _conversion_lr(add):
    """Learning rate of the conversion phase (FM pretraining at 1e-4): B at 1e19, M at 3e19 along its f-curve, XL at
    3e20 around the optimum."""
    for lr in LR_SWEEP:
        add(run_id=f"r3_B_1e19_{ftag(0.8)}_lr{lr:g}_s0", kind="convert", route="imf", size="B", seed=0, lr=lr,
            flops_end=1e19, parent=(mainline_id("fm", "B", 0), f"{ftag(0.8)}_1e19"), budget="1e19", f=0.8)
    for f, src in ((0.5, "r1_M_s0"), (0.65, "r1_M_ext3e19_s0"), (0.8, "r1_M_s0"), (0.95, "r1_M_s0"), (0.99, "r1_M_s0")):
        for lr in (5e-5, 2e-4, 4e-4):
            add(run_id=f"r3_M_3e19_{ftag(f)}_lr{lr:g}_s0", kind="convert", route="imf", size="M", seed=0, lr=lr,
                flops_end=3e19, parent=(src, f"{ftag(f)}_3e19"), budget="3e19", f=f,
                gpus=4 if f <= 0.65 else (2 if f == 0.8 else None))
    for f in (0.95, 0.99):
        for lr in (2e-4, 4e-4):
            add(run_id=f"r3_B_1e19_{ftag(f)}_lr{lr:g}_s0", kind="convert", route="imf", size="B", seed=0, lr=lr,
                flops_end=1e19, parent=("r1_B_s0", f"{ftag(f)}_1e19"), budget="1e19", f=f)
    for f, src in ((0.5, "r1_XL_s0"), (0.65, "r1_XL_ext3e20_s0"), (0.8, "r1_XL_s0")):
        add(run_id=f"r3_XL_3e20_{ftag(f)}_lr5e-05_s0", kind="convert", route="imf", size="XL", seed=0, lr=5e-5,
            flops_end=3e20, parent=(src, f"{ftag(f)}_3e20"), budget="3e20", f=f, gpus=8)


def _fm_reruns(add):
    """FM mainlines of B and M re-run to 2.5e18 FLOPs (an f = 0.25 branch point at 1e19), used by the shortcut f-curves
    and the f ~ 0.1 ablation."""
    for size, mid in (("B", "r1_B_rerun2_s0"), ("M", "r1_M_rerun_s0")):
        add(run_id=mid, kind="mainline", route="fm", size=size, seed=0, lr=LR, flops_end=2.5e18,
            milestones=[dict(flops=2.5e18, kind="weights", tag="f0.25_1e19")], parent=None, warmup_ref=0.85e19, gpus=2)


def _ablations(add):
    """(i) f ~ 0.1 at the compute-optimal conversion size M (3e19 from the 2.5e18 FM state, f = 0.083; 1e20 from the
    9.5e18 state, f = 0.095); (ii) the naive initialization (new conditioning embedders and the auxiliary head randomly
    initialized) against the function-preserving one, M at 3e19 and 1e20, f = 0.65."""
    _fm_reruns(add)
    for b, C, src, tag, fl, ft_ in (("3e19", 3e19, "r1_M_rerun_s0", "f0.25_1e19", 2.5e18, "f0.083"),
                                    ("1e20", 1e20, "r1_M_s0", "f0.95_1e19", 9.5e18, "f0.095")):
        add(run_id=f"r3_M_{b}_{ft_}_s0", kind="convert", route="imf", size="M", seed=0, lr=LR, flops_end=C,
            parent=(src, tag), budget=b, f=fl / C, gpus=4 if b == "3e19" else 8)
    for b, C, src in (("3e19", 3e19, "r1_M_ext3e19_s0"), ("1e20", 1e20, "r1_M_ext1e20_s0")):
        add(run_id=f"r3_M_{b}_f0.65_naive_s0", kind="convert", route="imf", size="M", seed=0, lr=LR, flops_end=C,
            parent=(src, f"f0.65_{b}"), budget=b, f=0.65, init_mode="naive", gpus=4 if b == "3e19" else 8)


def _runs_1e21(add):
    """Native XL; conversion at XL (FM stable phase of the grid's XL mainline continued to 0.6e21) and at H, f = 0.6;
    multi-step H. Sizes from a fitted FID(N, C) surface of the grid."""
    ms = [dict(flops=f * 1e21, kind="weights", tag=f"f{f:.2f}_1e21") for f in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)]
    ms.append(dict(flops=0.85e21, kind="full", tag="cd_1e21"))
    add(run_id="v2_H_fm_s0", kind="mainline", route="fm", size="H", seed=0, lr=LR, flops_end=0.85e21, milestones=ms,
        parent=None, gpus=8)
    add(run_id="v1_XL_main_s0", kind="mainline", route="imf", size="XL", seed=0, lr=LR, flops_end=0.85e21,
        milestones=[dict(flops=0.85e21, kind="full", tag="cd_1e21")], parent=None, gpus=8)
    add(run_id="v1_XL_1e21_cd_s0", kind="cooldown", route="imf", size="XL", seed=0, lr=LR, flops_end=1e21,
        parent=("v1_XL_main_s0", "cd_1e21"), budget="1e21", gpus=8)
    add(run_id="v2_XL_ext_s0", kind="extend", route="fm", size="XL", seed=0, lr=LR, flops_end=0.6e21,
        milestones=[dict(flops=0.6e21, kind="weights", tag="f0.60_1e21")], parent=("r1_XL_s0", "cd_3e20"),
        src_flops=0.85 * 3e20, gpus=8)
    add(run_id="v2_XL_1e21_f0.60_s0", kind="convert", route="imf", size="XL", seed=0, lr=LR, flops_end=1e21,
        parent=("v2_XL_ext_s0", "f0.60_1e21"), budget="1e21", f=0.6, gpus=8)
    add(run_id="v2_H_1e21_f0.60_s0", kind="convert", route="imf", size="H", seed=0, lr=LR, flops_end=1e21,
        parent=("v2_H_fm_s0", "f0.60_1e21"), budget="1e21", f=0.6, gpus=8)
    add(run_id="v2_H_1e21_cd_s0", kind="cooldown", route="fm", size="H", seed=0, lr=LR, flops_end=1e21,
        parent=("v2_H_fm_s0", "cd_1e21"), budget="1e21", gpus=8)


def _distillation(add):
    """DMD2 students (B, M) of FM teachers (L, XL at 1e20 and 3e20) with 1e19 ... 1e20 FLOPs of distillation, students
    initialized from the student-size FM model at 8.5e18 FLOPs; same-size students initialized from their teacher; the
    one-factor hyper-parameter scan (B teacher, 2e18 FLOPs) and learning-rate checks."""
    teachers = [("XL", "1e20"), ("L", "1e20"), ("XL", "3e20"), ("L", "3e20")]
    for ts, tb in teachers:
        for ss in ("B", "M"):
            for db, dfl in (("1e19", 1e19), ("3e19", 3e19), ("1e20", 1e20)):
                add(run_id=f"r4_{ss}_{ts}{tb}_d{db}_s0", kind="distill", route="dmd", size=ss, seed=0, lr=None,
                    flops_end=dfl, parent=None, budget=db, teacher=(f"r1_{ts}_{tb}_cd_s0", ts),
                    student_init=(mainline_id("fm", ss, 0), "cd_1e19"))
    for ts, tb in (("L", "1e20"), ("XL", "3e20")):
        add(run_id=f"r4_{ts}_{ts}{tb}_d3e19_selfinit_s0", kind="distill", route="dmd", size=ts, seed=0, lr=None, flops_end=3e19,
            parent=None, budget="3e19", teacher=(f"r1_{ts}_{tb}_cd_s0", ts), student_init=("TEACHER", None),
            gpus=8 if ts == "XL" else None)
    for ts in ("XL", "L"):
        add(run_id=f"r4_B_{ts}1e20_d1e19_init3e19_s0", kind="distill", route="dmd", size="B", seed=0, lr=None, flops_end=1e19,
            parent=None, budget="1e19", teacher=(f"r1_{ts}_1e20_cd_s0", ts), student_init=(mainline_id("fm", "B", 0), "cd_3e19"))
    scan = {"base": {}, "lr3e-6": dict(lr_g=3e-6, lr_f=3e-6), "lr3e-5": dict(lr_g=3e-5, lr_f=3e-5),
            "ttur3": dict(ttur=3), "ttur10": dict(ttur=10), "gan0": dict(gen_cls_w=0.0, cls_w=0.0),
            "gan3x": dict(gen_cls_w=1e-2, cls_w=3e-2)}
    for tag, hp in scan.items():
        add(run_id=f"r4s_B_B1e20_{tag}_s0", kind="distill", route="dmd", size="B", seed=0, lr=None, flops_end=2e18,
            parent=None, budget="2e18", teacher=("r1_B_1e20_cd_s0", "B"), student_init=(mainline_id("fm", "B", 0), "cd_1e19"),
            hp=hp, gpus=2)
    for tch, tag, hp in (("B", "lr1e-4", dict(lr_g=1e-4, lr_f=1e-4)), ("B", "lr3e-4", dict(lr_g=3e-4, lr_f=3e-4)),
                         ("L", "lr3e-5", dict(lr_g=3e-5, lr_f=3e-5)), ("L", "lr1e-4", dict(lr_g=1e-4, lr_f=1e-4))):
        add(run_id=f"r4s_B_{tch}1e20_{tag}_s0", kind="distill", route="dmd", size="B", seed=0, lr=None, flops_end=2e18,
            parent=None, budget="2e18", teacher=(f"r1_{tch}_1e20_cd_s0", tch), student_init=(mainline_id("fm", "B", 0), "cd_1e19"),
            hp=hp, gpus=2)
    hp = dict(lr_g=1e-4, lr_f=1e-4)
    for ts, tb in (("L", "1e20"), ("XL", "3e20")):
        add(run_id=f"r4_{ts}_{ts}{tb}_d3e19_selfinit_lr1e-4_s0", kind="distill", route="dmd", size=ts, seed=0, lr=None,
            flops_end=3e19, parent=None, budget="3e19", teacher=(f"r1_{ts}_{tb}_cd_s0", ts), student_init=("TEACHER", None),
            hp=hp, gpus=8 if ts == "XL" else None)
    for ss, db, dfl in (("M", "1e20", 1e20), ("B", "1e19", 1e19)):
        add(run_id=f"r4_{ss}_L1e20_d{db}_lr1e-4_s0", kind="distill", route="dmd", size=ss, seed=0, lr=None, flops_end=dfl,
            parent=None, budget=db, teacher=("r1_L_1e20_cd_s0", "L"), student_init=(mainline_id("fm", ss, 0), "cd_1e19"),
            hp=hp, gpus=8 if db == "1e20" else None)


def _sunk(add):
    """The FM model already exists: convert the cooled-down teachers of the same-size DMD2 students (selected EMA weights)
    with the new compute those students used (3e19 FLOPs)."""
    for size, src, C0, tag in (("XL", "r1_XL_3e20_cd_s0", 3e20, "3e20"), ("L", "r1_L_1e20_cd_s0", 1e20, "1e20")):
        C = C0 + 3e19
        add(run_id=f"r3_{size}_sunk{tag}_d3e19_s0", kind="convert", route="imf", size=size, seed=0, lr=LR, flops_end=C,
            parent=(src, "FINAL_SEL"), budget="sunk", f=C0 / C, gpus=8)


def _shortcut(add):
    """The f-curve with shortcut models: B at 1e19, M at 1e19 and 3e19, three seeds. Seed 0 branches from the FM runs of
    the iMF conversions; seeds 1 and 2 from one fresh FM mainline per size and seed that carries every branch point;
    native shortcut models train from scratch (mainline + cooldowns)."""
    _fm_reruns(add)
    F1, F3 = (0.25, 0.5, 0.8, 0.95, 0.99), (0.5, 0.65, 0.8, 0.95, 0.99)
    for seed in (0, 1, 2):
        src = {}
        if seed == 0:
            for size, mid in (("B", "r1_B_s0"), ("M", "r1_M_s0")):
                for f in F1:
                    src[(size, "1e19", f)] = (mid, f"{ftag(f)}_1e19")
            src[("B", "1e19", 0.25)] = ("r1_B_rerun2_s0", "f0.25_1e19")
            src[("M", "1e19", 0.25)] = ("r1_M_rerun_s0", "f0.25_1e19")
            for f in F3:
                src[("M", "3e19", f)] = ("r1_M_s0", f"{ftag(f)}_3e19")
            src[("M", "3e19", 0.65)] = ("r1_M_ext3e19_s0", "f0.65_3e19")
        else:
            for size, sizes_b in (("B", ("1e19",)), ("M", ("1e19", "3e19"))):
                mid = f"g3fm_{size}_s{seed}"
                ms = [dict(flops=f * 1e19, kind="weights", tag=f"{ftag(f)}_1e19") for f in F1]
                if size == "M":
                    ms += [dict(flops=f * 3e19, kind="weights", tag=f"{ftag(f)}_3e19") for f in F3]
                add(run_id=mid, kind="mainline", route="fm", size=size, seed=seed, lr=LR,
                    flops_end=max(m["flops"] for m in ms), milestones=ms, parent=None, warmup_ref=0.85e19,
                    gpus=2 if size == "B" else 4)
                for b in sizes_b:
                    for f in (F1 if b == "1e19" else F3):
                        src[(size, b, f)] = (mid, f"{ftag(f)}_{b}")
        src[("M", "3e19", 0.8e19 / 3e19)] = src[("M", "1e19", 0.8)]  # f = 0.27: the 8e18 branch point
        add(run_id=f"t2_B_s{seed}", kind="mainline", route="sc", size="B", seed=seed, lr=LR, flops_end=0.85e19,
            milestones=[dict(flops=0.85e19, kind="full", tag="cd_1e19")], parent=None, gpus=2)
        add(run_id=f"t2_B_1e19_cd_s{seed}", kind="cooldown", route="sc", size="B", seed=seed, lr=LR, flops_end=1e19,
            parent=(f"t2_B_s{seed}", "cd_1e19"), budget="1e19", gpus=2)
        add(run_id=f"t2_M_s{seed}", kind="mainline", route="sc", size="M", seed=seed, lr=LR, flops_end=0.85 * 3e19,
            milestones=[dict(flops=0.85e19, kind="full", tag="cd_1e19"), dict(flops=0.85 * 3e19, kind="full", tag="cd_3e19")],
            parent=None, gpus=4)
        for b, C in (("1e19", 1e19), ("3e19", 3e19)):
            add(run_id=f"t2_M_{b}_cd_s{seed}", kind="cooldown", route="sc", size="M", seed=seed, lr=LR, flops_end=C,
                parent=(f"t2_M_s{seed}", f"cd_{b}"), budget=b, gpus=4)
        for (size, b, f), (mid, tag) in sorted(src.items()):
            C = 1e19 if b == "1e19" else 3e19
            add(run_id=f"t3_{size}_{b}_{ftag(f)}_s{seed}", kind="convert", route="sc", size=size, seed=seed, lr=LR,
                flops_end=C, parent=(mid, tag), budget=b, f=f,
                gpus=4 if (1 - f) * C > 5e18 else (2 if (1 - f) * C > 3e18 else None))


_BUILDERS = {"grid": _grid, "split": _split, "edges": _edges, "stage_seeds": _stage_seeds, "lr": _lr,
             "conversion_lr": _conversion_lr, "ablations": _ablations, "runs_1e21": _runs_1e21,
             "distillation": _distillation, "sunk": _sunk, "shortcut": _shortcut}


def all_runs():
    """Every run of the paper, keyed by run id (a run shared by two groups belongs to the first)."""
    runs = {}
    for group in GROUPS:
        def add(**kw):
            if kw["run_id"] not in runs:
                runs[kw["run_id"]] = dict(kw, group=group)
        _BUILDERS[group](add)
    return runs


def dependencies(run):
    """Run ids a run needs before it can start: its parent, and for distillation the teacher and the student's init."""
    deps = [run["parent"][0]] if run.get("parent") else []
    if run["kind"] == "distill":
        deps.append(run["teacher"][0])
        if run["student_init"][0] != "TEACHER":
            deps.append(run["student_init"][0])
    return deps


def plan(groups=("grid",), with_ancestors=True):
    """The runs of the given groups, plus (by default) every run they depend on, in dependency order."""
    runs = all_runs()
    unknown = set(groups) - set(GROUPS)
    assert not unknown, f"unknown groups {sorted(unknown)}; choose from {GROUPS}"
    want = [rid for rid, r in runs.items() if r["group"] in groups]
    out, seen = [], set()

    def visit(rid):
        if rid in seen:
            return
        seen.add(rid)
        if with_ancestors:
            for d in dependencies(runs[rid]):
                visit(d)
        out.append(runs[rid])
    for rid in want:
        visit(rid)
    return out


def config_for(run):
    """Training config for osr.train (osr.train_dmd for distillation)."""
    route, size = run["route"], run["size"]
    if run["kind"] == "distill":
        tid, ts = run["teacher"]
        teacher = f"{RUNS}/{tid}/final_sel.pt"
        if not os.path.exists(teacher):
            teacher = f"{RUNS}/{tid}/final.pt"
        sid, stag = run["student_init"]
        student = teacher if sid == "TEACHER" else f"{RUNS}/{sid}/milestones/{stag}.pt"
        return dict(run_id=run["run_id"], out_root=RUNS, kind="distill", route="dmd", size=size, seed=run["seed"],
                    flops_end=float(run["flops_end"]), entry="osr.train_dmd",
                    data=dict(root=LATENTS, global_batch=DMD["batch"]), ema=dict(sigma_rels=[0.05, 0.10]),
                    dmd=dict(DMD["hp"], **run.get("hp", {}), teacher=teacher, teacher_size=ts, student_init=student),
                    ckpt_every_min=20, log_every=100, compile=True)
    cfg = dict(run_id=run["run_id"], out_root=RUNS, kind=run["kind"], route=route, size=size, seed=run["seed"],
               flops_end=float(run["flops_end"]),
               data=dict(root=LATENTS, global_batch=GLOBAL_BATCH, micro_batch=None),
               optim=dict(lr=run["lr"], betas=[0.9, 0.95], wd=SC_WD if route == "sc" else 0.0),
               ema=dict(sigma_rels=[0.05, 0.10]), ckpt_every_min=20, log_every=100, compile=True)
    cfg["data"]["micro_batch"] = min(GLOBAL_BATCH // GPUS[size], MICRO[route][size])
    if run["kind"] == "mainline":
        first = run.get("warmup_ref") or min(m["flops"] for m in run["milestones"])
        steps_first = first / (FLOPS_PER_SAMPLE[route][size] * GLOBAL_BATCH)
        cfg["schedule"] = dict(warmup_steps=int(min(WARMUP_STEPS, 0.1 * steps_first)))
        cfg["milestones"] = run["milestones"]
    elif run["kind"] == "extend":
        cfg["schedule"] = dict()
        cfg["milestones"] = run["milestones"]
        cfg["init"] = {"from": init_path(run["parent"])}
    elif run["kind"] == "cooldown":
        cfg["schedule"] = dict(cooldown_shape="linear")
        cfg["init"] = {"from": init_path(run["parent"])}
    else:  # convert
        cfg["schedule"] = dict(warmup_steps=2000, warmup_frac=0.1, cooldown_frac=COOLDOWN_FRAC, cooldown_shape="linear")
        cfg["init"] = {"from": init_path(run["parent"])}
        if run.get("init_mode"):
            cfg["init"]["mode"] = run["init_mode"]
    if route == "imf":
        cfg["loss"] = dict(cfg_beta=CFG_BETA[size], jvp_impl="manual")
    elif route == "sc":
        cfg["loss"] = dict(SC_LOSS, stratified=True)
    else:
        cfg["loss"] = dict(t_mean=-0.4, t_std=1.0, class_dropout=0.1)
    return cfg


def own_flops(run):
    """FLOPs a run itself spends (a mainline from 0, a branch from its start, an extension from its source)."""
    if run["kind"] in ("mainline", "distill"):
        return run["flops_end"]
    if run["kind"] == "extend":
        return run["flops_end"] - run["src_flops"]
    start = (1 - COOLDOWN_FRAC) * run["flops_end"] if run["kind"] == "cooldown" else run["f"] * run["flops_end"]
    return run["flops_end"] - start


if __name__ == "__main__":
    import sys
    groups = tuple(sys.argv[1:]) or GROUPS
    rs = plan(groups)
    by = {}
    for r in rs:
        by.setdefault((r["group"], r["kind"]), []).append(r)
    print(f"{len(rs)} runs (groups {', '.join(groups)} with their dependencies), {sum(own_flops(r) for r in rs):.3e} FLOPs")
    for (g, k), v in sorted(by.items()):
        print(f"  {g:14s} {k:9s} {len(v):4d} runs  {sum(own_flops(r) for r in v):.3e} FLOPs")
