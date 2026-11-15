# Pretrain, Then Convert: Compute-Optimal Training of One-Step Image Generators

Code for the paper. A one-step image generator can be trained from scratch with a one-step objective (here improved
MeanFlow, iMF), or by first training a multi-step flow-matching (FM) model and converting it. We measure how a fixed
training budget is best split between the two phases on class-conditional ImageNet 256x256 with one backbone family
(imfDiT, 21M to 1.36B parameters) and over 200 models trained with 1e19 to 3e20 FLOPs, plus runs at 1e21 FLOPs:

* pretraining and then converting beats training from scratch in every size and budget trained with both routes
  (1.08-2.14x lower FID);
* the best split spends about 60% of the budget on FM; at the compute-optimal size every split between 25% and 80%
  is within 9% of the best;
* the gain is an exchange of samples: an iMF step costs kappa = 13/6 + 8/D FM steps (depth D), early FM samples are
  worth as much as one-step samples, and at the best split the converted model equals a native model trained on the
  same number of samples;
* one compute law describes both one-step routes once conversion is credited with a compute multiplier of 1.78; at
  1e21 FLOPs a converted XL model reaches one-step FID 1.75 (native training: 1.91).

## Contents

```
osr/                    training and evaluation library
  models.py             imfDiT (port of the official iMF PyTorch model; loads the released iMF checkpoints)
  losses.py             flow matching, improved MeanFlow, shortcut models
  jvp.py                hand-written forward-mode JVP of the u-path (compilable), used by the iMF loss
  train.py              trainer for every route: mainlines, cooldowns, conversions, extensions (FLOP-triggered)
  train_dmd.py          DMD2 distillation (CC BY-NC-SA 4.0, see NOTICE.md)
  data.py, ema.py       cached-latent data pipeline; power-function EMA
  flops.py              measured training FLOPs (torch FlopCounterMode)
  sample.py, evaluate.py, extra_metrics.py   samplers; FID / IS, FD-DINOv2, precision / recall
  experiments.py        every training run of the paper (groups, run ids, configs)
  paths.py              where data, checkpoints and evaluation outputs live (environment variables)
scripts/
  launch.py             list / train / evaluate the runs of the paper, or write SLURM scripts
  encode_latents.py     ImageNet -> SD-VAE latents
  dino_ref_stats.py     FD-DINOv2 reference statistics and ADM reference features
  eval_run.py           main evaluation (EMA selection, guidance sweep, FID-50k)
  eval_gridext.py       extended guidance search where the selected guidance lies on the grid edge
  eval_gridpolish.py    shared candidate guidance points for multi-step models
  eval_extra.py         FD-DINOv2, ADM-FID, precision / recall at the selected point
  eval_fdsel.py         FD-DINOv2 with its own guidance selection
  eval_prcurve.py       precision / recall along the guidance sweep
  eval_steps.py         multi-step models at 1-32 sampling steps (FID against generation cost)
  eval_fid.py           FID of any checkpoint at given guidance points (also the official iMF checkpoints)
  fid_sample_noise.py   sampling noise of FID-50k (split halves of the stored Inception features)
  sample_grid.py        sample grids with the same noise for several models
  branch_diagnostics.py path straightness and one-step error at FM branch points
  flops_table.py        FLOPs per sample for every size and objective (osr/flops_table.json)
  bench_step.py         training throughput per size
  validate_pipeline.py  sanity checks of latents, VAE, training and the official iMF-B/2 FID
  compact_finals.py     keep only the selected weights of evaluated runs
  self_test.py          end-to-end training check on random latents (one GPU)
configs/examples/       training configs of a few runs
slurm/train.sbatch      SLURM template used by scripts/launch.py
tests/                  unit tests (pytest)
```

## Installation

Python 3.12 and a CUDA GPU (we used H100 / H200, PyTorch 2.14 with CUDA 13):

```
pip install -r requirements.txt
```

Locations are set with environment variables (osr/paths.py); by default everything goes under `./work`:

| variable | content | default |
|---|---|---|
| `OSR_WORK` | root of the defaults | `./work` |
| `OSR_LATENTS` | cached latents | `$OSR_WORK/latents/sdvae_mse_256` |
| `OSR_RUNS` | training runs (configs, logs, checkpoints) | `$OSR_WORK/runs` |
| `OSR_EVAL` | evaluation outputs | `$OSR_WORK/eval` |
| `OSR_REF` | reference statistics | `$OSR_WORK/ref_stats` |
| `OSR_VAE` | SD-VAE (directory or Hugging Face id) | `stabilityai/sd-vae-ft-mse` |
| `OSR_DINO_REPO` | DINOv2 for torch.hub | `facebookresearch/dinov2` |

## Data and reference statistics

1. **Latents.** ImageNet-1k train, ADM center crop 256, SD-VAE (`ft-mse`) posterior moments of each image and its
   horizontal flip (about 40 GB):
   ```
   torchrun --nproc_per_node 8 scripts/encode_latents.py --root /path/to/imagenet/train
   ```
2. **FID reference** (ImageNet-256 train statistics of JiT, as in the official iMF evaluation):
   ```
   wget -P $OSR_REF https://raw.githubusercontent.com/LTH14/JiT/refs/heads/main/fid_stats/jit_in256_stats.npz
   ```
3. **ADM reference batch** (ADM-FID, precision / recall):
   ```
   wget -P $OSR_REF https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```
4. **FD-DINOv2 reference** (100 ImageNet train images per class, DINOv2 ViT-L/14) and the Inception features of the
   ADM reference images: `torchrun --nproc_per_node 2 scripts/dino_ref_stats.py`.

## Checks

```
python -m pytest -q tests                  # CPU tests; the shortcut tests need a GPU
OSR_WORK=/tmp/osr_self_test python scripts/self_test.py   # a few training steps of every route, one GPU
python scripts/validate_pipeline.py        # latents, VAE and real-data training (OSR_OFFICIAL_IMF_B: also FID of iMF-B/2)
```
`tests/test_model_equiv.py` also compares imfDiT with the official implementation when `OSR_IMF_TORCH` points to a
clone of github.com/Lyy-iiis/imeanflow (branch `torch`).

## Training

Every run of the paper is defined in `osr/experiments.py`. Runs follow a warmup-stable-decay protocol: each route and
size has one constant-learning-rate mainline; the model for budget C branches from it at 0.85 C and decays the learning
rate to zero at C, and the conversion with split f branches from the FM mainline at f C and trains the one-step
objective until C. Every trigger is in measured training FLOPs. The conversion is function-preserving
(`imfDiT.add_v_head_from_u`, `imfDiT.freeze_new_condition_embedders`): the new embedders of the guidance scale, the
guidance interval and h start as constants at the values FM was trained with, and the auxiliary v-head starts as a copy
of the u-head, so the converted network starts exactly as the FM network.

Groups: `grid` (the IsoFLOP grid), `split`, `edges`, `stage_seeds`, `lr`, `conversion_lr`, `ablations`, `runs_1e21`,
`distillation`, `sunk`, `shortcut` (see the docstring of `osr/experiments.py`).

```
python -m osr.experiments grid                      # runs and FLOPs of a group
python scripts/launch.py list grid                  # with their state (done / ready / waiting)
python scripts/launch.py train r1_M_s0 --gpus 2     # train one run here (its parent checkpoint must exist)
python scripts/launch.py slurm grid --submit        # SLURM jobs for every run that is ready
```
Runs start from milestones of their parents, so `slurm` submits what is ready; run it again as parents finish. The
trainer saves a resumable checkpoint every 20 minutes and on SIGTERM / SIGUSR1 and requeues itself under SLURM.
A single config can also be trained directly:
`torchrun --nproc_per_node 2 -m osr.train --config configs/examples/convert_M_3e19_f0.65.yaml`.

All runs of the paper together used about 17.5k GPU-hours of training (H100, the last runs H200) and 1.3k of
evaluation.

## Evaluation

```
python scripts/launch.py eval r3_M_3e19_f0.65_s0    # or: torchrun --nproc_per_node 2 scripts/eval_run.py --run RUN_ID
torchrun --nproc_per_node 2 scripts/eval_gridext.py --run RUN_ID     # extended guidance search (edge optima)
torchrun --nproc_per_node 2 scripts/eval_gridpolish.py --run RUN_ID  # multi-step models: shared candidate points
torchrun --nproc_per_node 2 scripts/eval_extra.py --run RUN_ID       # FD-DINOv2, ADM-FID, precision / recall
torchrun --nproc_per_node 4 scripts/eval_steps.py --run RUN_ID       # multi-step models: 1-32 sampling steps
```
`eval_run.py` selects the EMA (5k samples), sweeps the guidance on 10k samples (seed 1) and reports FID-50k with
independent noise (seed 0) and class-balanced labels; one-step models sample with 1 NFE, multi-step models with 50
Euler steps and two-pass guidance on an interval. Results go to `$OSR_EVAL/<run>/summary.json`.
`scripts/eval_fid.py --official` evaluates the released iMF checkpoints (huggingface.co/Lyy0725/iMF) with the same
pipeline (our evaluation of iMF-XL/2: FID 1.73; published: 1.72).

## FLOP accounting

Training compute is measured, not estimated: `osr/flops.py` counts the matmul-class operations of one full training
step (forward, the no-gradient guidance passes, the JVP tangent and the backward pass) with `FlopCounterMode` on the
single-pass reference implementation, and the trainer accumulates it per step. `osr/flops_table.json` lists the
per-sample FLOPs of every size and objective (`python scripts/flops_table.py`); the ratio iMF / FM is
kappa = 13/6 + 8/D (2.34-2.87 measured).

## Checkpoints

The checkpoints of the paper's frontier models and of the 1e21-FLOP runs will be released.

## License

MIT (LICENSE), except `osr/train_dmd.py`, which adapts DMD2 and is distributed under CC BY-NC-SA 4.0. Third-party code
and models are listed in NOTICE.md.
