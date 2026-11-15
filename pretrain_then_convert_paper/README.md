# Pretrain, Then Convert: paper and analysis

LaTeX source of the paper and the analysis that produces every number, table and figure in it from the results of
the training runs. The training and evaluation code is in the code repository `pretrain_then_convert`.

```
paper/        LaTeX source, ready for Overleaf: main.tex, refs.bib, ICML 2026 style files,
              tables/ and figures/ (written by analysis/run_all.sh)
analysis/     analysis scripts; analysis/out/ holds intermediate results
data/         results of all runs (see below)
tools/        build.sh (build the PDF), check_pages.py (8-page limit of the main text)
docs/         PROJECT_DETAILS.md: experiments, setup, analysis methods and results in detail (Chinese)
```

## Reproducing the paper

Python 3.12 with numpy 2.5, scipy 1.18, matplotlib 3.11 (STIX fonts are bundled) and pillow 12. `repro_tables.py`
reads hyper-parameters from the code repository, so the code repository's requirements must also be installed; it
looks for the code next to this repository, or at `PTC_CODE`.

```
bash analysis/run_all.sh     # every table and figure (about 30 minutes on one CPU; the bootstraps dominate)
tools/build.sh               # build/main.pdf with pdflatex + bibtex, then the page check
```

## Data

| path | content |
|---|---|
| `data/runs.csv` | the 411 training runs: route, kind (mainline, cooldown, convert, distill, ...), size, seed, budget, split, learning rate, FLOPs, parent milestone |
| `data/eval/<run>/summary.json` | main evaluation: selected weights and guidance, FID-50k (`scripts/eval_run.py`; `grid1`: before the extended guidance search) |
| `data/eval/<run>/extra.json` | FD-DINOv2, ADM-FID, precision / recall at the selected point (`scripts/eval_extra.py`) |
| `data/eval/<run>/fdsel.json`, `prcurve.json`, `gridext.json` | FD-DINOv2 with its own guidance selection, precision / recall along the guidance sweep, extended guidance search |
| `data/eval/<run>/steps.json` | multi-step frontier models at 1-32 sampling steps (`scripts/eval_steps.py`) |
| `data/eval/official_imf/` | FID of the released iMF checkpoints with our pipeline |
| `data/run_epochs.json` | FLOPs per sample and epochs at the end of every run |
| `data/flops_table.json`, `data/throughput.json` | measured training FLOPs per sample of every size and objective; training throughput |
| `data/lit/` | published ImageNet-256 results, each with a verbatim quote from the paper's arXiv source and its location |
| `data/samples/` | lossless sample grids of the sample figures (`scripts/sample_grid.py`) |
| `data/frozen/` | results that need the checkpoints or the cluster: path straightness and one-step error at the FM branch points (`scripts/branch_diagnostics.py`), sampling noise of FID-50k of the 12 frontier models (`scripts/fid_sample_noise.py`), GPU-hours from the SLURM accounting |

Script names in the table refer to the code repository.

## Where each table and figure comes from

| script | output |
|---|---|
| `collect.py` | `analysis/out/suite_metrics.csv`: one row per evaluated run |
| `frontier.py`, `surface.py`, `noise.py` | modules: frontiers and floored laws, the (N, C) surface, the seed-noise model of the bootstraps |
| `tables.py` | Table 2 (`t2_fits`), `t4_repro`, `t6_distillation`, `ta_flops`, `ta_lr` |
| `paper_tables.py` | `t1_col`, `ta_all`, `ta_joint`, `ta_heldout`, `ta_nstar`, `ta_exchange`, `ta_efficient`, `ta_sunk`, `ta_init` |
| `figures.py` | shared figure style; `fig_isoflop`, `fig_fstar_stab` |
| `paper_figs.py` | `fig_mechanism`, `fig_laws`, `fig_landscape` |
| `generation_cost.py` | FID against generation and total (training + generation) compute: `fig_gencost`, `ta_steps` |
| `diagrams.py` | `fig_overview`, `fig_related`, `fig_method` |
| `joint_law.py` | one compute law for both one-step routes (compute multiplier) |
| `matched_steps.py` | equal-FLOPs and equal-samples comparison: `fig_matched`, `ta_matched` |
| `two_component.py` | two-component model of the conversion curves: `ta_twocomp` |
| `noise_model.py` | bootstrap intervals, optimal-split estimators: `ta_noise`, `ta_fstar_nc` |
| `robustness.py` | backtests, crossover, matched-size ratios, selection bias of the converted frontier |
| `repetition.py` | epochs per run and the effect of data repetition |
| `distillation.py` | DMD2 students against total compute: `fig_distillation` |
| `shortcut.py` | split curves with shortcut models: `fig_shortcut`, `ta_shortcut` |
| `conversion_lr.py` | conversion learning rate: `ta_ctune` |
| `branch_points.py` | FM branch points: `fig_branch_points` |
| `precision_recall.py`, `pr_curves.py` | precision / recall of the frontier models: `fig_precision_recall`, `fig_prcurve` |
| `fd_dinov2.py`, `stage_seeds_fd.py` | FD-DINOv2 versions of the main comparisons; stage-seed noise |
| `guidance_extension.py` | effect of the extended guidance search |
| `lit_table.py` | comparison with published results: `t3_compare_short`, `t8_compare`, `ta_lit_sources` |
| `repro_tables.py` | `ta_arch`, `ta_hparams`, `ta_eval`, `ta_compute` |
| `sample_figs.py` | sample figures |

## Overleaf

Upload the contents of `paper/` (or `pretrain_then_convert_overleaf.zip`) as a new project; the main file is
`main.tex` (pdfLaTeX, BibTeX).
