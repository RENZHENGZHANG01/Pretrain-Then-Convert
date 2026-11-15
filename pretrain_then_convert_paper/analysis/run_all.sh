#!/bin/bash
# Every number, table and figure of the paper from the data snapshot (data/), in dependency order.
# Outputs: analysis/out/ (intermediate results), paper/tables/, paper/figures/. About 30 minutes on one CPU
# (the parametric bootstraps dominate).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for s in collect guidance_extension conversion_lr fd_dinov2 stage_seeds_fd pr_curves precision_recall shortcut \
         repetition branch_points joint_law distillation matched_steps two_component noise_model tables robustness \
         figures paper_figs generation_cost paper_tables diagrams sample_figs lit_table repro_tables; do
    echo "== $s"
    $PY $s.py
done
