"""Paths of the paper repository: the data snapshot (data/), analysis outputs (analysis/out/), and the tables and
figures of the paper (paper/tables/, paper/figures/)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EVAL = os.path.join(DATA, "eval")          # evaluation results of every run (summary.json, extra.json, ...)
OUT = os.path.join(ROOT, "analysis", "out")
TAB = os.path.join(ROOT, "paper", "tables")
FIGS = os.path.join(ROOT, "paper", "figures")
for _d in (OUT, TAB, FIGS):
    os.makedirs(_d, exist_ok=True)
FT = json.load(open(os.path.join(DATA, "flops_table.json")))  # FLOPs per sample of every size and objective
NPAR = {k: v["N_inference"] for k, v in FT.items()}
BUD = {"1e19": 1e19, "3e19": 3e19, "1e20": 1e20, "3e20": 3e20}
