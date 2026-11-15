"""End-to-end check of the training code on one GPU with random latents (no data needed, a few minutes):
an FM mainline with a branch point, a conversion to iMF from it, a native iMF mainline with a cooldown, and a shortcut
conversion, each for a few steps through osr.train under torchrun; the runs must finish, and the converted model must
start out equal to the FM model (function-preserving conversion).

  OSR_WORK=/tmp/osr_self_test python scripts/self_test.py
"""
import copy
import os
import subprocess
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from osr import experiments as X  # noqa: E402
from osr.models import build_model  # noqa: E402
from osr.paths import RUNS  # noqa: E402

SIZE = "S"
PER_STEP = {"fm": X.FLOPS_PER_SAMPLE["fm"][SIZE] * X.GLOBAL_BATCH, "imf": X.FLOPS_PER_SAMPLE["imf"][SIZE] * X.GLOBAL_BATCH}


def config(run, steps):
    cfg = X.config_for(run)
    cfg["data"]["random"] = True
    cfg["compile"] = False
    cfg["log_every"] = 5
    cfg["run_id"] = "selftest_" + run["run_id"]
    if "init" in cfg:
        cfg["init"]["from"] = cfg["init"]["from"].replace(f"{RUNS}/", f"{RUNS}/selftest_")
    return cfg


def train(cfg):
    os.makedirs(f"{RUNS}/_configs", exist_ok=True)
    path = f"{RUNS}/_configs/{cfg['run_id']}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    subprocess.check_call(["torchrun", "--standalone", "--nproc_per_node", "1", "-m", "osr.train", "--config", path], cwd=REPO)
    st = open(f"{RUNS}/{cfg['run_id']}/status.txt").read()
    assert st.startswith("done"), st
    print(cfg["run_id"], "->", st.strip(), flush=True)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    n = 30  # steps of each run
    fm = dict(run_id="r1_S_s0", kind="mainline", route="fm", size=SIZE, seed=0, lr=X.LR, flops_end=n * PER_STEP["fm"],
              milestones=[dict(flops=(n // 2) * PER_STEP["fm"], kind="weights", tag="f0.50_test")], parent=None, warmup_ref=None)
    train(config(fm, n))
    conv = dict(run_id="r3_S_test_f0.50_s0", kind="convert", route="imf", size=SIZE, seed=0, lr=X.LR,
                flops_end=(n // 2) * PER_STEP["fm"] + n * PER_STEP["imf"], parent=("r1_S_s0", "f0.50_test"), f=0.5)
    train(config(conv, n))
    nat = dict(run_id="r2_S_s0", kind="mainline", route="imf", size=SIZE, seed=0, lr=X.LR, flops_end=n * PER_STEP["imf"],
               milestones=[dict(flops=(n // 2) * PER_STEP["imf"], kind="full", tag="cd_test")], parent=None)
    train(config(nat, n))
    cd = dict(run_id="r2_S_test_cd_s0", kind="cooldown", route="imf", size=SIZE, seed=0, lr=X.LR,
              flops_end=(n // 2) * PER_STEP["imf"] / (1 - X.COOLDOWN_FRAC), parent=("r2_S_s0", "cd_test"))
    train(config(cd, n))
    sc = dict(conv, run_id="t3_S_test_f0.50_s0", route="sc")
    train(config(sc, n))
    # function preservation at the branch point
    ms = torch.load(f"{RUNS}/selftest_r1_S_s0/milestones/f0.50_test.pt", map_location="cpu", weights_only=False)
    fmm = build_model(SIZE, with_v_head=False)
    fmm.load_state_dict(ms["model"])
    cv = copy.deepcopy(fmm)
    cv.add_v_head_from_u()
    cv.freeze_new_condition_embedders()
    B = 4
    z, y = torch.randn(B, 4, 32, 32), torch.randint(0, 1000, (B,))
    zeros, ones = torch.zeros(B), torch.ones(B)
    with torch.no_grad():
        u_fm, _ = fmm(z, zeros, zeros, ones, zeros, ones, y, heads=("u",))
        u_cv, v_cv = cv(z, zeros, torch.rand(B), 1 + 5 * torch.rand(B), 0.3 * torch.rand(B), 0.7 + 0.3 * torch.rand(B), y)
    assert (u_cv - u_fm).abs().max() < 1e-4 and (v_cv - u_fm).abs().max() < 1e-4
    print("self test passed", flush=True)


if __name__ == "__main__":
    main()
