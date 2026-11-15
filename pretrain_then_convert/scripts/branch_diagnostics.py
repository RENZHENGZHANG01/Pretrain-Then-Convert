"""What does FM pretraining hand to the conversion? At the FM branch points of the conversion grid (raw weights of
the FM mainline milestones f0.50/0.80/0.95/0.99, B and M at 1e19 and 3e19), on a fixed batch
(N noises, classes spread over ImageNet, conditional velocity, no guidance):

  S  straightness of the 50-step Euler ODE path: mean_k ||v_k - d||^2 / ||d||^2 with d = x(1) - x(0)
     (0 = perfectly straight; d is the displacement per unit time),
  G  one-step gap: ||(x1 - v(x1, 1)) - x0_ODE||^2 / ||x1 - x0_ODE||^2, the relative error of jumping with the
     instantaneous velocity. A converted model starts exactly there (its new h-embedder is frozen to h = 0),
     so G measures how much the conversion has to fix,
  dS / dG between consecutive branch points (how much the FM model still changes),

and pairs each branch point with the 1-NFE FID-50k of the conversion that started from it.
CPU with bf16 autocast (AMX); B and M are small.

  OMP_NUM_THREADS=32 python scripts/branch_diagnostics.py [n_samples]  ->  $OSR_EVAL/branch_diagnostics.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.models import build_model  # noqa: E402
from osr.paths import EVAL, RUNS  # noqa: E402

FS = (0.50, 0.80, 0.95, 0.99)


def load(path, size):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(size, with_v_head=False)
    m.load_state_dict(st["model"], strict=False)
    return m.eval()


@torch.no_grad()
def velocity(m, z, y):
    B = z.shape[0]
    zeros, ones = torch.zeros(B), torch.ones(B)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return m(z, zeros, zeros, ones, zeros, ones, y, heads=("u",))[0].float()


@torch.no_grad()
def measure(m, z, y, steps=50):
    ts = torch.linspace(1.0, 0.0, steps + 1)
    x, vs = z.clone(), []
    for i in range(steps):
        v = velocity(m, x, y)
        vs.append(v)
        x = x + (ts[i + 1] - ts[i]) * v
    x0 = x
    d = z - x0  # displacement per unit time (dz/dt points from data to noise)
    dn = d.flatten(1).pow(2).sum(1)
    S = torch.stack([(v - d).flatten(1).pow(2).sum(1) / dn for v in vs]).mean(0)
    one = z - vs[0]
    G = (one - x0).flatten(1).pow(2).sum(1) / dn
    return dict(S=float(S.mean()), S_se=float(S.std() / len(S) ** 0.5), G=float(G.mean()), G_se=float(G.std() / len(G) ** 0.5))


def conv_fid(run_id):
    """FID-50k of an evaluated run (scripts/eval_run.py), or None."""
    p = f"{EVAL}/{run_id}/summary.json"
    return json.load(open(p))["fid50k"] if os.path.exists(p) else None


def main(n=128):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 32)))
    g = torch.Generator().manual_seed(0)
    z = torch.randn(n, 4, 32, 32, generator=g)
    y = torch.arange(n) * (1000 // n) % 1000
    res = []
    for size in ("B", "M"):
        for b in ("1e19", "3e19"):
            for f in FS:
                p = f"{RUNS}/r1_{size}_s0/milestones/f{f:.2f}_{b}.pt"
                if not os.path.exists(p):
                    continue
                m = load(p, size)
                r = dict(size=size, budget=b, f=f, **measure(m, z, y),
                         conv_fid50k=conv_fid(f"r3_{size}_{b}_f{f:.2f}_s0"))
                res.append(r)
                print(json.dumps(r), flush=True)
                os.makedirs(EVAL, exist_ok=True)
                json.dump(res, open(f"{EVAL}/branch_diagnostics.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 128)
