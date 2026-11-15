"""Qualitative samples for the paper: the same classes and the same noise for several models, saved as lossless
PNG grids (rows = classes, columns = samples). Each model uses its selected weights and guidance setting
(eval/<run>/summary.json); one-step models sample with 1 NFE, multi-step models with 50 Euler steps.

  python scripts/sample_grid.py --runs v2_XL_1e21_f0.60_s0 v1_XL_1e21_cd_s0 v2_H_1e21_cd_s0 --out samples
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import EVAL, RUNS  # noqa: E402
from osr.data import LATENT_MEAN, LATENT_STD  # noqa: E402
from osr.evaluate import VAE, load_run_model  # noqa: E402
from osr.sample import fm_sample, meanflow_sample  # noqa: E402

# macaw, golden retriever, arctic fox, otter, lesser panda, giant panda, balloon, coral reef, volcano, cheeseburger,
# alp, bald eagle, lion, space shuttle, lakeside, strawberry
CLASSES = [88, 207, 279, 360, 387, 388, 417, 973, 980, 933, 970, 22, 291, 812, 975, 949]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--classes", nargs="+", type=int, default=CLASSES)
    ap.add_argument("--n", type=int, default=8, help="samples per class")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float32).to(dev).eval()
    lmean = torch.tensor(LATENT_MEAN, device=dev).view(1, 4, 1, 1)
    lstd = torch.tensor(LATENT_STD, device=dev).view(1, 4, 1, 1)
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(len(args.classes), args.n, 4, 32, 32, generator=g)  # shared by all models
    meta = {}
    for run in args.runs:
        summ = json.load(open(f"{EVAL}/{run}/summary.json"))
        ck = f"{RUNS}/{run}/final_sel.pt"
        if not os.path.exists(ck):
            ck = f"{RUNS}/{run}/final.pt"
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model = load_run_model(st, summ["weights"], dev)
        p = summ["point"]
        rows = []
        for ci, c in enumerate(args.classes):
            z = noise[ci].to(dev)
            y = torch.full((args.n,), c, device=dev, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if st["route"] == "imf":
                    x = meanflow_sample(model, z, y, nfe=1, omega=p.get("omega", 1.0), t_min=p.get("t_min", 0.0),
                                        t_max=p.get("t_max", 1.0))
                else:
                    x = fm_sample(model, z, y, steps=50, omega=p.get("omega", 1.0), t_lo=p.get("t_lo", 0.0),
                                  t_hi=p.get("t_hi", 1.0))
            img = vae.decode(x.float() * lstd + lmean).sample
            img = ((img + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
            rows.append(torch.cat(list(img), dim=2))  # [3, 256, 256 * n]
        grid = torch.cat(rows, dim=1).permute(1, 2, 0).numpy()
        Image.fromarray(grid).save(f"{args.out}/{run}.png", optimize=False)
        meta[run] = dict(route=st["route"], size=st["size"], weights=summ["weights"], point=p, fid50k=summ["fid50k"],
                         nfe=1 if st["route"] == "imf" else 50)
        print(run, meta[run], flush=True)
        del model, st
        torch.cuda.empty_cache()
    json.dump(dict(classes=args.classes, n=args.n, seed=args.seed, runs=meta), open(f"{args.out}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
