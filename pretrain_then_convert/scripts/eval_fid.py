"""Generate samples and compute FID / IS (torch-fidelity Inception, JiT/iMF reference stats).

  torchrun --nproc_per_node 8 scripts/eval_fid.py --ckpt RUN/final.pt --weights ema0.10 \
      --sampler meanflow --nfe 1 --grid "omega=6,7,8;t_min=0.4;t_max=0.65" --n 10000 --out EVALDIR

--grid is a ';'-separated list of comma-separated values; the cartesian product is
evaluated (each point with --n samples) after loading the model / VAE / Inception once.
Results are appended to EVALDIR/results.jsonl; with --save_features the Inception pool3
features of each point are saved (fp16) for bootstrap CIs.
--official loads an official iMF .pth (keys prefixed 'net.') for pipeline validation.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.data import LATENT_MEAN, LATENT_STD  # noqa: E402
from osr.models import build_model  # noqa: E402
from osr.paths import FID_REF as REF, VAE  # noqa: E402
from osr.sample import fm_sample, meanflow_sample  # noqa: E402


def parse_grid(s):
    keys, vals = [], []
    for part in s.split(";"):
        if not part.strip():
            continue
        k, v = part.split("=")
        keys.append(k.strip())
        vals.append([float(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def load_model(args, dev):
    if args.official:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
        model = build_model(args.size, eval_mode=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not [m for m in missing if not m.startswith("v_")], missing
        meta = dict(official=True, size=args.size)
    else:
        st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        size, route = st["size"], st["route"]
        model = build_model(size, with_v_head=(route == "imf"))
        if args.weights == "raw":
            sd = st["model"]
        else:
            i = [f"ema{s:.3f}".replace("0.", "") for s in st["ema"]["sigma_rels"]].index(args.weights)
            sd = {k: v for k, v in st["ema"]["shadows"][i].items()}
        model.load_state_dict(sd, strict=True)
        meta = dict(official=False, size=size, route=route, step=st["step"], flops=st["flops"],
                    run_id=st["cfg"]["run_id"], kind=st["cfg"]["kind"])
    return model.to(dev).eval(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--size", default="B")
    ap.add_argument("--weights", default="raw")
    ap.add_argument("--sampler", default="meanflow", choices=["meanflow", "fm"])
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--solver", default="euler")
    ap.add_argument("--grid", default="omega=1")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save_features", action="store_true")
    ap.add_argument("--save_png", type=int, default=0, help="save the first K images of each point")
    ap.add_argument("--vae", default=VAE)
    ap.add_argument("--ref", default=REF)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    model, meta = load_model(args, dev)
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(dev).eval()
    del vae.encoder
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_isc import isc_features_to_metric
    incep = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(dev).eval()
    ref = np.load(args.ref)
    ref_stats = {"mu": ref["mu"].astype(np.float64), "sigma": ref["sigma"].astype(np.float64)}
    lmean = torch.tensor(LATENT_MEAN, device=dev).view(1, 4, 1, 1)
    lstd = torch.tensor(LATENT_STD, device=dev).view(1, 4, 1, 1)

    n = args.n
    assert n % 1000 == 0
    labels_all = np.arange(1000).repeat(n // 1000)
    per = (n + world - 1) // world
    my = np.arange(rank * per, min(n, (rank + 1) * per))

    for point in parse_grid(args.grid):
        t0 = time.time()
        feats, logits, pngs = [], [], []
        for s in range(0, len(my), args.bs):
            idx = my[s:s + args.bs]
            g = [torch.Generator().manual_seed(args.seed * 10_000_000 + int(i)) for i in idx]
            z = torch.stack([torch.randn(4, 32, 32, generator=gi) for gi in g]).to(dev)
            y = torch.from_numpy(labels_all[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.sampler == "meanflow":
                    x = meanflow_sample(model, z, y, nfe=args.nfe, omega=point.get("omega", 1.0),
                                        t_min=point.get("t_min", 0.0), t_max=point.get("t_max", 1.0))
                else:
                    x = fm_sample(model, z, y, steps=args.nfe, solver=args.solver, omega=point.get("omega", 1.0),
                                  t_lo=point.get("t_lo", 0.0), t_hi=point.get("t_hi", 1.0), shift=point.get("shift", 1.0))
            with torch.no_grad():
                img = vae.decode(x.float() * lstd + lmean).sample
                img = ((img + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
                f, lg = incep(img)
            feats.append(f.double().cpu())
            logits.append(lg.double().cpu())
            if args.save_png and len(pngs) * args.bs < args.save_png and rank == 0:
                pngs.append(img[: args.save_png].cpu())
        feats = torch.cat(feats).to(dev)
        logits = torch.cat(logits).to(dev)
        # gather (pad to equal length)
        cnt = torch.tensor([feats.shape[0]], device=dev)
        cnts = [torch.zeros_like(cnt) for _ in range(world)]
        dist.all_gather(cnts, cnt)
        mx = int(max(c.item() for c in cnts))
        pad = lambda a: torch.cat([a, a.new_zeros(mx - a.shape[0], a.shape[1])])
        gf = [torch.zeros(mx, feats.shape[1], device=dev, dtype=feats.dtype) for _ in range(world)]
        gl = [torch.zeros(mx, logits.shape[1], device=dev, dtype=logits.dtype) for _ in range(world)]
        dist.all_gather(gf, pad(feats))
        dist.all_gather(gl, pad(logits))
        if rank == 0:
            F_ = torch.cat([a[: int(c.item())] for a, c in zip(gf, cnts)]).cpu()
            L_ = torch.cat([a[: int(c.item())] for a, c in zip(gl, cnts)]).cpu()
            st = fid_features_to_statistics(F_)
            fid = float(fid_statistics_to_metric(st, ref_stats, False)["frechet_inception_distance"])
            isc = isc_features_to_metric(L_, splits=10, shuffle=True, rng_seed=2020)
            rec = dict(meta, tag=args.tag, sampler=args.sampler, nfe=args.nfe, solver=args.solver, weights=args.weights,
                       n=n, seed=args.seed, fid=fid, inception_score=float(isc["inception_score_mean"]),
                       inception_score_std=float(isc["inception_score_std"]), ref=os.path.basename(args.ref),
                       eval_seconds=time.time() - t0, world=world, **point)
            print(json.dumps(rec), flush=True)
            with open(os.path.join(args.out, "results.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            key = "_".join(f"{k}{v:g}" for k, v in point.items())
            if args.save_features:
                np.save(os.path.join(args.out, f"inception_{args.weights}_nfe{args.nfe}_{key}_n{n}.npy"), F_.half().numpy())
            if pngs:
                from torchvision.utils import save_image
                save_image(torch.cat(pngs)[: args.save_png].float() / 255, os.path.join(args.out, f"samples_{args.weights}_{key}.png"), nrow=10)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
