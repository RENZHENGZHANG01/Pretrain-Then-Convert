"""Distributed sample-and-score: FID / IS with torch-fidelity's Inception (JiT / iMF reference
stats), optional FD-DINOv2 later. Used by scripts/eval_fid.py and scripts/eval_run.py.

Protocol: class-balanced labels (n/1000 per class), per-sample noise
seeds; selection on 10k samples with seed 1, reporting on 50k with seed 0 (independent).
"""
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .data import LATENT_MEAN, LATENT_STD
from .models import build_model
from .paths import FID_REF as REF, VAE
from .sample import fm_sample, meanflow_sample


def ema_tags(st):
    return [f"ema{s:.3f}".replace("0.", "") for s in st["ema"]["sigma_rels"]]


def load_run_model(st, weights, dev):
    model = build_model(st["size"], with_v_head=(st["route"] == "imf"))
    if "selected_weights" in st:  # compacted checkpoint (final_sel.pt): only the selected weights are stored
        assert weights in (st["selected_weights"], "selected"), (weights, st["selected_weights"])
        model.load_state_dict(st["model"], strict=True)
        return model.to(dev).eval()
    if weights == "raw":
        sd = st["model"]
    else:
        sd = st["ema"]["shadows"][ema_tags(st).index(weights)]
    model.load_state_dict(sd, strict=True)
    return model.to(dev).eval()


class Evaluator:
    def __init__(self, dev, vae=VAE, ref=REF, compile_model=True):
        from diffusers.models import AutoencoderKL
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.dev = dev
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float32).to(dev).eval()
        del self.vae.encoder
        self.incep = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(dev).eval()
        r = np.load(ref)
        self.ref = {"mu": r["mu"].astype(np.float64), "sigma": r["sigma"].astype(np.float64)}
        self.ref_name = os.path.basename(ref)
        self.lmean = torch.tensor(LATENT_MEAN, device=dev).view(1, 4, 1, 1)
        self.lstd = torch.tensor(LATENT_STD, device=dev).view(1, 4, 1, 1)
        self.compile_model = compile_model

    @torch.no_grad()
    def generate_scores(self, model, sampler, nfe, point, n, seed, bs=125, solver="euler", keep_png=0, extra=None):
        """Returns (features[N,2048] float64 on rank0 else None, logits, pngs[, extra feats]).
        `extra`: optional callable img_uint8 -> [B, D] features (e.g. DINOv2), gathered like the rest."""
        assert n % 1000 == 0
        labels_all = np.arange(1000).repeat(n // 1000)
        per = (n + self.world - 1) // self.world
        my = np.arange(self.rank * per, min(n, (self.rank + 1) * per))
        feats, logits, pngs, ext = [], [], [], []
        for s in range(0, len(my), bs):
            idx = my[s:s + bs]
            g = [torch.Generator().manual_seed(seed * 10_000_000 + int(i)) for i in idx]
            z = torch.stack([torch.randn(4, 32, 32, generator=gi) for gi in g]).to(self.dev)
            y = torch.from_numpy(labels_all[idx]).to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if sampler == "meanflow":
                    x = meanflow_sample(model, z, y, nfe=nfe, omega=point.get("omega", 1.0),
                                        t_min=point.get("t_min", 0.0), t_max=point.get("t_max", 1.0))
                else:
                    x = fm_sample(model, z, y, steps=nfe, solver=solver, omega=point.get("omega", 1.0),
                                  t_lo=point.get("t_lo", 0.0), t_hi=point.get("t_hi", 1.0), shift=point.get("shift", 1.0))
            img = self.vae.decode(x.float() * self.lstd + self.lmean).sample
            img = ((img + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
            f, lg = self.incep(img)
            feats.append(f.double())
            logits.append(lg.double())
            if extra is not None:
                ext.append(extra(img))
            if keep_png and self.rank == 0 and sum(p.shape[0] for p in pngs) < keep_png:
                pngs.append(img.cpu())
        if extra is not None:
            return self._gather(torch.cat(feats)), self._gather(torch.cat(logits)), pngs, self._gather(torch.cat(ext))
        return self._gather(torch.cat(feats)), self._gather(torch.cat(logits)), pngs

    def _gather(self, a):
        cnt = torch.tensor([a.shape[0]], device=self.dev)
        cnts = [torch.zeros_like(cnt) for _ in range(self.world)]
        dist.all_gather(cnts, cnt)
        mx = int(max(c.item() for c in cnts))
        pad = torch.cat([a, a.new_zeros(mx - a.shape[0], a.shape[1])])
        out = [torch.zeros_like(pad) for _ in range(self.world)]
        dist.all_gather(out, pad)
        if self.rank != 0:
            return None
        return torch.cat([o[: int(c.item())] for o, c in zip(out, cnts)]).cpu()

    def metrics(self, feats, logits):
        from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
        from torch_fidelity.metric_isc import isc_features_to_metric
        st = fid_features_to_statistics(feats)
        fid = float(fid_statistics_to_metric(st, self.ref, False)["frechet_inception_distance"])
        isc = isc_features_to_metric(logits, splits=10, shuffle=True, rng_seed=2020)
        return dict(fid=fid, inception_score=float(isc["inception_score_mean"]), ref=self.ref_name)

    def evaluate(self, model, sampler, nfe, point, n, seed, **kw):
        t0 = time.time()
        feats, logits, pngs = self.generate_scores(model, sampler, nfe, point, n, seed, **kw)
        out = None
        if self.rank == 0:
            out = self.metrics(feats, logits)
            out.update(sampler=sampler, nfe=nfe, n=n, seed=seed, seconds=time.time() - t0, **point)
        return out, feats, pngs
