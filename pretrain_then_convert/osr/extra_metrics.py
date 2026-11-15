"""Secondary metrics: FD-DINOv2 (dgm-eval protocol), FID against the ADM reference batch, and
improved precision / recall (k = 3, Inception pool features, ADM reference images).

DINOv2: ViT-L/14 through torch.hub (OSR_DINO_REPO: facebookresearch/dinov2 or a local clone; the weights come from
the torch.hub cache). Preprocessing as dgm-eval: bicubic resize to 224x224, ImageNet mean/std; feature = the CLS
embedding.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .paths import ADM_REF, DINO_REF, DINO_REPO

__all__ = ["ADM_REF", "DINO_REF", "DinoV2", "frechet", "precision_recall", "stats"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoV2:
    def __init__(self, dev):
        source = "local" if os.path.isdir(DINO_REPO) else "github"
        self.model = torch.hub.load(DINO_REPO, "dinov2_vitl14", source=source).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, img_u8):
        x = img_u8.to(self.dev).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        x = (x - _MEAN.to(self.dev)) / _STD.to(self.dev)
        return self.model(x).double()


def frechet(mu1, s1, mu2, s2):
    """Frechet distance. tr sqrt(S1 S2) = sum sqrt(eig(S1^1/2 S2 S1^1/2)) via symmetric eigensolvers: same value
    as the usual scipy sqrtm form, robust, and independent of the sqrtm API (SciPy 1.18 dropped `disp`)."""
    diff = mu1 - mu2
    w, v = np.linalg.eigh((s1 + s1.T) / 2)
    h = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = h @ s2 @ h
    tr_covmean = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def stats(feats):
    f = feats.cpu().numpy().astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


@torch.no_grad()
def precision_recall(ref, gen, k=3, dev="cuda", chunk=10000):
    """Improved precision/recall (Kynkaenniemi et al. 2019) with k-NN radii, as in the ADM evaluator."""
    ref = ref.to(dev).float(); gen = gen.to(dev).float()

    def radii(x):
        r = torch.empty(x.shape[0], device=dev)
        for i in range(0, x.shape[0], chunk):
            d = torch.cdist(x[i:i + chunk], x)
            r[i:i + chunk] = d.kthvalue(k + 1, dim=1).values  # +1: distance to itself is 0
        return r

    def coverage(a, ra, b):  # fraction of b inside the union of balls around a
        inside = torch.zeros(b.shape[0], dtype=torch.bool, device=dev)
        for i in range(0, b.shape[0], chunk):
            d = torch.cdist(b[i:i + chunk], a)
            inside[i:i + chunk] = (d <= ra[None, :]).any(dim=1)
        return inside.float().mean().item()

    r_ref, r_gen = radii(ref), radii(gen)
    return dict(precision=coverage(ref, r_ref, gen), recall=coverage(gen, r_gen, ref))
