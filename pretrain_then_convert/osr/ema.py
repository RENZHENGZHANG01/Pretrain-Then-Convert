"""Power-function EMA (Karras et al., EDM2). beta_t = (1 - 1/t)^(gamma + 1), where gamma is
set from the relative width sigma_rel. The averaging window is a fixed *fraction* of the
run so far, so EMAs stay comparable across budgets and WSD branches (no fixed decay).
"""
import numpy as np
import torch


def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


class PowerEMA:
    def __init__(self, model, sigma_rels=(0.05, 0.10)):
        self.sigma_rels = tuple(sigma_rels)
        self.gammas = [sigma_rel_to_gamma(s) for s in self.sigma_rels]
        self.shadows = [{k: v.detach().clone().float() for k, v in model.state_dict().items()
                         if v.dtype.is_floating_point} for _ in self.sigma_rels]

    @torch.no_grad()
    def update(self, model, t):
        """t = number of optimizer steps taken in this EMA's lifetime (t >= 1)."""
        sd = model.state_dict()
        for gamma, sh in zip(self.gammas, self.shadows):
            beta = (1 - 1 / t) ** (gamma + 1) if t > 1 else 0.0
            keys = list(sh.keys())
            src = [sd[k].float() for k in keys]
            dst = [sh[k] for k in keys]
            torch._foreach_lerp_(dst, src, 1 - beta)

    def state_dict(self):
        return {"sigma_rels": self.sigma_rels, "shadows": self.shadows}

    def load_state_dict(self, st):
        assert tuple(st["sigma_rels"]) == self.sigma_rels
        for sh, src in zip(self.shadows, st["shadows"]):
            for k in sh:
                sh[k].copy_(src[k])

    def tag(self, i):
        return f"ema{self.sigma_rels[i]:.3f}".replace("0.", "")
