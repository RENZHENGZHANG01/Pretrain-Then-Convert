"""Engineering sanity checks on real data before the grid is launched (not research gates).

  python scripts/validate_pipeline.py            (1 GPU)

1. VAE round trip: decode cached posterior means of random train images -> PSNR vs the
   ADM-cropped source image. A wrong latent layout / normalization gives ~10 dB or noise.
2. Normalized latent statistics (iMF constants): per-channel mean ~ 0, std ~ 1.
3. Short real-data training (B, batch 256): FM 400 steps and iMF 200 steps; losses must fall.
4. If the official iMF-B/2 checkpoint is staged: FID-10k at the README setting (omega 8,
   interval [0.4, 0.65]); the 50k value is published as 3.32 (torch) / 3.39 (paper), so a
   10k estimate of roughly 4-6 is expected (FID is biased upward at small n).
Writes $OSR_WORK/validate/validate.json and PASS or FAIL.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import LATENTS, VAE, WORK  # noqa: E402
from osr.data import LatentData  # noqa: E402
from osr.losses import fm_loss, imf_loss  # noqa: E402
from osr.models import build_model  # noqa: E402

LAT = LATENTS
OFFICIAL_B = os.environ.get("OSR_OFFICIAL_IMF_B", "")  # optional: the official iMF-B/2 checkpoint (iMF-B-2.pth)


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = {}
    ok = True
    from diffusers.models import AutoencoderKL
    from scripts.encode_latents import Files
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float32).to(dev).eval()
    files = open(f"{LAT}/files.txt").read().split("\n")
    mom = np.load(f"{LAT}/moments.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(files), 32, replace=False))
    # 1. round trip (orig and flipped copies)
    psnr = []
    with torch.no_grad():
        for i in idx:
            _, img = Files(files, int(i), int(i) + 1)[0]
            x = img.float().to(dev) / 127.5 - 1
            for flip in (0, 1):
                m = torch.from_numpy(np.asarray(mom[i, flip, :4], dtype=np.float32)).to(dev)[None]
                rec = vae.decode(m).sample[0].clamp(-1, 1)
                ref = torch.flip(x, dims=[2]) if flip else x
                mse = ((rec - ref) ** 2).mean().item() / 4.0  # images in [-1,1] -> scale to [0,1]
                psnr.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    out["roundtrip_psnr_mean"] = float(np.mean(psnr))
    out["roundtrip_psnr_min"] = float(np.min(psnr))
    ok &= out["roundtrip_psnr_mean"] > 20.0
    # 2. normalized latent statistics on 2048 random samples
    data = LatentData(LAT, 256, 0, 1, 0, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    zs = []
    for s in range(8):
        momb, _ = data.batch(s)
        zs.append(data.to_latent(momb, g))
    z = torch.cat(zs)
    out["latent_channel_mean"] = z.mean(dim=(0, 2, 3)).tolist()
    out["latent_channel_std"] = z.std(dim=(0, 2, 3)).tolist()
    ok &= all(abs(m) < 0.3 for m in out["latent_channel_mean"]) and all(0.6 < s < 1.5 for s in out["latent_channel_std"])
    # 3. short real-data training
    for route, fn, steps in (("fm", fm_loss, 400), ("imf", imf_loss, 200)):
        torch.manual_seed(0)
        model = build_model("B", with_v_head=(route == "imf")).to(dev)
        model.enable_compile()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
        key = "loss_fm" if route == "fm" else "loss_u"
        hist = []
        for s in range(steps):
            for gr in opt.param_groups:
                gr["lr"] = 1e-4 * min(1.0, (s + 1) / 50)
            momb, y = data.batch(s)
            gg = torch.Generator(device=dev).manual_seed(1000 + s)
            x = data.to_latent(momb, gg)
            y = y.to(dev)
            for k in range(4):  # micro-batches of 64
                sl = slice(64 * k, 64 * (k + 1))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, logs = fn(model, x[sl], y[sl], {"cfg_beta": 1.0}, gg)
                (loss / 4).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            hist.append(float(logs[key]))
        a, b = float(np.mean(hist[:30])), float(np.mean(hist[-30:]))
        out[f"train_{route}_{key}_first30"], out[f"train_{route}_{key}_last30"] = a, b
        ok &= b < a and np.isfinite(b)
        del model, opt
        torch.cuda.empty_cache()
    out["pass"] = bool(ok)
    os.makedirs(f"{WORK}/validate", exist_ok=True)
    with open(f"{WORK}/validate/validate.json", "w") as f:
        json.dump(out, f, indent=1)
    open(f"{WORK}/validate/{'PASS' if ok else 'FAIL'}", "w").close()
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
