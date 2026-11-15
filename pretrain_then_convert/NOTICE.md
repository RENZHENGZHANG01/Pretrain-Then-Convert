# Third-party code and models

| Component | Source | License | Used in |
|---|---|---|---|
| imfDiT model and the iMF objective | github.com/Lyy-iiis/imeanflow (JAX and `torch` branch), Copyright (c) Yiyang Lu | MIT | `osr/models.py` (port of the PyTorch model; parameter names kept so the released checkpoints load), `osr/losses.py` (port of the JAX objective) |
| DMD2 | github.com/tianweiy/DMD2 (`main/edm/edm_guidance.py`) | CC BY-NC-SA 4.0 | `osr/train_dmd.py` (adapted to flow matching and imfDiT); this file is distributed under CC BY-NC-SA 4.0 |
| Power-function EMA | Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (EDM2) | method from the paper | `osr/ema.py` (implementation of the published averaging rule) |
| Shortcut models | Frans et al., "One Step Diffusion via Shortcut Models" | method from the paper | `osr/losses.py` (`shortcut_loss`) |
| ADM center crop | github.com/openai/guided-diffusion | MIT | `scripts/encode_latents.py` |
| FID / IS (Inception) | torch-fidelity (pip package) | Apache-2.0 | `osr/evaluate.py` |
| FD-DINOv2 protocol | github.com/layer6ai-labs/dgm-eval | MIT | `osr/extra_metrics.py` (same preprocessing and features) |
| DINOv2 ViT-L/14 | github.com/facebookresearch/dinov2 (torch.hub) | Apache-2.0 | `osr/extra_metrics.py` |
| SD-VAE `ft-mse` | huggingface.co/stabilityai/sd-vae-ft-mse (diffusers) | CreativeML OpenRAIL-M | latents and decoding |
| FID reference statistics | `jit_in256_stats.npz` from github.com/LTH14/JiT | see the JiT repository | `osr/evaluate.py` |
| ADM reference batch | `VIRTUAL_imagenet256_labeled.npz` from guided-diffusion | MIT | ADM-FID, precision / recall |
