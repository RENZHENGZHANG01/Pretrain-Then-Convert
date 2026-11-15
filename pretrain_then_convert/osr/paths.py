"""Where data, checkpoints and evaluation outputs live.

Every location can be set with an environment variable; by default everything sits under ./work (OSR_WORK).

  OSR_WORK       root of the defaults below                                  (default: ./work)
  OSR_LATENTS    cached SD-VAE latents of ImageNet-256 (scripts/encode_latents.py)
  OSR_RUNS       training runs: one directory per run id (config, logs, checkpoints)
  OSR_EVAL       evaluation outputs: one directory per run id
  OSR_REF        reference statistics (FID reference, ADM reference batch, FD-DINOv2 statistics)
  OSR_VAE        SD-VAE weights: a local directory or a Hugging Face id     (default: stabilityai/sd-vae-ft-mse)
  OSR_DINO_REPO  DINOv2 code for torch.hub: a local clone or a GitHub repo  (default: facebookresearch/dinov2)
"""
import os

WORK = os.path.abspath(os.environ.get("OSR_WORK", "work"))
LATENTS = os.environ.get("OSR_LATENTS", os.path.join(WORK, "latents", "sdvae_mse_256"))
RUNS = os.environ.get("OSR_RUNS", os.path.join(WORK, "runs"))
EVAL = os.environ.get("OSR_EVAL", os.path.join(WORK, "eval"))
REF = os.environ.get("OSR_REF", os.path.join(WORK, "ref_stats"))
VAE = os.environ.get("OSR_VAE", "stabilityai/sd-vae-ft-mse")
DINO_REPO = os.environ.get("OSR_DINO_REPO", "facebookresearch/dinov2")

# reference files (see README, "Reference statistics")
FID_REF = os.path.join(REF, "jit_in256_stats.npz")                      # ImageNet-256 train statistics (JiT / iMF)
ADM_REF = os.path.join(REF, "VIRTUAL_imagenet256_labeled.npz")          # ADM reference batch (guided-diffusion)
ADM_FEATS = os.path.join(REF, "adm_ref_inception_pool.npy")             # its Inception pool features (dino_ref_stats.py)
DINO_REF = os.path.join(REF, "dinov2_vitl14_imagenet256_train100k.npz")  # FD-DINOv2 reference (dino_ref_stats.py)

FLOPS_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops_table.json")  # scripts/flops_table.py


def run_dir(run_id):
    return os.path.join(RUNS, run_id)


def eval_dir(run_id):
    return os.path.join(EVAL, run_id)
