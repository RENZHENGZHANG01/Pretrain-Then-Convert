"""Reference statistics for FD-DINOv2: 100 random ImageNet-train images per class (100k), ADM
center crop 256 (as for the latents), DINOv2 ViT-L/14 CLS features. Also caches Inception pool
features of the 10k ADM reference images (for precision / recall).

  torchrun --nproc_per_node 2 scripts/dino_ref_stats.py
"""
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.extra_metrics import ADM_REF, DINO_REF, DinoV2  # noqa: E402
from osr.paths import ADM_FEATS, LATENTS as LAT  # noqa: E402
from scripts.encode_latents import Files  # noqa: E402


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    files = open(f"{LAT}/files.txt").read().split("\n")
    labels = np.load(f"{LAT}/labels.npy")
    rng = np.random.default_rng(0)
    idx = np.sort(np.concatenate([rng.choice(np.where(labels == c)[0], 100, replace=False) for c in range(1000)]))
    mine = idx[rank::world]
    dino = DinoV2(dev)

    class Sub(torch.utils.data.Dataset):
        def __len__(self): return len(mine)
        def __getitem__(self, i): return Files(files, int(mine[i]), int(mine[i]) + 1)[0][1]

    dl = torch.utils.data.DataLoader(Sub(), batch_size=250, num_workers=10)
    feats = torch.cat([dino(b.to(dev)) for b in dl])
    allf = [torch.zeros(len(idx[r::world]), feats.shape[1], dtype=feats.dtype, device=dev) for r in range(world)]
    dist.all_gather(allf, feats)
    if rank == 0:
        f = torch.cat(allf).cpu().numpy()
        np.savez(DINO_REF, mu=f.mean(0), sigma=np.cov(f, rowvar=False), n=len(f))
        print("dinov2 ref:", f.shape, "->", DINO_REF, flush=True)
        # Inception pool features of the ADM reference images (10k, uint8 NHWC in arr_0)
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        incep = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(dev).eval()
        imgs = np.load(ADM_REF)["arr_0"]
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 500):
                x = torch.from_numpy(imgs[i:i + 500]).permute(0, 3, 1, 2).contiguous().to(dev)
                out.append(incep(x)[0].double().cpu())
        np.save(ADM_FEATS, torch.cat(out).numpy().astype(np.float32))
        print("adm ref inception feats:", len(imgs), "->", ADM_FEATS, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
