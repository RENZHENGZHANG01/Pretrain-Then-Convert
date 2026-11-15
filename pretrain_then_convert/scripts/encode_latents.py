"""Encode ImageNet-1k train (ADM center crop 256) + horizontal flips with SD-VAE.

  torchrun --nproc_per_node 8 scripts/encode_latents.py --root /path/to/imagenet/train
  (defaults: --vae $OSR_VAE, --out $OSR_LATENTS; see osr/paths.py)

Writes moments.npy (fp16 [N, 2, 8, 32, 32]: posterior mean|std for orig, flip), labels.npy,
files.txt and a done-marker per rank. Labels follow the sorted synset order (ImageFolder).
Ranks own contiguous index shards and write into one shared memmap; a rank skips chunks it
has already finished (chunk markers), so the job can be preempted and resubmitted.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import LATENTS, VAE  # noqa: E402

IMAGENET = os.environ.get("IMAGENET_TRAIN", "imagenet/train")  # ImageNet-1k train, one folder per synset


def center_crop_arr(pil_image, image_size):
    """ADM center crop (guided-diffusion image_datasets.py)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return arr[cy:cy + image_size, cx:cx + image_size]


class Files(torch.utils.data.Dataset):
    def __init__(self, files, lo, hi):
        self.files, self.lo, self.hi = files, lo, hi

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        j = self.lo + i
        img = Image.open(self.files[j]).convert("RGB")
        arr = center_crop_arr(img, 256)
        return j, torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)


def list_files(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                files.append(os.path.join(root, c, f))
                labels.append(ci)
    return classes, files, np.array(labels, dtype=np.int64)


def init_dataset(root, out, limit=0):
    """CPU-only: list files, write labels/files/classes and create the moments memmap."""
    classes, files, labels = list_files(root)
    if limit:
        files, labels = files[:limit], labels[:limit]
    os.makedirs(os.path.join(out, "chunks_done"), exist_ok=True)
    np.save(os.path.join(out, "labels.npy"), labels)
    with open(os.path.join(out, "files.txt"), "w") as f:
        f.write("\n".join(files))
    with open(os.path.join(out, "classes.txt"), "w") as f:
        f.write("\n".join(classes))
    mpath = os.path.join(out, "moments.npy")
    if not os.path.exists(mpath):
        np.lib.format.open_memmap(mpath, mode="w+", dtype=np.float16, shape=(len(files), 2, 8, 32, 32)).flush()
    print(f"init: {len(files)} files, {len(classes)} classes -> {out}")


def main_shard():
    """Independent 1-GPU shard (SLURM array): processes chunks[shard::nshards]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=4096)
    args, _ = ap.parse_known_args()
    files = open(os.path.join(args.out, "files.txt")).read().split("\n")
    N = len(files)
    mom = np.load(os.path.join(args.out, "moments.npy"), mmap_mode="r+")
    assert mom.shape[0] == N
    dev = torch.device("cuda")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(dev).eval()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    chunks = [(s, min(N, s + args.chunk)) for s in range(0, N, args.chunk)][args.shard::args.nshards]
    t0, done_imgs = time.time(), 0
    for (lo, hi) in chunks:
        marker = os.path.join(args.out, "chunks_done", f"{lo:08d}")
        if os.path.exists(marker):
            continue
        dl = torch.utils.data.DataLoader(Files(files, lo, hi), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
        for idx, img in dl:
            x = img.to(dev, non_blocking=True).float() / 127.5 - 1.0
            x = torch.cat([x, torch.flip(x, dims=[3])])
            with torch.no_grad():
                d = vae.encode(x).latent_dist
                m = torch.cat([d.mean, d.std], dim=1)
            b = idx.shape[0]
            mom[idx.numpy()] = torch.stack([m[:b], m[b:]], dim=1).half().cpu().numpy()
            done_imgs += b
        mom.flush()
        open(marker, "w").close()
        el = time.time() - t0
        print(f"[shard {args.shard}] chunk {lo}-{hi} done; {done_imgs} imgs, {done_imgs / el:.1f} img/s", flush=True)
    n_done = len(os.listdir(os.path.join(args.out, "chunks_done")))
    n_all = (N + args.chunk - 1) // args.chunk
    print(f"[shard {args.shard}] finished; chunks done {n_done}/{n_all}", flush=True)
    if n_done == n_all:
        open(os.path.join(args.out, "COMPLETE"), "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=VAE)
    ap.add_argument("--out", default=LATENTS)
    ap.add_argument("--root", default=IMAGENET)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    os.makedirs(os.path.join(args.out, "chunks_done"), exist_ok=True)

    lists = [None]
    if rank == 0:
        lists[0] = list_files(args.root)
    dist.broadcast_object_list(lists, src=0)
    classes, files, labels = lists[0]
    if args.limit:
        files, labels = files[:args.limit], labels[:args.limit]
    N = len(files)
    mpath = os.path.join(args.out, "moments.npy")
    if rank == 0:
        if not os.path.exists(mpath):
            np.lib.format.open_memmap(mpath, mode="w+", dtype=np.float16, shape=(N, 2, 8, 32, 32)).flush()
        np.save(os.path.join(args.out, "labels.npy"), labels)
        with open(os.path.join(args.out, "files.txt"), "w") as f:
            f.write("\n".join(files))
        with open(os.path.join(args.out, "classes.txt"), "w") as f:
            f.write("\n".join(classes))
    dist.barrier()
    mom = np.load(mpath, mmap_mode="r+")
    assert mom.shape[0] == N

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float32).to(dev).eval()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    chunks = [(s, min(N, s + args.chunk)) for s in range(0, N, args.chunk)]
    mine = chunks[rank::world]
    t0, done_imgs = time.time(), 0
    for (lo, hi) in mine:
        marker = os.path.join(args.out, "chunks_done", f"{lo:08d}")
        if os.path.exists(marker):
            continue
        dl = torch.utils.data.DataLoader(Files(files, lo, hi), batch_size=args.batch, num_workers=args.workers,
                                         pin_memory=True, persistent_workers=False)
        for idx, img in dl:
            x = img.to(dev, non_blocking=True).float() / 127.5 - 1.0
            x = torch.cat([x, torch.flip(x, dims=[3])])
            with torch.no_grad():
                d = vae.encode(x).latent_dist
                m = torch.cat([d.mean, d.std], dim=1)  # [2b, 8, 32, 32]
            b = idx.shape[0]
            out = torch.stack([m[:b], m[b:]], dim=1).half().cpu().numpy()
            mom[idx.numpy()] = out
            done_imgs += b
        mom.flush()
        open(marker, "w").close()
        if rank == 0:
            el = time.time() - t0
            print(f"[rank0] chunk {lo}-{hi} done; {done_imgs} imgs in {el:.0f}s ({done_imgs / el:.1f} img/s/rank)", flush=True)
    dist.barrier()
    if rank == 0:
        n_done = len(os.listdir(os.path.join(args.out, "chunks_done")))
        print(f"chunks done {n_done}/{len(chunks)}", flush=True)
        if n_done == len(chunks):
            open(os.path.join(args.out, "COMPLETE"), "w").close()
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    if "--init" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--init", action="store_true")
        ap.add_argument("--out", required=True)
        ap.add_argument("--root", default=IMAGENET)
        ap.add_argument("--limit", type=int, default=0)
        a = ap.parse_args()
        init_dataset(a.root, a.out, a.limit)
    elif "--shard" in sys.argv:
        main_shard()
    else:
        main()
