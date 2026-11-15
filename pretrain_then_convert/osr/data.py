"""Cached SD-VAE latents of ImageNet-256 (ADM center crop), original + horizontal flip.

On disk (written by scripts/encode_latents.py), under <root>/:
  moments.npy  float16 [N, 2, 8, 32, 32]   VAE posterior (mean[4], std[4]) for (orig, flip)
  labels.npy   int64   [N]
Training samples z = mean + std * eps (as in the official MeanFlow/iMF pipelines) and
normalizes per channel with the iMF constants. Every epoch uses a fresh permutation and a
random flip per image; both are a pure function of (seed, epoch), so resuming is exact.
"""
import numpy as np
import torch

# per-channel statistics of raw sd-vae-ft-mse latents used by iMF (utils/vae_util.py)
LATENT_MEAN = (0.86488, -0.27787343, 0.21616915, 0.3738409)
LATENT_STD = (4.85503674, 5.31922414, 3.93725398, 3.9870003)


def warm_page_cache(path, chunk=1 << 28):
    """Sequentially read a file once so the OS page cache holds it; afterwards the random
    per-sample reads are memory speed instead of Lustre-latency bound (~500 img/s/rank cold)."""
    buf = bytearray(chunk)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass


class LatentData:
    def __init__(self, root, global_batch, rank, world, seed, device):
        self.moments = np.load(f"{root}/moments.npy", mmap_mode="r")
        self.labels = torch.from_numpy(np.load(f"{root}/labels.npy").astype(np.int64))
        self.N = self.labels.shape[0]
        assert global_batch % world == 0
        self.gb, self.lb = global_batch, global_batch // world
        self.rank, self.world, self.seed, self.device = rank, world, seed, device
        self.steps_per_epoch = self.N // global_batch  # drop the ragged tail each epoch
        self.mean = torch.tensor(LATENT_MEAN, device=device).view(1, 4, 1, 1)
        self.std = torch.tensor(LATENT_STD, device=device).view(1, 4, 1, 1)
        self._epoch, self._perm, self._flip = None, None, None

    def _epoch_state(self, epoch):
        if self._epoch != epoch:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            self._perm = torch.randperm(self.N, generator=g)
            self._flip = torch.randint(0, 2, (self.N,), generator=g)
            self._epoch = epoch
        return self._perm, self._flip

    def batch(self, step):
        """Local batch for global step `step` -> (x [lb,4,32,32] normalized fp32, y [lb])."""
        epoch, it = divmod(step, self.steps_per_epoch)
        perm, flip = self._epoch_state(epoch)
        idx = perm[it * self.gb + self.rank * self.lb: it * self.gb + (self.rank + 1) * self.lb]
        order = torch.argsort(idx)  # one sorted fancy-index read (both flips), then pick the flip
        sidx = idx[order].numpy()
        rows = self.moments[sidx]  # [lb, 2, 8, 32, 32]
        mom = rows[np.arange(len(sidx)), flip[idx[order]].numpy()]
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order))
        mom = torch.from_numpy(np.ascontiguousarray(mom))[inv]
        y = self.labels[idx]
        return mom, y

    def to_latent(self, mom, gen):
        """posterior sample + per-channel normalization, on device."""
        mom = mom.to(self.device, non_blocking=True).float()
        mean, std = mom[:, :4], mom[:, 4:]
        z = mean + std * torch.randn(mean.shape, generator=gen, device=self.device)
        return (z - self.mean) / self.std

    def epochs(self, step):
        return step * self.gb / self.N


class Prefetcher:
    """Background thread that reads the next batches from the memmap."""

    def __init__(self, data, start_step, depth=4):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.data, self.step = data, start_step
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        s = self.step
        while not self.stop:
            mom, y = self.data.batch(s)
            self.q.put((s, mom.pin_memory(), y))
            s += 1

    def get(self):
        return self.q.get()


class RandomLatents:
    """Stand-in with the same interface, for throughput / FLOP tests before latents exist."""

    def __init__(self, global_batch, rank, world, seed, device):
        self.lb, self.gb, self.N, self.device = global_batch // world, global_batch, 1_281_167, device
        self.steps_per_epoch = self.N // global_batch
        self.rank, self.seed = rank, seed

    def batch(self, step):
        g = torch.Generator().manual_seed(self.seed + 7919 * step + self.rank)
        return torch.randn(self.lb, 8, 32, 32, generator=g).abs_() * 0.1, torch.randint(0, 1000, (self.lb,), generator=g)

    def to_latent(self, mom, gen):
        return torch.randn(mom.shape[0], 4, 32, 32, generator=gen, device=self.device)

    def epochs(self, step):
        return step * self.gb / self.N
