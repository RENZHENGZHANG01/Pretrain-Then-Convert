"""Throughput benchmark of one training step per size, 1 GPU.
   MODES=fm:1,imf:1 (route:compiled), MICRO_<size>=n overrides the micro-batch.
   Reports algorithmic TFLOP/s (FLOPs measured on the single-pass reference) and memory."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.flops import measure_step_flops
from osr.losses import fm_loss, imf_loss
from osr.models import build_model

torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device('cuda')
sizes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['B', 'XL']
mbs = {'S': 128, 'B': 64, 'M': 32, 'L': 32, 'XL': 32, 'H': 16}
modes = os.environ.get('MODES', 'fm:1,imf:1').split(',')


def bench(m, opt, fn, x, y, fl):
    def step(i):
        g = torch.Generator(device=dev).manual_seed(i)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = fn(m, x, y, {}, g)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    t0 = time.time()
    for i in range(3):
        step(i)
    torch.cuda.synchronize()
    tw = time.time() - t0
    n = 10
    t0 = time.time()
    for i in range(n):
        step(100 + i)
    torch.cuda.synchronize()
    return (time.time() - t0) / n, tw


for size in sizes:
    for mode in modes:
        route, comp = mode.split(':')
        comp = comp == '1'
        fn = fm_loss if route == 'fm' else imf_loss
        torch.manual_seed(0)
        m = build_model(size, with_v_head=(route == 'imf')).to(dev)
        if comp:
            m.enable_compile()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
        B = int(os.environ.get('MICRO_' + size, mbs[size]))
        x = torch.randn(B, 4, 32, 32, device=dev)
        y = torch.randint(0, 1000, (B,), device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fl = measure_step_flops(fn, m, x[:4], y[:4], {}, seed=1) * (B // 4)
        torch.cuda.reset_peak_memory_stats()
        try:
            dt, tw = bench(m, opt, fn, x, y, fl)
            r = dict(size=size, route=route, compiled=comp, micro=B, step_s=dt, tflops=fl / dt / 1e12, img_s=B / dt,
                     mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, warm_s=tw)
        except torch.OutOfMemoryError:
            r = dict(size=size, route=route, compiled=comp, micro=B, oom=True)
        print(json.dumps(r), flush=True)
        del m, opt
        torch.cuda.empty_cache()
