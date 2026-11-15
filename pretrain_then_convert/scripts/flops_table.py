"""Exact per-sample training FLOPs (FlopCounterMode) for every size and objective, on CPU with a tiny batch: FM and
iMF training steps (forward, guidance passes, JVP, backward), the inference forward of the u-path, the inference
parameter count, and kappa = iMF / FM. Writes osr/flops_table.json (used by osr.experiments for warmup lengths; the
training runs measure their own FLOPs).

  python scripts/flops_table.py
"""
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import osr.models as om  # noqa: E402
from osr.flops import measure_step_flops  # noqa: E402
from osr.losses import fm_loss, imf_loss  # noqa: E402
from osr.models import build_model, count_params  # noqa: E402
from osr.paths import FLOPS_TABLE  # noqa: E402


def main(batch=2):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 32)))
    out = {}
    for size in ["XS", "S", "B", "M", "L", "XL", "H"]:
        row = {}
        for route, fn in [("fm", fm_loss), ("imf", imf_loss)]:
            m = build_model(size, with_v_head=(route == "imf"))
            x = torch.randn(batch, 4, 32, 32)
            y = torch.randint(0, 1000, (batch,))
            row[route] = measure_step_flops(fn, m, x, y, {}, seed=0) / batch
            if route == "fm":
                om._ATTN_OVERRIDE[0] = "math"
                with FlopCounterMode(display=False) as fc, torch.no_grad():
                    z, o = torch.zeros(batch), torch.ones(batch)
                    m(x, z, z, o, z, o, y, heads=("u",))
                om._ATTN_OVERRIDE[0] = None
                row["fwd"] = fc.get_total_flops() / batch
                row["N_inference"] = count_params(m)
        row["imf_over_fm"] = row["imf"] / row["fm"]
        out[size] = row
        print(size, {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(out, open(FLOPS_TABLE, "w"), indent=1)


if __name__ == "__main__":
    main()
