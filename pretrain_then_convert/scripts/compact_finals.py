"""Compact evaluated branch checkpoints: final.pt holds raw weights + 2 EMAs (fp32). The
evaluation selected one of the three (eval/<run>/summary.json "weights"); store only that one
(fp32, unchanged) as final_sel.pt, verify it tensor-by-tensor against final.pt, and write a
manifest ($OSR_RUNS/compact_manifest.json); final.pt can then be deleted for the verified runs.

  python scripts/compact_finals.py            (CPU job)
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import EVAL, RUNS  # noqa: E402



def selected_state(st, weights):
    if weights == "raw":
        return st["model"]
    tags = [f"ema{s:.3f}".replace("0.", "") for s in st["ema"]["sigma_rels"]]
    return st["ema"]["shadows"][tags.index(weights)]


def main():
    manifest_path = f"{RUNS}/compact_manifest.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for rid in sorted(os.listdir(RUNS)):
        d = f"{RUNS}/{rid}"
        src, dst = f"{d}/final.pt", f"{d}/final_sel.pt"
        summ = f"{EVAL}/{rid}/summary.json"
        if not (os.path.exists(src) and os.path.exists(f"{EVAL}/{rid}/DONE") and os.path.exists(summ)):
            continue
        if manifest.get(rid, {}).get("verified") and os.path.exists(dst):
            continue
        weights = json.load(open(summ))["weights"]
        st = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
        if "ema" not in st:  # not a branch final with EMAs (should not happen for evaluated runs)
            continue
        sel = {k: v.clone() for k, v in selected_state(st, weights).items()}
        out = {"model": sel, "selected_weights": weights, "step": st["step"], "flops": st["flops"],
               "cfg": st["cfg"], "route": st["route"], "size": st["size"], "source": "final.pt (compacted)"}
        tmp = dst + ".tmp"
        torch.save(out, tmp)
        os.replace(tmp, dst)
        chk = torch.load(dst, map_location="cpu", weights_only=False)
        ref = selected_state(st, weights)
        ok = set(chk["model"]) == set(ref) and all(torch.equal(chk["model"][k], ref[k]) for k in ref)
        manifest[rid] = dict(weights=weights, verified=bool(ok), src_bytes=os.path.getsize(src), dst_bytes=os.path.getsize(dst))
        print(rid, weights, "verified" if ok else "MISMATCH", f"{os.path.getsize(src)/1e9:.2f} -> {os.path.getsize(dst)/1e9:.2f} GB", flush=True)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        del st, chk, sel, out
    n = sum(1 for v in manifest.values() if v["verified"])
    print(f"verified {n}/{len(manifest)}; saved {sum(v['src_bytes'] - v['dst_bytes'] for v in manifest.values() if v['verified'])/1e9:.1f} GB once originals are removed")


if __name__ == "__main__":
    main()
