"""Shared-candidate polish of the multi-step guidance selection (third evaluation step). The coordinate search of
eval_gridext.py is greedy on noisy 5k-sample scores, and seeds of one cell sometimes ended in different basins
(wide interval + low scale vs narrow interval + high scale). Here every multi-step model is scored on the same
set of candidate points, covering the best points the search found across all models, on 10k samples (seed 1);
the best point on 10k (previous stage-B and extension points included) is selected, and FID-50k (seed 0) and the
other metrics are recomputed there if the selection changed. The main protocol's values stay under "grid1".

  torchrun --nproc_per_node G scripts/eval_gridpolish.py --run RUN_ID
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import ADM_FEATS, EVAL, RUNS  # noqa: E402
from osr.evaluate import Evaluator, load_run_model  # noqa: E402

# (omega, t_lo, t_hi): both basins found by the extended search (the most frequent best points)
CAND = [(1.75, 0.2, 0.85), (2.5, 0.0, 0.75), (2.5, 0.3, 0.7), (3.5, 0.3, 0.6),
        (5.0, 0.2, 0.65), (5.0, 0.3, 0.65), (5.0, 0.4, 0.65), (7.0, 0.3, 0.55)]


def key_of(p):
    return tuple(sorted((k, round(float(v), 4)) for k, v in p.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nB", type=int, default=10000)
    ap.add_argument("--nC", type=int, default=50000)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out_dir = f"{EVAL}/{args.run}"
    if os.path.exists(f"{out_dir}/gridpolish.json"):
        if rank == 0:
            print("already polished", args.run)
        dist.destroy_process_group()
        return
    summ = json.load(open(f"{out_dir}/summary.json"))
    ck = f"{RUNS}/{args.run}/final.pt"
    if not os.path.exists(ck):
        ck = f"{RUNS}/{args.run}/final_sel.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    weights, route = summ["weights"], st["route"]
    assert route == "fm", route
    model = load_run_model(st, weights, dev)
    ev = Evaluator(dev)
    meta = dict(run_id=args.run, size=st["size"], route=route, kind=st["cfg"]["kind"], step=st["step"], flops=st["flops"])
    seen = {}
    for line in open(f"{out_dir}/results.jsonl"):
        r = json.loads(line)
        if r.get("weights") == weights and r.get("stage") in ("B", "Bx", "Bp"):
            p = {k: r[k] for k in ("omega", "t_lo", "t_hi")}
            seen[key_of(p)] = (r["fid"], p)
    old_point, old10 = summ["point"], summ["fid10k_sel"]
    tried = []
    for w, lo, hi in CAND:
        p = dict(omega=w, t_lo=lo, t_hi=hi)
        if key_of(p) in seen:
            continue
        res, _, _ = ev.evaluate(model, "fm", 50, p, args.nB, 1)
        fid = None
        if rank == 0:
            fid = res["fid"]
            with open(f"{out_dir}/results.jsonl", "a") as f:
                f.write(json.dumps(dict(meta, **res, weights=weights, stage="Bp")) + "\n")
        obj = [fid]; dist.broadcast_object_list(obj, src=0); fid = obj[0]
        seen[key_of(p)] = (fid, p); tried.append(dict(p, fid10k=fid))
    best10, best_p = min(seen.values(), key=lambda x: x[0])
    changed = key_of(best_p) != key_of(old_point) and best10 < old10
    rec = dict(run_id=args.run, weights=weights, candidates=CAND, old_point=old_point, old_fid10k=old10, tried=tried,
               best_point=best_p, best_fid10k=best10, changed=bool(changed))
    if changed:
        from osr.extra_metrics import ADM_REF, DINO_REF, DinoV2, frechet, precision_recall, stats
        adm_feats = ADM_FEATS
        feats, logits, pngs, dfeats = ev.generate_scores(model, "fm", 50, best_p, args.nC, seed=0, keep_png=100,
                                                         extra=DinoV2(dev))
        fid = None
        if rank == 0:
            m = ev.metrics(feats, logits)
            m.update(sampler="fm", nfe=50, n=args.nC, seed=0, **best_p)
            with open(f"{out_dir}/results.jsonl", "a") as f:
                f.write(json.dumps(dict(meta, **m, weights=weights, stage="Cp")) + "\n")
            fid = m["fid"]
            mu_g, s_g = stats(feats); adm = np.load(ADM_REF); dref = np.load(DINO_REF); mu_d, s_d = stats(dfeats)
            if os.path.exists(f"{out_dir}/extra.json") and not os.path.exists(f"{out_dir}/extra_grid1.json"):
                shutil.copy(f"{out_dir}/extra.json", f"{out_dir}/extra_grid1.json")
            extra = dict(run_id=args.run, weights=weights, point=best_p, n=args.nC, fid50k_stageC=fid,
                         fid_adm=frechet(mu_g, s_g, adm["mu"].astype(np.float64), adm["sigma"].astype(np.float64)),
                         fd_dinov2=frechet(mu_d, s_d, dref["mu"].astype(np.float64), dref["sigma"].astype(np.float64)),
                         inception_score=m["inception_score"],
                         **precision_recall(torch.from_numpy(np.load(adm_feats)), feats.float(), k=3, dev=dev))
            json.dump(extra, open(f"{out_dir}/extra.json", "w"), indent=1)
            dpath = f"{out_dir}/dinov2_{weights}_n{args.nC}.npy"
            if os.path.exists(dpath) and not os.path.exists(dpath.replace(".npy", "_grid1.npy")):
                os.replace(dpath, dpath.replace(".npy", "_grid1.npy"))
            np.save(dpath, dfeats.half().numpy())
            kk = "_".join(f"{k}{v:g}" for k, v in best_p.items())
            np.save(f"{out_dir}/inception_{weights}_fm50_{kk}_n{args.nC}.npy", feats.half().numpy())
            prev = {k: summ[k] for k in ("point", "fid50k", "fid10k_sel") if k in summ}
            grid1 = summ.get("grid1", prev)
            summ.update(grid1=grid1, point=best_p, fid50k=fid, fid10k_sel=best10)
            rec.update(new_fid50k=fid, old_fid50k=prev["fid50k"], grid1_fid50k=grid1["fid50k"])
    if rank == 0:
        summ["gridpolish"] = True
        json.dump(summ, open(f"{out_dir}/summary.json", "w"), indent=1)
        json.dump(rec, open(f"{out_dir}/gridpolish.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("tried", "candidates")}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
