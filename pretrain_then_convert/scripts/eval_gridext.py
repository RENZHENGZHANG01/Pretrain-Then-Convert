"""Extended guidance search (second evaluation step, after scripts/eval_run.py) where the main protocol's optimum
sits on the edge of its grid. Same selected weights, same stage-B protocol (10k samples, seed 1), new points only:
  imf: selected omega = 10.5 (largest swept) -> omega ladder 12.5, 15, 18, 22, 27 on the selected interval, while
       the best point is the largest omega tried; selected interval [0.4, 0.65] (the narrowest swept) -> coordinate
       search over the interval in steps of 0.05 (intervals containing t = 0.5, as in training);
  fm:  every model selected (0.2, 0.85), the only non-trivial interval swept -> coordinate search over the interval
       in steps of 0.1;
  coordinate search: lower bound one step either way, then the upper bound, each with three
       omegas around the current best, stepping further outward (up to three steps) while the edge keeps winning,
       then an omega extension at the best interval; on 5k samples (seed 1, the starting point included); its two
       best new points are re-scored on 10k samples (seed 1) for the comparison with the previous selection.
       Points evaluated earlier (stage B) are reused.
If a new point beats the previous selection on 10k samples, stage C is redone there (50k, seed 0, with the DINOv2
features and extra metrics), and summary.json is updated; the previous values are kept under "grid1" (and
extra_grid1.json / *_grid1.npy). Always writes eval/<run>/gridext.json.

  torchrun --nproc_per_node G scripts/eval_gridext.py --run RUN_ID
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

IMF_LADDER = (12.5, 15.0, 18.0, 22.0, 27.0)
IMF_OMEGAS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.0, 8.5, 10.5, 12.5, 15.0, 18.0, 22.0, 27.0)
FM_OMEGAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
VERSION = 2


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
    summ = json.load(open(f"{out_dir}/summary.json"))
    gx = f"{out_dir}/gridext.json"
    if os.path.exists(gx) and json.load(open(gx)).get("version", 1) >= VERSION:  # already extended
        if rank == 0:
            print("already extended", args.run)
        dist.destroy_process_group()
        return
    ck = f"{RUNS}/{args.run}/final.pt"
    if not os.path.exists(ck):
        ck = f"{RUNS}/{args.run}/final_sel.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    weights, route = summ["weights"], st["route"]
    model = load_run_model(st, weights, dev)
    sampler, nfe = ("meanflow", 1) if route == "imf" else ("fm", 50)
    ev = Evaluator(dev)
    meta = dict(run_id=args.run, size=st["size"], route=route, kind=st["cfg"]["kind"], step=st["step"], flops=st["flops"])

    # previous values (selected weights, seed 1): stage B and version-1 extension points (10k), 5k search points
    ks = ("omega", "t_min", "t_max") if route == "imf" else ("omega", "t_lo", "t_hi")
    seen, seen5 = {}, {}
    for line in open(f"{out_dir}/results.jsonl"):
        r = json.loads(line)
        if r.get("weights") != weights or r.get("stage") not in ("B", "Bx", "Bx5k"):
            continue
        p = {k: r[k] for k in ks}
        (seen5 if r["stage"] == "Bx5k" else seen)[key_of(p)] = (r["fid"], p)
    old_point, old10 = summ["point"], summ["fid10k_sel"]  # the current selection (version 1 may have moved it)
    tried = []

    def run_n(p, n, stage):
        res, _, _ = ev.evaluate(model, sampler, nfe, p, n, 1)
        fid = None
        if rank == 0:
            fid = res["fid"]
            with open(f"{out_dir}/results.jsonl", "a") as f:
                f.write(json.dumps(dict(meta, **res, weights=weights, stage=stage)) + "\n")
        obj = [fid]; dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def ev10(p):
        k = key_of(p)
        if k in seen:
            return seen[k][0]
        fid = run_n(p, args.nB, "Bx")
        seen[k] = (fid, p); tried.append(dict(p, fid10k=fid))
        return fid

    def ev5(p):
        k = key_of(p)
        if k not in seen5:
            fid = run_n(p, args.nB // 2, "Bx5k")
            seen5[k] = (fid, p); tried.append(dict(p, fid5k=fid))
        return seen5[k][0]

    def best_seen():
        return min(seen.values(), key=lambda x: x[0])

    lo_k, hi_k = ks[1], ks[2]
    ladder = IMF_OMEGAS if route == "imf" else FM_OMEGAS

    def P(w, lo, hi):
        return {"omega": w, lo_k: round(lo, 3), hi_k: round(hi, 3)}

    def valid(lo, hi):
        if route == "imf":  # trained with t_min in [0, 0.5] and t_max in [0.5, 1]
            return 0.0 <= lo <= 0.5 <= hi <= 1.0 and hi - lo >= 0.1 - 1e-9
        return 0.0 <= lo < hi <= 1.0 and hi - lo >= 0.1 - 1e-9

    def near(w, fac):
        return min(ladder, key=lambda v: abs(np.log(v) - np.log(w * fac)))

    def sweep(lo, hi, w0):
        for fac in (1 / 1.3, 1.0, 1.3):
            ev5(P(near(w0, fac), lo, hi))

    def best5():
        return min(seen5.values(), key=lambda x: x[0])[1]

    def coord(which, step):
        """Move one bound: one step either way, then further outward (up to 3 steps) while the edge wins."""
        b = best5()
        lo, hi, w0 = b[lo_k], b[hi_k], b["omega"]
        x0 = lo if which == "lo" else hi
        for d in (-step, step):
            c = (x0 + d, hi) if which == "lo" else (lo, x0 + d)
            if valid(*c):
                sweep(*c, w0)
        for _ in range(3):
            b = best5()
            x = b[lo_k] if which == "lo" else b[hi_k]
            if abs(x - x0) < 1e-9:
                break
            nxt = x + (x - x0) / abs(x - x0) * step
            c = (nxt, b[hi_k]) if which == "lo" else (b[lo_k], nxt)
            if not valid(*c):
                break
            sweep(*c, b["omega"])
            x0 = x

    if route == "imf" and old_point["omega"] >= 10.5:
        for w in IMF_LADDER:
            p = dict(omega=w, t_min=old_point["t_min"], t_max=old_point["t_max"])
            ev10(p)
            if best_seen()[1]["omega"] != w:  # the best is no longer the largest omega tried
                break
    narrow = route == "imf" and (old_point["t_min"], old_point["t_max"]) == (0.4, 0.65)
    if route == "fm" or narrow:
        step = 0.05 if route == "imf" else 0.1
        start = best_seen()[1] if route == "imf" else dict(old_point)
        ev5(dict(start))
        coord("lo", step)
        coord("hi", step)
        for _ in range(4):  # omega extension at the best interval while its best omega is the largest tried
            b = best5()
            same = [p["omega"] for _, p in seen5.values() if (p[lo_k], p[hi_k]) == (b[lo_k], b[hi_k])]
            if b["omega"] < max(same):
                break
            nxt = [v for v in ladder if v > b["omega"]]
            if not nxt:
                break
            ev5(P(nxt[0], b[lo_k], b[hi_k]))
        new5 = sorted((v for k, v in seen5.items() if k not in seen), key=lambda x: x[0])
        for _, p in new5[:2]:  # the two best points not yet scored on 10k
            ev10(p)

    best10, best_p = best_seen()
    changed = key_of(best_p) != key_of(old_point) and best10 < old10
    rec = dict(run_id=args.run, version=VERSION, weights=weights, old_point=old_point, old_fid10k=old10, tried=tried,
               best_point=best_p, best_fid10k=best10, changed=bool(changed))
    if changed:
        from osr.extra_metrics import ADM_REF, DINO_REF, DinoV2, frechet, precision_recall, stats
        adm_feats = ADM_FEATS
        feats, logits, pngs, dfeats = ev.generate_scores(model, sampler, nfe, best_p, args.nC, seed=0, keep_png=100,
                                                         extra=DinoV2(dev))
        fid = None
        if rank == 0:
            m = ev.metrics(feats, logits)
            m.update(sampler=sampler, nfe=nfe, n=args.nC, seed=0, **best_p)
            with open(f"{out_dir}/results.jsonl", "a") as f:
                f.write(json.dumps(dict(meta, **m, weights=weights, stage="Cx")) + "\n")
            fid = m["fid"]
            mu_g, s_g = stats(feats); adm = np.load(ADM_REF); dref = np.load(DINO_REF); mu_d, s_d = stats(dfeats)
            if os.path.exists(f"{out_dir}/extra.json") and not os.path.exists(f"{out_dir}/extra_grid1.json"):
                shutil.copy(f"{out_dir}/extra.json", f"{out_dir}/extra_grid1.json")  # keep the main protocol's values
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
            np.save(f"{out_dir}/inception_{weights}_{sampler}{nfe}_{kk}_n{args.nC}.npy", feats.half().numpy())
        obj = [fid]; dist.broadcast_object_list(obj, src=0); fid = obj[0]
        f2 = None
        if route == "imf":  # robustness metric at the new point
            res2, _, _ = ev.evaluate(model, sampler, 2, best_p, args.nB, 1)
            if rank == 0:
                f2 = res2["fid"]
                with open(f"{out_dir}/results.jsonl", "a") as f:
                    f.write(json.dumps(dict(meta, **res2, weights=weights, stage="C2nfex")) + "\n")
        if rank == 0:
            prev = {k: summ[k] for k in ("point", "fid50k", "fid10k_sel", "fid_2nfe_10k") if k in summ}
            grid1 = summ.get("grid1", prev)  # the main protocol's values stay under grid1
            summ.update(grid1=grid1, point=best_p, fid50k=fid, fid10k_sel=best10)
            if f2 is not None:
                summ["fid_2nfe_10k"] = f2
            rec.update(new_fid50k=fid, old_fid50k=prev["fid50k"], grid1_fid50k=grid1["fid50k"])
    if rank == 0:
        summ["gridext"] = True
        json.dump(summ, open(f"{out_dir}/summary.json", "w"), indent=1)
        json.dump(rec, open(f"{out_dir}/gridext.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in rec.items() if k != "tried"}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
