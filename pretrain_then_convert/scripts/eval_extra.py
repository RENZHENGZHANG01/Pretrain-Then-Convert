"""Secondary metrics for an already evaluated run: regenerate the stage-C samples (same selected
weights / guidance point / seed 0 / 50k) and compute FD-DINOv2, FID vs the ADM reference batch,
precision/recall (k=3, Inception pool, ADM reference images) and a re-check of FID (JiT stats).

  torchrun --nproc_per_node G scripts/eval_extra.py --run RUN_ID
Writes eval/<run>/extra.json and dinov2_*.npy (fp16 features).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osr.paths import ADM_FEATS, EVAL, RUNS  # noqa: E402
from osr.evaluate import Evaluator, load_run_model  # noqa: E402
from osr.extra_metrics import ADM_REF, DINO_REF, DinoV2, frechet, precision_recall, stats  # noqa: E402

ADM_FEATS = ADM_FEATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=50000)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out_dir = f"{EVAL}/{args.run}"
    summ = json.load(open(f"{out_dir}/summary.json"))
    ck = f"{RUNS}/{args.run}/final.pt"
    if not os.path.exists(ck):
        ck = f"{RUNS}/{args.run}/final_sel.pt"
    st = torch.load(ck, map_location="cpu", weights_only=False)
    model = load_run_model(st, summ["weights"], dev)
    sampler, nfe = ("meanflow", 1) if st["route"] in ("imf", "dmd", "sc") else ("fm", 50)
    ev = Evaluator(dev)
    dino = DinoV2(dev)
    feats, logits, _, dfeats = ev.generate_scores(model, sampler, nfe, summ["point"], args.n, seed=0, extra=dino)
    if rank == 0:
        m = ev.metrics(feats, logits)                       # FID vs JiT stats (re-check of fid50k)
        mu_g, s_g = stats(feats)
        adm = np.load(ADM_REF)
        fid_adm = frechet(mu_g, s_g, adm["mu"].astype(np.float64), adm["sigma"].astype(np.float64))
        dref = np.load(DINO_REF)
        mu_d, s_d = stats(dfeats)
        fd_dino = frechet(mu_d, s_d, dref["mu"].astype(np.float64), dref["sigma"].astype(np.float64))
        pr = precision_recall(torch.from_numpy(np.load(ADM_FEATS)), feats.float(), k=3, dev=dev)
        rec = dict(run_id=args.run, weights=summ["weights"], point=summ["point"], n=args.n,
                   fid_jit_recheck=m["fid"], fid50k_stageC=summ["fid50k"], fid_adm=fid_adm, fd_dinov2=fd_dino,
                   inception_score=m["inception_score"], **pr)
        json.dump(rec, open(f"{out_dir}/extra.json", "w"), indent=1)
        np.save(f"{out_dir}/dinov2_{summ['weights']}_n{args.n}.npy", dfeats.half().numpy())
        print(json.dumps(rec), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
