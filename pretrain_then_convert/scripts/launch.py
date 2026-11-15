"""Run the experiments of the paper (osr/experiments.py).

  python scripts/launch.py list  [GROUP ...]                 runs of the groups (and their dependencies) with their state
  python scripts/launch.py train RUN_ID [--gpus N] [--dry]   train one run on this machine (torchrun)
  python scripts/launch.py eval  RUN_ID [--gpus N] [--dry]   evaluate one finished run (scripts/eval_run.py)
  python scripts/launch.py slurm [GROUP ...] [--eval] [--submit] [--template slurm/train.sbatch]
                                                             one SLURM script per run that is ready to train (or, with
                                                             --eval, finished and not yet evaluated); --submit sbatches

States: done (status.txt says done), ready (every checkpoint the run starts from exists), started (status.txt exists,
not done: running, preempted or stopped; training resumes from its last checkpoint), waiting (a dependency is
missing). A run starts from a milestone of its parent; start runs in dependency order (`list` prints them in that
order) and re-run `slurm` as parents finish. The trainer saves a resumable checkpoint every 20 minutes and on
SIGTERM / SIGUSR1, and requeues itself under SLURM.
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from osr import experiments as X  # noqa: E402
from osr.paths import EVAL, RUNS  # noqa: E402

TFLOPS_PER_GPU = 150e12  # algorithmic training throughput used for the time limit (H100, compiled; see bench_step.py)


def status(run_id):
    p = f"{RUNS}/{run_id}/status.txt"
    return open(p).read().strip() if os.path.exists(p) else ""


def needed_files(run):
    """Checkpoints a run starts from."""
    if run["kind"] == "distill":
        cfg = X.config_for(run)
        return [cfg["dmd"]["teacher"], cfg["dmd"]["student_init"]]
    return [X.init_path(run["parent"])] if run.get("parent") else []


def state(run):
    s = status(run["run_id"])
    if s.startswith("done"):
        return "done"
    if s:
        return "started"
    return "ready" if all(os.path.exists(p) for p in needed_files(run)) else "waiting"


def gpus(run, override=None):
    return override or run.get("gpus") or X.GPUS[run["size"]]


def train_cmd(run, n_gpus):
    cfg = X.config_for(run)
    os.makedirs(f"{RUNS}/_configs", exist_ok=True)
    path = f"{RUNS}/_configs/{run['run_id']}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    entry = cfg.get("entry", "osr.train")
    return f"torchrun --standalone --nproc_per_node {n_gpus} -m {entry} --config {path}"


def eval_cmd(run, n_gpus):
    return f"torchrun --standalone --nproc_per_node {n_gpus} scripts/eval_run.py --run {run['run_id']}"


def run_here(cmd, dry):
    print(cmd, flush=True)
    if not dry:
        sys.exit(subprocess.call(cmd, shell=True, cwd=REPO))


def write_slurm(run, cmd, n_gpus, template, kind):
    hours = 4 if kind == "eval" else max(2, min(48, int(1.3 * X.own_flops(run) / (n_gpus * TFLOPS_PER_GPU) / 3600) + 1))
    os.makedirs(f"{RUNS}/_slurm", exist_ok=True)
    name = f"{kind}_{run['run_id']}"
    text = open(template).read().format(job_name=name, gpus=n_gpus, cpus=min(96, 12 * n_gpus), mem=f"{64 * n_gpus}G",
                                        hours=hours, log=f"{RUNS}/_slurm/{name}-%j.out", repo=REPO, command=cmd)
    path = f"{RUNS}/_slurm/{name}.sbatch"
    with open(path, "w") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "train", "eval", "slurm"])
    ap.add_argument("targets", nargs="*", help="groups (list, slurm) or one run id (train, eval)")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="print the command without running it")
    ap.add_argument("--eval", action="store_true", help="slurm: evaluation jobs for finished runs")
    ap.add_argument("--submit", action="store_true", help="slurm: submit the written scripts with sbatch")
    ap.add_argument("--template", default=f"{REPO}/slurm/train.sbatch")
    args = ap.parse_args()

    if args.action in ("train", "eval"):
        runs = X.all_runs()
        assert len(args.targets) == 1 and args.targets[0] in runs, "give one run id (python scripts/launch.py list)"
        run = runs[args.targets[0]]
        if args.action == "train":
            missing = [p for p in needed_files(run) if not os.path.exists(p)]
            assert not missing or args.dry, f"missing start checkpoints {missing}: train the parent runs first"
            run_here(train_cmd(run, gpus(run, args.gpus)), args.dry)
        else:
            assert status(run["run_id"]).startswith("done") or args.dry, "the run has not finished training"
            run_here(eval_cmd(run, args.gpus or 2), args.dry)
        return

    groups = tuple(args.targets) or X.GROUPS
    runs = X.plan(groups)
    if args.action == "list":
        counts = {}
        for r in runs:
            st = state(r)
            ev = " evaluated" if os.path.exists(f"{EVAL}/{r['run_id']}/DONE") else ""
            counts[st] = counts.get(st, 0) + 1
            print(f"{r['run_id']:44s} {r['group']:13s} {r['kind']:9s} C={r['flops_end']:.2e} gpus={gpus(r)}  {st}{ev}")
        print(f"{len(runs)} runs:", ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        return

    written = []
    for r in runs:
        st = state(r)
        if args.eval:
            if st != "done" or r["kind"] in ("mainline", "extend") or os.path.exists(f"{EVAL}/{r['run_id']}/DONE"):
                continue
            n = args.gpus or 2
            written.append(write_slurm(r, eval_cmd(r, n), n, args.template, "eval"))
        elif st in ("ready", "started"):
            n = gpus(r, args.gpus)
            written.append(write_slurm(r, train_cmd(r, n), n, args.template, "train"))
    for p in written:
        print(p)
        if args.submit:
            subprocess.call(["sbatch", p])
    print(f"{len(written)} job scripts" + (" submitted" if args.submit else f" in {RUNS}/_slurm (submit with --submit)"))


if __name__ == "__main__":
    main()
