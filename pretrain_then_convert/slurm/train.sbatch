#!/bin/bash
# Template used by scripts/launch.py slurm (training and evaluation jobs). Add the account / partition / QOS lines of
# your cluster. The trainer saves a checkpoint and requeues itself on SIGUSR1 (sent 5 minutes before the time limit).
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={log}

cd {repo}
{command}
