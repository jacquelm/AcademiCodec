#!/bin/sh
#SBATCH --job-name=speechtoken_test     # job name
#SBATCH --output=job/speechtoken%j.out  # output file name
#SBATCH --error=job/speechtoken%j.err   # error file name
#SBATCH --constraint=a100            # reserve 80 GB A100 GPUs
#SBATCH --nodes=1                    # reserve x nodes
#SBATCH --ntasks=1                   # reserve x tasks (or processes)
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # reserve x GPUs per node
#SBATCH --cpus-per-task=8           # reserve x CPUs per task (and associated memory)
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
# #SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
# #SBATCH --exclusive                  # reserve the nodes
#SBATCH --qos=qos_gpu_a100-t3        # reserve the partition with a100 gpus

# cd ${SLURM_SUBMIT_DIR}

module purge                         # purgemodules inherited by default
module load arch/a100

conda activate speechcodec

set -x                               # activate echo of launched commands
python -u scripts/train_speechtokenizer.py -c configs/speechtokenizer/spt_mhubert.json

