#!/bin/bash
# Job name:
#SBATCH --job-name=JOINT
#
# Account:
#SBATCH -A m3691
#
# Partition:
#SBATCH -C gpu
#
# Number of nodes:
#SBATCH -N 1
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH -c 2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH -G 1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#SBATCH -o out/slurm.%N.%j.out # STDOUT
#SBATCH -e out/slurm.%N.%j.err # STDERR
## Command(s) to run (example):
export PYTHONUNBUFFERED=1
conda activate maze_smaac
cd fp
# Evaluate
srun -N 1 -n 1 python AC_JOINT/evaluate.py -data="/data/rkunani/smaac_data" --ap=50 --c_suffix=100000 -n=joint_wcci-run-0_GAT-k20_100k_0

# srun -N 1 -n 1 --gres=gpu:1 python AC_JOINT/evaluate.py \
# --ap=50 \
# --c_suffix=20000 \
# -n=joint_wcci-run-0_GAT-k200_100k_requeue_0

# srun -N 1 -n 1 --gres=gpu:1 python AC_JOINT/evaluate.py -n=joint_wcci_run_0
# srun -N 1 -n 1 --gres=gpu:1 python AC_JOINT/evaluate.py -n=joint_wcci_run_ap20_0
# srun -N 1 -n 1 --gres=gpu:1 python AC_JOINT/evaluate.py -n=joint_wcci_run_ap200_0
