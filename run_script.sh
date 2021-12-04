#!/bin/bash
# Job name:
#SBATCH --job-name=SMAAC
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
#SBATCH --time=04:00:00
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
## Command(s) to run (example):
export PYTHONUNBUFFERED=1
conda activate maze_smaac
cd SMAAC
python test.py -n=wcci_run -s=0 -c=wcci
