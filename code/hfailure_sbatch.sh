#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 2-01:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH -o /base_parent_path/output/hfailure_slurm.%j.out
#SBATCH -e /base_parent_path/output/hfailure_slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.mamun@asu.edu

#SBATCH --export=NONE

module load mamba/latest
source activate tensorflow-gpu-2.10.0
pip install imblearn
python main.py --exp_name full_cycle_1 --task train_all --arch lstm --num_epochs 10 --dataset nih --seed 42
