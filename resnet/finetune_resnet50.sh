#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

#SBATCH --time=03:00:00

#SBATCH -J "resnet_finetune"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

mkdir -p logs

/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python finetune_resnet50.py \
    --epochs 15 \
    --batch_size 32 \
    --output_dir checkpoints