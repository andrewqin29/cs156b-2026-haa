#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

#SBATCH --time=00:30:00

#SBATCH -J "resnet_inference"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

mkdir -p logs

cd /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet

/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --csv /resnick/groups/CS156b/from_central/2026/haa/preprocessed_test_labels.csv \
    --output submission.csv