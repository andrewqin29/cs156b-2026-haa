#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu

#SBATCH --time=05:00:00

#SBATCH -J "resnet_finetune_alex"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-alex-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-alex-%j.err

mkdir -p /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs

cd /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/train

/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python resnet_full.py \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-5 \
    --img_size 512 \
    --output_dir /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/checkpoints_512_multiview \
    --train_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_neg_FIXED/manifests_preprocessed/train_manifest_preprocessed.csv \
    --val_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_neg_FIXED/manifests_preprocessed/val_manifest_preprocessed.csv
