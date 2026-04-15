#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

#SBATCH --time=04:00:00

#SBATCH -J "resnet_finetune_alex"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/logs/slurm-alex-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/logs/slurm-alex-%j.err

mkdir -p /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/logs

cd /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/train

/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python finetune_resnet50_alex_preprocessed.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 5e-5 \
    --output_dir /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/checkpoints_resnet_alex_manifests \
    --train_csv /resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/train_manifest_preprocessed.csv \
    --val_csv /resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/val_manifest_preprocessed.csv
