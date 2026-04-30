#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH -J "resnet_multiview_mse"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-multiview-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-multiview-%j.err

cd /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/train

PYTHON_BIN=/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python

$PYTHON_BIN train_multiview.py \
    --epochs 50 \
    --batch_size 128 \
    --lr 5e-5 \
    --dropout 0.4 \
    --img_size 512 \
    --unfreeze_epoch 6 \
    --patience 7 \
    --num_workers 8 \
    --skip_frontal \
    --consistency_lambda 0.0 \
    --output_dir /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/checkpoints_v3 \
    --train_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG/manifests_preprocessed/train_manifest_preprocessed.csv \
    --val_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG/manifests_preprocessed/val_manifest_preprocessed.csv