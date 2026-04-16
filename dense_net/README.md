# DenseNet Pipeline

This directory contains a manifest-driven DenseNet transfer learning workflow
for the 9-label multilabel chest X-ray task.

## Layout

- `common.py`: shared label constants, masked loss, metrics, and utilities
- `data.py`: manifest-aware dataset and transforms
- `model.py`: DenseNet model builder and freeze/unfreeze helpers
- `data_preprocessing/build_densenet_manifests.py`: patient-level train/val/test manifest builder
- `data_preprocessing/prepare_densenet_images.py`: optional cached-image builder
- `train/train_densenet.py`: training entrypoint
- `inference/densenet_inference.py`: inference entrypoint

## Design choices

- Multilabel setup with sigmoid outputs
- Missing or uncertain labels encoded with sentinel `-999`
- Patient-level train/val split handled in manifest generation
- Frontal-only filtering enabled by default in manifest generation
- ImageNet normalization and transfer learning from `torchvision` DenseNet models
- No hardcoded cluster paths inside training or inference scripts

## Example commands

```bash
python -m dense_net.train.train_densenet \
  --output_dir /resnick/groups/CS156b/from_central/2026/haa/dense_net_runs/densenet121_run1
```

```bash
python -m dense_net.inference.densenet_inference \
  --checkpoint /resnick/groups/CS156b/from_central/2026/haa/dense_net_runs/densenet121_run1/best_model.pt \
  --output /resnick/groups/CS156b/from_central/2026/haa/dense_net_runs/densenet121_run1/submission.csv
```

## Notes

- `densenet121` is the default backbone.
- Training and inference default to the shared preprocessed manifests already
  created under `/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed`.
- DenseNet accepts either raw manifests with `abs_path` or cached manifests with
  `preprocessed_path`.
- For cluster jobs, prefer `python -m ...` from the repo root so imports stay stable.
