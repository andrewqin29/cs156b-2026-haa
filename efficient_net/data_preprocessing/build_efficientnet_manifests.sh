#!/bin/bash
set -euo pipefail

# Adjust python path for your environment if needed.
PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_efficientnet_manifests.py" \
  --output_root /resnick/groups/CS156b/from_central/2026/haa/efficient_net_data \
  --val_split 0.15 \
  --seed 42

echo "Manifests created under /resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests"
