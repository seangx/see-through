#!/usr/bin/env bash
# See-through Mac (MPS) one-shot setup
# Usage: bash setup_mac.sh

set -e

PY=${PY:-python3.12}
VENV=${VENV:-.venv}

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[setup_mac] $PY not found. Install with: brew install python@3.12"
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "[setup_mac] creating venv at $VENV with $PY..."
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "[setup_mac] upgrading pip..."
pip install -q --upgrade pip wheel setuptools

echo "[setup_mac] installing PyTorch (MPS)..."
pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"

echo "[setup_mac] installing requirements-mac.txt..."
pip install -r requirements-mac.txt

python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"mps available: {torch.backends.mps.is_available()}")
print(f"mps built:     {torch.backends.mps.is_built()}")
PY

echo
echo "[setup_mac] DONE. To run inference:"
echo "  source $VENV/bin/activate"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 python inference/scripts/inference_psd.py \\"
echo "    --srcp your_image.png --resolution 768 --save_to_psd"
