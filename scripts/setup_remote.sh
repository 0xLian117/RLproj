#!/usr/bin/env bash
# One-time environment setup on the remote GPU box.
#
#   bash scripts/setup_remote.sh
#
# Creates a venv that REUSES the base image's CUDA torch (--system-site-packages)
# and installs the rest. vLLM is attempted but optional (training falls back to
# HuggingFace generate if it is missing).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== RLVE-lite setup in $ROOT =="

PYBIN="${PYBIN:-python3}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
  echo "== creating venv ($VENV, reusing system site-packages for CUDA torch) =="
  "$PYBIN" -m venv "$VENV" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip setuptools wheel

echo "== installing core dependencies =="
pip install "transformers>=4.51" "trl>=0.18" "datasets>=2.19" \
            "accelerate>=0.34" "peft>=0.12" "numpy>=1.24" "matplotlib>=3.7"

# Editable install so `import rlve` works from anywhere.
pip install -e . || true

echo "== checking torch / CUDA =="
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available — training will be extremely slow.")
except Exception as e:
    print("ERROR importing torch:", e)
PY

echo "== attempting to install vLLM (optional) =="
if pip install "vllm"; then
  python -c "import vllm; print('vLLM', vllm.__version__, 'installed OK')" \
    || echo "vLLM imported with issues; training will fall back to HF generate."
else
  echo "vLLM install failed — that's OK, training falls back to HF generate."
fi

echo "== running CPU self-test (no GPU needed) =="
PYTHONPATH=. python tools/selftest.py

echo
echo "== setup complete. Next: bash scripts/run_all.sh =="
