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

# Use a fast PyPI mirror by default (Tsinghua). Override with PIP_INDEX=...,
# or set PIP_INDEX="" to use the default PyPI.
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_ARGS=()
if [ -n "$PIP_INDEX" ]; then
  PIP_HOST="$(echo "$PIP_INDEX" | sed -E 's#https?://([^/]+)/.*#\1#')"
  PIP_ARGS=(-i "$PIP_INDEX" --trusted-host "$PIP_HOST")
  echo "== using pip index: $PIP_INDEX =="
fi

# NO_VENV=1 -> install into the currently active env (e.g. a conda env that
# already has CUDA torch). Recommended on clusters where you already have a
# working conda env. Otherwise we create a venv that reuses system torch.
if [ "${NO_VENV:-0}" = "1" ]; then
  echo "== NO_VENV=1: installing into the active Python env: $(which python) =="
else
  if [ ! -f "$VENV/bin/activate" ]; then
    rm -rf "$VENV"   # remove any half-created venv
    echo "== creating venv ($VENV, reusing system site-packages for CUDA torch) =="
    "$PYBIN" -m venv "$VENV" --system-site-packages
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

python -m pip install "${PIP_ARGS[@]}" -U pip setuptools wheel

echo "== installing core dependencies =="
pip install "${PIP_ARGS[@]}" "transformers>=4.51" "trl>=0.18" "datasets>=2.19" \
            "accelerate>=0.34" "peft>=0.12" "numpy>=1.24" "matplotlib>=3.7"

# Editable install so `import rlve` works from anywhere.
pip install "${PIP_ARGS[@]}" -e . || true

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

echo "== vLLM (optional, faster generation) =="
if python -c "import vllm" 2>/dev/null; then
  python -c "import vllm; print('vLLM already present:', vllm.__version__)"
elif [ "${WITH_VLLM:-0}" = "1" ]; then
  echo "WITH_VLLM=1: installing vLLM (NOTE: this may change the torch version)"
  if pip install "${PIP_ARGS[@]}" "vllm"; then
    python -c "import vllm; print('vLLM', vllm.__version__, 'installed OK')" \
      || echo "vLLM imported with issues; training will fall back to HF generate."
  else
    echo "vLLM install failed — that's OK, training falls back to HF generate."
  fi
else
  echo "vLLM not installed and WITH_VLLM!=1 -> training will use HF generate"
  echo "(slower). To install it (may alter torch): WITH_VLLM=1 bash scripts/setup_remote.sh"
fi

echo "== running CPU self-test (no GPU needed) =="
PYTHONPATH=. python tools/selftest.py

echo
echo "== setup complete. Next: bash scripts/run_all.sh =="
