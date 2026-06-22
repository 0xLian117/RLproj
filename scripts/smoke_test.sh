#!/usr/bin/env bash
# Tiny end-to-end GPU validation BEFORE the real (multi-hour) run.
# Trains STAD for 3 steps with a tiny batch and runs a tiny eval, so any
# TRL / vLLM / model issue surfaces in ~2 minutes instead of hours.
#
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate || true
export PYTHONPATH="${PYTHONPATH:-.}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LORA_FLAG="${LORA_FLAG:-}"
SMOKE_DIR="${SMOKE_DIR:-results/runs/_smoke}"

echo "== SMOKE: tiny GRPO train (model=$MODEL lora='${LORA_FLAG}') =="
python -m rlve.train \
  --condition smoke --controller stad --sampler lp $LORA_FLAG \
  --model "$MODEL" --max-steps 3 \
  --num-generations 4 --prompts-per-step 2 \
  --max-prompt-length 256 --max-completion-length 128 \
  --vllm-gpu-mem 0.30 --logging-steps 1 \
  --output-dir "$SMOKE_DIR"

echo "== SMOKE: tiny eval =="
python -m rlve.evaluate \
  --model "$SMOKE_DIR" --tag _smoke \
  --eval-set results/_smoke_eval_set.json --n-per 2 \
  --max-tokens 128 --out-dir results/eval

echo "== SMOKE PASSED: GPU training + eval pipeline works end-to-end =="
