#!/usr/bin/env bash
# ============================================================================
# Distributed (DDP) training across N GPUs for ONE model, via `accelerate`.
# Both cards train (data-parallel); each rank generates from its own copy of the
# adaptive curriculum and only rank 0 writes logs. Generation defaults to
# HuggingFace (NO_VLLM=1) to avoid the vLLM/driver version maze — robust but
# slower. Set NO_VLLM=0 to use vLLM colocate if you have a TRL-compatible vLLM.
#
#   OUT_ROOT=/big/disk/rlve_out MODEL=/path/to/Qwen3.5-4B bash scripts/run_ddp.sh
#
# Knobs: NPROC (default 2), MODEL, OUT_ROOT, MAX_STEPS, PROMPTS_PER_STEP,
#        MAX_COMPLETION_LEN, NO_VLLM (default 1), EVAL_N_PER.
#
# TIP: do a quick end-to-end validation first with a tiny run, then scale up:
#   MAX_STEPS=20 EVAL_N_PER=4 OUT_ROOT=... MODEL=... bash scripts/run_ddp.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NPROC="${NPROC:-2}"
export MODEL="${MODEL:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
export OUT_ROOT="${OUT_ROOT:-.}"
export RESULTS_DIR="${RESULTS_DIR:-$OUT_ROOT/results_ddp}"
export PROFILE="${PROFILE:-4090x48}"
export MAX_STEPS="${MAX_STEPS:-200}"
export NO_VLLM="${NO_VLLM:-1}"           # HF generate by default (robust)
# Per-rank batch: with DDP the global batch = PROMPTS_PER_STEP * NPROC, so a
# 4B fits more easily (each card holds half the work).
export PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
# Run training under accelerate (DDP). Eval/sim stay single-process.
export TRAIN_LAUNCHER="accelerate launch --num_processes ${NPROC} --multi_gpu rlve/train.py"
# Do NOT pin a single GPU — DDP needs all of them visible.
unset GPU || true

echo "== DDP run: NPROC=$NPROC model=$MODEL no_vllm=$NO_VLLM out=$RESULTS_DIR =="
echo "   global prompts/step = $PROMPTS_PER_STEP x $NPROC"
bash scripts/run_all.sh
