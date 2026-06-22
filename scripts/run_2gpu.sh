#!/usr/bin/env bash
# ============================================================================
# Two-GPU launcher: run TWO independent single-GPU experiments in parallel,
# one model per card. This is the most robust way to use two cards — each job
# is the fully-validated single-GPU colocate path, with no multi-process /
# DDP complications, and you get a model-size comparison (4B vs 0.5B) for free.
#
#   bash scripts/run_2gpu.sh
#
# Override the local model paths if needed:
#   MODEL_A=/path/to/big   MODEL_B=/path/to/small   bash scripts/run_2gpu.sh
#
# IMPORTANT — storage: training/eval write large files (LoRA adapters are small,
# but each eval briefly merges a FULL model to disk). If your CODE directory is
# on a small disk, point OUT_ROOT at a big disk (e.g. where your conda env lives)
# so nothing large is written next to the code:
#   OUT_ROOT=/inspire/hdd/global_user/<you>/rlve_out  bash scripts/run_2gpu.sh
#
# GPU 0 -> MODEL_A (bigger), results in  $OUT_ROOT/results_a/
# GPU 1 -> MODEL_B (smaller), results in $OUT_ROOT/results_b/
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# Local model paths (NO download). Edit these to your actual paths.
MODEL_A="${MODEL_A:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
MODEL_B="${MODEL_B:-/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-0.5B-Instruct}"
PROFILE="${PROFILE:-4090x48}"
MAX_STEPS="${MAX_STEPS:-200}"
OUT_ROOT="${OUT_ROOT:-.}"        # put results on a BIG disk if code disk is small
mkdir -p "$OUT_ROOT"

echo "== launching 2 parallel jobs (out_root=$OUT_ROOT) =="
echo "  GPU0: $MODEL_A  -> $OUT_ROOT/results_a/"
echo "  GPU1: $MODEL_B  -> $OUT_ROOT/results_b/"

# Job A on GPU 0
GPU=0 MODEL="$MODEL_A" RESULTS_DIR="$OUT_ROOT/results_a" PROFILE="$PROFILE" MAX_STEPS="$MAX_STEPS" \
  bash scripts/run_all.sh > "$OUT_ROOT/results_a_console.log" 2>&1 &
PID_A=$!

# Job B on GPU 1
GPU=1 MODEL="$MODEL_B" RESULTS_DIR="$OUT_ROOT/results_b" PROFILE="$PROFILE" MAX_STEPS="$MAX_STEPS" \
  bash scripts/run_all.sh > "$OUT_ROOT/results_b_console.log" 2>&1 &
PID_B=$!

echo "  job A pid=$PID_A (tail -f $OUT_ROOT/results_a_console.log)"
echo "  job B pid=$PID_B (tail -f $OUT_ROOT/results_b_console.log)"
echo "== waiting for both to finish (Ctrl-C will NOT stop the jobs; use kill) =="

RC=0
wait $PID_A || { echo "job A (GPU0, $MODEL_A) exited non-zero"; RC=1; }
wait $PID_B || { echo "job B (GPU1, $MODEL_B) exited non-zero"; RC=1; }

echo "==================================================="
echo " BOTH JOBS DONE."
echo "   $OUT_ROOT/results_a/REPORT.md  (+ $OUT_ROOT/results_a_results.tgz)   model: $MODEL_A"
echo "   $OUT_ROOT/results_b/REPORT.md  (+ $OUT_ROOT/results_b_results.tgz)   model: $MODEL_B"
echo "==================================================="
exit $RC
