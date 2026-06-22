#!/usr/bin/env bash
# ============================================================================
# Run ONE 4B model using BOTH 48GB cards:
#   * GPU 1 (GEN_GPU)  : a dedicated vLLM generation server (`trl vllm-serve`)
#   * GPU 0 (TRAIN_GPU): single-process GRPO training (our adaptive curriculum)
#
# This is the clean two-card setup for a single model: training stays
# single-process (no DDP), and freeing the training card from a colocated vLLM
# lets the 4B use a full-size batch.
#
#   OUT_ROOT=/big/disk/rlve_out  bash scripts/run_4b_2gpu.sh
#
# Env knobs: MODEL, OUT_ROOT, TRAIN_GPU, GEN_GPU, PORT, MAX_STEPS,
#            PROMPTS_PER_STEP, MAX_COMPLETION_LEN.
# Fallback: if the server won't start, just run on one card with
#   GPU=0 MODEL=... PROFILE=4090x48 PROMPTS_PER_STEP=4 VLLM_MEM=0.4 \
#     bash scripts/run_all.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
OUT_ROOT="${OUT_ROOT:-.}"
TRAIN_GPU="${TRAIN_GPU:-0}"
GEN_GPU="${GEN_GPU:-1}"
PORT="${PORT:-8000}"
MAX_STEPS="${MAX_STEPS:-200}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

mkdir -p "$OUT_ROOT"
SERVER_LOG="$OUT_ROOT/vllm_server.log"

echo "== starting vLLM server on GPU $GEN_GPU (model=$MODEL, port=$PORT) =="
CUDA_VISIBLE_DEVICES="$GEN_GPU" trl vllm-serve --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "== stopping vLLM server (pid $SERVER_PID) =="; kill $SERVER_PID 2>/dev/null' EXIT

echo "== waiting for the server to come up (model load can take a few minutes) =="
READY=0
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "!! vLLM server process died. Last lines of $SERVER_LOG:"; tail -n 30 "$SERVER_LOG"
    echo "!! Falling back is recommended: run on one card (see header)."; exit 1
  fi
  if grep -qE "Uvicorn running on|Application startup complete|vLLM is ready" "$SERVER_LOG" 2>/dev/null; then
    READY=1; echo "== server is up (after ~$((i*5))s) =="; break
  fi
  sleep 5
done
[ "$READY" = "1" ] || echo "WARN: didn't detect a readiness line in $SERVER_LOG; proceeding anyway (the smoke test will confirm the connection)."

echo "== launching 4B training on GPU $TRAIN_GPU (server mode) =="
GPU="$TRAIN_GPU" MODEL="$MODEL" RESULTS_DIR="$OUT_ROOT/results_4b" \
  PROFILE=4090x48 MAX_STEPS="$MAX_STEPS" \
  VLLM_MODE=server VLLM_HOST=127.0.0.1 VLLM_PORT="$PORT" \
  bash scripts/run_all.sh

echo "==================================================="
echo " 4B run done -> $OUT_ROOT/results_4b/REPORT.md (+ $OUT_ROOT/results_4b_results.tgz)"
echo " vLLM server log: $SERVER_LOG"
echo "==================================================="
