#!/usr/bin/env bash
# ============================================================================
# RLVE-lite: full experiment launcher (run this on the remote GPU box).
#
#   bash scripts/run_all.sh                              # single GPU, defaults
#   PROFILE=4090x48 bash scripts/run_all.sh              # one 48GB card (LoRA)
#   MAX_STEPS=120 bash scripts/run_all.sh                # shorter run
#   SKIP_SMOKE=1 bash scripts/run_all.sh                 # skip pre-flight smoke
#
#   # use a LOCAL model (no download) and pin to one GPU + its own results dir:
#   GPU=0 MODEL=/path/to/Qwen3.5-4B  RESULTS_DIR=results_4b \
#     PROFILE=4090x48 bash scripts/run_all.sh
#
# Two cards in parallel = run this twice (once per GPU) with different GPU /
# MODEL / RESULTS_DIR. See scripts/run_2gpu.sh for a ready-made launcher.
#
# Local models never download. We also export HF_HUB_OFFLINE=1 by default; set
# HF_HUB_OFFLINE=0 if you DO want to pull a model from the Hugging Face Hub.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM=false
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# shellcheck disable=SC1091
source configs/profiles.sh

RESULTS="${RESULTS_DIR:-results}"
VLLM_MODE="${VLLM_MODE:-colocate}"   # colocate | server
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
mkdir -p "$RESULTS/logs" "$RESULTS/runs" "$RESULTS/eval" "$RESULTS/sim"
START=$(date +%s)
echo "================ RLVE-lite run_all ================"
echo " profile=$PROFILE  model=$MODEL  max_steps=$MAX_STEPS"
echo " gpu='${CUDA_VISIBLE_DEVICES:-all}'  results=$RESULTS  offline=$HF_HUB_OFFLINE"
echo " num_gen=$NUM_GEN  prompts/step=$PROMPTS_PER_STEP  lora='${LORA_FLAG}'"
echo "==================================================="

fatal() { echo "FATAL: $*" >&2; exit 1; }

# --- 0. CPU self-test (fail fast) -------------------------------------------
echo "### [0/6] self-test"
python tools/selftest.py 2>&1 | tee "$RESULTS/logs/selftest.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fatal "self-test failed"

# --- 1. CPU simulation (mechanism validation, always works) -----------------
echo "### [1/6] CPU simulation"
python tools/simulate.py --out "$RESULTS/sim" 2>&1 | tee "$RESULTS/logs/sim.log"

# --- 2. smoke test (fail fast on GPU pipeline) ------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "### [2/6] GPU smoke test"
  MODEL="$MODEL" LORA_FLAG="$LORA_FLAG" RESULTS_DIR="$RESULTS" \
    VLLM_MODE="$VLLM_MODE" VLLM_HOST="$VLLM_HOST" VLLM_PORT="$VLLM_PORT" \
    bash scripts/smoke_test.sh 2>&1 | tee "$RESULTS/logs/smoke.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || fatal "smoke test failed — fix before the full run"
else
  echo "### [2/6] smoke test SKIPPED"
fi

# --- 3. fixed eval set + base-model eval ------------------------------------
echo "### [3/6] build eval set + evaluate base model"
python -m rlve.eval_set --out "$RESULTS/eval_set.json" --n-per "$EVAL_N_PER" \
  2>&1 | tee "$RESULTS/logs/eval_set.log"
python -m rlve.evaluate --model "$MODEL" --tag base \
  --eval-set "$RESULTS/eval_set.json" --n-per "$EVAL_N_PER" \
  --max-tokens "$MAX_COMPLETION_LEN" --out-dir "$RESULTS/eval" \
  2>&1 | tee "$RESULTS/logs/eval_base.log" || echo "WARN: base eval failed"

# --- 4. train + evaluate each condition -------------------------------------
echo "### [4/6] train + evaluate conditions"
OK_CONDS=()
for spec in "${CONDITIONS[@]}"; do
  IFS='|' read -r tag controller sampler extra <<< "$spec"
  echo "--- condition: $tag (controller=$controller sampler=$sampler $extra) ---"
  # shellcheck disable=SC2086
  if python -m rlve.train \
      --condition "$tag" --controller "$controller" --sampler "$sampler" \
      $LORA_FLAG $extra \
      --model "$MODEL" --max-steps "$MAX_STEPS" \
      --num-generations "$NUM_GEN" --prompts-per-step "$PROMPTS_PER_STEP" \
      --max-prompt-length "$MAX_PROMPT_LEN" \
      --max-completion-length "$MAX_COMPLETION_LEN" \
      --vllm-gpu-mem "$VLLM_MEM" \
      --vllm-mode "$VLLM_MODE" --vllm-server-host "$VLLM_HOST" --vllm-server-port "$VLLM_PORT" \
      --output-dir "$RESULTS/runs/$tag" 2>&1 | tee "$RESULTS/logs/train_$tag.log"; then
    if python -m rlve.evaluate --model "$RESULTS/runs/$tag" --tag "$tag" \
        --eval-set "$RESULTS/eval_set.json" --n-per "$EVAL_N_PER" \
        --max-tokens "$MAX_COMPLETION_LEN" --out-dir "$RESULTS/eval" \
        2>&1 | tee "$RESULTS/logs/eval_$tag.log"; then
      OK_CONDS+=("$tag")
    else
      echo "WARN: eval failed for $tag"
    fi
  else
    echo "WARN: training failed for $tag — continuing with remaining conditions"
  fi
done

# --- 5. figures + report ----------------------------------------------------
echo "### [5/6] figures + report"
python tools/plot_results.py --results "$RESULTS" 2>&1 | tee "$RESULTS/logs/plot.log"

# --- 6. package lightweight results (no model weights) ----------------------
echo "### [6/6] package results"
TARBALL="${RESULTS}_results.tgz"
tar czf "$TARBALL" \
  --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
  --exclude="$RESULTS/eval/_work_*" --exclude="$RESULTS/runs/*/merged" \
  --exclude="$RESULTS/runs/*/checkpoint-*" \
  "$RESULTS/sim" "$RESULTS/eval" "$RESULTS/figures" "$RESULTS/REPORT.md" \
  "$RESULTS/logs" "$RESULTS/runs" 2>/dev/null || true

END=$(date +%s)
echo "==================================================="
echo " DONE in $(( (END-START)/60 )) min. Conditions OK: ${OK_CONDS[*]:-none}"
echo " Summary report : $RESULTS/REPORT.md"
echo " Figures        : $RESULTS/figures/"
echo " Return file    : $TARBALL  (copy this back / commit kept files)"
echo "==================================================="
cat "$RESULTS/REPORT.md" 2>/dev/null || true
