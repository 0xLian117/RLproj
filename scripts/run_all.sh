#!/usr/bin/env bash
# ============================================================================
# RLVE-lite: full experiment launcher (run this on the remote GPU box).
#
#   bash scripts/run_all.sh                 # default: single H200, 200 steps
#   PROFILE=4090 bash scripts/run_all.sh    # 24GB cards (LoRA, smaller batch)
#   MAX_STEPS=120 bash scripts/run_all.sh   # shorter run
#   SKIP_SMOKE=1 bash scripts/run_all.sh    # skip the pre-flight smoke test
#
# Produces (under results/):
#   sim/            CPU mechanism-validation logs
#   runs/<cond>/    training logs (+ model weights, git-ignored)
#   eval/<tag>.json per-condition accuracy on the fixed eval set
#   figures/*.png   plots,  REPORT.md  summary,  logs/  raw stdout
# At the end it writes rlve_results.tgz (lightweight: logs+figures+jsons, NO
# model weights) — copy that back, or `git add` the kept results files.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate || true
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM=false

# shellcheck disable=SC1091
source configs/profiles.sh

mkdir -p results/logs results/runs results/eval results/sim
START=$(date +%s)
echo "================ RLVE-lite run_all ================"
echo " profile=$PROFILE  model=$MODEL  max_steps=$MAX_STEPS"
echo " num_gen=$NUM_GEN  prompts/step=$PROMPTS_PER_STEP  lora='${LORA_FLAG}'"
echo "==================================================="

fatal() { echo "FATAL: $*" >&2; exit 1; }

# --- 0. CPU self-test (fail fast) -------------------------------------------
echo "### [0/6] self-test"
python tools/selftest.py 2>&1 | tee results/logs/selftest.log
[ "${PIPESTATUS[0]}" -eq 0 ] || fatal "self-test failed"

# --- 1. CPU simulation (mechanism validation, always works) -----------------
echo "### [1/6] CPU simulation"
python tools/simulate.py --out results/sim 2>&1 | tee results/logs/sim.log

# --- 2. smoke test (fail fast on GPU pipeline) ------------------------------
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "### [2/6] GPU smoke test"
  MODEL="$MODEL" LORA_FLAG="$LORA_FLAG" bash scripts/smoke_test.sh \
    2>&1 | tee results/logs/smoke.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || fatal "smoke test failed — fix before the full run"
else
  echo "### [2/6] smoke test SKIPPED"
fi

# --- 3. fixed eval set + base-model eval ------------------------------------
echo "### [3/6] build eval set + evaluate base model"
python -m rlve.eval_set --out results/eval_set.json --n-per "$EVAL_N_PER" \
  2>&1 | tee results/logs/eval_set.log
python -m rlve.evaluate --model "$MODEL" --tag base \
  --eval-set results/eval_set.json --n-per "$EVAL_N_PER" \
  --max-tokens "$MAX_COMPLETION_LEN" --out-dir results/eval \
  2>&1 | tee results/logs/eval_base.log || echo "WARN: base eval failed"

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
      --output-dir "results/runs/$tag" 2>&1 | tee "results/logs/train_$tag.log"; then
    if python -m rlve.evaluate --model "results/runs/$tag" --tag "$tag" \
        --eval-set results/eval_set.json --n-per "$EVAL_N_PER" \
        --max-tokens "$MAX_COMPLETION_LEN" --out-dir results/eval \
        2>&1 | tee "results/logs/eval_$tag.log"; then
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
python tools/plot_results.py --results results 2>&1 | tee results/logs/plot.log

# --- 6. package lightweight results (no model weights) ----------------------
echo "### [6/6] package results"
tar czf rlve_results.tgz \
  --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
  --exclude='results/eval/_work_*' --exclude='results/runs/*/merged' \
  --exclude='results/runs/*/checkpoint-*' \
  results/sim results/eval results/figures results/REPORT.md \
  results/logs results/runs 2>/dev/null || true

END=$(date +%s)
echo "==================================================="
echo " DONE in $(( (END-START)/60 )) min. Conditions OK: ${OK_CONDS[*]:-none}"
echo " Summary report : results/REPORT.md"
echo " Figures        : results/figures/"
echo " Return file    : rlve_results.tgz  (copy this back / commit kept files)"
echo "==================================================="
cat results/REPORT.md 2>/dev/null || true
