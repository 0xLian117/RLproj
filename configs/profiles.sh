#!/usr/bin/env bash
# Hardware profiles + experiment matrix, sourced by scripts/run_all.sh.
# Override any variable from the environment, e.g.:
#   PROFILE=4090 MAX_STEPS=150 bash scripts/run_all.sh

# ----------------------------------------------------------------------------
# Hardware profile. Default targets a single H200 (~141GB): full fine-tuning of
# a 1.5B model with vLLM colocated. The 4090 profile uses LoRA + smaller batches
# so it fits on 24GB cards.
# ----------------------------------------------------------------------------
PROFILE="${PROFILE:-h200}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-200}"

case "$PROFILE" in
  h200)
    NUM_GEN="${NUM_GEN:-8}"
    PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-8}"     # -> pdtbs = 64
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-384}"
    MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-640}"
    VLLM_MEM="${VLLM_MEM:-0.35}"
    LORA_FLAG="${LORA_FLAG:-}"                     # full fine-tune
    ;;
  4090)
    # Two 4090s OR a single 24GB card. LoRA keeps memory in check.
    NUM_GEN="${NUM_GEN:-6}"
    PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"      # -> pdtbs = 24
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-320}"
    MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-512}"
    VLLM_MEM="${VLLM_MEM:-0.40}"
    LORA_FLAG="${LORA_FLAG:---lora}"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (use 'h200' or '4090')" >&2; exit 1;;
esac

# ----------------------------------------------------------------------------
# Experiment matrix: "tag|controller|sampler|extra-args"
#   static     : fixed medium difficulty (degenerate baseline)
#   threshold  : faithful RLVE adaptive controller (Zeng et al., 2025)
#   stad       : OUR signal-targeting controller (uniform env sampler)
#   stad_lp    : OUR controller + learning-progress env sampler (full method)
# ----------------------------------------------------------------------------
CONDITIONS=(
  "static|static|uniform|--static-level 4"
  "threshold|threshold|uniform|"
  "stad|stad|uniform|"
  "stad_lp|stad|lp|"
)

# Evaluation set size (problems per env per difficulty level).
EVAL_N_PER="${EVAL_N_PER:-16}"
