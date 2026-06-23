#!/usr/bin/env bash
# ============================================================================
# Run the full difficulty-strategy comparison on SCALER (single GPU):
#   adaptive    = SCALER's own controller          (SCALER-train.json)
#   static-lo/mid/hi = frozen difficulty            (SCALER-static-*.json)
#   freeenergy  = our Gibbs / free-energy controller (SCALER-train.json, DIFFICULTY_MODE=freeenergy)
#
# Prereqs:
#   * verl env active; SandboxFusion up on :8080; g++ available
#   * arms generated:  python scaler_addon/scaler_make_arms.py --in SCALER-data/train/SCALER-8.json --out arms --n-train 5
#   * patch applied:   python scaler_addon/apply_freeenergy_patch.py --scaler <SCALER_DIR>
#
# Usage:
#   SCALER_DIR=/path/to/SCALER MODEL=/path/to/Qwen2.5-3B-Instruct bash scaler_addon/run_arms.sh
# ============================================================================
set -uo pipefail

SCALER_DIR=${SCALER_DIR:-$PWD}
ADDON=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-3B-Instruct}
CKPT_ROOT=${CKPT_ROOT:-/inspire/hdd/global_user/chenglian-253104020001/ckpts}
STEPS=${STEPS:-40}
GPU=${GPU:-0}
ARM_SH="$ADDON/scaler_arm.sh"

cd "$SCALER_DIR"
mkdir -p ~/runs_out "$CKPT_ROOT"

# sandbox check
python -c "import socket,sys;sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8080))==0 else 1)" \
  || { echo "ERROR: SandboxFusion not on :8080"; exit 1; }
# arms check
for a in train static-lo static-mid static-hi; do
  [ -f "arms/SCALER-$a.json" ] || { echo "ERROR: missing arms/SCALER-$a.json (run scaler_make_arms.py)"; exit 1; }
done

run () {  # name  arm-json-suffix  difficulty_mode
  local name=$1 suf=$2 mode=$3
  echo "######## RUN $name (mode=$mode) $(date) ########"
  ray stop --force 2>/dev/null; pkill -9 -f raylet 2>/dev/null; sleep 3
  DIFFICULTY_MODE="$mode" \
  GPU="$GPU" STEPS="$STEPS" exp_name="$name" \
  RAY_DATA_HOME="$SCALER_DIR" MODEL_PATH="$MODEL" \
  CKPTS_DIR="$CKPT_ROOT/$name" \
  TRAIN_FILE="$SCALER_DIR/arms/SCALER-$suf.json" \
    bash "$ARM_SH" 2>&1 | tee ~/runs_out/$name.log
  echo "######## DONE $name exit=${PIPESTATUS[0]} $(date) ########"
}

# 4 SCALER-controller arms + 1 free-energy arm
run adaptive    train      ""
run static_lo   static-lo  ""
run static_mid  static-mid ""
run static_hi   static-hi  ""
run freeenergy  train      freeenergy

echo "ALL DONE. logs: ~/runs_out/  |  analyze: python $ADDON/analyze.py --logs ~/runs_out --out results --G 8"
