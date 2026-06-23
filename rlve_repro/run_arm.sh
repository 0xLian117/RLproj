#!/usr/bin/env bash
# ============================================================================
# RLVE 单卡(H200)一臂训练。在【RLVE 仓库根目录】运行(SLIME 容器内)。
# 把官方 8 卡脚本就地改单卡 + 指向你的模型 + 按 ARM 设置难度策略。
#
#   ARM=adaptive   NUM_ENV=16 bash run_arm.sh RLVE          # RLVE 基线(acc≥0.9 升难)
#   ARM=static     NUM_ENV=16 STATIC_D=4 bash run_arm.sh RLVE   # 静态难度(冻结在 d=4)
#   ARM=signal     NUM_ENV=16 bash run_arm.sh RLVE          # Signal-RLVE 消融(需先 apply_patch.py)
#   ARM=fep        NUM_ENV=16 bash run_arm.sh RLVE          # FEP-RLVE(我们,需先 apply_patch.py)
#
# 依赖:已 `python apply_patch.py --rlve $PWD`(signal/fep 臂必须);权重已转换。
# ============================================================================
set -euo pipefail

WANDB_PROJECT=${1:-RLVE}
ARM=${ARM:-adaptive}                 # adaptive(RLVE-90) | static | signal(Signal-RLVE) | fep(FEP-RLVE,ours)
NUM_ENV=${NUM_ENV:-16}               # 1 / 4 / 16 / 256 / 400
STATIC_D=${STATIC_D:-4}              # 仅 static 臂:冻结的难度等级
# 单卡负载(OOM 就调小)
RESP_LEN=${RESP_LEN:-8192}
ROLLOUT_BSZ=${ROLLOUT_BSZ:-32}
N_SAMPLES=${N_SAMPLES:-8}
MAX_TOK=${MAX_TOK:-8192}
# 你的模型路径
MODEL_HF=${MODEL_HF:-/inspire/hdd/global_user/chenglian-253104020001/models/Nemotron-Research-Reasoning-Qwen-1.5B-v2}
MODEL_DIST=${MODEL_DIST:-${MODEL_HF}_torch_dist}

MODELDIR=Nemotron-Research-Reasoning-Qwen-1.5B-v2
RLVE_SH=scripts/training/${MODELDIR}/rlve.sh
[ -f "$RLVE_SH" ] || { echo "ERROR: 不在 RLVE 仓库根目录(缺 $RLVE_SH)"; exit 1; }

# 每次从原始脚本开始改(幂等)
[ -f "${RLVE_SH}.orig" ] || cp "$RLVE_SH" "${RLVE_SH}.orig"
cp "${RLVE_SH}.orig" "$RLVE_SH"

echo "== [1] 8 卡 → 单卡 + 缩小负载 =="
sed -i -E 's/--num-gpus 8/--num-gpus 1/' "$RLVE_SH"
sed -i -E 's/--actor-num-gpus-per-node 8/--actor-num-gpus-per-node 1/' "$RLVE_SH"
sed -i -E 's/--context-parallel-size 8/--context-parallel-size 1/' "$RLVE_SH"
sed -i -E "s/--rollout-max-response-len 24576/--rollout-max-response-len ${RESP_LEN}/" "$RLVE_SH"
sed -i -E "s/--rollout-batch-size 128/--rollout-batch-size ${ROLLOUT_BSZ}/" "$RLVE_SH"
sed -i -E "s/--n-samples-per-prompt 16/--n-samples-per-prompt ${N_SAMPLES}/" "$RLVE_SH"
sed -i -E "s/--max-tokens-per-gpu 3072/--max-tokens-per-gpu ${MAX_TOK}/" "$RLVE_SH"

echo "== [2] 指向你的模型 =="
# 用 # 作分隔符,免去路径里 / 的转义;先换更长的 ref-load(_torch_dist),再换 hf-checkpoint
sed -i -E "s#--ref-load \.\./${MODELDIR}_torch_dist#--ref-load ${MODEL_DIST}#" "$RLVE_SH"
sed -i -E "s#--hf-checkpoint \.\./${MODELDIR}#--hf-checkpoint ${MODEL_HF}#" "$RLVE_SH"

echo "== [3] 按 ARM 设难度策略 =="
case "$ARM" in
  adaptive)    EXTRA="";  unset DIFFICULTY_MODE || true ;;
  static)      EXTRA="--initial-difficulty ${STATIC_D} --difficulty-sliding-window-size 1 --min-metric-to-increase-difficulty 2.0"; unset DIFFICULTY_MODE || true ;;
  signal)      EXTRA="";  export DIFFICULTY_MODE=signal ;;
  fep)         EXTRA="";  export DIFFICULTY_MODE=fep ;;
  *) echo "ERROR: ARM 必须是 adaptive|static|signal|fep"; exit 1 ;;
esac
# 把额外 RLVE 参数注入到 train.py 调用(--colocate 之后)
if [ -n "$EXTRA" ]; then
  sed -i -E "s/^   --colocate \\\\$/   --colocate \\\\\n   ${EXTRA} \\\\/" "$RLVE_SH"
fi

echo "== 核对关键改动 =="
grep -nE "num-gpus|actor-num-gpus-per-node|context-parallel-size|hf-checkpoint|ref-load|initial-difficulty|rollout-max-response-len|n-samples-per-prompt" "$RLVE_SH" | head -20
echo "ARM=$ARM  NUM_ENV=$NUM_ENV  DIFFICULTY_MODE=${DIFFICULTY_MODE:-<unset>}  STATIC_D=${STATIC_D}"

echo "== 启动 num-environment=${NUM_ENV} =="
bash "scripts/training/${MODELDIR}/rlve/num-environment=${NUM_ENV}.sh" "$WANDB_PROJECT"
