#!/usr/bin/env bash
# ============================================================================
# 官方 RLVE 复现:ProRL-1.5B-v2,单卡(H200)版。
# 把官方 8 卡 Megatron 脚本就地改成单卡(cp/tp/pp=1, ray --num-gpus 1)+ 缩短
# 响应长度/批大小,然后跑 num-environment=1(最小臂:仅 "Multiplication")。
#
# 必须在【SLIME docker 容器内 + RLVE 仓库根目录】运行(见 rlve_repro/README.md)。
#   bash rlve_repro/run_prorl_1gpu.sh RLVE_PROJ
#
# 可用环境变量覆盖:RESP_LEN / ROLLOUT_BSZ / N_SAMPLES / MAX_TOK / NUM_ENV
# ============================================================================
set -euo pipefail

WANDB_PROJECT=${1:-RLVE}
NUM_ENV=${NUM_ENV:-1}                 # 1 / 4 / 16 / 256 / 400
RESP_LEN=${RESP_LEN:-8192}            # 官方 24576,单卡先缩短
ROLLOUT_BSZ=${ROLLOUT_BSZ:-32}        # 官方 128
N_SAMPLES=${N_SAMPLES:-8}             # 官方 16(GRPO 组大小)
MAX_TOK=${MAX_TOK:-8192}              # max-tokens-per-gpu

RLVE_SH=scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve.sh
[ -f "$RLVE_SH" ] || { echo "ERROR: 不在 RLVE 仓库根目录(找不到 $RLVE_SH)"; exit 1; }

# 备份一次
[ -f "${RLVE_SH}.orig" ] || cp "$RLVE_SH" "${RLVE_SH}.orig"
cp "${RLVE_SH}.orig" "$RLVE_SH"     # 每次从原始开始改,幂等

echo "== 把 8 卡配置改成单卡 + 缩小负载 =="
sed -i -E 's/--num-gpus 8/--num-gpus 1/' "$RLVE_SH"
sed -i -E 's/--actor-num-gpus-per-node 8/--actor-num-gpus-per-node 1/' "$RLVE_SH"
sed -i -E 's/--context-parallel-size 8/--context-parallel-size 1/' "$RLVE_SH"
sed -i -E "s/--rollout-max-response-len 24576/--rollout-max-response-len ${RESP_LEN}/" "$RLVE_SH"
sed -i -E "s/--rollout-batch-size 128/--rollout-batch-size ${ROLLOUT_BSZ}/" "$RLVE_SH"
sed -i -E "s/--n-samples-per-prompt 16/--n-samples-per-prompt ${N_SAMPLES}/" "$RLVE_SH"
sed -i -E "s/--max-tokens-per-gpu 3072/--max-tokens-per-gpu ${MAX_TOK}/" "$RLVE_SH"

echo "== 核对改动 =="
grep -nE "num-gpus|gpus-per-node|context-parallel-size|rollout-max-response-len|rollout-batch-size|n-samples-per-prompt|max-tokens-per-gpu" "$RLVE_SH"

echo "== 启动 num-environment=${NUM_ENV} =="
bash "scripts/training/Nemotron-Research-Reasoning-Qwen-1.5B-v2/rlve/num-environment=${NUM_ENV}.sh" "$WANDB_PROJECT"
