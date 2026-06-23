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
ROLLOUT_BSZ=${ROLLOUT_BSZ:-8}        # 保守默认:单卡 colocate 训练 batch(原 32)
N_SAMPLES=${N_SAMPLES:-8}
MAX_TOK=${MAX_TOK:-2048}             # 保守默认:训练单卡峰值 token(原 8192;OOM 就靠它)
OVERSAMPLE=${OVERSAMPLE:-16}          # DAPO 超采样 prompt 数(原版 384!→ 每步生成 OVERSAMPLE×N_SAMPLES 条)
SGLANG_MEM=${SGLANG_MEM:-0.4}        # 保守默认:SGLang 静态显存占比(原 0.7);留更多给训练侧,治 colocate OOM
if [ "$OVERSAMPLE" -lt "$ROLLOUT_BSZ" ]; then OVERSAMPLE=$ROLLOUT_BSZ; fi   # 防 over<rollout 的 assert
# 时长控制:总步数 / 存档间隔 / 评测间隔
MAX_STEPS=${MAX_STEPS:-50}                  # --num-rollout(对照只需分化,~50 步足够)
SAVE_EVERY=${SAVE_EVERY:-${MAX_STEPS}}      # 默认只在最后存一次,省盘省时
EVAL_EVERY=${EVAL_EVERY:-25}                # held-out 环境评测间隔(给泛化曲线)
# 你的模型路径
MODEL_HF=${MODEL_HF:-/inspire/hdd/global_user/chenglian-253104020001/models/Nemotron-Research-Reasoning-Qwen-1.5B-v2}
MODEL_DIST=${MODEL_DIST:-${MODEL_HF}_torch_dist}
# checkpoint 存大盘(别写 ../,会落在 /root 撑爆);wandb 离线
SAVE_ROOT=${SAVE_ROOT:-/inspire/hdd/global_user/chenglian-253104020001/rlve_ckpts}
export WANDB_MODE=${WANDB_MODE:-offline}             # offline 不连服务器(disabled 会被 SLIME shared 模式无视→挂)
export WANDB_API_KEY=${WANDB_API_KEY:-offline}
export WANDB_DIR=${WANDB_DIR:-/inspire/hdd/global_user/chenglian-253104020001/wandb}  # 离线文件夹写大盘,别堆 RLVE 根
mkdir -p "$SAVE_ROOT" "$WANDB_DIR"

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
sed -i -E "s/--over-sampling-batch-size 384/--over-sampling-batch-size ${OVERSAMPLE}/" "$RLVE_SH"
sed -i -E "s/--max-tokens-per-gpu 3072/--max-tokens-per-gpu ${MAX_TOK}/" "$RLVE_SH"
# 时长控制:总步数 / 存档 / 评测间隔
sed -i -E "s/--num-rollout 1000000/--num-rollout ${MAX_STEPS}/" "$RLVE_SH"
sed -i -E "s/--save-interval 1/--save-interval ${SAVE_EVERY}/" "$RLVE_SH"
sed -i -E "s/--eval-interval 20/--eval-interval ${EVAL_EVERY}/" "$RLVE_SH"

echo "== [2] 指向你的模型 =="
# 用 # 作分隔符,免去路径里 / 的转义;先换更长的 ref-load(_torch_dist),再换 hf-checkpoint
sed -i -E "s#--ref-load \.\./${MODELDIR}_torch_dist#--ref-load ${MODEL_DIST}#" "$RLVE_SH"
sed -i -E "s#--hf-checkpoint \.\./${MODELDIR}#--hf-checkpoint ${MODEL_HF}#" "$RLVE_SH"
# --save / --load 改到大盘,且每个 ARM 独立目录(否则各臂写死同名目录→串档/resume 冲突)
# SAVE_ROOT 与 ARM 在 run_arm.sh 里展开(烤成绝对路径),${RUN_NAME} 留给 rlve.sh 运行时展开
sed -i "s#\.\./\${RUN_NAME}/#${SAVE_ROOT}/\${RUN_NAME}_${ARM}/#g" "$RLVE_SH"

# 没设 WANDB_API_KEY 时给个占位,避免 --wandb-key 传空报错(配合 WANDB_MODE=offline)
sed -i 's/--wandb-key "${WANDB_API_KEY}"/--wandb-key "${WANDB_API_KEY:-offline}"/' "$RLVE_SH"

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
