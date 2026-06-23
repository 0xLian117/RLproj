#!/usr/bin/env bash
# ============================================================================
# SCALER 四臂实验(单卡,如 H200):adaptive vs static-lo / static-mid / static-hi
#
# 用法(在集群、verl 环境里):
#   conda activate verl
#   # 确保 SandboxFusion 已在本机 8080 起好(sandbox 环境: make run-online)
#   nohup bash run_scaler_4arms.sh > ~/runs_out/all.log 2>&1 &
#   tail -f ~/runs_out/all.log
#
# 前提:① verl 环境激活 ② 沙箱在 localhost:8080 ③ 模型是受支持架构(qwen2/qwen3),
#       不能是 qwen3_5(Qwen3.5-*),否则 transformers/vllm 不认。
# ============================================================================
set -uo pipefail

# ----------------------- 按需修改这几行 -----------------------
SCALER_DIR=/inspire/hdd/project/machine-behavior/chenglian-253104020001/SCALER
# 4B 模型(必须受支持):先下载 Qwen3-4B(下面注释里的命令),或换成本地受支持模型
MODEL=/inspire/hdd/global_user/chenglian-253104020001/models/Qwen3-4B
CKPT_ROOT=/inspire/hdd/global_user/chenglian-253104020001/ckpts   # 存大盘,别写满项目盘
STEPS=40            # 训练步数(想更充分改 60/80;4B 较慢,时间紧可减到 30)
GPU=0               # 用哪张卡
#
# 没下 Qwen3-4B 的话,先跑一次:
#   HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-4B --local-dir $MODEL
# 不想下载 → 用本地现成且受支持的(H200 141G 轻松):
#   MODEL=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct
# 注意:/inspire/.../Qwen/Qwen3.5-4B 是 qwen3_5,不支持,别用。
# --------------------------------------------------------------

cd "$SCALER_DIR"
S=recipe/environment/qwen3-1.7b-8-envs.sh
mkdir -p ~/runs_out "$CKPT_ROOT"

echo "================= 前置检查 ================="
# 1) 沙箱必须在线(代码环境算 reward 要用)
python -c "import socket,sys;sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8080))==0 else 1)" \
  || { echo "ERROR: SandboxFusion 不在 8080。先在本机 sandbox 环境 'make run-online' 起好再跑。"; exit 1; }
# 2) 模型可加载(挡掉 qwen3_5 / 路径不存在)
python - "$MODEL" <<'PY' || { echo "ERROR: 模型加载失败 —— 可能是不支持的架构(如 qwen3_5)或路径不存在。请用 Qwen3-4B 或 Qwen2.5-7B-Instruct。"; exit 1; }
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("OK  model_type:", c.model_type, "| architectures:", c.architectures)
PY

echo "================= 配置 SCALER 脚本(单卡 / 评前+后 / ckpt 大盘 / 4B 显存)================="
sed -i -E 's/^num_gpus=.*/num_gpus=1/' $S
sed -i -E 's/^tensor_model_parallel_size=.*/tensor_model_parallel_size=1/' $S
sed -i -E 's/^val_before_train=.*/val_before_train=True/' $S
sed -i -E "s/^epoch=.*/epoch=$STEPS/" $S
sed -i -E "s/^test_and_save_freq=.*/test_and_save_freq=$STEPS/" $S
sed -i -E 's/^n_resp_per_prompt=.*/n_resp_per_prompt=8/' $S
sed -i -E 's/^train_prompt_bsz=.*/train_prompt_bsz=16/' $S
sed -i -E 's/^train_prompt_mini_bsz=.*/train_prompt_mini_bsz=8/' $S
sed -i -E 's/^max_prompt_length=.*/max_prompt_length=$((1024 * 1))/' $S
sed -i -E 's/^max_response_length=.*/max_response_length=$((1024 * 2))/' $S   # 4B 留点思考空间
sed -i -E 's/^actor_ppo_max_token_len=.*/actor_ppo_max_token_len=12288/' $S
sed -i -E 's/^infer_ppo_max_token_len=.*/infer_ppo_max_token_len=12288/' $S
sed -i -E 's/^gpu_memory_utilization=.*/gpu_memory_utilization=0.4/' $S
echo "--- 关键参数 ---"
grep -nE '^(num_gpus|tensor_model_parallel_size|epoch|test_and_save_freq|n_resp_per_prompt|train_prompt_bsz|max_response_length|actor_ppo_max_token_len|gpu_memory_utilization|val_before_train)=' $S

echo "================= 依次跑四臂(train = adaptive)================="
for pair in "train:adaptive" "static-lo:static_lo" "static-mid:static_mid" "static-hi:static_hi"; do
  suf="${pair%%:*}"; name="${pair##*:}"
  echo "######## RUN $name   arms/SCALER-$suf.json   ($(date)) ########"
  ray stop --force 2>/dev/null; pkill -9 -f raylet 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=$GPU exp_name="$name" \
    RAY_DATA_HOME="$SCALER_DIR" MODEL_PATH="$MODEL" \
    CKPTS_DIR="$CKPT_ROOT/$name" \
    TRAIN_FILE="$SCALER_DIR/arms/SCALER-$suf.json" \
    bash "$S" 2>&1 | tee ~/runs_out/$name.log
  echo "######## DONE $name  exit=${PIPESTATUS[0]}   ($(date)) ########"
done

echo "================= 全部完成 ================="
echo " 日志: ~/runs_out/{adaptive,static_lo,static_mid,static_hi}.log"
echo " ckpt: $CKPT_ROOT/"
echo " 取数: grep -E 'reward|distance|val' ~/runs_out/*.log"
