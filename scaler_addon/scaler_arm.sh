#!/usr/bin/env bash
# ============================================================================
# 单臂 SCALER 训练 recipe(单卡 / Qwen2.5 友好设置 / 评测前+后 / ckpt 存大盘)
# 直接放到 SCALER 仓库根目录,在那里(cwd = SCALER 根)用循环调用,换 TRAIN_FILE/exp_name 即可。
#
# 可被环境变量覆盖:MODEL_PATH / TRAIN_FILE / exp_name / CKPTS_DIR / RAY_DATA_HOME / STEPS / GPU
#
# 例(在 SCALER 根目录):
#   M=/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-3B-Instruct
#   for pair in "train:adaptive" "static-lo:static_lo" "static-mid:static_mid" "static-hi:static_hi"; do
#     suf=${pair%%:*}; name=${pair##*:}
#     ray stop --force 2>/dev/null; sleep 2
#     exp_name=$name MODEL_PATH=$M TRAIN_FILE=$PWD/arms/SCALER-$suf.json \
#       CKPTS_DIR=/inspire/hdd/global_user/chenglian-253104020001/ckpts/$name \
#       bash scaler_arm.sh 2>&1 | tee ~/runs_out/$name.log
#   done
# ============================================================================
set -uxo pipefail
export WANDB_MODE=offline

# ---------- 单卡 + 显存/步数设置(给 Qwen2.5-3B/7B 在单卡上用)----------
num_gpus=1
tensor_model_parallel_size=1
val_before_train=True
STEPS=${STEPS:-40}                 # 训练步数(想更充分改 60/80)
epoch=${STEPS}
test_and_save_freq=${STEPS}        # 评测/存档:训练前(val_before_train)+ 第 STEPS 步
n_resp_per_prompt=8
train_prompt_bsz=16
train_prompt_mini_bsz=8
num_environment_per_step=5         # 我们训练用 5 个环境
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))
gpu_memory_utilization=0.4
actor_ppo_max_token_len=12288
infer_ppo_max_token_len=12288

# ---------- 其余沿用官方默认 ----------
project_name='SCALER_course'
lr=1e-6
lr_warmup_steps=20
with_instruction=True
enable_windows_sample=False
enable_weighted_sample=False
clip_ratio_low=0.2
clip_ratio_high=0.2
entropy_coeff=0
loss_agg_mode="token-mean"
use_dynamic_bsz=True
max_num_gen_batches=100
offload=True
ref_offload=True
sandboxfusion_url="http://localhost:8080/run_code"

# ---------- 这些可被环境变量覆盖(循环里换它们)----------
GPU=${GPU:-0}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}"}                         # 默认当前目录(应是 SCALER 根)
MODEL_PATH=${MODEL_PATH:-"/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-3B-Instruct"}
exp_name=${exp_name:-"adaptive"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/arms/SCALER-train.json"}
CKPTS_DIR=${CKPTS_DIR:-"/inspire/hdd/global_user/chenglian-253104020001/ckpts/${exp_name}"}
TEST_FILE=${TEST_FILE:-["${RAY_DATA_HOME}/SCALER-data/test/bbeh_data.parquet","${RAY_DATA_HOME}/SCALER-data/test/think_MATH-500_MATH-500-processed.parquet","${RAY_DATA_HOME}/SCALER-data/test/think_amc23_amc23_test.parquet","${RAY_DATA_HOME}/SCALER-data/test/think_aime24_aime24_test.parquet","${RAY_DATA_HOME}/SCALER-data/test/MMLU-Pro-Valid.parquet","${RAY_DATA_HOME}/SCALER-data/test/GPQA-Diamond-Test.parquet"]}

export CUDA_VISIBLE_DEVICES=${GPU}

PYTHONUNBUFFERED=1 python3 -m recipe.environment.main_dapo \
    +data.setting_filename="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    +data.num_environment_per_step=${num_environment_per_step} \
    +data.with_instruction=${with_instruction} \
    +data.sandboxfusion_url="${sandboxfusion_url}" \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=acc \
    +algorithm.update_train_configs=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    reward_model.overlong_buffer.len=0 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_training_steps=${epoch} \
    +trainer.enable_weighted_sample=${enable_weighted_sample} \
    +trainer.enable_windows_sample=${enable_windows_sample} \
    +trainer.windows_continous_zero_correct_limit=5 \
    +trainer.windows_continous_max_distance_limit=5
