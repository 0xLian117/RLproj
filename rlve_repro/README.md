# 官方 RLVE 复现:ProRL-1.5B-v2(单卡 H200)

直接复现作业参考的 **RLVE**(`github.com/Zhiyuan-Zeng/RLVE`),从 **ProRL-1.5B-v2**
起点训练。官方是 SLIME+Megatron、8×80GB;这里把它压到**单卡 H200**。

> 诚实提示:这条路的主要工作量是**搭 SLIME 环境(Docker)**——和之前 verl 那套
> 完全不同、要重搭。跑通后再把自由能控制器接进难度调度(见最后一节)。

## 0. 模型(可下载)
ProRL-1.5B-v2 = `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` 的 **`v2`** 修订:
```bash
hf download nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --revision v2 \
  --local-dir ../Nemotron-Research-Reasoning-Qwen-1.5B-v2
```

## 1. 环境:SLIME docker(官方唯一安装路径)
```bash
docker pull slimerl/slime:v0.5.0rc0-cu126        # 你能 build/推 Harbor,也可走那条
docker run -d --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 --name RLVE \
  slimerl/slime:v0.5.0rc0-cu126 tail -f /dev/null
docker exec -it RLVE bash
# —— 以下都在容器内 ——
cd /root && git clone https://github.com/Zhiyuan-Zeng/RLVE.git && cd RLVE && pip install -e .
```

## 2. HF → Megatron 权重转换(一次,单卡即可)
```bash
source scripts/models/deepseek-r1-distill-qwen-1.5B.sh    # ProRL 用这个 arch
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint ../Nemotron-Research-Reasoning-Qwen-1.5B-v2 \
    --save ../Nemotron-Research-Reasoning-Qwen-1.5B-v2_torch_dist
```

## 3. 单卡训练(用本目录脚本)
把 `run_prorl_1gpu.sh` 放进 RLVE 仓库根目录运行(它把官方 8 卡脚本就地改单卡):
```bash
# 在 RLVE 仓库根目录
cp /path/to/run_prorl_1gpu.sh .
NUM_ENV=1 bash run_prorl_1gpu.sh RLVE
```
脚本做的事:`ray --num-gpus 8→1`、`actor-num-gpus-per-node 8→1`、
`context-parallel-size 8→1`、缩短 `rollout-max-response-len`/`rollout-batch-size`/
`n-samples-per-prompt`、`over-sampling-batch-size`、`max-tokens-per-gpu`,并降低
`sglang-mem-fraction-static`,然后跑 `num-environment=1`(最小臂,仅 `Multiplication` 环境)。
跑通后逐步放大:`NUM_ENV=4 / 16 / 256 / 400`。

如果看到 `ActorDiedError` / `Worker unexpectedly exits` / `SIGKILL`,优先当作
CPU/GPU OOM 处理。单卡先用脚本默认的保守配置:
`RESP_LEN=4096 ROLLOUT_BSZ=8 N_SAMPLES=4 OVERSAMPLE=8 MAX_TOK=4096 SGLANG_MEM=0.4`。
仍然 OOM 就继续降:`RESP_LEN=2048`、`MAX_TOK=2048`、`OVERSAMPLE=4`。
稳定后再放大,例如:
```bash
NUM_ENV=1 RESP_LEN=8192 ROLLOUT_BSZ=16 N_SAMPLES=8 OVERSAMPLE=8 MAX_TOK=4096 bash run_prorl_1gpu.sh RLVE
```

## 4. 评测(复现 RLVE 的泛化结论)
```bash
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_BENCHMARKS.sh \
    "../[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=1]/iter_xxxx"
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_HELD-OUT_ENVIRONMENTS.sh <ckpt>
```

## 5. 创新点:FEP-RLVE(Expected Free Energy 难度选择,对真实 RLVE 代码实现)
RLVE 的难度调度全在 **`slime/ray/rollout_data_source.py` 的 `RLVEManager`**:
`generate_problem()` 选难度、`update()` 按 acc≥0.9 升难(只升不降)。我们把它换成
**active inference / Expected Free Energy** 选难度:每环境维护能力后验 `q(s_e)`,按
`π(d)∝exp(−G_e(d)/T)` 采样,`G_e(d)=−λ_signal·U_K(p̂)−λ_info·I(s;o|d)+λ_cost·C(d)`,
`U_K(p)=1−p^K−(1−p)^K`(= RLVE/DAPO 的有效 prompt 比例,0.5 处最大),
`I` 为对能力的信息增益。可升可降、主动定位能力(explore→exploit 自动涌现)。

本目录文件:
- `active_inference_controller.py` — `FEPRLVEController`,纯 Python,已离线验证:
  `U_K` 与信息增益都在 p≈0.5 峰值;belief 跟踪上升能力、σ 收缩。
- `active_inference_manager.py` — `FEPRLVEManager(RLVEManager)`,只重写
  `generate_problem()`/`update()`(按真实 RLVE 接口:`Sample.metadata`、`reward["accuracy"]`、
  `ParameterController.update()/get_parameter_list()`)。
- `apply_patch.py` — 把上两个文件拷进 RLVE 根目录,并在 `RLVEManager` 实例化处加
  env 变量开关(幂等,默认不改 RLVE 行为)。

接入(一次):
```bash
python /path/to/rlve_repro/apply_patch.py --rlve "$PWD"   # 在 RLVE 仓库根目录
```
- `DIFFICULTY_MODE` 不设 → 原样 RLVE-90;`=signal` → Signal-RLVE 消融(λ_info=0);`=fep` → FEP-RLVE。
- 超参(env 变量):`FE_DMAX(16) FE_K(=n_samples_per_prompt) FE_SLOPE(1.0) FE_LSIG(1.0) FE_LINFO(1.0) FE_LCOST(0.0) FE_T(0.25)`。

## 6. 实验怎么跑(四臂,见 EXPERIMENT_PLAN.md)
用 `run_arm.sh`(已指向你的模型路径):
```bash
ARM=adaptive NUM_ENV=1  bash run_arm.sh RLVE              # 先跑通最小臂
ARM=adaptive NUM_ENV=16 bash run_arm.sh RLVE             # RLVE-90 基线
ARM=static   NUM_ENV=16 STATIC_D=4 bash run_arm.sh RLVE  # Static
ARM=signal   NUM_ENV=16 bash run_arm.sh RLVE             # Signal-RLVE 消融(先 apply_patch)
ARM=fep      NUM_ENV=16 bash run_arm.sh RLVE             # FEP-RLVE(我们)
```
`run_arm.sh` 默认也走保守单卡配置;稳定后可显式提高 `RESP_LEN`、`MAX_TOK`、
`OVERSAMPLE`、`ROLLOUT_BSZ` 做正式长跑。

先把 §0–§4 原始复现(`ARM=adaptive NUM_ENV=1`)跑通,再做 §5–§6。
