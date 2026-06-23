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
NUM_ENV=1 RESP_LEN=8192 ROLLOUT_BSZ=32 N_SAMPLES=8 bash run_prorl_1gpu.sh RLVE
```
脚本做的事:`ray --num-gpus 8→1`、`actor-num-gpus-per-node 8→1`、
`context-parallel-size 8→1`、缩短 `rollout-max-response-len`/`rollout-batch-size`/
`n-samples-per-prompt`,然后跑 `num-environment=1`(最小臂,仅 `Multiplication` 环境)。
跑通后逐步放大:`NUM_ENV=4 / 16 / 256 / 400`。

OOM 就调小:`RESP_LEN=4096`、`ROLLOUT_BSZ=16`、`MAX_TOK=4096`。

## 4. 评测(复现 RLVE 的泛化结论)
```bash
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_BENCHMARKS.sh \
    "../[Nemotron-Research-Reasoning-Qwen-1.5B-v2]_[num-environment=1]/iter_xxxx"
bash scripts/evaluation/Nemotron-Research-Reasoning-Qwen-1.5B-v2/eval_HELD-OUT_ENVIRONMENTS.sh <ckpt>
```

## 5. 创新点:自由能难度控制器(已对真实 RLVE 代码实现)
RLVE 的难度调度全在 **`slime/ray/rollout_data_source.py` 的 `RLVEManager`**:
`generate_problem()` 选难度、`update()` 按 acc≥0.9 升难(只升不降)。我们的创新是把它
换成按 **Gibbs 分布 `q(d)∝exp(U(d)/T)`** 采样难度的控制器(`U(d)=1−p^G−(1−p)^G`,0.5 处最大),
可升可降、对准信号最优带。

本目录文件:
- `freeenergy_manager.py` — `FreeEnergyRLVEManager(RLVEManager)`,只重写 `generate_problem()`/`update()`
  (按真实 RLVE 接口:`Sample.metadata{environment,problem_difficulty}`、`reward["accuracy"]`、
  `ParameterController.update()/get_parameter_list()`),对真实代码已离线验证逻辑正确。
- `apply_patch.py` — 把上面的 manager 拷进 RLVE 根目录,并在 `RLVEManager` 实例化处加一个
  env 变量开关:`DIFFICULTY_MODE=freeenergy` 时用我们的 manager,否则原样 RLVE(幂等)。
- `freeenergy_controller.py` — 同算法的独立纯 Python 版(便于单测/讲解)。

接入(一次):
```bash
# 在 RLVE 仓库根目录
python /path/to/rlve_repro/apply_patch.py --rlve "$PWD"
# 之后:DIFFICULTY_MODE=freeenergy 那一臂才用自由能,不设则逐字保持 RLVE 原行为
```
超参经 env 变量:`FE_DMAX(16) FE_G(=n_samples_per_prompt) FE_T0(0.6) FE_TMIN(0.1) FE_ANNEAL(60) FE_EMA(0.7)`。

## 6. 实验怎么跑
见 **`EXPERIMENT_PLAN.md`**:A 组(环境 scaling 复现)+ B 组(static / RLVE / free-energy 三臂对照)。
用 `run_arm.sh`(本目录,已指向你的模型路径):
```bash
ARM=adaptive   NUM_ENV=1  bash run_arm.sh RLVE     # 先跑通最小臂
ARM=adaptive   NUM_ENV=16 bash run_arm.sh RLVE     # RLVE 基线
ARM=static     NUM_ENV=16 STATIC_D=4 bash run_arm.sh RLVE
ARM=freeenergy NUM_ENV=16 bash run_arm.sh RLVE     # 需先 apply_patch.py
```

先把 §0–§4 原始复现(A1)跑通,再做 §5–§6。
