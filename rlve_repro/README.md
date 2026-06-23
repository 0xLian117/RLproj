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

## 5. 之后:把自由能放进 RLVE(创新点)
创新算法已实现为 **`freeenergy_controller.py`**(本目录,纯 Python、无 SLIME/torch 依赖、可单测):
给定各难度的成功率估计,按 `q(d) ∝ exp(U(d)/T)` 采样下一批难度,`U(d)=1−p^G−(1−p)^G`。
`T→0` 时退化回 RLVE 的固定升难点(已在脚本 self-check 验证)。

接入点(RLVE 官方 + SLIME 后端):
- **每环境难度**:`Gym/parameter_controllers/<env>/parameter_controller.py`,`update()`=难度+1,`get_parameter_list()`=当前难度的参数集;
- **自适应升难调度**(论文里维护难度区间 `[ℓ,h]`、按成功率升难)在 **SLIME rollout 侧**(`slime/rollout/`,`rlve_rm.py` 只是 verifier);
- **改法**:在那段"用各难度成功率决定下一批出多难"的逻辑里:
  1. 维护每(环境,难度)的成功率 EMA → 喂给 `FreeEnergyController.update({d: success})`;
  2. 用 `FreeEnergyController.sample_difficulties(n)` 决定下一批难度,替换"acc≥0.9 升一档";
  3. 驱动各环境的 `ParameterController` 到采样到的难度。
```python
from freeenergy_controller import FreeEnergyController
fe = FreeEnergyController(d_min=0, d_max=20, G=8)   # G = n-samples-per-prompt
ds = fe.sample_difficulties(batch)                  # 本步要出的难度
# rollout + 判分后:
fe.update({d: observed_success_rate[d] for d in set(ds)})
```

先把 §0–§4 的原始复现跑通,再做 §5。
