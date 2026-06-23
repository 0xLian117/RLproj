# 实验设计 — FEP-RLVE(单卡 H200,ProRL-1.5B-v2)

方法:**FEP-RLVE** —— 把 RLVE 的"准确率>0.9 才升难"规则,换成按 **Expected Free
Energy** 选难度:`G_e(d) = −λ_signal·U_K(p̂) − λ_info·I(s;o|d) + λ_cost·C(d)`,
`π(d|e) ∝ exp(−G_e(d)/T)`,`U_K(p)=1−p^K−(1−p)^K`。每环境维护能力后验 `q(s_e)`,
rollout 后贝叶斯更新。FEP 只控制"下一批题从哪来",RL 算法仍是 DAPO。

模型:`/inspire/hdd/global_user/chenglian-253104020001/models/Nemotron-Research-Reasoning-Qwen-1.5B-v2`
框架:官方 RLVE(SLIME)。所有臂**同模型、同算力**,只改难度采样器。

## 四臂(核心对照,固定环境数,如 NUM_ENV=16)
| 臂 | 难度策略 | 命令 | 隔离了什么 |
|---|---|---|---|
| **Static** | 冻结难度 d | `ARM=static STATIC_D=4 NUM_ENV=16 bash run_arm.sh RLVE` | 固定难度会失败 |
| **RLVE-90** | acc≥0.9 升难(原版) | `ARM=adaptive NUM_ENV=16 bash run_arm.sh RLVE` | 论文基线 |
| **Signal-RLVE** | 只 `U_K`(λ_info=0) | `ARM=signal NUM_ENV=16 bash run_arm.sh RLVE` | 对准 0.5 信号带,但**无信息增益** |
| **FEP-RLVE(ours)** | `U_K`+信息增益+Gibbs | `ARM=fep NUM_ENV=16 bash run_arm.sh RLVE` | 完整 EFE |

**为什么必须有 Signal-RLVE**:它是"把 0.9 换成对准信号带"的版本。只有 FEP-RLVE
**比 Signal-RLVE 还好**,才能证明 **信息增益(epistemic / FEP 的核心)真的有贡献**,
而不是"只是把阈值从 0.9 改成 0.5"。这一对照是"它到底算不算 FEP"的关键。

(跑 signal/fep 前先 `python apply_patch.py --rlve $PWD`。)

## (可选)A 组 · 复现 RLVE 环境 scaling → 泛化
RLVE-90 在 NUM_ENV=1/4/16 上各跑一次,复现"环境越多 held-out 越好"。时间紧可省略或只做 1 vs 16。

## 评测(每臂:训练前 + 后)
```bash
M=Nemotron-Research-Reasoning-Qwen-1.5B-v2
bash scripts/evaluation/$M/eval_BENCHMARKS.sh        <ckpt>   # AIME/OMEGA/OlympiadBench/LiveCodeBench/BBEH
bash scripts/evaluation/$M/eval_HELD-OUT_ENVIRONMENTS.sh <ckpt>  # 50 held-out 环境
```

## 指标
1. **effective prompt ratio**(机理,RLVE 自己也用):fep/signal 臂日志直接打印
   `FEP/effective_sample_ratio`;RLVE-90/Static 从 `RLVE/<env>/accuracy` 反推 p→算 `U_K`。
   预期:Static 塌、RLVE-90≈0.57(停在 0.9)、Signal/FEP≈0.99(对准 0.5)。
2. **能力信念轨迹**(仅 FEP):`FEP/<env>/competence_mean ± std`、`expected_difficulty`
   —— σ 收缩 = 主动定位能力(explore→exploit)。
3. **泛化**:held-out 环境 + 6 benchmark 的训练前后 Δ;主张 FEP ≥ Signal > RLVE-90 > Static。
4. **难度熵 / 环境覆盖**(多环境时):sampler 有没有塌缩到少数难度/环境。
5. **温度/权重消融(可选)**:扫 `FE_T`、`FE_LINFO`。

## 优先级(deadline)
1. **RLVE-90 NUM_ENV=1 跑通**(验证环境/权重转换/单卡)。
2. **RLVE-90 NUM_ENV=16** + 评测(基线)。
3. **FEP-RLVE NUM_ENV=16** + 评测(我们)→ 头条对比。
4. **Signal-RLVE** + **Static** → 补齐四臂(隔离 FEP 贡献)。

**最小可交付** = RLVE-90 vs FEP-RLVE(2 臂,同模型);**可信交付** = 四臂(加 Signal-RLVE 隔离信息增益)。
