# 实验设计 — RLVE 复现 + 自由能创新(单卡 H200,ProRL-1.5B-v2)

模型:`/inspire/hdd/global_user/chenglian-253104020001/models/Nemotron-Research-Reasoning-Qwen-1.5B-v2`
框架:官方 RLVE(SLIME 后端)。所有臂同模型、同算力,只改难度策略。

## 两组实验

### A 组 · 复现 RLVE 核心论点(环境 scaling → 泛化)
| 实验 | 环境数 | 控制器 | 命令 |
|---|---|---|---|
| A1 | 1 | RLVE 原版 | `ARM=adaptive NUM_ENV=1 bash run_arm.sh RLVE` |
| A2 | 4 | RLVE 原版 | `ARM=adaptive NUM_ENV=4 bash run_arm.sh RLVE` |
| A3 | 16 | RLVE 原版 | `ARM=adaptive NUM_ENV=16 bash run_arm.sh RLVE` |

→ 复现「环境越多,held-out 泛化越好」。时间紧可只做 A1 + A3 两点画趋势。

### B 组 · 创新对照(固定 16 环境,只换难度控制器)
| 实验 | 控制器 | 命令 |
|---|---|---|
| B1 | static-d(冻结 d=4) | `ARM=static NUM_ENV=16 STATIC_D=4 bash run_arm.sh RLVE` |
| B2 | RLVE 原版(acc≥0.9 升难) | = A3,不重跑 |
| B3 | **free-energy(我们)** | `ARM=freeenergy NUM_ENV=16 bash run_arm.sh RLVE` |

(跑 B3 前先 `python apply_patch.py --rlve $PWD`。)

## 三种难度策略怎么落到 RLVE 参数
- **adaptive(基线)**:RLVE 默认 —— 难度上界 h,在窗内随机出题;上界处 acc≥`--min-metric-to-increase-difficulty`(0.9)就 h+1,只升不降。
- **static**:`--initial-difficulty D --difficulty-sliding-window-size 1 --min-metric-to-increase-difficulty 2.0` —— 难度永远钉在 D(阈值 2.0 永不触发升难)。
- **free-energy**:`DIFFICULTY_MODE=freeenergy` 切到 `FreeEnergyRLVEManager`,按 `q(d)∝exp(U(d)/T)` 采样难度、EMA 估 `p̂(d)`、退火 T,可升可降。无视上面三个 RLVE 难度参数。

## 评测(每个臂:训练前 + 训练后)
```bash
M=Nemotron-Research-Reasoning-Qwen-1.5B-v2
bash scripts/evaluation/$M/eval_BENCHMARKS.sh        <ckpt>   # AIME/OMEGA/OlympiadBench/LiveCodeBench/BBEH
bash scripts/evaluation/$M/eval_HELD-OUT_ENVIRONMENTS.sh <ckpt>  # 50 held-out 环境
```
报告各臂相对训练前的 Δ。

## 核心指标
1. **有效样本率** `1−p^G−(1−p)^G`:free-energy 臂训练日志里直接打印
   `FreeEnergy/effective_sample_ratio`;baseline/static 从 `RLVE/<env>/accuracy` 反推
   `p`→算 U。预期:static 塌、RLVE≈0.57(停在 0.9)、free-energy≈0.99(对准 0.5)。
2. **难度轨迹**:`RLVE/<env>/difficulty`(baseline 升到饱和)、`FreeEnergy/<env>/expected_difficulty`(对准 0.5 带)。
3. **泛化**:held-out + benchmark 训练前后。
4. **温度消融(可选)**:`FE_T0`/`FE_TMIN`/`FE_ANNEAL` 扫 T;数值上 `T→0` 退化为固定 0.5 定点。

## 优先级(deadline)
1. **A1 跑通**(验证环境/权重转换/单卡)。
2. **A3(=B2)** RLVE 基线 16 环境 + 评测。
3. **B1 + B3**(同 16 环境)→ static / RLVE / free-energy 三臂对照(创新核心)。
4. 余力:A2 补 scaling 点 + 温度消融。

**最小可交付** = A1 跑通 + B1/B2/B3 三臂 + 评测。
