# RLproj — 复现 RLVE 并加入自由能难度控制

课程项目(RL for LLM reasoning with **environment scaling**)。主线:复现作业参考的
**RLVE**(arXiv:2511.07317;可验证环境 + 自适应难度的 RL),从 **ProRL-1.5B-v2**
起点在单卡上训练;创新点:把 RLVE 的「准确率超阈值就升难」启发式,替换为一个
有理论依据的 **期望自由能(Active Inference / Friston FEP)难度控制器**——RLVE 的固定升难
是它的退化特例(`λ_info=0`、`T→0`、偏好高准确率)。

## 仓库结构
```
rlve_repro/                官方 RLVE 复现 + 我们的创新
  README.md                单卡(H200)复现 ProRL-1.5B-v2 的完整步骤(SLIME, 8→1 卡)
  run_arm.sh                单卡四臂启动器(Static / RLVE-90 / Signal-RLVE / FEP-RLVE)
  active_inference_controller.py  ★创新:期望自由能(EFE)难度控制器(纯 Python,已 self-check)
  active_inference_manager.py     把 EFE 控制器接进 SLIME 的 RLVEManager
  analyze_rlve.py           解析 worker 日志 → results/ 图表
paper/                     mini-paper 提纲(NeurIPS)+ 自由能→G(d) 推导(derivation.md)
poster/                    海报提纲(poster模板.pptx 为模板)
idea.html                  研究 proposal(可交互,讲清:问题→自由能方法→实验设计)
des.md                     作业要求
```

## 思路一句话
RL 训练在难度偏离模型能力时停滞:太易→整组全对、太难→整组全错,两者都给不出 GRPO 梯度。
一组 K 个 rollout 非退化(有对有错→有梯度)的概率 `U_K(p)=1−p^K−(1−p)^K`,在成功率 0.5 最大。
RLVE 把难度调到高成功率(≥0.9)才升档,停在低信号区;**我们**把「选哪个难度」当成
**active inference 的行动选择**,对每个难度算**期望自由能**
`G(d)=λ_pref·(−log U_K(p̂)) − λ_info·I(s;o|d) + λ_cost·C(d)`(实用=偏好风险 −log U_K + 认知信息增益 I + 成本),
再按 active inference 自带的**策略后验** `π(d)∝exp(−G(d)/T)`(softmax,精度 γ=1/T)采样。
能力信念 `q(s)` 由 IRT 似然 `σ(a(s−d))` 贝叶斯更新——即 FEP 的「感知=最小化变分自由能」。
温度 T:`T→0` 贪心、`T→∞` 均匀探索。推导见 `paper/derivation.md`,proposal 见 `idea.html`。

## 怎么跑
完整步骤见 **`rlve_repro/README.md`**:
1. 下模型 `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --revision v2`;
2. SLIME docker 环境 + `pip install -e .`;
3. HF→Megatron 权重转换;
4. `bash run_prorl_1gpu.sh RLVE`(单卡,先 `num-environment=1`,再放大);
5. 评测复现「环境越多泛化越好」;
6. 接入 `freeenergy_controller.py`(README §5)做创新对照。

基线:RLVE (arXiv:2511.07317) · github.com/Zhiyuan-Zeng/RLVE
