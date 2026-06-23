# RLproj — 复现 RLVE 并加入自由能难度控制

课程项目(RL for LLM reasoning with **environment scaling**)。主线:复现作业参考的
**RLVE**(arXiv:2511.07317;可验证环境 + 自适应难度的 RL),从 **ProRL-1.5B-v2**
起点在单卡上训练;创新点:把 RLVE 的「准确率超阈值就升难」启发式,替换为一个
有理论依据的 **自由能 / Gibbs 难度控制器**——RLVE 的固定升难点是它在零温下的特例。

## 仓库结构
```
rlve_repro/                官方 RLVE 复现 + 我们的创新
  README.md                单卡(H200)复现 ProRL-1.5B-v2 的完整步骤(SLIME, 8→1 卡)
  run_prorl_1gpu.sh         把官方 8 卡脚本就地改单卡并启动
  freeenergy_controller.py  ★创新:自由能/Gibbs 难度控制器(纯 Python,已 self-check)
paper/                     mini-paper 提纲(NeurIPS)+ p(1-p)→自由能 推导(derivation.md)
poster/                    海报提纲(poster模板.pptx 为模板)
idea.html                  研究 proposal(可交互,讲清:问题→自由能方法→实验设计)
des.md                     作业要求
```

## 思路一句话
RL 训练在难度偏离模型能力时停滞:太易→整组全对、太难→整组全错,两者都给不出 GRPO 梯度。
一组 G 个 rollout 的学习信号 ∝ 组内奖励方差 `p(1−p)`,在成功率 0.5 最大。RLVE 把难度调到
高成功率(≥0.9)才升档;**我们**最小化自由能 `F[q]=−E_q[U]−T·H[q]`,最优解是 Gibbs 分布
`q(d)∝exp(U(d)/T)`,`U(d)=1−p^G−(1−p)^G`。温度 T:`T→0` 复现 RLVE 的定点,`T→∞` 为均匀探索。
推导见 `paper/derivation.md`,proposal 见 `idea.html`。

## 怎么跑
完整步骤见 **`rlve_repro/README.md`**:
1. 下模型 `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B --revision v2`;
2. SLIME docker 环境 + `pip install -e .`;
3. HF→Megatron 权重转换;
4. `bash run_prorl_1gpu.sh RLVE`(单卡,先 `num-environment=1`,再放大);
5. 评测复现「环境越多泛化越好」;
6. 接入 `freeenergy_controller.py`(README §5)做创新对照。

基线:RLVE (arXiv:2511.07317) · github.com/Zhiyuan-Zeng/RLVE
