## 大作业具体信息

- 选题：给定题目四选一，或者自选题目

- 打分原则：在特定研究方向上做出适当创新

- 提交材料

- Mini paper (40%) 

- NeurIPS template: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh 

- 正文不少于4页，不多于8页，后可接不限页数的参考文献和附录

- Poster (PPT or PDF) (10%) 

- 用于2026年6月30日poster活动展示交流

- 代码包（用于对抄袭事件的辅助判定）


## 2. RL LLMs with Env Scaling

## 核心动机：RL训练不是只缺更多题，而是缺能持续产生“适中难度”学习信号的环境

## WHY

## 固定题库会很快失效

- 太易：全对，reward 无梯度

- 太难：全错，更新停住

- 人工答案贵，题库规模有限

## WHAT

## RLVE：可验证环境+自适应难度

- 环境程序生成无限问题

- verifier 自动给 reward

- 正确率超过阈值后自动升难

- RLVE-Gym: 400 个环境

## SO WHAT

## 环境 scale 带来更强泛化

- ProRL-1.5B-v2：平均 +3.37%

- 原 RLVR: +0.49%，且用 >3× compute

- 环境工程成为 RL scaling 入口

## 推荐框架&Baseline

https://github.com/Zhiyuan-Zeng/RLVE 

1张H200即可复现，2X4090也可以。（感谢24级同学苏浩阳同学帮忙复现）

## 2. RLVE 怎么做：生成题目、验证奖励、动态升难

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-22/c763ce74-51c2-42dc-a6a8-8190fd5d21aa/fd163627c23ec998ab72760f06c26836339257e858021eaf54981079895629fb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-22/c763ce74-51c2-42dc-a6a8-8190fd5d21aa/a58fda8eb97746ce459ef790f0ccb459107df93702b86653825e2887e40f4907.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-22/c763ce74-51c2-42dc-a6a8-8190fd5d21aa/105509b925ccec010cbc17a219dfd21036243cc7df597196036a4e0faacef855.jpg)


课程项目落点：设计 3-5 个 verifiable envs；比较 static difficulty vs adaptive difficulty；评估训练前后与 held-out env 泛化。

## 2. RL LLMs with Env Scaling

## 推荐框架&Baseline

https://github.com/Zhiyuan-Zeng/RLVE 

1张H200即可复现，2X40490也可以。（感谢24级同学苏浩阳同学帮忙复现）

## 推荐扩展方向

1 拓展到大模型上，现有是1.5b小模型

2 非手工设计去scaling env(文章中也指出了他们尝试了llm 设计env失败了)

3 ..... 


## Report 评分


Report rubric


<table><tr><td>Category</td><td>Poor (0–59%)</td><td>Fair (60–79%)</td><td>Good (80–89%)</td><td>Excellent (90–100%)</td></tr><tr><td>Method explanation</td><td>Descriptions are vague, incomplete, or mathematically incorrect.</td><td>The main methods are explained, but some motivation, equations, or implementation details are still thin.</td><td>Methods are described correctly, though some motivation or details may be thin.</td><td>Methods are described clearly and correctly, with enough mathematical and algorithmic detail that the reader can follow the design choices.</td></tr><tr><td>Empirical analysis</td><td>Results are shown but not interpreted, or the comparisons are not fair.</td><td>The report contains useful results, but some comparisons or conclusions are still limited.</td><td>Results are compared sensibly and the main conclusions are supported.</td><td>Results are compared carefully, failure cases are discussed honestly, and conclusions are clearly supported by both quantitative and qualitative evidence.</td></tr><tr><td>Communication quality</td><td>The report is hard to follow, missing important details, or poorly organized.</td><td>The report is readable, but organization or completeness still needs work.</td><td>The report is readable and mostly complete.</td><td>The report is well organized, concise, easy to follow, and answers the required questions directly.</td></tr></table>
