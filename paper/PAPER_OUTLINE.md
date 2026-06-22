# Mini-paper outline (NeurIPS 2024 template, 4–8 pages)

Title: **Targeting the Learning Signal: Adaptive Difficulty as Closed-Loop
Control of Reward Variance in RL for Language Models**

Paste into the Overleaf NeurIPS template
(https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh). Sections
below give the argument, the equations to typeset, and exactly which numbers to
drop in from `results/REPORT.md`. Text in〈angle brackets〉is a fill-in.

---

## Abstract (≈150 words)
RL post-training of LMs depends on a reward signal that vanishes when problems
are too easy (all-correct) or too hard (all-wrong). For group-relative methods
(GRPO/DAPO) the per-prompt gradient is proportional to within-group reward
variance, which for binary rewards is `p(1−p)`, maximal at success rate `p=0.5`.
RLVE keeps problems solvable via procedurally generated verifiable environments
and raises difficulty once the model is *proficient* (acc ≥ 0.9). We argue 0.9
is the wrong target and introduce **STAD**, a closed-loop controller that
servo-controls difficulty to the signal-maximizing band `p*≈0.5`, plus a
learning-progress environment sampler. On 5 training + 3 held-out verifiable
environments with Qwen2.5-1.5B-Instruct, STAD 〈sustains a higher effective
sample ratio and improves held-out accuracy by X% over the RLVE bump rule and
Y% over static difficulty〉. A controlled synthetic study isolates the mechanism.

## 1. Introduction
- Problem: "RL不是只缺更多题，而是缺能持续产生适中难度学习信号的环境." Fixed
  question banks saturate (all-correct → no gradient) or stall (all-wrong).
- RLVE's answer: infinite procedurally-generated verifiable environments with
  adaptive difficulty; environment engineering as an RL-scaling axis.
- **Gap we attack:** RLVE's controller is a *heuristic* — raise the difficulty
  ceiling once accuracy ≥ τ=0.9, never lower it, sample environments uniformly.
  We reframe "适中难度" as a precise control objective: hold each environment at
  the success rate that maximizes the GRPO learning signal.
- **Contributions:**
  1. A derivation identifying `p*=0.5` as the difficulty that maximizes the
     group-relative gradient signal for binary-verifiable rewards (§2).
  2. **STAD**, a bidirectional proportional-integral difficulty controller that
     targets `p*`, and a **learning-progress** environment sampler (§3).
  3. A single-GPU, pip-installable RLVE reimplementation (TRL+vLLM) with 8
     verifiable environments and a fair held-out evaluation protocol (§4).
  4. Empirical evidence (real GRPO runs + a controlled synthetic study) that
     targeting the signal band beats both the bump rule and static difficulty,
     with an honest analysis of when the sampler does and does not help (§5).

## 2. Background and the signal-variance argument  ← *key "method" content*
**2.1 GRPO/DAPO.** For a prompt `x`, sample a group of `G` completions; with
rewards `r_1..r_G`, the group-relative advantage of completion `j` is
```
A_j = (r_j − mean_i r_i) / (std_i r_i + ε).
```
The policy-gradient magnitude contributed by the group scales with the spread of
its rewards; a group with zero reward variance contributes zero advantage. DAPO
makes this explicit by *discarding* groups whose rollouts share an identical
reward (dynamic sampling).

**2.2 Binary rewards ⇒ signal is `p(1−p)`.** With a verifier giving `r∈{0,1}`
and group success probability `p`, the number correct `k∼Binomial(G,p)`. The
group is *informative* iff `0<k<G`; its probability is
```
P_inf(p) = 1 − p^G − (1−p)^G ,           argmax_p P_inf = 0.5 .
```
The expected within-group reward variance equals the Bernoulli variance
`p(1−p)`, also maximized at `p=0.5`. Hence the difficulty that maximizes the
expected learning signal is the one yielding **50% success** — not the 90%
"proficiency" that RLVE waits for. *(Typeset `P_inf` and a small plot of
`p(1−p)` and `P_inf` for `G=8`; both peak at 0.5.)*

**2.3 RLVE controller (baseline we extend).** Per environment, keep a difficulty
range `[ℓ,h]`; sample `d∼UniformInt(ℓ,h)`; track `(a,b)` = (correct, total) at
the upper bound `h`; once `b≥τ_num` and `a/b≥τ_acc=0.9`, set `h←h+1` and slide
`ℓ←h−d_Δ+1` (`d_Δ=4`). One-directional; environments sampled uniformly.

## 3. Method
**3.1 Effective sample ratio (the quantity we optimize).** Per step, over all
groups, `ESR = (#groups with 0<k<G)/(#groups)`. This is the trainable fraction
of the rollout budget — the operational reading of RLVE's "effective prompt
ratio".

**3.2 STAD controller (ours).** Hold a *continuous* difficulty `μ` per env.
Measure the EMA-smoothed pooled success rate `s̄` at the current operating point
and apply a PI update toward `p*`:
```
s̄_t   = β·s̄_{t−1} + (1−β)·s_t
μ_{t+1} = clip( μ_t + k_p·(s̄_t − p*) + k_i·Σ_τ(s̄_τ − p*),  [d_min, d_max] )
d ∼ stochastic-round(μ)  with prob ε pick a neighbor ±1   (exploration)
```
`s̄>p*` ⇒ too easy ⇒ μ increases; `s̄<p*` ⇒ too hard ⇒ μ decreases. Defaults:
`p*=0.5, k_p=1.5, k_i=0, β=0.7, ε=0.15`. Contrast with the bump rule: STAD is
**bidirectional** and targets the *signal-maximizing* rate, not proficiency.

**3.3 Learning-progress sampler (ours).** Replace uniform env selection with a
softmax bandit over a per-env signal score (EMA of that env's ESR), with a
uniform floor so no env starves:
```
score_e ← (1−δ)·score_e + δ·ESR_e ;  w_e = (1−η)·softmax(score_e/T)_e + η/N .
```
Spends more rollouts where the model currently extracts the most gradient.

## 4. Experimental setup
- **Environments.** 5 training (`arithmetic, sorting, gcd, linear_equation,
  counting`) + 3 held-out (`base_conversion, interval_scheduling, modular_exp`);
  each procedurally generated, parametric difficulty `d∈[0,d_max]`, algorithmic
  verifier extracting `\boxed{}`. Table of generators/verifiers in App. B.
- **Model & RL.** 〈Qwen2.5-1.5B-Instruct〉; GRPO with `dr_grpo` loss,
  clip-higher `ε_high=0.28`, no KL (`β=0`), `G=8`, 8 prompts/step,
  lr `1e-6`, 〈200〉 steps, vLLM rollouts.
- **Conditions.** `static (d=4)`, `threshold (RLVE)`, `stad`, `stad_lp` — all
  start from `d=0`. Identical compute (same #steps × #rollouts).
- **Evaluation (fair).** One fixed eval set, identical across models: per env,
  difficulties `{1,3,5,7,9}`, 16 problems each, greedy pass@1. Report (a) train
  envs, (b) **unseen high difficulties** of train envs, (c) **held-out envs**.
  Improvements measured vs the **base model** on the same set.

## 5. Results
**5.1 Mechanism (synthetic study, `tools/simulate.py`).** Drive the *exact same*
controllers with a synthetic policy whose per-env ability rises in proportion to
informative groups received (the GRPO premise). 〈Insert `results/figures/
simulation.png` + the simulation table from REPORT.md.〉 Representative run:

| condition | mean final ability | ESR (all) | ESR (last 50) | success (last 50) |
|---|---|---|---|---|
| static-easy | 5.24 | 0.67 | 0.49 | 0.85 |
| static-hard | 0.06 | 0.01 | 0.01 | 0.00 |
| threshold (RLVE) | 4.25 | 0.55 | 0.33 | 0.95 |
| **STAD (ours)** | **8.36** | **0.97** | **0.97** | **0.54** |
| STAD+LP | 8.36 | 0.96 | 0.97 | 0.54 |

Reading: static-hard never learns (no signal); static-easy and the bump rule
both drift to high success (low ESR); **STAD holds success at ≈0.5 and ESR≈1,
roughly doubling learned ability at equal compute.** LP helps the *static*
controller (5.24→5.37, ESR 0.49→0.63) but is redundant once STAD already
equalizes signal across envs — an honest, informative negative.

**5.2 Real GRPO runs.** 〈Insert `results/figures/training_dynamics.png` and
`eval_accuracy.png`, plus the eval table from REPORT.md.〉 State for each:
- *Training dynamics:* effective sample ratio and success over steps per
  condition (hypothesis H2: STAD's ESR stays highest; threshold's success sits
  near 0.9; static saturates or stalls).
- *Difficulty trajectories:* `results/figures/difficulty_trajectories.png` — STAD
  `μ` settles where success≈0.5; threshold `h` only climbs.
- *Held-out accuracy (headline):* base vs static vs threshold vs STAD vs STAD+LP.
  Report absolute Δ over base on held-out envs (H1: STAD ≥ threshold > static).
- *Difficulty generalization:* accuracy at unseen high difficulties.

**5.3 Failure cases & honesty.** Discuss: (i) where STAD ≈ threshold (if the
base model is already near 50% at low d); (ii) LP redundancy under STAD; (iii)
difficulty-range exhaustion — when STAD pushes an env to `d_max` and signal
collapses, motivating *environment scaling* (more/expandable envs), tying back to
RLVE's central thesis.

## 6. Related work
RLVR (verifiable rewards), RLVE (adaptive verifiable environments; the baseline),
GRPO/DAPO, automatic curriculum learning / teacher-student & learning-progress
curricula (the lineage of our sampler), self-play / PCG difficulty adaptation.
Position STAD as bringing *control-theoretic* difficulty targeting to RLVE.

## 7. Limitations
Single 1.5B model and small env suite (compute-bound); synthetic study makes a
modeling assumption (learning ∝ informative groups); `p*=0.5` assumes binary
rewards (shaped/continuous rewards shift the optimum); verifier coverage; LP's
situational value.

## 8. Conclusion
Reframing "适中难度" as closed-loop control of reward variance gives a simple,
principled, bidirectional difficulty rule that extracts more signal per rollout
than RLVE's proficiency bump, at single-GPU cost.

---
### Appendix A — derivation of `argmax_p P_inf(p)=0.5` and of `E[Var]=p(1−p)`.
### Appendix B — per-environment generators, difficulty maps, verifiers (from `rlve/envs/`).
### Appendix C — full hyperparameters (`results/runs/*/run_config.json`).
### Appendix D — extra plots, per-env/per-difficulty accuracy tables.

### Rubric self-check
- *Method explanation:* §2–3 give equations + motivation (variance argument,
  PI control, sampler) → aim "Excellent".
- *Empirical analysis:* §5 fair fixed-eval comparison, base-relative deltas,
  held-out generalization, **failure cases discussed honestly**, synthetic study
  isolating the mechanism.
- *Communication:* one figure per claim, tables mirror `REPORT.md`, ≤8 pages.
