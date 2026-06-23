# Mini-paper outline (NeurIPS template, 4–8 pages)

Title: **Free-Energy Difficulty Control for RL on Verifiable Environments:
Generalizing SCALER's Success Set-Point**

Paste into the NeurIPS 2024 Overleaf template. Sections give the argument, the
equations to typeset, and which numbers to drop in from `results/REPORT.md` /
`results/metrics.csv` / `results/figures/*` (produced by `scaler_addon/analyze.py`).
〈angle brackets〉= fill-in once runs finish.

---

## Abstract (~150 words)
RL post-training of reasoning LLMs stalls when problem difficulty drifts off the
model's frontier: too-easy groups are all-correct, too-hard groups all-wrong, and
both give zero group-relative gradient. RLVE/SCALER keep problems solvable via
procedurally generated *verifiable environments* with adaptive difficulty; SCALER
regulates difficulty to a fixed success set-point (≈0.5). We show that set-point is
the `T→0` limit of a more general objective: minimizing a **free energy**
`F[q]=−E_q[U]−T·H[q]` over difficulty levels, whose optimum is the Gibbs policy
`q(d)∝exp(U(d)/T)` with utility `U(d)=1−p^G−(1−p)^G` (the informative-group
probability). On SCALER's verifiable environments with 〈Qwen2.5-3B-Instruct〉, we
compare static, SCALER-adaptive, and free-energy difficulty, measuring training-time
**effective sample ratio** and held-out generalization. 〈Free-energy keeps the
effective sample ratio highest and improves held-out accuracy by X% over static and
Y% over SCALER's set-point.〉

## 1. Introduction
- Problem: RL needs a *sustained* supply of mid-difficulty signal, not just more
  data. Fixed difficulty saturates (all-correct → no gradient) or stalls
  (all-wrong → no gradient).
- Line of work: RLVR → **RLVE** (adaptive verifiable environments) → **SCALER**
  (synthesizes verifiable environments from real code; verl-based; controls
  in-environment difficulty toward success≈0.5 and curates environments).
- Gap: SCALER's controller is a fixed set-point regulator with no explicit
  objective. Is 0.5 optimal, and how should difficulty *and* environment selection
  be chosen jointly?
- Contributions:
  1. A free-energy objective for difficulty (and environment) selection whose
     optimum is a Gibbs policy `q(d)∝exp(U(d)/T)`; **SCALER's set-point is the
     `T→0` special case** (§3, App. B).
  2. An online free-energy controller, a drop-in for SCALER's difficulty module
     (`scaler_addon/`), switchable at runtime, loading SCALER's own env JSONs.
  3. A controlled study on SCALER's verifiable environments — static vs adaptive
     vs free-energy — using the **effective sample ratio** as the mechanistic
     metric plus held-out generalization (§4–5).

## 2. Background
- **Verifiable environments**: procedural problem generator + algorithmic verifier
  (here SCALER's code-derived envs; rewards from unit tests via SandboxFusion);
  scalar difficulty `d`.
- **GRPO/DAPO**: group of `G` rollouts per prompt, group-relative advantage; a
  group with zero reward variance contributes no gradient (DAPO discards it).
- **SCALER controller**: continuous `d`, success-rate EMA, proportional update
  toward target 0.5; environment curation by difficulty-slope + recency.

## 3. Method — Free-Energy Difficulty Control  *(core)*
**3.1 Learning utility.** For binary reward with group success `p(d)`, the group is
informative iff `0<k<G`; its probability `U(d)=1−p(d)^G−(1−p(d))^G` (= a multiple of
the expected reward variance `p(1−p)`), maximized at `p=0.5` (App. A). Typeset
`U(d)`; small plot of `p(1−p)` and `U` vs `p` for `G=8`.

**3.2 Free-energy objective.** Choose a distribution `q` over difficulty levels
minimizing `F[q]=−E_q[U]−T·H[q]`. The optimum is `q*(d)∝exp(U(d)/T)` with
`F[q*]=−T·logΣexp(U/T)` (App. B). Limits: `T→0` → argmax (≈SCALER's 0.5 set-point);
`T→∞` → uniform; annealed `T` trades focus vs diversity and abandons saturated /
hopeless levels. Joint `U(e,d)` makes environment curation fall out of the same
objective (marginal = negative free energy).

**3.3 Online controller.** EMA estimate `p̂(d)` from observed group success →
recompute `U(d)` → sample next difficulties from `q*` → anneal `T_0→T_min`. Drop-in
for SCALER (`freeenergy_difficulty.py`; env-var gated via `apply_freeenergy_patch.py`).

## 4. Experimental setup
- **Environments**: 5 SCALER training envs (`scaler_make_arms.py`) + 3 held-out
  envs; rewards verified by SandboxFusion (g++ for C++ reference solutions).
- **Model / RL**: 〈Qwen2.5-3B-Instruct〉, GRPO (`adv_estimator=grpo`),
  `clip_high=0.2`, no KL, `G=8`, single GPU, 〈40〉 steps, identical compute per arm.
- **Arms** (same recipe, only difficulty strategy changes):
  `static-lo/mid/hi` (frozen `d`) · `SCALER-adaptive` (set-point 0.5) ·
  `free-energy` (ours, `q∝exp(U/T)`).
- **Metrics**: training-time **effective sample ratio** `1−p^G−(1−p)^G` and
  success rate per step; **mean proposed difficulty** per step; held-out
  benchmarks (MATH-500, AMC23, AIME24, MMLU-Pro, BBEH, GPQA) **before vs after**.

## 5. Results
〈Insert `results/figures/effective_sample_ratio.png`, `success_rate.png`,
`difficulty.png`, and the table from `results/REPORT.md`.〉
- **Effective sample ratio** (headline mechanism): static-easy collapses (→0,
  saturated), static-hard stays low (stalled), SCALER-adaptive holds it up,
  free-energy 〈≥ adaptive〉. (Synthetic-format sanity check already reproduces this
  ordering — easy 0.22 / hard 0.49 / adaptive 0.99.)
- **Difficulty trajectory**: static = flat lines; SCALER = climbs to the 0.5 band;
  free-energy = distribution over levels (report mean ± spread).
- **Held-out accuracy**: base (before) vs each arm (after); Δ per benchmark.
  Hypothesis: free-energy ≥ SCALER-adaptive > best static.
- **Ablation**: temperature `T` (and `T→0` numerically reproducing SCALER).

## 6. Discussion / failure cases
- When the model outgrows `d_max`, every level saturates and `U→0` for all `d`:
  signal collapses regardless of controller → motivates *environment scaling*
  (more/expandable envs) — ties back to the RLVE/SCALER thesis.
- Binary vs partial-credit rewards: with unit-test pass-fraction rewards the
  variance-maximizing point shifts off 0.5, where free-energy's explicit `U` is a
  cleaner target than a hand-set 0.5.

## 7. Related work
RLVR; RLVE (arXiv:2511.07317); SCALER (arXiv:2601.04809); GRPO/DAPO; automatic
curriculum / learning-progress; max-entropy RL & Boltzmann exploration (the Gibbs
form); free-energy / active inference (objective framing).

## 8. Conclusion
Casting difficulty (and environment) selection as free-energy minimization gives a
principled, single-knob generalization of SCALER's set-point, recovering it at
`T→0` and—〈per our results〉—sustaining more learning signal and better
generalization at equal compute.

---
### Appendices
- **A** `argmax_p U=0.5`, `U=1−p^G−(1−p)^G`, variance link (`paper/derivation.md`).
- **B** free-energy → Gibbs derivation, limits, env-curation unification (`derivation.md`).
- **C** exact SCALER changes + hyperparameters + env install notes (`scaler_addon/README.md`).
- **D** per-benchmark / per-difficulty tables (`results/`).

### Rubric self-check
- *Method*: §3 + App. A/B give objective, closed-form optimum, limits, online algo.
- *Empirical*: §5 fair same-compute arms, effective-sample-ratio mechanism,
  held-out before/after, honest failure cases (§6).
- *Communication*: one figure per claim; tables mirror `results/REPORT.md`; ≤8pp.
