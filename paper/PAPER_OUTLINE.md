# Mini-paper outline (NeurIPS template, 4–8 pages)

Title: **Free-Energy Difficulty Control for RL on Verifiable Environments:
Beyond RLVE's Accuracy-Threshold Heuristic**

Paste into the NeurIPS 2024 Overleaf template. 〈angle brackets〉= fill-in once
runs finish. Numbers come from the RLVE reproduction logs / eval scripts.

---

## Abstract (~150 words)
RL post-training of reasoning LLMs stalls when difficulty drifts off the model's
frontier: too-easy groups are all-correct, too-hard all-wrong, and both give zero
group-relative gradient. RLVE keeps problems solvable via 400 hand-engineered
verifiable environments with adaptive difficulty — raising difficulty only once
accuracy exceeds a threshold (τ_acc=0.9). We show that operating point is
sub-optimal: the group learning signal is the within-group reward variance
`p(1-p)`, maximal at success `p=0.5`, where the informative-group probability
`U(d)=1-p^G-(1-p)^G` ≈ 0.99 (G=8) vs ≈ 0.57 at p=0.9. We recast difficulty
selection as **free-energy minimization** `F[q]=-E_q[U]-T·H[q]`, whose optimum is
the Gibbs policy `q(d)∝exp(U(d)/T)`; temperature T trades exploitation of the
signal-optimal band against exploration. Reproducing RLVE from ProRL-1.5B-v2, we
compare RLVE's threshold rule, static difficulty, and free-energy on training-time
**effective sample ratio** and held-out generalization.

## 1. Introduction
- Problem: RL needs a *sustained* supply of mid-difficulty signal; fixed difficulty
  saturates (all-correct→no gradient) or stalls (all-wrong→no gradient).
- Line of work: RLVR → **RLVE** (Zeng et al., 2025): 400 verifiable environments,
  procedurally generated, algorithmic verifiers, adaptive difficulty; environment
  scaling improves generalization.
- Gap: RLVE's controller is a one-directional heuristic — raise difficulty when
  accuracy ≥ 0.9. It parks the model at high success (≈0.9), a *low-signal* regime,
  and never lowers difficulty; no explicit objective, no exploration term.
- Contributions:
  1. Show RLVE's 0.9 threshold is sub-optimal via the variance/informative-group
     argument; the signal-optimal operating point is p=0.5 (§3, App. A).
  2. A free-energy objective for difficulty selection with closed-form Gibbs
     optimum `q(d)∝exp(U(d)/T)`; T→0 = a hard set-point at the optimal 0.5, T→∞ =
     uniform exploration (§3, App. B).
  3. A dependency-free controller (`rlve_repro/freeenergy_controller.py`) dropped
     into RLVE's difficulty scheduling, and a controlled comparison —
     RLVE-threshold vs static vs free-energy — using the **effective sample
     ratio** as the mechanistic metric plus held-out generalization (§4–5).

## 2. Background
- **RLVE verifiable environments**: procedural generator + algorithmic verifier;
  scalar difficulty `d`; reward continuous/binary, env-specific. RLVE-Gym = 400
  environments + 50 held-out.
- **GRPO/DAPO**: group of `G` rollouts per prompt, group-relative advantage; a
  zero-variance group (all-correct/all-wrong) contributes no gradient (DAPO
  discards it).
- **RLVE difficulty controller**: per env, range `[ℓ,h]`, `d∼Uniform(ℓ,h)`; track
  accuracy `a/b` at the upper bound `h`; when `b≥τ_num` and `a/b≥τ_acc=0.9`,
  `h←h+1`, slide `ℓ` (window `d_Δ=4`). One-directional.

## 3. Method — Free-Energy Difficulty Control  *(core)*
**3.1 Learning utility.** For binary reward, group success `p(d)`, the group is
informative iff `0<k<G`; `U(d)=1−p^G−(1−p)^G` (a multiple of `p(1−p)`), max at 0.5
(App. A). RLVE's `p≈0.9` gives `U≈0.57 ≪ U(0.5)≈0.99` (G=8) → most groups wasted.
Typeset `U(d)` and a `p(1−p)` / `U` vs `p` plot for G=8.

**3.2 Free-energy objective.** Choose a distribution `q` over difficulty levels
minimizing `F[q]=−E_q[U]−T·H[q]`. Optimum `q*(d)∝exp(U(d)/T)`,
`F[q*]=−T·logΣexp(U/T)` (App. B). T→0 → hard set-point at argmax U (the 0.5 band,
not RLVE's 0.9); T→∞ → uniform; annealed T focuses on informative levels while
probing neighbours and abandoning saturated / hopeless ones. Joint `U(e,d)` yields
an environment-selection weight (negative free energy) from the same objective.

**3.3 Online controller.** EMA `p̂(d)` from observed group success → recompute
`U(d)` → sample next difficulties from `q*` → anneal `T_0→T_min`. Implemented as
`rlve_repro/freeenergy_controller.py` (no SLIME/torch dep; self-checked: T→0
concentrates on the p=0.5 level). Integration point = RLVE's SLIME rollout
difficulty scheduler (replaces the acc≥0.9 bump).

## 4. Experimental setup
- **Reproduction**: official RLVE (`Zhiyuan-Zeng/RLVE`, SLIME backend) from
  **ProRL-1.5B-v2**, single H200 (8→1 GPU config; see `rlve_repro/`).
- **Environments**: a subset of RLVE-Gym for training + held-out environments;
  scale env count (1/4/16/...) to also reproduce RLVE's env-scaling trend.
- **Arms** (same recipe, only the difficulty controller changes):
  static-d (frozen) · **RLVE-threshold** (acc≥0.9 bump, baseline) ·
  **free-energy** (ours, `q∝exp(U/T)`).
- **Metrics**: training-time **effective sample ratio** `1−p^G−(1−p)^G` and
  success rate per step; mean proposed difficulty per step; held-out benchmarks
  (AIME 2024/25, OMEGA, OlympiadBench, LiveCodeBench, BBEH) before vs after.

## 5. Results  〈fill from logs〉
- **Effective sample ratio** (headline): static collapses (saturated/stalled),
  RLVE-threshold sits at the 0.9 operating point (`U≈0.57`), free-energy holds the
  0.5 band (`U≈0.99`). Plot vs step.
- **Difficulty trajectory**: static flat; RLVE bumps up then sits at ~0.9 success;
  free-energy = distribution targeting 0.5, contracting as T anneals.
- **Held-out accuracy / env-scaling**: base (before) vs each arm (after); Δ per
  benchmark; reproduce "more environments → better held-out".
- **Temperature ablation**: sweep T; numerically show T→0 = fixed 0.5 set-point.

## 6. Discussion / failure cases
- When the model outgrows `d_max`, all levels saturate, `U→0` for every `d`: no
  controller has signal to exploit → motivates *environment scaling* (RLVE's
  thesis), the natural extension.
- Binary vs partial-credit rewards: the variance-max point shifts off 0.5, where
  free-energy's explicit `U` is a cleaner target than a hand-set threshold.

## 7. Related work
RLVR; **RLVE** (arXiv:2511.07317); GRPO/DAPO; ProRL; automatic curriculum /
learning-progress; max-entropy RL & Boltzmann exploration (the Gibbs form);
free-energy / active inference (objective framing).

## 8. Conclusion
Casting difficulty selection as free-energy minimization replaces RLVE's
accuracy-threshold heuristic with a principled, single-knob controller that targets
the signal-optimal operating point and—〈per our results〉—sustains more learning
signal and better generalization at equal compute.

---
### Appendices
- **A** `argmax_p U=0.5`, `U=1−p^G−(1−p)^G`, variance link, and why RLVE's 0.9 is
  low-signal (`paper/derivation.md`).
- **B** free-energy → Gibbs derivation, temperature limits, env-curation
  unification (`derivation.md`).
- **C** exact RLVE single-GPU reproduction steps + controller integration
  (`rlve_repro/README.md`).
- **D** per-benchmark / per-difficulty tables.

### Rubric self-check
- *Method*: §3 + App. A/B — objective, closed-form optimum, limits, online algo.
- *Empirical*: §5 same-compute arms, effective-sample-ratio mechanism, held-out
  before/after, honest failure cases (§6).
- *Communication*: one figure per claim; ≤8 pp.
