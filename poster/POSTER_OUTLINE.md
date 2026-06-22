# Poster outline (for `poster模板.pptx`, June 30 2026 session)

A poster is read in 60 seconds — one bold claim, one equation, two figures.

## Title bar
**Targeting the Learning Signal: Adaptive Difficulty as Closed-Loop Control of
Reward Variance in RL for LMs** · 〈names〉 · RLVE-lite

## Panel 1 — Motivation (top-left)
- Cartoon: easy box (✓✓✓✓ → flat, no gradient) | hard box (✗✗✗✗ → no gradient) |
  middle box (✓✓✗✗ → gradient!).
- One line: *"RL doesn't just need more problems — it needs problems at the
  difficulty that keeps producing a learning signal."*

## Panel 2 — Key insight (the equation, center-top, make it big)
- `Signal ∝ within-group reward variance = p(1−p)`, maximal at **p* = 0.5**.
- `P(informative group) = 1 − pᴳ − (1−p)ᴳ`, also peaks at 0.5.
- Small plot of `p(1−p)` and `P_inf` for G=8 (mark the peak at 0.5).
- Punchline: *RLVE waits for 90% accuracy to raise difficulty — we target 50%,
  the signal-maximizing band.*

## Panel 3 — Method (left-center)
- **STAD** controller (one box): `μ ← clip(μ + k_p(s̄ − p*))`, bidirectional
  (↑ if too easy, ↓ if too hard). vs RLVE's one-way "bump at 0.9".
- **Learning-progress sampler** (one box): softmax bandit over each env's
  effective sample ratio + uniform floor.
- Diagram: generate → verify (✓/✗) → measure success → adjust difficulty (loop).

## Panel 4 — Setup (small, left-bottom)
- 8 verifiable envs (5 train / 3 held-out), Qwen2.5-1.5B-Instruct, GRPO,
  single GPU (TRL+vLLM). 4 conditions: static / RLVE-threshold / STAD / STAD+LP.

## Panel 5 — Results (right half, the biggest panels)
- **Fig A:** effective sample ratio vs steps — STAD stays high, threshold/static
  decay (`results/figures/training_dynamics.png`).
- **Fig B:** held-out accuracy bars, base vs 4 conditions
  (`results/figures/eval_accuracy.png`).
- Headline number box: *"STAD: +〈X〉% held-out vs RLVE bump, +〈Y〉% vs static, at
  equal compute."*
- Synthetic-study mini-table (mechanism): STAD ≈2× learned ability, ESR 0.97 vs
  0.55.

## Panel 6 — Takeaways (bottom-right)
1. "适中难度" = a control target, not a heuristic: hold success at p*≈0.5.
2. Bidirectional > one-way bump; targets signal, not proficiency.
3. Honest: the env sampler helps weak controllers but is redundant once STAD
   equalizes signal; pushing difficulty to the ceiling motivates env scaling.

Footer: arXiv:2511.07317 (RLVE baseline) · code: github.com/0xLian117/RLproj
