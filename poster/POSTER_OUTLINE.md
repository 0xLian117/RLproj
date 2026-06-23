# Poster outline (for `poster模板.pptx`, June 30 2026)

Read in 60 seconds: one claim, one equation, two figures.

## Title bar
**Free-Energy Difficulty Control for RL on Verifiable Environments** ·
〈names〉 · generalizing SCALER's success set-point

## Panel 1 — Motivation (top-left)
Cartoon: easy box (✓✓✓✓ → flat, no gradient) | hard box (✗✗✗✗ → no gradient) |
mid box (✓✓✗✗ → gradient!). Line: *RL needs a sustained supply of mid-difficulty
signal, not just more data.*

## Panel 2 — Key idea (center-top, make the equation big)
- Group learning signal `∝ p(1−p)`; informative-group prob `U(d)=1−pᴳ−(1−p)ᴳ`,
  peak at `p=0.5` (plot for G=8).
- Don't pick one set-point — minimize **free energy** `F[q]=−E_q[U]−T·H[q]`
  ⇒ **`q(d) ∝ exp(U(d)/T)`**.
- Punchline: *SCALER's 0.5 set-point = the `T→0` limit of our objective.*

## Panel 3 — Method (left-center)
- Free-energy controller: EMA `p̂(d)` → `U(d)` → sample `q∝exp(U/T)` → anneal `T`.
- `T→0` exploit (≈SCALER) · `T→∞` explore (diverse) · same objective also gives an
  environment-curation weight (negative free energy).
- Diagram: generate → verify (SandboxFusion ✓/✗) → estimate U → sample difficulty (loop).

## Panel 4 — Setup (small, left-bottom)
SCALER verifiable envs (5 train / 3 held-out), Qwen2.5-3B-Instruct, GRPO, single GPU.
Arms: static-lo/mid/hi · SCALER-adaptive · **free-energy (ours)**.

## Panel 5 — Results (right half, biggest)
- **Fig A**: effective sample ratio vs step — static collapses, adaptive holds,
  free-energy 〈highest〉 (`results/figures/effective_sample_ratio.png`).
- **Fig B**: held-out accuracy, base vs arms (bars).
- Headline box: *free-energy 〈+X%〉 held-out vs static, 〈+Y%〉 vs SCALER set-point,
  equal compute.*

## Panel 6 — Takeaways (bottom-right)
1. Difficulty = a free-energy control target, not a hand-set 0.5.
2. One temperature knob unifies exploit/explore and difficulty/environment.
3. SCALER is recovered at `T→0`; honest limit — when the model outgrows `d_max`,
   signal collapses → need environment scaling.

Footer: SCALER arXiv:2601.04809 · RLVE arXiv:2511.07317 · code: github.com/0xLian117/RLproj
