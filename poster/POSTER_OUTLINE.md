# Poster outline (for `poster模板.pptx`, June 30 2026)

Read in 60 seconds: one claim, one equation, two figures.

## Title bar
**Free-Energy Difficulty Control for RL on Verifiable Environments** ·
〈names〉 · beyond RLVE's accuracy-threshold heuristic

## Panel 1 — Motivation (top-left)
Cartoon: easy box (✓✓✓✓ → flat, no gradient) | hard box (✗✗✗✗ → no gradient) |
mid box (✓✓✗✗ → gradient!). Line: *RL needs a sustained supply of mid-difficulty
signal, not just more data.*

## Panel 2 — Key idea (center-top, make the equation big)
- Group learning signal `∝ p(1−p)`; informative-group prob `U(d)=1−pᴳ−(1−p)ᴳ`,
  peak at `p=0.5` (plot for G=8).
- RLVE raises difficulty only at **acc ≥ 0.9** → sits at `U(0.9)≈0.57`, low signal.
- Don't pick a threshold — minimize **free energy** `F[q]=−E_q[U]−T·H[q]`
  ⇒ **`q(d) ∝ exp(U(d)/T)`**, targeting the `U`-max band.

## Panel 3 — Method (left-center)
- Free-energy controller: EMA `p̂(d)` → `U(d)` → sample `q∝exp(U/T)` → anneal `T`.
- `T→0` = hard set-point at the optimal 0.5 (vs RLVE's 0.9) · `T→∞` = explore ·
  same objective also gives an environment-selection weight (negative free energy).
- Diagram: generate → verify (RLVE verifier ✓/✗) → estimate U → sample difficulty (loop).

## Panel 4 — Setup (small, left-bottom)
Reproduce RLVE (SLIME) from **ProRL-1.5B-v2**, single H200; RLVE-Gym verifiable
envs + held-out. Arms: static-d · RLVE-threshold(0.9) · **free-energy (ours)**.

## Panel 5 — Results (right half, biggest)
- **Fig A**: effective sample ratio vs step — static collapses, RLVE sits at 0.9,
  free-energy holds 0.5 〈highest〉.
- **Fig B**: held-out accuracy, base vs arms (bars) + env-scaling trend.
- Headline box: *free-energy 〈+X%〉 held-out vs RLVE threshold, equal compute.*

## Panel 6 — Takeaways (bottom-right)
1. Difficulty = a free-energy control target, not an accuracy threshold.
2. RLVE's 0.9 is a low-signal operating point; the signal-optimum is 0.5.
3. One temperature knob unifies exploit/explore and difficulty/environment;
   honest limit — when the model outgrows `d_max`, signal collapses → need env scaling.

Footer: RLVE arXiv:2511.07317 · github.com/Zhiyuan-Zeng/RLVE · code: github.com/0xLian117/RLproj
