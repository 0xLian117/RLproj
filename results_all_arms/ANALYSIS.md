# Why these results look the way they do — mechanism & improvement analysis

*Companion to `REPORT.md`. The scoreboard (mean success, eff-ratio, held-out) is
not self-explanatory; this file reads the **process** behind it so we don't draw
results-only conclusions. Backing figure: `figures/diagnostics.png`.*

Run: ProRL-1.5B-v2, 4 training envs, 3 arms, **30 rollout steps** each, GRPO
K=4, `dmax=16`, `RESP_LEN=2048`, eval every ~10 steps (steps 0/9/19/29 on 128
held-out envs). Controllers confirmed active in the logs:
`[FEP-RLVE] mode=fep … lambda_info=1.0` and `… mode=signal … lambda_info=0.0`.

## 1. The headline number is an artifact of the controller, not of learning

`REPORT.md` shows training success parked near **0.5** for all arms and a small
FEP edge in last-5 success (0.531). Read literally this looks like "stable, well-
regulated training." The logs say otherwise:

| diagnostic | what the logs show | what it means |
|---|---|---|
| held-out reward (4 evals/arm) | adaptive −0.906→−0.891, signal −0.922→−0.813, fep −0.906→−0.875 | **flat & floored** (~3–9% success); no arm generalizes; gaps are noise |
| `pg_loss` magnitude | ~±2–3e-4 across all arms/steps | policy-gradient signal ~2 orders below a healthy RL run |
| `rollout/truncated` | pinned **0.7–0.9** for most of the run | most rollouts hit the ~2048 length cap |
| `entropy_loss` | adaptive 0.93→0.54, fep 1.13→0.65 (~40% drop) | policy **is** moving — it sharpens — but not toward transfer |
| training success trend | adaptive −0.075, fep +0.097, signal +0.020 (first5→last5) | tiny, and confounded by difficulty selection |

The key reframing: **with the policy barely improving on the task, the only way
training success can sit at ≈0.5 is that the difficulty controller keeps feeding
the model problems at its current ability.** A flat 0.5 curve is therefore
evidence the *controller* works — not that the *model* is getting better. The
arms can't separate on capability because none of them moved capability much in
30 steps.

So FEP's last-5 advantage is real but it is a statement about **regulation
quality** (FEP parks success nearest the maximally-informative 0.5 set-point in
steady state, exactly as the Gibbs-optimal theory predicts), **not** a downstream
capability gain. We should claim the former and not over-claim the latter.

## 2. Mechanism chain — why learning stalled

```
1.5B model on hard math/code envs
      │
      ▼  generates long chains, no early answer
truncation 0.7–0.9 at the ~2048 cap
      │
      ▼  truncated == failure (reward conflates "out of tokens" with "wrong")
many GRPO groups are DEGENERATE: most/all K=4 samples fail
      │
      ▼  identical reward within a group ⇒ zero advantage
effective gradient collapses (pg_loss ~2e-4)
      │
      ▼  30 steps is far too short to escape this regime
held-out stays at the floor; entropy collapses (premature commitment)
```

## 3. The theory ↔ reality gap (the important part for the paper)

Our effective-sample-ratio and the free-energy utility both assume each of the
K samples in a group is an **independent Bernoulli(p) draw of correctness**, so
the expected informative-group fraction is `U_K(p) = 1 − p^K − (1−p)^K ≈ 0.85`
here — which is exactly why eff-ratio looks healthy (~0.85) and *identical*
across all three arms.

But truncation breaks the independence/identifiability assumption: when 70–90%
of completions are cut off, group outcomes become **correlated and degenerate**
(whole groups truncate → all fail), so the **realized** informative fraction is
far below the theoretical `U_K`. The controller is optimizing an *expected*
informativeness that the truncation regime never delivers. eff-ratio can't see
this because it is computed from the same closed-form `U_K` — it is blind to the
degeneracy by construction. **This gap, not the choice of difficulty arm, is the
binding constraint in this run.**

## 4. What this run does and does not establish

- **Does:** the FEP controller regulates difficulty as designed — it holds
  success closest to the informative 0.5 point in steady state, and T→0 / λ_info=0
  ablations behave as the theory says. Controller validation ✓.
- **Does not:** any generalization win. Held-out is floored and flat for all
  arms; 30 steps + truncation-dominated reward leave no room for transfer to
  appear. Do not put a "FEP generalizes better" claim on the poster from this run.

## 4b. Reconciliation with `ANALYSIS_REPORT.md`

A sibling results-level report (`ANALYSIS_REPORT.md`) headlines **Signal-RLVE**
as "the strongest empirical result" because its held-out delta is largest
(−0.922 → −0.812, **+0.109**). We read the *same numbers* more conservatively,
and the two docs should be reconciled rather than both quoted:

- Held-out is reward on **128 envs**. −0.922 → −0.812 is success ~3.9% → ~9.4%,
  i.e. ≈ 5 more problems solved out of 128. The binomial SE at p≈0.07, n=128 is
  ≈ 2.3% (~3 problems), so +5.5pp is **~1.7 SE** — suggestive, not significant,
  from a **single seed** with only **4 eval points** and a noisy trajectory
  (signal first *dropped* to −0.953 at eval 9 before rising).
- All held-out evals are themselves heavily truncated, so the held-out scale is
  truncation-suppressed for every arm (§2–§3).

**Where the two reports agree (and what to actually put on the poster):** (i)
truncation is the binding caveat; (ii) the arm with the best *training* success
(adaptive/RLVE-90) is *not* the best on held-out — which is the project's core
motivation that training accuracy is the wrong curriculum signal; (iii) results
are preliminary/short-budget. The defensible joint claim is *"training success
does not predict generalization, and difficulty objectives that target the
informative band are the right direction"* — **not** a ranked "Signal > FEP"
verdict, which this run's noise floor cannot support.

## 5. Improvement directions (prioritized, each tied to a diagnosis above)

1. **Kill the truncation bottleneck first (highest leverage).** Raise
   `max_response_length` (2048 → 4k–8k) and/or add length budgeting so the model
   emits a boxed answer before the cap. Until groups stop being degenerate-
   truncated, *no* difficulty controller can extract signal (§2, §3).
2. **Decouple "truncated" from "wrong" in the reward.** Give truncation a
   distinct/milder penalty or partial credit for correct-but-cut reasoning, so
   within-group reward variance — and therefore advantage — survives (§2).
3. **Make the controller chase *realized* informativeness, not expected `U_K(p)`.**
   Feed the *measured* non-degenerate-group fraction (or batch advantage variance)
   back into the utility: `U(d) ← realized informative-group rate at difficulty d`.
   This closes the §3 gap and is a clean, novel extension of the free-energy
   formulation — arguably the strongest paper contribution to come out of this run.
4. **Train ≥200–300 steps** before reading held-out. 30 steps cannot move
   generalization; re-run the FEP arm long as the headline curve (§1).
5. **Get held-out off the floor so arms can separate.** Either scale the model or
   build a held-out band within a 1.5B model's reach; the current 128-env set sits
   at ~3–9% success, too hard to resolve between-arm differences (§1).
6. **Instrument process metrics as first-class panels** (now in
   `figures/diagnostics.png`; should also live in `analyze_rlve.py`): truncated-
   fail rate, non-degenerate-group fraction, advantage std, entropy, held-out
   trajectory, difficulty-vs-step. Keeps future analysis process-aware.
7. **Watch entropy collapse over a long run** (it fell ~40% in 30 steps while
   held-out stayed flat → premature commitment). Keep an entropy bonus / KL-to-ref
   to preserve exploration when the run is extended (§2).

_Evidence regenerable from the logs: `python rlve_repro/diagnostics.py`
(pg_loss / entropy from the per-step driver logs, truncated/success from
`metrics.csv`, held-out from the worker `*.out`)._
