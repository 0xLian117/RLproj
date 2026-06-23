# Mini-paper outline (NeurIPS template, 4–8 pages)

Title: **FEP-RLVE: Difficulty Selection as Active Inference for RL on Verifiable
Environments**

Paste into the NeurIPS Overleaf template. 〈angle brackets〉= fill-in once runs
finish. Numbers come from the RLVE reproduction logs / `results/`.

> **Framing note.** This is Friston's *Free Energy Principle* (active inference),
> **not** a thermodynamic "energy − entropy" objective. Difficulty selection is an
> *action*: maintain a belief over the model's competence, update it by minimizing
> **variational** free energy (perception), and pick the next difficulty by
> minimizing **expected** free energy (pragmatic + epistemic). The RL algorithm
> (GRPO/DAPO) is unchanged.

---

## Abstract (~150 words)
RL post-training of reasoning LLMs stalls when difficulty drifts off the model's
capability frontier: too-easy groups are all-correct, too-hard all-wrong, and both
give zero group-relative gradient (Zeng et al., 2025; SCALER, 2026). RLVE keeps
problems solvable via verifiable environments with adaptive difficulty — but raises
difficulty only once accuracy exceeds a threshold (τ_acc=0.9), a one-directional
heuristic that parks the model in a low-signal regime (informative-group
probability `U_K(0.9)≈0.57 ≪ U_K(0.5)≈0.99`, K=8). We recast difficulty selection
as **active inference**: the trainer keeps a belief `q(s_e)` over each
environment's latent competence, updates it by variational free energy
(Bayesian), and selects the next difficulty by minimizing **expected free
energy** `G_e(d) = λ_pref·(−log U_K(p̂)) − λ_info·I(s;o|d) + λ_cost·C(d)`, sampling
`π(d)∝exp(−G/T)`. The pragmatic term is the preference risk `−log U_K`, minimal at
the signal-optimal `p=0.5` band and `→+∞` for degenerate difficulties; the
epistemic term actively probes competence and moves difficulty *both* ways.
Reproducing RLVE from ProRL-1.5B-v2 on a single H200, we compare static, RLVE's
threshold, a pragmatic-only ablation (Signal-RLVE), and full FEP-RLVE on
training-time effective sample ratio and held-out generalization.

## 1. Introduction
- Problem: RL needs a *sustained* supply of mid-difficulty signal; difficulty
  misaligned with capability → reward sparsity (too hard) or signal decay /
  pattern-collapse (too easy) (SCALER, arXiv:2601.04809).
- Line of work: RLVR → **RLVE** (Zeng et al., 2025, arXiv:2511.07317): verifiable
  environments, procedural generation, algorithmic verifiers, adaptive difficulty;
  environment scaling improves generalization.
- Gap: RLVE's controller is a one-directional *heuristic* — raise difficulty when
  accuracy ≥ 0.9. It parks the model at high success (≈0.9), a *low-signal* regime,
  never lowers difficulty, has no explicit objective and no exploration term.
- Contributions:
  1. Show RLVE's 0.9 threshold is sub-optimal via the variance/informative-group
     argument; the signal-optimal operating point is p=0.5 (§3, App. A).
  2. Recast difficulty selection as **active inference**: a generative model of
     competence, variational-free-energy belief update, and an **expected free
     energy** objective whose pragmatic term is the preference risk `−log U_K` and whose epistemic term is
     the information gain about competence (§3, App. B).
  3. A dependency-free EFE controller (`rlve_repro/active_inference_controller.py`)
     dropped into RLVE's difficulty scheduler, and a controlled 4-arm comparison
     (static · RLVE-90 · Signal-RLVE · FEP-RLVE) isolating the epistemic term, on
     the **effective sample ratio** plus held-out generalization (§4–5).

## 2. Background
- **RLVE verifiable environments**: procedural generator + algorithmic verifier;
  scalar difficulty `d`; reward continuous/binary, env-specific.
- **GRPO/DAPO**: group of `K` rollouts per prompt, group-relative advantage; a
  zero-variance group (all-correct/all-wrong) contributes no gradient (DAPO
  discards it). Hence the trainable rollouts live in *informative* groups.
- **RLVE difficulty controller**: per env, range `[ℓ,h]`, `d∼Uniform(ℓ,h)`; track
  accuracy at the upper bound `h`; when enough samples and `acc≥τ_acc=0.9`,
  `h←h+1`, slide the window. One-directional.
- **Active inference / FEP** (Friston): perception = minimize variational free
  energy (≈ Bayesian belief update); action = minimize expected free energy
  (pragmatic preference + epistemic information gain). (Friston 2010; Da Costa et
  al. on EFE / Bayesian optimal design.)

## 3. Method — Difficulty Selection as Active Inference  *(core)*
**3.1 Learning utility.** For binary reward and group success `p(d)`, a group is
informative iff `0<k<K`; `U_K(d)=1−p^K−(1−p)^K` (a multiple of `p(1−p)`), max at
0.5 (App. A). RLVE's `p≈0.9` gives `U_K≈0.57 ≪ U_K(0.5)≈0.99` (K=8). Plot
`p(1−p)` / `U_K` vs `p` for K=8.

**3.2 Generative model + belief update (perception).** Latent competence `s_e`;
IRT likelihood `P(o=1|s_e,d)=σ(a(s_e−d))`; belief `q(s_e)` on a grid. Each batch:
`q(s)←q(s)·∏_k P(o_k|s,d_k)/Z` — the variational-free-energy (Bayesian) update
(App. B.2).

**3.3 Expected free energy (action).** Score each difficulty by
`G_e(d) = λ_pref·(−log U_K(p̂(d))) − λ_info·I_q(s_e;o|d) + λ_cost·C(e,d)`, with
`p̂(d)=E_q[σ(a(s−d))]`, pragmatic = preference risk `−log U_K` (preference for
*informative* outcomes; →+∞ at degenerate difficulties, App. B.3), epistemic term
`I=H[Ber(p̂)]−E_{s∼q}H[Ber(σ(a(s−d)))]`. Sample
`π(d|e)∝exp(−G_e(d)/T)` — the active-inference policy (App. B.4). RLVE = the
degenerate case `λ_info=0, T→0`, preference for high accuracy (App. B.5).

**3.4 Online controller.** Per-environment grid belief; `observe()` updates it;
`G()`/`policy()` score and sample. Implemented in
`rlve_repro/active_inference_controller.py` (no SLIME/torch dep; self-checked:
`U_K` and info-gain both peak at p=0.5; belief σ contracts 1.2→0.06 as competence
rises). Integration point = RLVE's SLIME rollout scheduler
(`active_inference_manager.py`, replaces the acc≥0.9 bump).

## 4. Experimental setup
- **Reproduction**: official RLVE (`Zhiyuan-Zeng/RLVE`, SLIME backend) from
  **ProRL-1.5B-v2** (`nvidia/Nemotron-Research-Reasoning-Qwen-1.5B` rev v2),
  single H200 (8→1 GPU config; see `rlve_repro/`).
- **Environments**: a subset of RLVE-Gym for training + held-out environments;
  optionally scale env count to reproduce RLVE's env-scaling trend.
- **Arms** (same recipe, *only* the difficulty controller changes):
  1. **Static** — frozen difficulty.
  2. **RLVE-90** — acc≥0.9 threshold bump (the RLVE baseline).
  3. **Signal-RLVE** — EFE with `λ_info=0` (pragmatic `U_K` only; targets 0.5 but
     no information gain). *Key ablation.*
  4. **FEP-RLVE** (ours) — full EFE (`λ_info>0`).
  Signal-RLVE matters: only if FEP-RLVE beats it is the *epistemic* term (the
  Friston-specific part) doing work, rather than "just move 0.9 → 0.5".
- **Metrics**: training-time **effective sample ratio** `U_K(p̂)` and group success
  rate per step; held-out reward before vs after.

## 5. Results  〈from `results/`〉
- **Success rate / effective ratio** (headline mechanism): RLVE-90 drifts to
  0.7–0.83 success early (low-signal); Signal-RLVE and FEP-RLVE hold near the 0.5
  signal-optimal band (mean success 0.46 / 0.44; mean `U_K` 0.974 / 0.970 vs
  RLVE-90's 0.962). Plot vs step (`results/figures/success_rate.png`).
- **Held-out generalization** (before→after): all three trained arms improve from
  ≈−0.92; Signal-RLVE reaches the best final value (−0.77); FEP-RLVE improves most
  monotonically (`results/figures/heldout.png`).
- **Honest caveat**: at this scale (4 envs / 30 steps / 1.5B), effective-ratio gaps
  are modest and held-out differences are within run-to-run noise; eval uses
  sampled decoding so single-point comparisons are noisy — compare Δ and trends.
- 〈Temperature / λ_info ablation if time permits.〉

## 6. Discussion / failure cases
- When the model outgrows `d_max`, all levels saturate (`U_K→0` for every `d`): no
  controller has signal → motivates *environment scaling* (RLVE/SCALER thesis).
- Competence modelled as a single scalar `s_e` is a simplification (real skills are
  multi-dimensional); the controller relies on a usable success-rate signal.
- Binary vs partial-credit rewards: the variance-max point shifts off 0.5, where an
  explicit `U_K` target is cleaner than a hand-set accuracy threshold.

## 7. Related work
RLVR; **RLVE** (arXiv:2511.07317); **SCALER** (arXiv:2601.04809, adaptive
multi-environment difficulty tracking the capability frontier); GRPO/DAPO; ProRL;
automatic curriculum / learning-progress; **active inference / Free Energy
Principle** (Friston 2010; expected free energy & Bayesian optimal design).

## 8. Conclusion
Casting difficulty selection as active inference replaces RLVE's accuracy-threshold
heuristic with a principled objective (belief update + expected free energy) that
targets the signal-optimal operating point, actively probes competence, and moves
difficulty both ways — at near-zero extra compute and with the RL core unchanged.

---
### Appendices
- **A** `argmax_p U_K=0.5`, `U_K=1−p^K−(1−p)^K`, variance link, why RLVE's 0.9 is
  low-signal (`paper/derivation.md`).
- **B** active-inference (FEP) derivation: generative model, variational-FE belief
  update, expected-FE objective (pragmatic + epistemic), softmax policy, RLVE as a
  special case (`derivation.md`).
- **C** exact RLVE single-GPU reproduction steps + controller integration
  (`rlve_repro/README.md`).
- **D** per-benchmark / per-difficulty tables.

### Rubric self-check
- *Method*: §3 + App. A/B — generative model, belief update, EFE objective, policy,
  RLVE special case.
- *Empirical*: §5 same-compute 4 arms, effective-sample-ratio mechanism, held-out
  before/after, **Signal-RLVE ablation** isolating the epistemic term, honest
  failure cases (§6).
- *Communication*: one figure per claim; ≤8 pp.
