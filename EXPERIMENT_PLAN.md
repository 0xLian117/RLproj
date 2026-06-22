# Experiment plan — course landing point on the SCALER (verl) stack

Graded core (from the assignment): **(1) design 3–5 verifiable environments;
(2) compare STATIC vs ADAPTIVE difficulty; (3) evaluate generalization before/after
training and on held-out environments.** Innovation ("适当创新") rides on top.

We realize all three inside SCALER's stack (so the tech stack follows the public
repo), using its released data — no synthesis needed for the core study.

## (1) 3–5 verifiable environments
Take 5 of SCALER's released verifiable environments (others held out). The helper
`tools/scaler_make_arms.py` (tested on the real `SCALER-8.json`) splits them and
builds every arm file:
```bash
python tools/scaler_make_arms.py --in SCALER-data/train/SCALER-8.json \
    --out arms --n-train 5 --lo 3 --mid 9 --hi 15
# -> arms/SCALER-train.json     (5 train envs, ADAPTIVE controller, untouched)
#    arms/SCALER-heldout.json   (3 held-out envs, for generalization)
#    arms/SCALER-static-{lo,mid,hi}.json (5 train envs, controller FROZEN at d)
```
The static files freeze each env's difficulty by pinning `dmin=dmax=d` (and
`k=0, step_cap=0, state.d=d`) so difficulty never moves — robust regardless of
which kwargs SCALER's JSON object-hook honors.

## (2) STATIC vs ADAPTIVE difficulty
Run the **same** recipe (`recipe/environment/qwen3-1.7b-8-envs.sh`, edited for
2 GPUs) at identical compute, only swapping `TRAIN_FILE`:

| Arm | TRAIN_FILE | difficulty |
|---|---|---|
| ADAPTIVE (SCALER) | `arms/SCALER-train.json` | controller adapts toward success 0.5 |
| STATIC-easy | `arms/SCALER-static-lo.json` | frozen low |
| STATIC-mid | `arms/SCALER-static-mid.json` | frozen medium |
| STATIC-hard | `arms/SCALER-static-hi.json` | frozen high |

Expected story (the paper's motivation): static-easy saturates (success →1, signal
→0), static-hard stalls (success →0, signal →0), adaptive sustains an informative
success regime. Log per step: success rate and **effective sample ratio** =
fraction of non-degenerate groups (verl/DAPO already filters zero-std groups, so
this is observable from the dynamic-sampling stats).

In the script, set: `num_gpus=2`, `tensor_model_parallel_size=2`, `offload=True`,
`TRAIN_FILE=<arm>`, `MODEL_PATH=<Qwen3-1.7B-Base>`, and the SandboxFusion URL.

## (3) Generalization (before vs after, held-out)
- **Held-out benchmarks** (already wired in the recipe's `TEST_FILE`): MATH-500,
  AMC23, AIME24, MMLU-Pro, BBEH, GPQA — eval the **base** model (before) and each
  trained arm (after). This is cross-distribution generalization.
- **Held-out environments**: evaluate each trained model on `arms/SCALER-heldout.json`
  envs (never trained on) — same-family but unseen tasks. (Generate a small fixed
  problem set per held-out env at a few difficulties and score via the sandbox.)
- Report Δ over base on both, per arm.

## Innovation on top ("适当创新") — optional third arm
Because SCALER already does adaptive-to-0.5 + curation, a genuine improvement must
go beyond it. Lowest-risk, highest-fit option: add an **ADAPTIVE+ (SMDC)** arm by
editing `recipe/environment/difficulty_control.py` — control difficulty to maximize
the *measured* learning signal (informative-group fraction / reward variance)
rather than a fixed 0.5 target, with a PI (vs proportional) update to cut lag.
Theory backbone in `paper/derivation.md`. Compare ADAPTIVE+ vs ADAPTIVE vs STATIC
on the same arms. (Confirm direction before implementing.)

## Compute note (2×48GB, ~8 days)
1.7B + FSDP offload + 8k response fits 2×48GB. Start with a **short validation**
(a handful of steps, `test_and_save_freq` small) on the ADAPTIVE arm to confirm the
loop (rollout → SandboxFusion verify → reward → update) works, then launch all
arms. If OOM: lower `train_prompt_bsz`, `max_response_length`, or `gpu_memory_utilization`.

> Fallback for the deadline: the RLVE-lite repo (TRL) already implements exactly
> this 3-arm static/adaptive + held-out study and runs on your hardware today.
