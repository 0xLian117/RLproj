# `scaler_addon` — our additions on top of SCALER

This folder is the **only original code** of the project. It sits on top of two
public repos (not vendored here):

* **SCALER** — `github.com/ALEX-nlp/SCALER` (verl-based RL with synthesized,
  difficulty-controllable verifiable environments). Paper: arXiv:2601.04809.
* **SandboxFusion** — `github.com/bytedance/SandboxFusion` (code-execution server
  that runs reference solutions / unit tests to produce verifiable rewards).

Everything in this folder is what *we* wrote or changed; SCALER/SandboxFusion are
used as-is from their checkouts.

---

## What this adds

1. **A static-vs-adaptive difficulty study** on SCALER's released environments
   (course requirement: 3–5 verifiable envs; static vs adaptive; held-out
   generalization).
2. **An innovation: a Free-Energy difficulty controller** that generalizes
   SCALER's fixed-0.5 set-point into an explicit objective, with SCALER recovered
   as the zero-temperature limit.
3. Tooling to build the experiment arms, run them, and turn the logs into the
   report's tables/figures (incl. the **effective sample ratio** metric).

## Files

| file | role |
|---|---|
| `scaler_make_arms.py` | from SCALER's `SCALER-data/train/SCALER-8.json`, build the arms: `SCALER-train.json` (adaptive, untouched) · `SCALER-static-{lo,mid,hi}.json` (difficulty frozen by pinning `dmin=dmax`) · `SCALER-heldout.json` (held-out envs) |
| `freeenergy_difficulty.py` | **our controller.** Drop-in for SCALER's `recipe/environment/difficulty_control.py` interface; samples difficulty from `q(d) ∝ exp(U(d)/T)`. |
| `apply_freeenergy_patch.py` | copies the controller into the SCALER checkout and adds an env-var-gated hook so `DIFFICULTY_MODE=freeenergy` switches controllers (no-op otherwise). Idempotent. |
| `scaler_arm.sh` | self-contained single-arm recipe (single-GPU, Qwen2.5 settings, eval before+after, ckpt to a big disk). Env-overridable. |
| `run_arms.sh` | runs all five arms (adaptive / static-lo/mid/hi / free-energy). |
| `analyze.py` | parses `~/runs_out/*.log` → `metrics.csv`, figures, `REPORT.md` (reward, success rate, **effective sample ratio**, mean difficulty, held-out scores). |

## Changes made to the SCALER checkout (for transparency / reproducibility)

These are applied to a fresh SCALER clone; they are small and listed here so the
diff against upstream is clear:

1. **`recipe/environment/dapo_ray_trainer.py`** — line ~179 hard-codes
   `random.sample(list(self.train_configs.keys()), 8)` (assumes ≥8 envs). With our
   5-env arms set `num_environment_per_step=5` (we do, in `scaler_arm.sh`) and, if
   you train on <8 envs, change `8` → `min(8, len(self.train_configs))`.
2. **`recipe/environment/dapo_ray_trainer.py`** — the free-energy hook
   (added automatically by `apply_freeenergy_patch.py`; reversible).
3. **Environment install notes** (verl stack we used; the public install script
   leaves a few gaps): pin `numpy==1.26.4` + `scipy==1.12.0` (vLLM/megatron need
   numpy<2); install `flash_attn==2.8.1` wheel matching torch cxx11abi;
   `math_verify` + `latex2sympy2_extended`; make `transformers`’
   `MistralForSequenceClassification` import optional in `verl/utils/model.py`
   (removed in transformers≥4.56). SandboxFusion needs `g++`/`gcc` (the env
   problems’ reference solutions are mostly C++). 4090 multi-GPU needs
   `NCCL_P2P_DISABLE=1`; single H200/one-GPU avoids all of that.

## Reproduce

```bash
# 0) prerequisites: verl env active; SandboxFusion up on :8080 (with g++); a
#    supported instruct model (e.g. Qwen2.5-3B-Instruct; NOT qwen3_5/Qwen3.5-*).
SCALER=/path/to/SCALER
cd "$SCALER"

# 1) build arms (5 train envs + 3 held-out)
python /path/to/scaler_addon/scaler_make_arms.py \
    --in SCALER-data/train/SCALER-8.json --out arms --n-train 5

# 2) wire in the free-energy controller (env-var gated; default behaviour unchanged)
python /path/to/scaler_addon/apply_freeenergy_patch.py --scaler "$SCALER"

# 3) run all arms (single GPU)
SCALER_DIR="$SCALER" MODEL=/path/to/Qwen2.5-3B-Instruct \
    bash /path/to/scaler_addon/run_arms.sh

# 4) build tables + figures for the report
python /path/to/scaler_addon/analyze.py --logs ~/runs_out --out results --G 8
```

## The innovation in one paragraph

For a group of `G` binary-reward rollouts with success rate `p`, the GRPO/DAPO
gradient signal is proportional to the within-group reward variance `p(1-p)` and to
the informative-group probability `U(d)=1-p^G-(1-p)^G`, both maximised at `p=0.5`
(derivation in `../paper/derivation.md`). SCALER regulates difficulty to a fixed
success set-point ≈0.5. We instead treat the per-level utility `U(d)` as an energy
and draw difficulties from the free-energy-minimising Gibbs distribution
`q(d) ∝ exp(U(d)/T)`: temperature `T` trades off exploiting the most informative
level against keeping neighbouring levels alive, and **`T→0` recovers SCALER's
set-point as a special case** while the same objective yields an environment-level
weight (negative free energy) that unifies difficulty control with environment
curation. We compare static vs SCALER-adaptive vs free-energy on training-time
effective sample ratio and on held-out generalization.

> Note: tunables are env vars (`FE_G, FE_T0, FE_TMIN, FE_ANNEAL, FE_EMA`); no JSON
> regeneration needed — the free-energy controller loads SCALER's existing arm JSONs.
