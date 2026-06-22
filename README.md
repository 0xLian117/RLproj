# RLVE-lite: RL for LMs with Signal-Targeting Adaptive Verifiable Environments

A lightweight, **single-GPU** re-implementation of RLVE (Reinforcement Learning
with Verifiable Environments; [Zeng et al., 2025, arXiv:2511.07317](https://arxiv.org/abs/2511.07317))
plus a novel contribution:

> **STAD — Signal-Targeting Adaptive Difficulty.** Instead of RLVE's
> one-directional "raise difficulty once the model is 90% accurate" rule, STAD
> is a closed-loop controller that drives each environment's difficulty toward
> the success-rate band that **maximizes the GRPO learning signal** (group reward
> variance `p(1−p)`, maximal at `p* ≈ 0.5`), moving difficulty **both up and
> down**. We pair it with a **learning-progress environment sampler** (a softmax
> bandit) that spends rollouts where the model is currently learning most.

This repo is designed so you can **design locally, run on one rented GPU
(H200 or 2×4090), and return only lightweight results** (logs, figures, JSON —
no model weights).

---

## 1. The idea in one paragraph

RL post-training stalls when the reward signal vanishes: too-easy problems give
all-correct groups (zero advantage), too-hard problems give all-wrong groups
(zero advantage). For group-relative methods (GRPO/DAPO) the per-prompt gradient
signal is proportional to the within-group reward **variance**, which for binary
rewards is `p(1−p)` — maximal exactly at success rate `p = 0.5`. RLVE keeps
problems solvable by procedurally generating them and ramping difficulty once the
model is *proficient* (accuracy ≥ 0.9). We argue that 0.9 is the wrong target:
it spends most rollouts on near-saturated, low-signal groups. **STAD instead
servo-controls difficulty to hold the success rate at the signal-maximizing band
`p* ≈ 0.5`.**

## 2. What's implemented

- **8 verifiable environments** (`rlve/envs/`), each = (problem generator,
  parametric difficulty, algorithmic verifier):
  - *Training* (5): `arithmetic`, `sorting`, `gcd`, `linear_equation`, `counting`.
  - *Held-out* (3, structurally different, for generalization): `base_conversion`,
    `interval_scheduling`, `modular_exp`.
- **3 difficulty controllers** (`rlve/difficulty/controllers.py`):
  `static` (degenerate baseline), `threshold` (faithful RLVE), `stad` (ours).
- **2 environment samplers** (`rlve/difficulty/sampler.py`):
  `uniform` (RLVE), `lp` learning-progress bandit (ours).
- **GRPO training** on TRL + vLLM (`rlve/train.py`), single GPU, with automatic
  fallback to HuggingFace generate if vLLM is unavailable, and optional LoRA.
- **Fair evaluation** (`rlve/evaluate.py`) on a fixed, shared eval set covering
  seen/unseen difficulties and held-out environments.
- **CPU mechanism simulation** (`tools/simulate.py`) and a **CPU self-test**
  (`tools/selftest.py`) that validate the whole non-GPU half with no model.

## 3. Experiment matrix (the "appropriate innovation")

| Condition | Controller | Sampler | Question it answers |
|---|---|---|---|
| `static`    | fixed d=4         | uniform | Does a fixed difficulty saturate/stall? (motivation) |
| `threshold` | RLVE bump @0.9    | uniform | The published RLVE baseline |
| `stad`      | **ours, p*=0.5**  | uniform | Does targeting the signal band beat the bump rule? |
| `stad_lp`   | **ours, p*=0.5**  | **lp**  | Does the learning-progress sampler add more? |

All four are compared on: (a) training-time **effective sample ratio** (fraction
of informative groups) and success rate, (b) **post-training accuracy** vs the
**base model** on training envs, on **unseen high difficulties**, and on
**held-out environments**.

---

## 4. How to run it (remote GPU)

### Step 0 — get the code onto the box
```bash
git clone https://github.com/0xLian117/RLproj.git
cd RLproj
```

### Step 1 — one-time setup (creates a venv, installs deps, runs the CPU self-test)
```bash
bash scripts/setup_remote.sh
```
`vLLM` is installed if possible but is **optional** — training falls back to HF
generation automatically.

### Step 2 — launch everything
```bash
# default: single H200, full fine-tune of Qwen2.5-1.5B-Instruct, 200 steps
bash scripts/run_all.sh

# single 48GB card (RTX 4090 48GB): LoRA + full-size batch (recommended for 48GB)
PROFILE=4090x48 bash scripts/run_all.sh

# 24GB cards (2×4090 or one 4090): LoRA + smaller batches
PROFILE=4090 bash scripts/run_all.sh

# shorter / cheaper smoke of the science:
MAX_STEPS=120 bash scripts/run_all.sh
```

#### Local models (no download) + two GPUs
`--model` accepts a **local path** as well as a HF id; local paths never
download, and the run scripts export `HF_HUB_OFFLINE=1` by default (set
`HF_HUB_OFFLINE=0` to allow Hub downloads). Pick a card with `GPU=<id>` and an
output dir with `RESULTS_DIR=<dir>`:
```bash
GPU=0 MODEL=/path/to/local/Qwen-model RESULTS_DIR=results_x \
  PROFILE=4090x48 bash scripts/run_all.sh
```
**Two GPUs (recommended way to use two cards):** run one model per card in
parallel — each job is the fully-validated single-GPU path (no DDP), and you get
a model-size comparison for the "scale to larger models" extension:
```bash
# GPU0 -> 4B (results_a/), GPU1 -> 0.5B-Instruct (results_b/)
bash scripts/run_2gpu.sh
# or override the paths:
MODEL_A=/path/to/big MODEL_B=/path/to/small bash scripts/run_2gpu.sh
```
Tips for a bigger / reasoning model: if it emits long chain-of-thought, give it
room with `MAX_COMPLETION_LEN=1280`; if it OOMs, lower `PROMPTS_PER_STEP=6` and
`VLLM_MEM=0.25`.

**Both cards on ONE model (e.g. a single 4B run):** dedicate one GPU to a vLLM
generation server and train on the other. Training stays single-process (the
adaptive curriculum is untouched) and the training card is freed from a
colocated vLLM, so the 4B can use a full-size batch:
```bash
OUT_ROOT=/big/disk/rlve_out  MODEL=/path/to/Qwen3.5-4B  bash scripts/run_4b_2gpu.sh
```
(GPU1 runs `trl vllm-serve`; GPU0 trains in `--vllm-mode server`. If the server
won't start, fall back to one card:
`GPU=0 MODEL=... PROFILE=4090x48 PROMPTS_PER_STEP=4 VLLM_MEM=0.4 bash scripts/run_all.sh`.)
`run_all.sh` does, in order: CPU self-test → CPU simulation → **GPU smoke test
(fails fast if the stack is broken)** → build eval set → eval base model →
train+eval the 4 conditions → make figures + `results/REPORT.md` → package
`rlve_results.tgz`.

Estimated time on one H200: ~2–3 h for 4×200 steps + evals. The smoke test alone
is ~2 min — if it fails, stop and send me the log before burning GPU hours.

### Step 3 — return the results
Everything you need is lightweight (no weights):
```bash
# Option A: copy the tarball back
scp <box>:RLproj/rlve_results.tgz .

# Option B: commit the kept results (weights are git-ignored)
git add results/REPORT.md results/figures results/eval results/sim results/runs
git commit -m "experiment results"
git push
```
Then send me `results/REPORT.md` + `results/figures/` and I'll write up the paper
numbers.

---

## 5. Repo layout
```
rlve/
  envs/            8 verifiable environments + registry (train/held-out split)
  difficulty/      controllers (static / threshold / STAD) + samplers (uniform / LP)
  stats.py         group-level signal bookkeeping (effective sample ratio)
  curriculum.py    orchestrator shared by trainer and simulator
  data.py          adaptive dataset feeding GRPO from the live curriculum
  reward.py        verifier-based reward function for TRL
  callbacks.py     per-step controller/sampler updates during training
  train.py         GRPO training entrypoint (TRL + vLLM, LoRA optional)
  eval_set.py      fixed, shared evaluation set
  evaluate.py      greedy pass@1 eval (vLLM or HF), per-env/difficulty/split
tools/
  selftest.py      CPU validation of envs/controllers/integration (no GPU)
  simulate.py      CPU mechanism study on a synthetic learning policy
  plot_results.py  figures + REPORT.md
scripts/
  setup_remote.sh  env setup
  smoke_test.sh    ~2-min GPU pre-flight
  run_all.sh       full experiment launcher (single process)
  run_2gpu.sh      two models, one per card, in parallel
  run_4b_2gpu.sh   one model on two cards (vLLM server + trainer)
  run_ddp.sh       distributed (DDP) training of one model across N GPUs
configs/profiles.sh   hardware profiles (h200 / 4090x48 / 4090) + experiment matrix
paper/PAPER_OUTLINE.md    NeurIPS-style write-up skeleton (with the math + protocol)
poster/POSTER_OUTLINE.md  poster content
```

## 6. Validate locally without a GPU
```bash
pip install numpy matplotlib            # torch optional for the selftest
python tools/selftest.py                # all green = envs/controllers correct
python tools/simulate.py --out results/sim   # synthetic mechanism study
python tools/plot_results.py --results results
```

## 7. Troubleshooting
- **OOM on 24GB cards:** use `PROFILE=4090` (LoRA), and if still tight, set
  `MODEL=Qwen/Qwen2.5-0.5B-Instruct` or lower `MAX_COMPLETION_LEN`.
- **vLLM install/import fails:** ignore it — training & eval fall back to HF
  generate (slower). You can also pass `--no-vllm`.
- **TRL version drift:** `rlve/train.py` filters unknown `GRPOConfig` fields and
  prints which it dropped, so newer/older TRL still runs.
- **vLLM won't import / `driver too old` / `'aimv2' already used`:** a vLLM↔torch↔
  driver↔TRL version clash. Easiest: run with `NO_VLLM=1` (HF generate, no vLLM).
  If you want vLLM, match it to your TRL's supported range (it prints the range)
  AND to a CUDA your driver supports (`nvidia-smi` top-right; install a torch
  whose `torch.version.cuda` ≤ that).
- **Distributed training:** `bash scripts/run_ddp.sh` (DDP via `accelerate`,
  defaults to `NO_VLLM=1`). Both cards train; rank 0 owns logging. Good for 4B/7B
  with LoRA. Do a tiny `MAX_STEPS=20 EVAL_N_PER=4 ... bash scripts/run_ddp.sh`
  first to validate before a long run.
- **A condition crashes mid-run:** `run_all.sh` continues with the remaining
  conditions and reports which succeeded; rerun a single one with the matching
  `python -m rlve.train ...` line from `results/logs/train_<cond>.log`.

## 8. Credit / baseline
Method and `RLVE-Gym` baseline: Zeng et al., *RLVE: Scaling Up Reinforcement
Learning for Language Models with Adaptive Verifiable Environments*,
arXiv:2511.07317, 2025. Reference framework:
https://github.com/Zhiyuan-Zeng/RLVE (SLIME+Megatron, 8×80GB). This repo is an
independent lightweight reimplementation for a course project, not their code.
