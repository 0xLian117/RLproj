# Reproducing SCALER on 2×48GB (verl), then innovating on top

Main line for the course project: **reproduce SCALER** (arXiv:2601.04809, ACL2026
Findings, RLVE follow-up, repo `github.com/ALEX-nlp/SCALER`) at small scale on
**2×48GB RTX 4090**, then add our innovation. SCALER is built on **verl**
(FSDP, fits 2 GPUs) — far lighter than the official RLVE (SLIME+Megatron, 8×80GB).

The repo `github.com/0xLian117/RLproj` (RLVE-lite, TRL) stays as the **deadline
fallback** — it already runs and gives a full method-reproduction + write-up.

---

## 0. Why this is feasible
- **Data is released** in the repo: `SCALER-data/train/SCALER-{8,64,512,2739}.json`
  (synthesized environments) + test parquets (MATH-500, AMC23, AIME24, MMLU-Pro,
  BBEH, GPQA). → you can train directly, **no need to run the synthesis pipeline**.
- **Docker image matches your driver (CUDA 12.8):**
  `verlai/verl:app-verl0.5-sglang0.4.9-mcore0.12.2` (cu126, torch2.7.1, fa2.8.0,
  sglang0.4.8 / vllm0.8.5).
- **Small config exists:** `recipe/environment/qwen3-1.7b-8-envs.sh` (1.7B, 8 envs).

## 1. Prerequisites (on the big disk)
```bash
# (a) verl runtime — pull the matching image, then build/package as you like
docker pull verlai/verl:app-verl0.5-sglang0.4.9-mcore0.12.2

# (b) SCALER repo + data (parquets are git-LFS)
git lfs install
git clone https://github.com/ALEX-nlp/SCALER.git
cd SCALER && git lfs pull
pip install -e .                      # inside the verl container

# (c) base model (you have network)
hf download Qwen/Qwen3-1.7B-Base --local-dir ../models/Qwen/Qwen3-1.7B-Base

# (d) SandboxFusion code-execution server (REQUIRED: code-env rewards run unit
#     tests through it at localhost:8080/run_code). Run it (it has a Docker image):
#     see github.com/bytedance/SandboxFusion ; expose port 8080.
#     Verify: curl -s localhost:8080/run_code -X POST ... returns JSON.
```

## 2. Edit the 8-env script for 2 GPUs
In `recipe/environment/qwen3-1.7b-8-envs.sh` change:
```bash
num_gpus=2                      # was 4
tensor_model_parallel_size=2    # was 4  (vLLM rollout TP; 2 or 1)
# point paths at YOUR locations:
RAY_DATA_HOME=/big/disk/SCALER                 # repo root (has SCALER-data/)
MODEL_PATH=/big/disk/models/Qwen/Qwen3-1.7B-Base
sandboxfusion_url="http://localhost:8080/run_code"
# keep offload=True / ref_offload=True (FSDP CPU offload — important for 48GB).
# if OOM: lower train_prompt_bsz (64→32), or max_response_length (8192→4096),
#         or gpu_memory_utilization (0.7→0.6).
```
`offload=True` + 1.7B fits comfortably on 2×48GB. The script already uses
`adv_estimator=grpo`, `clip_ratio_high`, DAPO reward manager — the RLVE-line recipe.

## 3. Validate small, then run
```bash
# quick sanity: set test_and_save_freq low and stop after a few steps to confirm
# the loop (rollout → sandbox verify → reward → update) works end to end.
bash recipe/environment/qwen3-1.7b-8-envs.sh
# checkpoints -> CKPTS_DIR (put on the big disk). Watch wandb (offline) / console.
```
Expect: training reward rising, the in-env difficulty `d` drifting up to hold
success near the controller target. Eval runs every `test_and_save_freq` steps on
the test parquets.

## 4. Where our innovation goes
`recipe/environment/difficulty_control.py` — `class DifficultyControl`:
- `update(distance_correct_avg_len_dict, now_step)`: the control law
  `err = batch_avg_correct - target(0.5)`; `delta = clip(k*err, ±step_cap)`;
  `d += delta`. **Proportional control to a fixed 0.5.**
- `_compute_effective_weight()`: env curation by difficulty-slope; `w_recency`.

Our contribution modifies THIS file (and its config knobs in the recipe), so the
tech stack stays 100% SCALER/verl. See the innovation options below.

## 5. Innovation (because SCALER already does "difficulty→0.5 + curation")
SCALER's controller already targets success 0.5 (the binary signal-max point) and
curates envs by learning progress — i.e. it already does what we'd planned as
"STAD + LP sampler". So our novelty must go **beyond** it. Candidate angles (pick
one with the course advisor / see chat):
- **SMDC — Signal-Maximizing Difficulty Control:** control `d` to maximize the
  *measured* GRPO learning signal (informative-group fraction / within-group
  reward variance) instead of regressing to a fixed 0.5, with a **PI / frontier-
  tracking** controller (vs SCALER's proportional one) to cut lag when capability
  drifts. Rationale: 0.5 is the variance-max point only for *binary* rewards; the
  code-env unit-test rewards are partial-credit, so the signal-max difficulty
  shifts. Theory in `paper/derivation.md`. A/B vs SCALER default on the 8-env run.
- **Principled env curation:** replace the difficulty-slope heuristic with a
  regret-minimizing learning-progress bandit over environments.
- **Synthesis (extension ②):** improve `SCALER/` env synthesis for difficulty-
  coverage / skill diversity (needs an LLM API + sandbox; heavier).

Metric to report either way: **effective sample ratio** (fraction of non-degenerate
groups) over training, plus downstream benchmark gains at equal compute.

> Honest note: I cannot test verl/SandboxFusion on your cluster from here. Run the
> small validation in §3 first; if anything breaks, send me the console log.
