# RLproj — Free-Energy Difficulty Control on SCALER

Course project on **RL for LLM reasoning with environment scaling**. We reproduce
**SCALER** (verl-based RL over synthesized, difficulty-controllable *verifiable
environments*; arXiv:2601.04809), run a **static-vs-adaptive difficulty** study with
**held-out generalization**, and add an innovation: a **Free-Energy difficulty
controller** that generalizes SCALER's fixed success set-point and recovers it as a
zero-temperature special case.

This repo is an **add-on package**, not a fork: it contains only the code we wrote,
and uses two public repos as-is.

## Layout
```
scaler_addon/      ← all of OUR code (controller, arm builder, runner, analysis) + its README
paper/             ← mini-paper outline (NeurIPS template) + the p(1-p) derivation
poster/            ← poster outline   (poster模板.pptx is the template)
des.md             ← the assignment brief
```
The experiment lives on top of:
- **SCALER** — `github.com/ALEX-nlp/SCALER` (training framework + released
  verifiable environments + benchmarks).
- **SandboxFusion** — `github.com/bytedance/SandboxFusion` (code-execution server
  that verifies solutions → rewards).

## Quick start
See **`scaler_addon/README.md`** for the full reproduce steps and the exact changes
made to the SCALER checkout. In short:
```bash
SCALER=/path/to/SCALER; cd "$SCALER"
python scaler_addon/scaler_make_arms.py --in SCALER-data/train/SCALER-8.json --out arms --n-train 5
python scaler_addon/apply_freeenergy_patch.py --scaler "$SCALER"
SCALER_DIR="$SCALER" MODEL=/path/to/Qwen2.5-3B-Instruct bash scaler_addon/run_arms.sh
python scaler_addon/analyze.py --logs ~/runs_out --out results --G 8
```

## The idea
RL stalls when difficulty drifts off the model's frontier: too-easy groups are
all-correct, too-hard groups all-wrong — both give zero GRPO gradient. The signal a
group of `G` rollouts carries is the within-group reward variance `p(1-p)`, maximal
at success `p=0.5`. SCALER regulates difficulty to that set-point; **we** maximize an
explicit free-energy objective `F[q] = -E_q[U] - T·H[q]` whose optimum is the Gibbs
policy `q(d) ∝ exp(U(d)/T)` over difficulty levels, with `U(d)=1-p^G-(1-p)^G`. Static
difficulty, SCALER-adaptive, and free-energy are compared on the **effective sample
ratio** (fraction of informative groups) during training and on held-out accuracy.

Details + math: `paper/PAPER_OUTLINE.md`, `paper/derivation.md`, `scaler_addon/README.md`.
