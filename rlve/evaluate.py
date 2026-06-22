"""Evaluate a model (base or trained) on the fixed verifiable-env eval set.

Computes greedy pass@1 accuracy per environment, per difficulty and per split
(train envs vs held-out envs). Uses vLLM if available, else HF generate. LoRA
adapter directories are merged into the base model before loading.

Outputs results/eval/<tag>.json with the full breakdown.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict

from rlve.envs.registry import make_env
from rlve.eval_set import get_or_build


def is_lora_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def is_full_model_dir(path):
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "config.json")) and
        not is_lora_dir(path))


def resolve_model(path, work_dir):
    """Return a path/id vLLM can load directly. Merges LoRA adapters if needed."""
    if not is_lora_dir(path):
        return path  # HF hub id, or a full saved model dir
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with open(os.path.join(path, "run_config.json")) as f:
        base = json.load(f)["model"]
    print(f"[eval] merging LoRA adapter {path} into base {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, path)
    model = model.merge_and_unload()
    merged = os.path.join(work_dir, "merged")
    os.makedirs(merged, exist_ok=True)
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(merged)
    del model
    return merged


def generate_vllm(model_path, prompts_text, max_tokens):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, trust_remote_code=True,
              gpu_memory_utilization=0.85, dtype="bfloat16", max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.generate(prompts_text, sp)
    return [o.outputs[0].text for o in outs]


def generate_hf(model_path, prompts_text, max_tokens, batch_size=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    texts = []
    for i in range(0, len(prompts_text), batch_size):
        batch = prompts_text[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            texts.append(tok.decode(gen, skip_special_tokens=True))
    return texts


def render_prompts(items, model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return [tok.apply_chat_template(it["prompt"], tokenize=False,
                                    add_generation_prompt=True) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or trained run dir")
    ap.add_argument("--tag", required=True, help="name for the output json")
    ap.add_argument("--eval-set", default="results/eval_set.json")
    ap.add_argument("--n-per", type=int, default=16)
    ap.add_argument("--out-dir", default="results/eval")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--no-vllm", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    items = get_or_build(args.eval_set, n_per=args.n_per)
    print(f"[eval] {args.tag}: {len(items)} problems from {args.eval_set}")

    work_dir = os.path.join(args.out_dir, f"_work_{args.tag}")
    os.makedirs(work_dir, exist_ok=True)
    model_path = resolve_model(args.model, work_dir)

    prompts_text = render_prompts(items, model_path)

    use_vllm = not args.no_vllm
    if use_vllm:
        try:
            import vllm  # noqa: F401
        except Exception as e:
            print(f"[eval] vLLM unavailable ({e}); using HF generate.")
            use_vllm = False
    completions = (generate_vllm if use_vllm else generate_hf)(
        model_path, prompts_text, args.max_tokens)

    # score
    envs = {}
    cell = defaultdict(lambda: [0, 0])      # (env, difficulty) -> [correct, total]
    split_tot = defaultdict(lambda: [0, 0])
    env_tot = defaultdict(lambda: [0, 0])
    for it, comp in zip(items, completions):
        name = it["env"]
        if name not in envs:
            envs[name] = make_env(name)
        from rlve.envs.base import Problem
        prob = Problem(question="", answer=it["answer"], difficulty=it["difficulty"],
                       env_name=name)
        correct = envs[name].verify(comp, prob).correct
        cell[(name, it["difficulty"])][0] += int(correct)
        cell[(name, it["difficulty"])][1] += 1
        split_tot[it["split"]][0] += int(correct)
        split_tot[it["split"]][1] += 1
        env_tot[name][0] += int(correct)
        env_tot[name][1] += 1

    per_env = {}
    for (name, d), (c, t) in sorted(cell.items()):
        per_env.setdefault(name, {})[str(d)] = round(c / t, 4)
    result = {
        "tag": args.tag,
        "model": args.model,
        "n_problems": len(items),
        "backend": "vllm" if use_vllm else "hf",
        "per_env_difficulty": per_env,
        "per_env_overall": {n: round(c / t, 4) for n, (c, t) in sorted(env_tot.items())},
        "train_avg": round(split_tot["train"][0] / max(1, split_tot["train"][1]), 4),
        "heldout_avg": round(split_tot["heldout"][0] / max(1, split_tot["heldout"][1]), 4),
        "overall_avg": round(
            (split_tot["train"][0] + split_tot["heldout"][0]) /
            max(1, split_tot["train"][1] + split_tot["heldout"][1]), 4),
    }
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] {args.tag}: train={result['train_avg']} "
          f"heldout={result['heldout_avg']} overall={result['overall_avg']} "
          f"-> {out_path}")

    # Free disk: delete any merged-model scratch dir (can be several GB).
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
