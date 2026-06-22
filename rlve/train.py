"""GRPO training entrypoint for RLVE-lite.

Trains a small LM on the procedurally generated verifiable environments with one
of three difficulty controllers (static / threshold / STAD) and one of two
environment samplers (uniform / learning-progress). Uses TRL's GRPOTrainer with
vLLM generation (falling back to HF generate if vLLM is unavailable).

Example:
    python -m rlve.train --condition stad --controller stad --sampler uniform \
        --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 200 --output-dir runs/stad
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="stad", help="name tag for this run")
    p.add_argument("--controller", default="stad",
                   choices=["static", "threshold", "stad"])
    p.add_argument("--sampler", default="uniform", choices=["uniform", "lp"])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="runs/run")
    p.add_argument("--envs", default="", help="comma-sep subset of train envs (default all)")

    # GRPO / optimisation
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8,
                   help="unique prompts per optimizer step")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.0, help="KL coeff (0 => no ref model)")
    p.add_argument("--max-prompt-length", type=int, default=384)
    p.add_argument("--max-completion-length", type=int, default=640)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--binary-reward", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # difficulty controller / sampler knobs
    p.add_argument("--init-level", type=int, default=0)
    p.add_argument("--static-level", type=int, default=4)
    p.add_argument("--p-star", type=float, default=0.5)
    p.add_argument("--kp", type=float, default=1.5)
    p.add_argument("--tau-acc", type=float, default=0.9)
    p.add_argument("--tau-num", type=int, default=0,
                   help="min samples at the upper difficulty before a bump; "
                        "0 => 8 * num_generations (the RLVE paper default)")

    # systems
    p.add_argument("--lora", action="store_true")
    p.add_argument("--no-vllm", action="store_true")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3)
    p.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="colocate: vLLM shares the training GPU; "
                        "server: connect to a separate `trl vllm-serve` process")
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--logging-steps", type=int, default=2)
    return p.parse_args()


def build_curriculum(args):
    from rlve.curriculum import Curriculum
    controller_kw = dict(init_level=args.init_level, static_level=args.static_level,
                         p_star=args.p_star, kp=args.kp, tau_acc=args.tau_acc,
                         tau_num=args.tau_num)
    env_names = [e for e in args.envs.split(",") if e] or None
    return Curriculum(controller_kind=args.controller, sampler_kind=args.sampler,
                      env_names=env_names, controller_kw=controller_kw)


def filter_config_kwargs(GRPOConfig, kw):
    """Keep only kwargs that are valid fields of GRPOConfig (version-tolerant)."""
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    kept = {k: v for k, v in kw.items() if k in valid}
    dropped = sorted(set(kw) - set(kept))
    if dropped:
        print(f"[train] note: GRPOConfig ignores unsupported args: {dropped}")
    return kept


def main():
    args = parse_args()
    if args.tau_num <= 0:                       # RLVE default: 8 x rollouts
        args.tau_num = 8 * args.num_generations
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from rlve.callbacks import AdaptiveCallback
    from rlve.data import AdaptiveDataset
    from rlve.reward import CurriculumReward

    set_seed(args.seed)

    # vLLM availability ----------------------------------------------------
    use_vllm = not args.no_vllm
    if use_vllm:
        try:
            import vllm  # noqa: F401
        except Exception as e:  # pragma: no cover
            print(f"[train] vLLM unavailable ({e}); falling back to HF generate.")
            use_vllm = False

    bf16 = args.bf16 and torch.cuda.is_available()
    pdtbs = args.prompts_per_step * args.num_generations

    curriculum = build_curriculum(args)
    dataset = AdaptiveDataset(curriculum, seed=args.seed)
    reward = CurriculumReward(curriculum, binary=args.binary_reward)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=pdtbs,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        beta=args.beta,
        temperature=args.temperature,
        top_p=1.0,
        num_iterations=1,
        loss_type="dr_grpo",          # unbiased length-normalised GRPO (DAPO-style)
        epsilon_high=0.28,            # DAPO "clip-higher"
        mask_truncated_completions=True,
        scale_rewards=True,
        use_vllm=use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        report_to=(["wandb"] if args.wandb else []),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        log_completions=False,
        seed=args.seed,
        model_init_kwargs={"torch_dtype": "bfloat16"} if bf16 else {},
    )
    config = GRPOConfig(**filter_config_kwargs(GRPOConfig, cfg_kwargs))

    peft_config = None
    if args.lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward],
        args=config,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )

    log_path = os.path.join(args.output_dir, "curriculum_log.jsonl")
    trainer.add_callback(AdaptiveCallback(
        curriculum, dataset, log_path=log_path,
        log_every=args.logging_steps, use_wandb=args.wandb))

    is_main = trainer.accelerator.is_main_process

    # persist run config for reproducibility / eval (main process only)
    if is_main:
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump({**vars(args), "use_vllm": use_vllm, "bf16": bf16,
                       "per_device_train_batch_size": pdtbs,
                       "world_size": trainer.accelerator.num_processes}, f, indent=2)
        print(f"[train] condition={args.condition} controller={args.controller} "
              f"sampler={args.sampler} model={args.model} vllm={use_vllm} "
              f"lora={args.lora} steps={args.max_steps} pdtbs={pdtbs} "
              f"world_size={trainer.accelerator.num_processes}", flush=True)

    trainer.train()
    trainer.save_model(args.output_dir)            # distributed-safe (saves once)
    if is_main:
        tok.save_pretrained(args.output_dir)

    if not is_main:
        return
    # final summary (main process only)
    hist = curriculum.history
    tail = hist[-min(len(hist), 50):] if hist else []
    def _avg(key):
        vals = [h[key] for h in tail if key in h]
        return sum(vals) / len(vals) if vals else 0.0
    summary = {
        "condition": args.condition,
        "controller": args.controller,
        "sampler": args.sampler,
        "model": args.model,
        "steps": len(hist),
        "final50_success_rate": round(_avg("success_rate"), 4),
        "final50_effective_ratio": round(_avg("effective_ratio"), 4),
        "final50_mean_reward_var": round(_avg("mean_reward_var"), 4),
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] DONE {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
