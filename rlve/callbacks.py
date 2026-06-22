"""Trainer callback that drives the adaptive curriculum during GRPO training.

  * on_step_begin: bump the dataset's round so the next step's prompts reflect
    the latest difficulty/sampler state (and the per-step cache is cleared).
  * on_step_end: drain the signal bus, update every controller + the sampler,
    log per-step curriculum metrics to a JSONL file (and wandb if active).
"""
from __future__ import annotations

import json
import os

from transformers import TrainerCallback

from rlve.curriculum import Curriculum
from rlve.data import AdaptiveDataset


class AdaptiveCallback(TrainerCallback):
    def __init__(self, curriculum: Curriculum, dataset: AdaptiveDataset,
                 log_path: str, log_every: int = 5, use_wandb: bool = False):
        self.curriculum = curriculum
        self.dataset = dataset
        self.log_path = log_path
        self.log_every = log_every
        self.use_wandb = use_wandb
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # truncate any previous log
        open(self.log_path, "w").close()
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except Exception:
                self._wandb = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.dataset.new_round()

    def on_step_end(self, args, state, control, **kwargs):
        metrics = self.curriculum.step_update()
        metrics["global_step"] = int(state.global_step)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._wandb is not None:
            try:
                self._wandb.log({f"curriculum/{k}": v for k, v in metrics.items()
                                 if isinstance(v, (int, float))},
                                step=int(state.global_step))
            except Exception:
                pass
        if state.global_step % self.log_every == 0:
            print(f"[curriculum] step={metrics['global_step']} "
                  f"succ={metrics.get('success_rate', 0):.3f} "
                  f"eff={metrics.get('effective_ratio', 0):.3f} "
                  f"n_groups={metrics.get('n_groups', 0)}", flush=True)
