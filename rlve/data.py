"""Adaptive dataset feeding the GRPO trainer from a live Curriculum.

TRL's GRPOTrainer uses a RepeatSampler that requires a map-style dataset with a
fixed length and *deterministic* ``__getitem__`` (each prompt index is fetched
``num_generations`` times to form a group, and all repeats must return the SAME
prompt). To allow the difficulty/sampler state to evolve while respecting this,
we:

  * key generated samples by ``(round, index)`` and cache them, so repeated
    fetches of the same index within a step return identical prompts;
  * bump ``round`` once per optimizer step (via ``AdaptiveCallback``), which
    clears the cache so the next step's prompts reflect the updated curriculum.

This keeps grouping correct regardless of gradient-accumulation / micro-batch
ordering, as long as the dataloader runs in the main process
(``dataloader_num_workers=0``).
"""
from __future__ import annotations

import random
from typing import Dict

from torch.utils.data import Dataset

from rlve.curriculum import Curriculum

SYSTEM_PROMPT = (
    "You are a careful problem solver. Reason step by step, then put your "
    "final answer inside \\boxed{...}. Output only the requested answer format "
    "inside the box."
)


def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


class AdaptiveDataset(Dataset):
    def __init__(self, curriculum: Curriculum, virtual_len: int = 100_000,
                 seed: int = 0):
        self.curriculum = curriculum
        self.virtual_len = virtual_len
        self.round = 0
        self._cached_round = -1
        self._samples: Dict[int, dict] = {}
        self._rng = random.Random(seed)

    def __len__(self):
        return self.virtual_len

    def new_round(self):
        """Called once per optimizer step: invalidate the per-step cache."""
        self.round += 1

    def _ensure_round(self):
        if self._cached_round != self.round:
            self._samples = {}
            self._cached_round = self.round

    def __getitem__(self, idx: int) -> dict:
        self._ensure_round()
        if idx not in self._samples:
            s = self.curriculum.next_problem(self._rng)
            self._samples[idx] = {
                "prompt": build_prompt(s.problem.question),
                "env": s.env_name,
                "answer": s.problem.answer,
                "difficulty": int(s.difficulty),
                "uid": f"{self.round}-{idx}",
            }
        return self._samples[idx]
