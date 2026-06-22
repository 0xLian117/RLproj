"""Reward function bridging TRL completions to our verifiers + curriculum.

TRL calls a reward function as ``fn(prompts, completions, **columns)`` where
each dataset column is broadcast to one value per completion. We:
  1. verify each completion with its environment's verifier,
  2. record (uid -> correct) into the curriculum's signal bus (grouped by uid so
     each prompt's ``num_generations`` rollouts form one group),
  3. return a continuous reward in [0, 1].

Reward shaping: 1.0 if correct, else 0.1 if the answer was at least well
formatted (inside \\boxed{}), else 0.0. The difficulty statistics use the strict
binary ``correct`` flag, not the shaped reward.
"""
from __future__ import annotations

from typing import List

from rlve.curriculum import Curriculum
from rlve.envs.base import Problem


def _completion_text(c) -> str:
    # Conversational completions arrive as [{"role": "assistant", "content": ...}].
    if isinstance(c, list):
        return "".join(turn.get("content", "") for turn in c if isinstance(turn, dict))
    return c if isinstance(c, str) else str(c)


class CurriculumReward:
    __name__ = "curriculum_reward"  # TRL uses this for logging

    def __init__(self, curriculum: Curriculum, binary: bool = False):
        self.curriculum = curriculum
        self.binary = binary

    def __call__(self, prompts=None, completions=None, **kwargs) -> List[float]:
        envs = kwargs["env"]
        answers = kwargs["answer"]
        diffs = kwargs["difficulty"]
        uids = kwargs["uid"]
        rewards: List[float] = []
        for i, comp in enumerate(completions):
            env_name = envs[i]
            env = self.curriculum.envs[env_name]
            problem = Problem(question="", answer=answers[i],
                              difficulty=int(diffs[i]), env_name=env_name)
            vr = env.verify(_completion_text(comp), problem)
            self.curriculum.record(uids[i], env_name, int(diffs[i]), vr.correct)
            if self.binary:
                rewards.append(1.0 if vr.correct else 0.0)
            else:
                rewards.append(vr.reward)
        return rewards
