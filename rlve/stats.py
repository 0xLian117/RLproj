"""Group-level signal bookkeeping shared between the reward fn and callbacks.

In GRPO/DAPO the policy gradient signal from a *group* of G samples drawn for
one prompt is proportional to the within-group reward variance. For binary
rewards with group success rate p, that variance is p(1-p), which is maximal at
p = 0.5 and zero at p in {0, 1}. A group with all-correct or all-wrong rollouts
contributes a zero advantage and is *non-informative* ("effective prompt" in the
RLVE paper). We therefore track, per environment and per difficulty level:

  * the pooled success rate, and
  * the fraction of *informative* groups (the "effective sample ratio"),

which the difficulty controllers and the environment sampler consume.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GroupOutcome:
    env: str
    difficulty: int
    n: int            # group size G
    n_correct: int

    @property
    def success_rate(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def informative(self) -> bool:
        # non-degenerate group: not all-correct and not all-wrong
        return 0 < self.n_correct < self.n

    @property
    def reward_variance(self) -> float:
        p = self.success_rate
        return p * (1.0 - p)


class SignalBus:
    """Accumulates per-group outcomes within a single training step.

    The reward function records one entry per (prompt, completion) outcome; the
    bus groups them by prompt so that each unique prompt becomes one GroupOutcome
    when drained at the end of the step.
    """

    def __init__(self):
        # prompt-key -> (env, difficulty, n, n_correct)
        self._pending: Dict[str, List] = {}

    def record(self, prompt_key: str, env: str, difficulty: int, correct: bool):
        slot = self._pending.get(prompt_key)
        if slot is None:
            self._pending[prompt_key] = [env, int(difficulty), 1, int(correct)]
        else:
            slot[2] += 1
            slot[3] += int(correct)

    def drain(self) -> List[GroupOutcome]:
        groups = [GroupOutcome(env=e, difficulty=d, n=n, n_correct=c)
                  for (e, d, n, c) in self._pending.values()]
        self._pending.clear()
        return groups


def summarize(groups: List[GroupOutcome]) -> Dict[str, float]:
    """Aggregate metrics over a list of groups (e.g. all groups in a step)."""
    if not groups:
        return {"n_groups": 0, "success_rate": 0.0, "effective_ratio": 0.0,
                "mean_reward_var": 0.0}
    n_groups = len(groups)
    eff = sum(1 for g in groups if g.informative) / n_groups
    total_n = sum(g.n for g in groups)
    total_c = sum(g.n_correct for g in groups)
    var = sum(g.reward_variance for g in groups) / n_groups
    return {
        "n_groups": n_groups,
        "success_rate": total_c / total_n if total_n else 0.0,
        "effective_ratio": eff,
        "mean_reward_var": var,
    }


def group_by_env(groups: List[GroupOutcome]) -> Dict[str, List[GroupOutcome]]:
    out: Dict[str, List[GroupOutcome]] = defaultdict(list)
    for g in groups:
        out[g.env].append(g)
    return dict(out)
