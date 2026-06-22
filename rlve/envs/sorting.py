"""Sorting environment: sort a list of integers ascending or descending.

Difficulty scales list length and value range.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem, _norm


class SortingEnv(Environment):
    name = "sorting"
    min_difficulty = 0
    max_difficulty = 14
    answer_spec = "the sorted list as comma-separated integers, e.g. 1, 2, 3"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        length = 4 + d
        hi = 50 + 10 * d
        nums = [rng.randint(0, hi) for _ in range(length)]
        ascending = rng.random() < 0.5
        order = "ascending" if ascending else "descending"
        gold = sorted(nums, reverse=not ascending)

        question = (
            f"Sort the following list of integers in {order} order:\n"
            f"{', '.join(map(str, nums))}\n"
            f"Give {self.answer_spec}."
        )
        return Problem(question=question, answer=", ".join(map(str, gold)),
                       difficulty=d, env_name=self.name,
                       info={"length": length, "ascending": ascending})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        # Parse any integers (commas/spaces/brackets tolerated) and compare.
        import re
        pred = re.findall(r"-?\d+", extracted)
        gold = re.findall(r"-?\d+", problem.answer)
        return pred == gold
