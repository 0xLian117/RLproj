"""Greatest-common-divisor environment.

Difficulty scales the magnitude of the numbers and how many numbers are given.
Numbers are constructed to share a non-trivial common factor so the answer is
rarely 1 (which would make the task degenerate).
"""
from __future__ import annotations

import math
import random
from functools import reduce

from rlve.envs.base import Environment, Problem, parse_int


class GCDEnv(Environment):
    name = "gcd"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer (the greatest common divisor)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        count = 2 + d // 4
        digits = 2 + d // 2
        hi = 10 ** digits - 1
        g = rng.randint(2, 9 + d)  # planted common factor
        nums = []
        for _ in range(count):
            k = rng.randint(2, max(3, hi // max(g, 1)))
            nums.append(g * k)
        gold = reduce(math.gcd, nums)

        question = (
            "Compute the greatest common divisor (GCD) of the following "
            f"integers:\n{', '.join(map(str, nums))}\nGive {self.answer_spec}."
        )
        return Problem(question=question, answer=str(gold), difficulty=d,
                       env_name=self.name, info={"count": count, "digits": digits})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
