"""Counting-by-inclusion-exclusion environment.

"How many integers in [1, N] are divisible by at least one of {p1, ..., pk}?"
Difficulty scales N and the number of divisors k (which controls how many
inclusion-exclusion terms are needed).
"""
from __future__ import annotations

import math
import random
from itertools import combinations

from rlve.envs.base import Environment, Problem, parse_int

_PRIMES = [2, 3, 5, 7, 11, 13]


def _count_divisible_by_any(n: int, divs):
    """Exact count via inclusion-exclusion."""
    total = 0
    k = len(divs)
    for r in range(1, k + 1):
        for combo in combinations(divs, r):
            lcm = 1
            for x in combo:
                lcm = lcm * x // math.gcd(lcm, x)
            sign = 1 if r % 2 == 1 else -1
            total += sign * (n // lcm)
    return total


class CountingEnv(Environment):
    name = "counting"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer (the count)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        n = rng.randint(10 ** (2 + d // 3), 10 ** (2 + d // 3) * 9)
        k = 1 + d // 3
        k = min(k, len(_PRIMES))
        divs = sorted(rng.sample(_PRIMES, k))
        gold = _count_divisible_by_any(n, divs)

        div_str = ", ".join(map(str, divs))
        question = (
            f"How many integers from 1 to {n} (inclusive) are divisible by at "
            f"least one of: {div_str}?\nGive {self.answer_spec}."
        )
        return Problem(question=question, answer=str(gold), difficulty=d,
                       env_name=self.name, info={"n": n, "divs": divs})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
