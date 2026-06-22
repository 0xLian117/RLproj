"""HELD-OUT environment: modular exponentiation a^e mod m.

Tests a multiplicative / modular reasoning skill absent from training envs.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem, parse_int


class ModularExpEnv(Environment):
    name = "modular_exp"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer in [0, m)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        a = rng.randint(2, 9 + d)
        e = rng.randint(2, 3 + d)
        m = rng.randint(10 + 5 * d, 50 + 20 * d)
        gold = pow(a, e, m)
        question = (
            f"Compute {a}^{e} mod {m} (the remainder of {a} raised to the "
            f"power {e}, divided by {m}).\nGive {self.answer_spec}."
        )
        return Problem(question=question, answer=str(gold), difficulty=d,
                       env_name=self.name, info={"a": a, "e": e, "m": m})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
