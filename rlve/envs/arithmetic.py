"""Multi-step integer arithmetic environment (additive chains).

Difficulty scales the number of terms and their digit count. Kept additive
(+/-) so the difficulty curve is smooth and partially solvable by a small
model, rather than cliff-hard as multiplication of large numbers would be.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem, parse_int


class ArithmeticEnv(Environment):
    name = "arithmetic"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer (the value of the expression)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        n_terms = 2 + d
        digits = 1 + d // 2
        lo = 10 ** (digits - 1) if digits > 1 else 0
        hi = 10 ** digits - 1

        terms = [rng.randint(lo, hi) for _ in range(n_terms)]
        ops = [rng.choice(["+", "-"]) for _ in range(n_terms - 1)]

        expr = str(terms[0])
        value = terms[0]
        for op, t in zip(ops, terms[1:]):
            expr += f" {op} {t}"
            value = value + t if op == "+" else value - t

        question = (
            f"Compute the value of the following expression:\n{expr}\n"
            f"Give {self.answer_spec}."
        )
        return Problem(question=question, answer=str(value), difficulty=d,
                       env_name=self.name, info={"n_terms": n_terms, "digits": digits})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
