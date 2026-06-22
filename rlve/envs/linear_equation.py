"""Linear-equation solving environment.

Low difficulty: one-sided ``a*x + b = c``.
Higher difficulty: two-sided ``a*x + b = a2*x + e`` with larger coefficients.
Equations are constructed around an integer solution x0 so the answer is always
an exact integer.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem, parse_int


class LinearEquationEnv(Environment):
    name = "linear_equation"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer (the value of x)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        span = 5 + 3 * d
        x0 = rng.randint(-span, span)
        a = rng.randint(2, 2 + d)
        b = rng.randint(-span, span)

        two_sided = d >= 3
        if two_sided:
            a2 = rng.randint(1, a - 1)  # keep a - a2 != 0
            # a*x + b = a2*x + e  =>  e = (a - a2)*x0 + b
            e = (a - a2) * x0 + b
            lhs = f"{a}*x + ({b})"
            rhs = f"{a2}*x + ({e})"
        else:
            # a*x + b = c  =>  c = a*x0 + b
            c = a * x0 + b
            lhs = f"{a}*x + ({b})"
            rhs = f"{c}"

        question = (
            f"Solve the following linear equation for x:\n{lhs} = {rhs}\n"
            f"The solution is an integer. Give {self.answer_spec}."
        )
        return Problem(question=question, answer=str(x0), difficulty=d,
                       env_name=self.name, info={"two_sided": two_sided})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
