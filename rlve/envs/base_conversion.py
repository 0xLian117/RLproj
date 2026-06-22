"""HELD-OUT environment: convert an integer between number bases.

Structurally distinct from any training env (digit manipulation rather than
arithmetic / sorting), so it tests generalization rather than memorization.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _to_base(n: int, base: int) -> str:
    if n == 0:
        return "0"
    out = []
    while n > 0:
        out.append(_DIGITS[n % base])
        n //= base
    return "".join(reversed(out))


class BaseConversionEnv(Environment):
    name = "base_conversion"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "the converted number as a string of digits (use uppercase A-F for bases > 10)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        from_base, to_base = rng.sample([2, 8, 10, 16], 2)
        value = rng.randint(10 ** (1 + d // 2), 10 ** (1 + d // 2) * 9)
        src = _to_base(value, from_base)
        gold = _to_base(value, to_base)
        question = (
            f"Convert the number {src} from base {from_base} to base {to_base}.\n"
            f"Give {self.answer_spec}."
        )
        return Problem(question=question, answer=gold, difficulty=d,
                       env_name=self.name,
                       info={"from": from_base, "to": to_base})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = extracted.strip().upper().lstrip("0") or "0"
        gold = problem.answer.upper().lstrip("0") or "0"
        return pred == gold
