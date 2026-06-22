"""HELD-OUT environment: maximum set of mutually non-overlapping intervals.

Classic activity-selection / greedy-by-earliest-end-time problem. Tests
algorithmic reasoning not present in the training envs.
"""
from __future__ import annotations

import random

from rlve.envs.base import Environment, Problem, parse_int


def _max_non_overlapping(intervals):
    chosen = 0
    last_end = float("-inf")
    for s, e in sorted(intervals, key=lambda iv: iv[1]):
        if s >= last_end:
            chosen += 1
            last_end = e
    return chosen


class IntervalSchedulingEnv(Environment):
    name = "interval_scheduling"
    min_difficulty = 0
    max_difficulty = 12
    answer_spec = "a single integer (the maximum number of non-overlapping intervals)"

    def generate(self, difficulty: int, rng: random.Random) -> Problem:
        d = self.clamp(difficulty)
        n = 3 + d
        hi = 10 + 2 * d
        intervals = []
        for _ in range(n):
            s = rng.randint(0, hi)
            e = s + rng.randint(1, 1 + hi // 2)
            intervals.append((s, e))
        gold = _max_non_overlapping(intervals)
        iv_str = "; ".join(f"[{s}, {e}]" for s, e in intervals)
        question = (
            "You are given the following intervals (each as [start, end]):\n"
            f"{iv_str}\n"
            "Select the maximum number of intervals such that no two selected "
            "intervals overlap (touching endpoints, e.g. [1,3] and [3,5], do "
            f"NOT count as overlapping). Give {self.answer_spec}."
        )
        return Problem(question=question, answer=str(gold), difficulty=d,
                       env_name=self.name, info={"n": n})

    def _is_correct(self, extracted: str, problem: Problem) -> bool:
        pred = parse_int(extracted)
        return pred is not None and pred == int(problem.answer)
