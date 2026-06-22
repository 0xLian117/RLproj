"""Small shared dataclasses (kept separate to avoid import cycles)."""
from __future__ import annotations

from dataclasses import dataclass

from rlve.envs.base import Problem


@dataclass
class Sample:
    env_name: str
    difficulty: int
    problem: Problem
