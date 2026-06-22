"""Environment registry and the train / held-out split.

Training environments are seen during RL. Held-out environments are NEVER used
for training and exist purely to measure generalization (a fresh environment
the policy has never been optimized against). We also evaluate generalization
to *unseen difficulty levels* of the training environments (see evaluate.py).
"""
from __future__ import annotations

from typing import Dict, List

from rlve.envs.arithmetic import ArithmeticEnv
from rlve.envs.base import Environment
from rlve.envs.base_conversion import BaseConversionEnv
from rlve.envs.counting import CountingEnv
from rlve.envs.gcd import GCDEnv
from rlve.envs.interval_scheduling import IntervalSchedulingEnv
from rlve.envs.linear_equation import LinearEquationEnv
from rlve.envs.modular_exp import ModularExpEnv
from rlve.envs.sorting import SortingEnv

TRAIN_ENV_CLASSES = [
    ArithmeticEnv,
    SortingEnv,
    GCDEnv,
    LinearEquationEnv,
    CountingEnv,
]

HELDOUT_ENV_CLASSES = [
    BaseConversionEnv,
    IntervalSchedulingEnv,
    ModularExpEnv,
]

_ALL = {c.name: c for c in TRAIN_ENV_CLASSES + HELDOUT_ENV_CLASSES}


def make_env(name: str) -> Environment:
    if name not in _ALL:
        raise KeyError(f"Unknown environment '{name}'. Known: {sorted(_ALL)}")
    return _ALL[name]()


def train_env_names() -> List[str]:
    return [c.name for c in TRAIN_ENV_CLASSES]


def heldout_env_names() -> List[str]:
    return [c.name for c in HELDOUT_ENV_CLASSES]


def make_train_envs() -> Dict[str, Environment]:
    return {c.name: c() for c in TRAIN_ENV_CLASSES}


def make_heldout_envs() -> Dict[str, Environment]:
    return {c.name: c() for c in HELDOUT_ENV_CLASSES}
