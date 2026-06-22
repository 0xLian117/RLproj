"""Difficulty controllers.

Three controllers, all sharing the same interface:

    sample_difficulty(rng) -> int     # difficulty level for the next problem
    update(groups)                     # adapt internal state from this env's groups
    state() -> dict                    # for logging / plotting

1. StaticController          -- fixed difficulty (no adaptation). The
                                degenerate baseline that "固定题库会很快失效".
2. ThresholdBumpController   -- faithful re-implementation of RLVE's controller
                                (Zeng et al., 2025): one-directional, increments
                                the upper bound h once accuracy >= tau_acc.
3. STADController (OURS)     -- Signal-Targeting Adaptive Difficulty: a closed-
                                loop proportional controller that drives the
                                success rate toward the band that MAXIMIZES the
                                GRPO learning signal (p* ~ 0.5), moving difficulty
                                both up AND down.
"""
from __future__ import annotations

import random
from typing import List

from rlve.envs.base import Environment
from rlve.stats import GroupOutcome


class DifficultyController:
    def __init__(self, env: Environment):
        self.env = env

    def sample_difficulty(self, rng: random.Random) -> int:
        raise NotImplementedError

    def update(self, groups: List[GroupOutcome]) -> None:
        raise NotImplementedError

    def state(self) -> dict:
        return {}


class StaticController(DifficultyController):
    """Fixed difficulty level throughout training."""

    def __init__(self, env: Environment, level: int = 4):
        super().__init__(env)
        self.level = env.clamp(level)

    def sample_difficulty(self, rng: random.Random) -> int:
        return self.level

    def update(self, groups: List[GroupOutcome]) -> None:
        pass

    def state(self) -> dict:
        return {"level": self.level}


class ThresholdBumpController(DifficultyController):
    """Faithful RLVE controller.

    Maintains a difficulty range [l, h]; samples d ~ UniformInt(l, h). Tracks
    (a correct, b total) of rollouts generated at the *upper bound* h. Once
    b >= tau_num and a/b >= tau_acc, increments h by 1 (and slides l up so that
    h - l + 1 <= d_delta). Difficulty only ever increases.
    """

    def __init__(self, env: Environment, l: int = 0, tau_acc: float = 0.9,
                 tau_num: int = 64, d_delta: int = 4):
        super().__init__(env)
        self.l = env.clamp(l)
        self.h = self.l
        self.tau_acc = tau_acc
        self.tau_num = tau_num
        self.d_delta = d_delta
        self.a = 0  # correct at difficulty h
        self.b = 0  # total at difficulty h

    def sample_difficulty(self, rng: random.Random) -> int:
        return rng.randint(self.l, self.h)

    def update(self, groups: List[GroupOutcome]) -> None:
        for g in groups:
            if g.difficulty == self.h:
                self.a += g.n_correct
                self.b += g.n
        if self.b >= self.tau_num and (self.a / self.b) >= self.tau_acc:
            if self.h < self.env.max_difficulty:
                self.h += 1
                if self.h - self.l + 1 > self.d_delta:
                    self.l = self.h - self.d_delta + 1
            self.a = 0
            self.b = 0

    def state(self) -> dict:
        return {"l": self.l, "h": self.h, "a": self.a, "b": self.b}


class STADController(DifficultyController):
    """Signal-Targeting Adaptive Difficulty (ours).

    Holds a *continuous* difficulty level mu. Each step it measures the pooled
    success rate s at the current operating point (EMA-smoothed) and applies a
    proportional (optionally PI) update that pushes the success rate toward the
    target p_star that maximizes the GRPO group-reward variance:

        mu  <-  mu + kp * (s_ema - p_star) + ki * I
        I   <-  I  + (s_ema - p_star)

    s_ema > p_star  =>  too easy  =>  mu increases (harder problems)
    s_ema < p_star  =>  too hard  =>  mu decreases (easier problems)

    Problems are generated at a stochastic rounding of mu with small +/-1
    exploration so neighbouring levels are probed.
    """

    def __init__(self, env: Environment, init: float = 0.0, p_star: float = 0.5,
                 kp: float = 1.5, ki: float = 0.0, ema_beta: float = 0.7,
                 explore: float = 0.15):
        super().__init__(env)
        self.mu = float(env.clamp(int(round(init))))
        self.p_star = p_star
        self.kp = kp
        self.ki = ki
        self.ema_beta = ema_beta
        self.explore = explore
        self.s_ema = p_star      # initialise at target so early steps are gentle
        self._I = 0.0
        self._seen = False

    def sample_difficulty(self, rng: random.Random) -> int:
        lo = int(self.mu)
        frac = self.mu - lo
        d = lo + (1 if rng.random() < frac else 0)
        if rng.random() < self.explore:
            d += rng.choice([-1, 1])
        return self.env.clamp(d)

    def update(self, groups: List[GroupOutcome]) -> None:
        total_n = sum(g.n for g in groups)
        total_c = sum(g.n_correct for g in groups)
        if total_n == 0:
            return
        s_step = total_c / total_n
        if not self._seen:
            self.s_ema = s_step
            self._seen = True
        else:
            self.s_ema = self.ema_beta * self.s_ema + (1 - self.ema_beta) * s_step
        err = self.s_ema - self.p_star
        self._I += err
        self.mu += self.kp * err + self.ki * self._I
        self.mu = max(float(self.env.min_difficulty),
                      min(float(self.env.max_difficulty), self.mu))

    def state(self) -> dict:
        return {"mu": round(self.mu, 3), "s_ema": round(self.s_ema, 3),
                "I": round(self._I, 3)}


def make_controller(kind: str, env: Environment, **kw) -> DifficultyController:
    kind = kind.lower()
    if kind == "static":
        return StaticController(env, level=kw.get("static_level", 4))
    if kind in ("threshold", "threshold_bump", "rlve"):
        return ThresholdBumpController(
            env, l=kw.get("init_level", 0), tau_acc=kw.get("tau_acc", 0.9),
            tau_num=kw.get("tau_num", 64), d_delta=kw.get("d_delta", 4))
    if kind == "stad":
        return STADController(
            env, init=kw.get("init_level", 0), p_star=kw.get("p_star", 0.5),
            kp=kw.get("kp", 1.5), ki=kw.get("ki", 0.0),
            ema_beta=kw.get("ema_beta", 0.7), explore=kw.get("explore", 0.15))
    raise ValueError(f"Unknown controller kind: {kind}")
