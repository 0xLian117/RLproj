"""Environment samplers: which environment to draw the next problem from.

* UniformSampler            -- RLVE's default: pick an environment uniformly.
* LearningProgressSampler   -- (OURS) a softmax bandit that allocates more
                               rollouts to environments currently producing the
                               most learning signal (highest effective sample
                               ratio), with a uniform exploration floor so no
                               environment is ever starved.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List


class EnvSampler:
    def __init__(self, env_names: List[str]):
        self.env_names = list(env_names)

    def sample_env(self, rng: random.Random) -> str:
        raise NotImplementedError

    def update(self, per_env_signal: Dict[str, float]) -> None:
        pass

    def weights(self) -> Dict[str, float]:
        n = len(self.env_names)
        return {e: 1.0 / n for e in self.env_names}


class UniformSampler(EnvSampler):
    def sample_env(self, rng: random.Random) -> str:
        return rng.choice(self.env_names)


class LearningProgressSampler(EnvSampler):
    """Softmax-over-signal bandit with a uniform floor.

        score_e <- (1 - decay) * score_e + decay * signal_e          (EMA)
        w_e     <- (1 - eps) * softmax(score_e / temp) + eps / N

    ``signal_e`` defaults to the effective sample ratio of environment e in the
    most recent step (fraction of informative groups). Environments that are
    saturated (all-correct) or hopeless (all-wrong) get a low signal and are
    sampled less, while still retaining the eps/N exploration floor.
    """

    def __init__(self, env_names: List[str], temp: float = 0.3,
                 eps: float = 0.15, decay: float = 0.3):
        super().__init__(env_names)
        self.temp = temp
        self.eps = eps
        self.decay = decay
        self.score = {e: 0.5 for e in self.env_names}

    def update(self, per_env_signal: Dict[str, float]) -> None:
        for e in self.env_names:
            if e in per_env_signal:
                self.score[e] = ((1 - self.decay) * self.score[e]
                                 + self.decay * per_env_signal[e])

    def weights(self) -> Dict[str, float]:
        n = len(self.env_names)
        mx = max(self.score.values())
        exps = {e: math.exp((self.score[e] - mx) / self.temp) for e in self.env_names}
        z = sum(exps.values())
        return {e: (1 - self.eps) * (exps[e] / z) + self.eps / n
                for e in self.env_names}

    def sample_env(self, rng: random.Random) -> str:
        w = self.weights()
        r = rng.random()
        acc = 0.0
        for e in self.env_names:
            acc += w[e]
            if r <= acc:
                return e
        return self.env_names[-1]


def make_sampler(kind: str, env_names: List[str], **kw) -> EnvSampler:
    kind = kind.lower()
    if kind == "uniform":
        return UniformSampler(env_names)
    if kind in ("lp", "learning_progress", "bandit"):
        return LearningProgressSampler(
            env_names, temp=kw.get("temp", 0.3), eps=kw.get("eps", 0.15),
            decay=kw.get("decay", 0.3))
    raise ValueError(f"Unknown sampler kind: {kind}")
