"""Free-Energy difficulty controller — our innovation, for the RLVE pipeline.

This is a small, dependency-free core that decides, given the model's current
per-difficulty success estimates, **what distribution of difficulties to sample
next**. It is the drop-in replacement for RLVE's "raise difficulty when accuracy
≥ 0.9" escalation rule.

Background (see ../paper/derivation.md):
  For a group of G binary-reward rollouts with success prob p, the GRPO/DAPO
  learning signal ∝ within-group reward variance p(1-p), and the probability the
  group is *informative* (not all-correct / all-wrong) is
      U(d) = 1 - p(d)^G - (1 - p(d))^G ,
  both maximised at p = 0.5.

RLVE picks a single difficulty set-point (escalate past 0.9). We instead choose a
*distribution* over difficulty levels that minimises the free energy
      F[q] = - E_q[U] - T * H[q]
whose optimum is the Gibbs / Boltzmann distribution
      q*(d) ∝ exp( U(d) / T ).
  * T → 0   → argmax_d U(d) ≈ the p=0.5 level  (recovers RLVE's set-point);
  * T → ∞   → uniform over difficulties        (max exploration);
  * annealed T → focus on informative levels while still probing neighbours,
    and automatically abandon saturated (p→1) / hopeless (p→0) levels.

Integration with RLVE (official repo, SLIME backend):
  RLVE's per-environment difficulty is driven by
  `Gym/parameter_controllers/<env>/parameter_controller.py` (`update()` = +1 level,
  `get_parameter_list()` = params at the current level). The *adaptive escalation*
  (track success at the upper difficulty, escalate when high) lives on the SLIME
  rollout side. To apply free energy:
    1. maintain per-(env, difficulty) success EMA `p_hat` from observed rollouts;
    2. each step, call `FreeEnergyController.sample_difficulties(n)` to choose the
       difficulties to generate next, instead of the fixed-range uniform sampling;
    3. drive each env's `ParameterController` to the sampled level via `update()`.
  See README §5. This module has no SLIME/torch dependency so it can be unit-tested
  and dropped into the rollout loop directly.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List


class FreeEnergyController:
    def __init__(self, d_min: int = 0, d_max: int = 20, G: int = 8,
                 T0: float = 0.5, T_min: float = 0.1, anneal_steps: int = 40,
                 ema: float = 0.7):
        self.d_min, self.d_max = int(d_min), int(d_max)
        self.G = int(G)
        self.T0, self.T_min = float(T0), float(T_min)
        self.anneal_steps = int(anneal_steps)
        self.ema = float(ema)
        self.t = 0                       # step counter (drives annealing)
        # per-level success estimate; unseen levels default to 0.5 (max utility
        # → explored first). Keyed by int difficulty.
        self.p_hat: Dict[int, float] = {}

    # ---- core quantities -------------------------------------------------
    def levels(self) -> List[int]:
        return list(range(self.d_min, self.d_max + 1))

    def p(self, d: int) -> float:
        return self.p_hat.get(int(d), 0.5)

    def utility(self, d: int) -> float:
        """U(d) = 1 - p^G - (1-p)^G ∈ [0,1]; informative-group probability."""
        p = min(1.0, max(0.0, self.p(d)))
        return 1.0 - p ** self.G - (1.0 - p) ** self.G

    def temperature(self) -> float:
        if self.anneal_steps <= 0:
            return max(self.T_min, self.T0)
        frac = min(1.0, self.t / float(self.anneal_steps))
        return max(self.T_min, self.T0 * (1.0 - frac) + self.T_min * frac)

    def distribution(self):
        """Return (levels, q) with q(d) ∝ exp(U(d)/T)."""
        levels = self.levels()
        T = max(1e-6, self.temperature())
        us = [self.utility(d) for d in levels]
        m = max(us)
        ws = [math.exp((u - m) / T) for u in us]
        z = sum(ws) or 1.0
        return levels, [w / z for w in ws]

    # ---- API used by the rollout loop ------------------------------------
    def sample_difficulties(self, n: int, rng: random.Random | None = None) -> List[int]:
        """Sample n difficulty levels from the Gibbs distribution q(d)."""
        rng = rng or random
        levels, q = self.distribution()
        out = []
        for _ in range(int(n)):
            r = rng.random()
            acc = 0.0
            pick = levels[-1]
            for d, p in zip(levels, q):
                acc += p
                if r <= acc:
                    pick = d
                    break
            out.append(int(pick))
        return out

    def update(self, observed: Dict[int, float]) -> dict:
        """Update per-level success EMA from observed {difficulty: success_rate},
        advance the annealing clock. Returns a small metrics dict for logging."""
        for d, succ in observed.items():
            di = int(d)
            prev = self.p_hat.get(di)
            self.p_hat[di] = float(succ) if prev is None \
                else self.ema * prev + (1.0 - self.ema) * float(succ)
        self.t += 1
        levels, q = self.distribution()
        exp_d = sum(d * w for d, w in zip(levels, q))
        return {"T": round(self.temperature(), 4),
                "expected_difficulty": round(exp_d, 3),
                "p_hat": {k: round(v, 3) for k, v in sorted(self.p_hat.items())}}

    def env_weight(self) -> float:
        """Negative free energy T·logΣ_d exp(U(d)/T): an environment-selection
        weight from the same objective (high while informative levels remain).
        Use across environments to also unify curation if desired."""
        T = max(1e-6, self.temperature())
        us = [self.utility(d) for d in self.levels()]
        m = max(us)
        return float(m + T * math.log(sum(math.exp((u - m) / T) for u in us)))


if __name__ == "__main__":
    # self-check: adapts toward the informative band; T→0 concentrates at p=0.5.
    rng = random.Random(0)
    c = FreeEnergyController(d_min=0, d_max=20)
    print("sample:", c.sample_difficulties(8, rng))
    print("update:", c.update({0: 0.9, 1: 0.5, 2: 0.1}))
    cold = FreeEnergyController(d_min=0, d_max=20, T0=0.02, T_min=0.02, anneal_steps=0)
    for d in range(21):
        cold.p_hat[d] = 0.5 if d == 5 else (0.99 if d < 5 else 0.01)
    lv, q = cold.distribution()
    top = max(zip(lv, q), key=lambda x: x[1])
    print(f"T->0 concentrates on level {top[0]} (p=0.5 level), prob={top[1]:.3f}")
