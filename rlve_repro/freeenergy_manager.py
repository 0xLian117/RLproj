"""FreeEnergyRLVEManager — drop-in replacement for RLVE's RLVEManager that
selects difficulty by free-energy / Gibbs sampling instead of the
"raise difficulty when accuracy >= 0.9" heuristic.

It subclasses the real `slime.ray.rollout_data_source.RLVEManager` and overrides
exactly two methods:
  * generate_problem(): pick env uniformly (as RLVE does), then choose the
    difficulty level from q(d) ∝ exp(U(d)/T) over [0, d_max], drive that env's
    Gym ParameterController to that level, and generate a problem there.
  * update(samples): update a per-(env, difficulty) success EMA p̂, anneal T.
    No upper-bound bump — difficulty is re-sampled from the Gibbs policy each
    time, so it can go both up and down toward the signal-optimal band.

U(d) = 1 - p̂(d)^G - (1 - p̂(d))^G  (informative-group probability; max at p=0.5).
See ../paper/derivation.md and ./freeenergy_controller.py.

Activated by env var DIFFICULTY_MODE=freeenergy (see apply_patch.py). Tunables via
env vars: FE_DMAX (default 16), FE_G (default = args.n_samples_per_prompt),
FE_T0 (0.6), FE_TMIN (0.1), FE_ANNEAL (steps, 60), FE_EMA (0.7).
"""
from __future__ import annotations

import copy
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from slime.ray.rollout_data_source import RLVEManager
from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from Gym.parameter_controller import ParameterController
from Gym.parameter_controllers import identifier2controller
from slime.utils.types import Sample


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class FreeEnergyRLVEManager(RLVEManager):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.fe_dmax = int(_envf("FE_DMAX", 16))
        self.fe_G = int(_envf("FE_G", getattr(args, "n_samples_per_prompt", 8)))
        self.fe_T0 = _envf("FE_T0", 0.6)
        self.fe_Tmin = _envf("FE_TMIN", 0.1)
        self.fe_anneal = int(_envf("FE_ANNEAL", 60))
        self.fe_ema = _envf("FE_EMA", 0.7)
        # per-env, per-difficulty success-rate EMA; unseen -> 0.5 (max utility)
        self.env_phat: Dict[str, Dict[int, float]] = {
            e: {} for e in args.environment_list}
        self.fe_step = 0
        print(f"[FreeEnergy] active: dmax={self.fe_dmax} G={self.fe_G} "
              f"T0={self.fe_T0} Tmin={self.fe_Tmin} anneal={self.fe_anneal}", flush=True)

    # ---- free-energy quantities ----
    def _T(self) -> float:
        if self.fe_anneal <= 0:
            return max(self.fe_Tmin, self.fe_T0)
        frac = min(1.0, self.fe_step / float(self.fe_anneal))
        return max(self.fe_Tmin, self.fe_T0 * (1 - frac) + self.fe_Tmin * frac)

    def _U(self, env: str, d: int) -> float:
        p = self.env_phat[env].get(int(d), 0.5)
        p = min(1.0, max(0.0, p))
        return 1.0 - p ** self.fe_G - (1.0 - p) ** self.fe_G

    def _gibbs(self, env: str):
        levels = list(range(0, self.fe_dmax + 1))
        T = max(1e-6, self._T())
        us = [self._U(env, d) for d in levels]
        m = max(us)
        ws = [math.exp((u - m) / T) for u in us]
        z = sum(ws) or 1.0
        return levels, [w / z for w in ws]

    def _sample_difficulty(self, env: str) -> int:
        levels, q = self._gibbs(env)
        r = random.random()
        acc = 0.0
        for d, p in zip(levels, q):
            acc += p
            if r <= acc:
                return int(d)
        return int(levels[-1])

    # ---- overrides ----
    def generate_problem(self) -> Tuple[str, Optional[VerifiableEnvironment]]:
        environment: str = random.choice(self.args.environment_list)
        target_d: int = self._sample_difficulty(environment)

        # drive the env's controller up to target_d, collect its parameter list
        parameter_controller: ParameterController = identifier2controller[environment]()
        for _ in range(target_d):
            parameter_controller.update()
        parameter_list = parameter_controller.get_parameter_list()
        if not parameter_list:
            return environment, target_d, None
        parameter: Dict = random.choice(parameter_list)

        problem: VerifiableEnvironment = identifier2environment[environment]()
        if problem.generator(seed=self.problem_generation_seed, parameter=parameter):
            generated = problem
        else:
            generated = None
            print(f"[FreeEnergy] gen failed env={environment} d={target_d} "
                  f"param={parameter}", flush=True)
        self.problem_generation_seed += 1
        return environment, target_d, generated

    def update(self, samples: List[Sample]) -> Dict[str, Any]:
        """EMA-update per-(env, difficulty) success; anneal T. No bump rule."""
        log_dict: Dict[str, Any] = {}
        # accumulate this round's success per (env, difficulty)
        acc: Dict[Tuple[str, int], List[int]] = {}
        for s in samples:
            env = s.metadata["environment"]
            d = int(s.metadata["problem_difficulty"])
            a = float(s.reward["accuracy"])
            acc.setdefault((env, d), [0.0, 0])
            acc[(env, d)][0] += a
            acc[(env, d)][1] += 1

        eff_sum, eff_n = 0.0, 0
        for (env, d), (ssum, n) in acc.items():
            obs = ssum / n if n else 0.0
            prev = self.env_phat[env].get(d)
            self.env_phat[env][d] = obs if prev is None \
                else self.fe_ema * prev + (1 - self.fe_ema) * obs
            eff_sum += self._U(env, d) * n
            eff_n += n

        self.fe_step += 1
        log_dict["rollout/problem_generation_seed"] = self.problem_generation_seed
        log_dict["FreeEnergy/temperature"] = round(self._T(), 4)
        log_dict["FreeEnergy/effective_sample_ratio"] = round(eff_sum / eff_n, 4) if eff_n else 0.0
        for env in self.args.environment_list:
            levels, q = self._gibbs(env)
            exp_d = sum(d * w for d, w in zip(levels, q))
            log_dict[f"FreeEnergy/{env}/expected_difficulty"] = round(exp_d, 3)
        return log_dict

    def get_state(self) -> Dict[str, Any]:
        st = super().get_state()
        st["env_phat"] = {e: dict(v) for e, v in self.env_phat.items()}
        st["fe_step"] = self.fe_step
        return st

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state)
        if "env_phat" in state:
            self.env_phat = {e: {int(k): float(v) for k, v in d.items()}
                             for e, d in state["env_phat"].items()}
        self.fe_step = int(state.get("fe_step", 0))
