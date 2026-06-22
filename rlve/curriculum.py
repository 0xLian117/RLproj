"""Curriculum orchestrator.

Ties together the per-environment difficulty controllers, the environment
sampler and the per-step signal bus. It is the single object shared between the
real GRPO trainer (``rlve/train.py``) and the CPU simulator
(``tools/simulate.py``), so the adaptive-difficulty logic is validated
identically in both.

Lifecycle each training step:
    1. ``next_problem(rng)``  is called B times to build the prompt batch.
    2. the reward fn calls ``record(...)`` once per rollout outcome.
    3. ``step_update()`` drains the bus, updates every controller and the
       sampler, and returns a metrics dict (also appended to ``self.history``).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from rlve.curriculum_types import Sample
from rlve.difficulty import make_controller, make_sampler
from rlve.envs.registry import make_train_envs
from rlve.stats import GroupOutcome, SignalBus, group_by_env, summarize


class Curriculum:
    def __init__(self, controller_kind: str = "stad", sampler_kind: str = "uniform",
                 env_names: Optional[List[str]] = None, controller_kw: Optional[dict] = None,
                 sampler_kw: Optional[dict] = None):
        all_envs = make_train_envs()
        if env_names is None:
            env_names = list(all_envs.keys())
        self.env_names = env_names
        self.envs = {n: all_envs[n] for n in env_names}
        controller_kw = controller_kw or {}
        sampler_kw = sampler_kw or {}
        self.controller_kind = controller_kind
        self.sampler_kind = sampler_kind
        self.controllers = {
            n: make_controller(controller_kind, self.envs[n], **controller_kw)
            for n in env_names
        }
        self.sampler = make_sampler(sampler_kind, env_names, **sampler_kw)
        self.bus = SignalBus()
        self.history: List[dict] = []
        self._step = 0

    # --- problem generation ----------------------------------------------
    def next_problem(self, rng: random.Random) -> Sample:
        env_name = self.sampler.sample_env(rng)
        env = self.envs[env_name]
        d = self.controllers[env_name].sample_difficulty(rng)
        problem = env.generate(d, rng)
        return Sample(env_name=env_name, difficulty=d, problem=problem)

    def verify(self, env_name: str, completion: str, problem) -> bool:
        return self.envs[env_name].verify(completion, problem).correct

    # --- signal recording -------------------------------------------------
    def record(self, prompt_key: str, env_name: str, difficulty: int, correct: bool):
        self.bus.record(prompt_key, env_name, difficulty, correct)

    # --- per-step update --------------------------------------------------
    def step_update(self) -> dict:
        groups: List[GroupOutcome] = self.bus.drain()
        by_env = group_by_env(groups)
        per_env_signal: Dict[str, float] = {}
        for name, ctrl in self.controllers.items():
            env_groups = by_env.get(name, [])
            ctrl.update(env_groups)
            per_env_signal[name] = summarize(env_groups)["effective_ratio"]
        self.sampler.update(per_env_signal)

        metrics = summarize(groups)
        metrics["step"] = self._step
        metrics["controller"] = self.controller_kind
        metrics["sampler"] = self.sampler_kind
        # per-env operating point + signal
        for name in self.env_names:
            st = self.controllers[name].state()
            metrics[f"diff/{name}"] = st.get("mu", st.get("h", st.get("level", 0)))
            metrics[f"eff/{name}"] = round(per_env_signal.get(name, 0.0), 3)
        if self.sampler_kind != "uniform":
            for name, w in self.sampler.weights().items():
                metrics[f"w/{name}"] = round(w, 3)
        self.history.append(metrics)
        self._step += 1
        return metrics
