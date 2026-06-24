"""FEPRLVEManager — drop-in replacement for RLVE's RLVEManager that selects
difficulty by minimizing Expected Free Energy (active inference), instead of the
"raise difficulty when accuracy >= 0.9" rule.

Per environment it keeps a posterior over the model's latent competence
(FEPRLVEController) and, each step, samples difficulty from π(d) ∝ exp(−G_e(d)/T),
    G_e(d) = + λ_signal·(−log U_K(p̂)) − λ_info·I(s;o|d) + λ_cost·C(d).
The pragmatic term is the active-inference preference risk −log P_pref(o) with the
preference set on "the group is informative" (predicted prob U_K(p̂)); −log U_K
sends risk →+∞ for degenerate (all-pass/all-fail) difficulties. FE_LSIG = λ_signal.
After rollouts it Bayesian-updates the competence belief from observed correct/wrong.

Subclasses the real `slime.ray.rollout_data_source.RLVEManager` and overrides only
generate_problem() and update(); reuses RLVE's Gym interface
(ParameterController.update()/get_parameter_list(), Sample.metadata, reward["accuracy"]).

Activated via env var DIFFICULTY_MODE:
  * fep     → full EFE (λ_info > 0): FEP-RLVE (ours)
  * signal  → λ_info = 0: Signal-RLVE ablation (target signal band, no info gain)
Other env-var tunables: FE_DMAX(16) FE_K(=n_samples_per_prompt) FE_SLOPE(1.0)
  FE_LSIG(1.0) FE_LINFO(1.0) FE_LCOST(0.0) FE_T(0.25).
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from slime.ray.rollout_data_source import RLVEManager
from Gym.environment import VerifiableEnvironment
from Gym.environments import identifier2environment
from Gym.parameter_controller import ParameterController
from Gym.parameter_controllers import identifier2controller
from slime.utils.types import Sample

from active_inference_controller import FEPRLVEController


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class FEPRLVEManager(RLVEManager):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        mode = os.environ.get("DIFFICULTY_MODE", "fep").lower()
        l_info = _envf("FE_LINFO", 1.0)
        if mode == "signal":          # Signal-RLVE ablation: no epistemic term
            l_info = 0.0
        self.fe_mode = mode
        self.fe_K = int(_envf("FE_K", getattr(args, "n_samples_per_prompt", 8)))
        dmax = int(_envf("FE_DMAX", 16))
        slope = _envf("FE_SLOPE", 1.0)
        l_sig = _envf("FE_LSIG", 1.0)
        l_cost = _envf("FE_LCOST", 0.0)
        T = _envf("FE_T", 0.25)
        self.ctrl: Dict[str, FEPRLVEController] = {
            e: FEPRLVEController(d_min=0, d_max=dmax, K=self.fe_K, slope=slope,
                                 lambda_signal=l_sig, lambda_info=l_info,
                                 lambda_cost=l_cost, T=T)
            for e in args.environment_list}
        print(f"[FEP-RLVE] mode={mode} dmax={dmax} K={self.fe_K} "
              f"lambda_signal={l_sig} lambda_info={l_info} T={T}", flush=True)

    # ---- overrides ----
    def generate_problem(self) -> Tuple[str, Optional[VerifiableEnvironment]]:
        environment: str = random.choice(self.args.environment_list)
        target_d: int = self.ctrl[environment].sample_difficulties(1)[0]

        pc: ParameterController = identifier2controller[environment]()
        for _ in range(target_d):
            pc.update()
        plist = pc.get_parameter_list()
        if not plist:
            return environment, target_d, None
        parameter: Dict = random.choice(plist)

        problem: VerifiableEnvironment = identifier2environment[environment]()
        if problem.generator(seed=self.problem_generation_seed, parameter=parameter):
            gen = problem
        else:
            gen = None
            print(f"[FEP-RLVE] gen failed env={environment} d={target_d}", flush=True)
        self.problem_generation_seed += 1
        return environment, target_d, gen

    def update(self, samples: List[Sample]) -> Dict[str, Any]:
        """Belief update per environment from observed (difficulty, correct)."""
        log: Dict[str, Any] = {"rollout/problem_generation_seed": self.problem_generation_seed}
        by_env: Dict[str, List[Tuple[int, int]]] = {}
        for s in samples:
            e = s.metadata["environment"]
            d = int(s.metadata["problem_difficulty"])
            a = int(round(float(s.reward["accuracy"])))
            by_env.setdefault(e, []).append((d, a))

        eff_sum, eff_n = 0.0, 0
        for e, outs in by_env.items():
            c = self.ctrl[e]
            # log the realized effective-sample-ratio at the difficulties used
            for d, a in outs:
                eff_sum += c.signal(d)
                eff_n += 1
            c.observe(outs)
            log[f"FEP/{e}/competence_mean"] = round(c.competence_mean(), 3)
            log[f"FEP/{e}/competence_std"] = round(c.competence_std(), 3)
            log[f"FEP/{e}/expected_difficulty"] = round(c.expected_difficulty(), 3)
        log["FEP/effective_sample_ratio"] = round(eff_sum / eff_n, 4) if eff_n else 0.0
        log["FEP/mode"] = self.fe_mode
        # Also echo belief metrics to stdout so they survive in the worker .out logs
        # (wandb is the primary sink, but plot_competence.py can then run offline too).
        belief = {k: v for k, v in log.items() if k.startswith("FEP/")}
        print(f"[FEP-RLVE] belief {belief}", flush=True)
        return log

    def get_state(self) -> Dict[str, Any]:
        st = super().get_state()
        st["fep_belief"] = {e: list(c.q) for e, c in self.ctrl.items()}
        return st

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state)
        for e, q in state.get("fep_belief", {}).items():
            if e in self.ctrl and len(q) == len(self.ctrl[e].q):
                self.ctrl[e].q = [float(x) for x in q]
