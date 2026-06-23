"""Free-Energy difficulty controller — a drop-in alternative to SCALER's
``recipe/environment/difficulty_control.py`` :class:`DifficultyControl`.

Motivation
----------
SCALER's controller is a fixed set-point regulator: it nudges the per-environment
difficulty so that the *success rate* tracks a constant target (≈0.5). That target
is justified because, for a group of ``G`` binary-reward rollouts, the GRPO/DAPO
learning signal is proportional to the within-group reward variance ``p(1-p)`` (and
to the probability the group is *informative*, ``1 - p^G - (1-p)^G``), both maximised
at ``p = 0.5``. See ``paper/derivation.md``.

We generalise that single set-point into an explicit objective. Define a per-level
utility (the informative-group probability)::

    U(d) = 1 - p(d)^G - (1 - p(d))^G

and, instead of collapsing to ``argmax_d U`` (=0.5), sample the next difficulties
from the Gibbs / max-entropy distribution that minimises the free energy
``F[q] = -E_q[U] - T*H[q]``::

    q(d) ∝ exp( U(d) / T )

* ``T → 0``  recovers ``argmax`` ≈ SCALER's 0.5 set-point (pure exploitation);
* ``T → ∞`` recovers uniform difficulty (pure exploration / diversity);
* intermediate, annealed ``T`` trades off "focus on the most informative levels"
  against "keep probing neighbouring levels", and naturally abandons saturated
  (p→1) and hopeless (p→0) levels because their ``U`` is low.

The same quantity yields an environment-level weight (negative free energy
``T·logΣ_d exp(U/T)``), so difficulty control and environment curation come from
one objective — though our recipe keeps uniform env sampling, so that weight is
only logged unless ``enable_weighted_sample=True``.

Interface
---------
This class mirrors the public surface the SCALER trainer touches:
``propose_distances(num)``, ``update(dist_dict, step)``, ``state['d']``, ``dmax``,
``get_weight(step)``, ``empty_histroy()``, ``get_windows_state()`` and the
``json_object_hook`` / ``json_default`` (de)serialisers. It also *reads* SCALER's
own version-1/2 payloads, so it can be loaded from the existing arm JSONs simply
by swapping the object hook (see ``apply_freeenergy_patch.py``). Tunables come from
env vars so no JSON regeneration is needed:

    FE_G        group size G (rollouts per prompt), default 8
    FE_T0       initial temperature,                default 0.5
    FE_TMIN     floor temperature,                  default 0.1
    FE_ANNEAL   steps to anneal T0 -> TMIN,         default 40
    FE_EMA      EMA factor for per-level success,   default 0.7
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class FreeEnergyDifficultyControl:
    def __init__(self, dmin=0.0, dmax=22.0, state=None,
                 # accepted for payload compatibility (SCALER fields); unused here
                 target=0.5, k=2, step_cap=1, jitter=0.0, ema_beta=0.0,
                 activate_function="freeenergy", history_len=20, slope_scale=0.05,
                 age_scale=100, std_scale=0.8, alpha=1.0, beta=0.5,
                 # free-energy specific (usually injected via env vars)
                 G=None, T0=None, T_min=None, anneal_steps=None, ema_u=None,
                 phat=None, **_ignored):
        self.dmin, self.dmax = int(dmin), int(dmax)
        # keep SCALER fields so re-serialisation is faithful
        self.target = target
        self.k = k
        self.step_cap = step_cap
        self.jitter = jitter
        self.ema_beta = ema_beta
        self.activate_function = activate_function
        self.history_len = history_len
        self.slope_scale = slope_scale
        self.age_scale = age_scale
        self.std_scale = std_scale
        self.alpha = alpha
        self.beta = beta

        # free-energy hyper-params (env var overrides win unless explicitly given)
        self.G = int(G if G is not None else _envf("FE_G", 8))
        self.T0 = float(T0 if T0 is not None else _envf("FE_T0", 0.5))
        self.T_min = float(T_min if T_min is not None else _envf("FE_TMIN", 0.1))
        self.anneal_steps = int(anneal_steps if anneal_steps is not None
                                else _envf("FE_ANNEAL", 40))
        self.ema_u = float(ema_u if ema_u is not None else _envf("FE_EMA", 0.7))

        if state:
            self.state = state
            self.state.setdefault("d", float(self.dmin))
            self.state.setdefault("t", 0)
            self.state.setdefault("last_step", -1)
            self.state.setdefault("distance_history", [])
            self.state.setdefault("correct_history", [])
        else:
            self.state = {"d": float(self.dmin), "t": 0, "last_step": -1,
                          "distance_history": [], "correct_history": []}

        # per-level success-rate estimate p_hat[d]; missing => optimistic 0.5
        # (0.5 => max utility => unseen levels get explored first).
        self.phat: Dict[int, float] = {int(k): float(v) for k, v in (phat or {}).items()}

    # ----------------------- core free-energy machinery -----------------------
    def _levels(self) -> List[int]:
        return list(range(self.dmin, self.dmax + 1))

    def _p(self, d: int) -> float:
        return self.phat.get(int(d), 0.5)

    def _utility(self, d: int) -> float:
        """Informative-group probability U(d) = 1 - p^G - (1-p)^G in [0,1]."""
        p = min(1.0, max(0.0, self._p(d)))
        return 1.0 - p ** self.G - (1.0 - p) ** self.G

    def temperature(self) -> float:
        t = self.state.get("t", 0)
        if self.anneal_steps <= 0:
            return max(self.T_min, self.T0)
        frac = min(1.0, t / float(self.anneal_steps))
        return max(self.T_min, self.T0 * (1.0 - frac) + self.T_min * frac)

    def _gibbs(self):
        """Return (levels, probs) with q(d) ∝ exp(U(d)/T)."""
        levels = self._levels()
        T = max(1e-6, self.temperature())
        us = [self._utility(d) for d in levels]
        m = max(us)
        ws = [math.exp((u - m) / T) for u in us]
        z = sum(ws) or 1.0
        return levels, [w / z for w in ws]

    # ----------------------- trainer-facing API -----------------------
    def propose_distances(self, batch_size) -> List[int]:
        """Sample ``batch_size`` integer difficulties from the Gibbs distribution."""
        n = int(batch_size)
        levels, probs = self._gibbs()
        # expected difficulty, logged via state['d'] for continuity with SCALER
        self.state["d"] = float(sum(d * p for d, p in zip(levels, probs)))
        out = []
        for _ in range(n):
            r = random.random()
            acc = 0.0
            pick = levels[-1]
            for d, p in zip(levels, probs):
                acc += p
                if r <= acc:
                    pick = d
                    break
            out.append(int(pick))
        return out

    def update(self, distance_correct_avg_len_dict, now_step=None):
        """Update per-level success EMA from observed (avg_correct, count), anneal T.

        ``distance_correct_avg_len_dict``: {distance: (correct_avg, count)}.
        Returns [mean_utility_observed, temperature, expected_d] for logging.
        """
        observed_u = []
        for d, val in distance_correct_avg_len_dict.items():
            try:
                correct_avg = float(val[0])
            except (TypeError, IndexError):
                correct_avg = float(val)
            di = int(round(float(d)))
            prev = self.phat.get(di, None)
            self.phat[di] = correct_avg if prev is None \
                else self.ema_u * prev + (1.0 - self.ema_u) * correct_avg
            self.state["distance_history"].append(float(di))
            self.state["correct_history"].append(float(correct_avg))
            observed_u.append(self._utility(di))

        if len(self.state["distance_history"]) > self.history_len:
            self.state["distance_history"] = self.state["distance_history"][-self.history_len:]
            self.state["correct_history"] = self.state["correct_history"][-self.history_len:]

        self.state["t"] = self.state.get("t", 0) + 1
        if now_step is not None:
            self.state["last_step"] = now_step

        levels, probs = self._gibbs()
        self.state["d"] = float(sum(d * p for d, p in zip(levels, probs)))
        mean_u = sum(observed_u) / len(observed_u) if observed_u else 0.0
        return [round(mean_u, 4), round(self.temperature(), 4), round(self.state["d"], 3)]

    def get_weight(self, global_step):
        """Environment-level weight = negative free energy T·logΣ_d exp(U(d)/T) ≥ 0.

        Higher when the environment still has informative difficulty levels; low
        when every level is saturated/hopeless. Used only if the trainer enables
        weighted environment sampling.
        """
        T = max(1e-6, self.temperature())
        us = [self._utility(d) for d in self._levels()]
        m = max(us)
        lse = m + T * math.log(sum(math.exp((u - m) / T) for u in us))
        return max(0.0, float(lse))

    # ----------------------- windows-mode shims (unused unless enabled) ------
    def empty_histroy(self):  # noqa: keep SCALER's spelling
        self.state["distance_history"] = []
        self.state["correct_history"] = []

    def get_windows_state(self):
        hist = self.state.get("distance_history", [])
        max_distance_count = sum(1 for d in reversed(hist) if abs(d - self.dmax) <= 1e-7)
        ch = self.state.get("correct_history", [])
        zero_correct_count = 0
        for c in reversed(ch):
            if c == 0:
                zero_correct_count += 1
            else:
                break
        slope = 0.0
        L = len(hist)
        if L >= 2:
            tm = (L - 1) / 2.0
            dm = sum(hist) / L
            num = sum((i - tm) * (d - dm) for i, d in enumerate(hist))
            den = sum((i - tm) ** 2 for i in range(L))
            slope = num / den if den > 1e-8 else 0.0
        return {"zero_correct_count": zero_correct_count,
                "max_distance_count": max_distance_count, "distance_slope": slope}

    # ----------------------- serialisation -----------------------
    def _to_serializable(self):
        return {
            "version": 3,
            "mode": "freeenergy",
            "params": {
                "target": self.target, "k": self.k,
                "dmin": self.dmin, "dmax": self.dmax,
                "step_cap": self.step_cap, "jitter": self.jitter,
                "ema_beta": self.ema_beta, "state": dict(self.state),
                "activate_function": self.activate_function,
                "history_len": self.history_len, "slope_scale": self.slope_scale,
                "age_scale": self.age_scale, "std_scale": self.std_scale,
                "alpha": self.alpha, "beta": self.beta,
                "G": self.G, "T0": self.T0, "T_min": self.T_min,
                "anneal_steps": self.anneal_steps, "ema_u": self.ema_u,
                "phat": {str(k): v for k, v in self.phat.items()},
            },
        }

    @classmethod
    def _from_serializable(cls, payload):
        return cls(**payload["params"])

    @staticmethod
    def json_default(o):
        if isinstance(o, FreeEnergyDifficultyControl):
            return o._to_serializable()
        raise TypeError(f"{type(o)} is not JSON serializable")

    @staticmethod
    def json_object_hook(d):
        # Accepts SCALER's version 1/2 payloads AND our version 3 payload, so the
        # existing arm JSONs load straight into the free-energy controller.
        if isinstance(d, dict) and d.get("version") in (1, 2, 3) and "params" in d:
            try:
                return FreeEnergyDifficultyControl._from_serializable(d)
            except Exception as e:  # pragma: no cover
                print("FreeEnergy json_object_hook error:", e)
                return d
        return d


if __name__ == "__main__":
    # quick self-check
    import json
    c = FreeEnergyDifficultyControl(dmin=0, dmax=10)
    s = json.dumps(c, default=FreeEnergyDifficultyControl.json_default)
    c2 = json.loads(s, object_hook=FreeEnergyDifficultyControl.json_object_hook)
    print("roundtrip ok:", isinstance(c2, FreeEnergyDifficultyControl))
    print("propose:", c2.propose_distances(8))
    print("update:", c2.update({0: (0.9, 8), 1: (0.5, 8)}, now_step=1))
