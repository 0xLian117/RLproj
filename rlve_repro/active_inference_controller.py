"""FEP-RLVE — Expected-Free-Energy difficulty selection for RLVE.

Replaces RLVE's rule-based curriculum ("if accuracy > 0.9: difficulty += 1") with
a belief-based curriculum: maintain a posterior over the model's latent competence
per environment, and pick the next difficulty by minimizing Expected Free Energy
(EFE). Active inference (Friston et al.): action is chosen to minimize EFE, which
combines *pragmatic* value (goal) and *epistemic* value (information gain).

EFE per (environment e, difficulty d):

    G_e(d) = − λ_signal · U_K(p̂_e(d))        # pragmatic: training signal
             − λ_info   · I_e(d)              # epistemic: info gain about competence
             + λ_cost   · C_e(d)              # cost (optional; rollout/verify expense)

    π(d | e) ∝ exp( − G_e(d) / T )

where, for a group of K rollouts at predicted success p̂, the DAPO/RLVE
"effective prompt ratio" (prob the group is non-degenerate → keeps a gradient) is

    U_K(p) = 1 − p^K − (1−p)^K          (maximized at p = 0.5)

and the epistemic information gain about latent competence s_e is

    I_e(d) = H[q(s_e)] − E_{o∼p(o|d)} H[q(s_e | o, d)]
           = H[Bern(p̄)] − E_{q(s)} H[Bern(σ(a(s−d)))]   (mutual info I(s;o|d))

Belief (version B / IRT): competence s_e on a grid; likelihood
P(correct | s, d) = σ(a·(s − d)); Bayesian update from observed correct/wrong.

Ablations (same code, flags):
  * RLVE-90      : not this controller (the original threshold rule).
  * Static       : not this controller (frozen difficulty).
  * Signal-RLVE  : λ_info = 0          (target the signal band, no info gain).
  * FEP-RLVE     : λ_info > 0          (full EFE: signal + info gain + Gibbs).

Dependency-free (no torch/slime); see active_inference_manager.py for SLIME wiring.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _Hb(p: float) -> float:                       # binary entropy
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def U_K(p: float, K: int) -> float:               # effective prompt ratio
    return 1.0 - p ** K - (1.0 - p) ** K


class FEPRLVEController:
    def __init__(self, d_min: int = 0, d_max: int = 16, K: int = 8, slope: float = 1.0,
                 lambda_signal: float = 1.0, lambda_info: float = 1.0,
                 lambda_cost: float = 0.0, T: float = 0.25,
                 ability_pad: int = 3, grid_step: float = 0.5):
        self.d_min, self.d_max = int(d_min), int(d_max)
        self.K = int(K)
        self.a = float(slope)
        self.l_sig = float(lambda_signal)
        self.l_info = float(lambda_info)
        self.l_cost = float(lambda_cost)
        self.T = float(T)
        lo, hi = d_min - ability_pad, d_max + ability_pad
        n = int(round((hi - lo) / grid_step)) + 1
        self.s_grid = [lo + i * grid_step for i in range(n)]
        self.q = [1.0 / n for _ in range(n)]       # belief q(s): uniform → explore
        self.cost = {}                              # optional per-d cost C_e(d)

    # ---- likelihood / prediction ----
    def _p(self, s: float, d: int) -> float:
        return _sigmoid(self.a * (s - d))

    def p_hat(self, d: int) -> float:               # p̄ = E_{q(s)}[P(correct|s,d)]
        return sum(qi * self._p(s, d) for qi, s in zip(self.q, self.s_grid))

    # ---- EFE terms ----
    def signal(self, d: int) -> float:              # pragmatic: U_K(p̂)
        return U_K(self.p_hat(d), self.K)

    def info_gain(self, d: int) -> float:           # epistemic: I(s; o | d)
        pbar = self.p_hat(d)
        cond = sum(qi * _Hb(self._p(s, d)) for qi, s in zip(self.q, self.s_grid))
        return _Hb(pbar) - cond

    def C(self, d: int) -> float:
        return float(self.cost.get(d, 0.0))

    def G(self, d: int) -> float:                   # expected free energy (minimize)
        return (-self.l_sig * self.signal(d)
                - self.l_info * self.info_gain(d)
                + self.l_cost * self.C(d))

    def policy(self) -> Tuple[List[int], List[float]]:
        ds = list(range(self.d_min, self.d_max + 1))
        scores = [-self.G(d) / max(1e-6, self.T) for d in ds]
        m = max(scores)
        ws = [math.exp(s - m) for s in scores]
        z = sum(ws) or 1.0
        return ds, [w / z for w in ws]

    # ---- rollout-loop API ----
    def sample_difficulties(self, n: int, rng: random.Random | None = None) -> List[int]:
        rng = rng or random
        ds, pi = self.policy()
        out = []
        for _ in range(int(n)):
            r, acc, pick = rng.random(), 0.0, ds[-1]
            for d, p in zip(ds, pi):
                acc += p
                if r <= acc:
                    pick = d
                    break
            out.append(int(pick))
        return out

    def observe(self, outcomes: List[Tuple[int, int]]) -> None:
        """Bayesian belief update from [(difficulty, correct∈{0,1}), ...]."""
        if not outcomes:
            return
        logq = [math.log(max(qi, 1e-300)) for qi in self.q]
        for d, correct in outcomes:
            for i, s in enumerate(self.s_grid):
                p = self._p(s, d)
                logq[i] += math.log(max(p if correct else (1 - p), 1e-12))
        m = max(logq)
        ws = [math.exp(l - m) for l in logq]
        z = sum(ws) or 1.0
        self.q = [w / z for w in ws]

    # ---- diagnostics ----
    def competence_mean(self) -> float:
        return sum(qi * s for qi, s in zip(self.q, self.s_grid))

    def competence_std(self) -> float:
        mu = self.competence_mean()
        return math.sqrt(max(0.0, sum(qi * (s - mu) ** 2
                                      for qi, s in zip(self.q, self.s_grid))))

    def expected_difficulty(self) -> float:
        ds, pi = self.policy()
        return sum(d * p for d, p in zip(ds, pi))


if __name__ == "__main__":
    rng = random.Random(0)
    print("=== peaks: U_K(signal) and info-gain both maximal at p≈0.5 ===")
    c = FEPRLVEController(d_min=0, d_max=16, K=8, slope=1.0)
    for i, s in enumerate(c.s_grid):  # pin belief at competence ≈ 8
        c.q[i] = math.exp(-0.5 * ((s - 8) / 0.6) ** 2)
    z = sum(c.q); c.q = [x / z for x in c.q]
    print(" d  p̂(d)  U_K(signal)  info_gain")
    for d in [4, 6, 7, 8, 9, 10, 12]:
        print(f"{d:3d}  {c.p_hat(d):.2f}   {c.signal(d):.3f}       {c.info_gain(d):.3f}")

    print("\n=== FEP-RLVE: belief tracks rising competence; explore→exploit ===")
    c = FEPRLVEController(d_min=0, d_max=16, K=8, slope=1.2, T=0.25)
    s_true = 3.0
    print("step | belief μ±σ | E[d] | p̂@E[d]")
    for step in range(12):
        ds = c.sample_difficulties(32, rng)
        c.observe([(d, 1 if rng.random() < _sigmoid(1.2 * (s_true - d)) else 0) for d in ds])
        s_true += 0.7
        if step % 3 == 0 or step == 11:
            ed = c.expected_difficulty()
            print(f"{step:4d} | {c.competence_mean():4.1f}±{c.competence_std():.2f} "
                  f"| {ed:4.1f} | {c.p_hat(round(ed)):.2f}")

    print("\n=== ablation: Signal-RLVE (λ_info=0) vs FEP-RLVE (λ_info=1) ===")
    for li in (0.0, 1.0):
        c = FEPRLVEController(d_min=0, d_max=16, K=8, lambda_info=li, T=0.25)
        # flat/uncertain belief → info-gain term should widen exploration when on
        ds, pi = c.policy()
        ent = -sum(p * math.log(p) for p in pi if p > 0)
        print(f"  lambda_info={li}: policy entropy over difficulties = {ent:.3f}")
    print("OK")
