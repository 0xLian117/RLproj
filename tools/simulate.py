"""CPU-only simulation of the curriculum on a synthetic learning policy.

This is NOT the real experiment (that needs a GPU; see ``rlve/train.py``).
It is a *mechanism validation*: it exercises exactly the same Curriculum,
controllers and sampler used in training, driving them with a synthetic policy
whose per-environment ability rises over time.

Modeling assumption (the premise of GRPO itself): a group of G rollouts only
produces a learning gradient when it is *informative* (not all-correct / not
all-wrong). We therefore let the policy's ability on an environment improve in
proportion to the number of informative groups it received that step:

    P(correct | env, d) = sigmoid( ability[env] - d )
    ability[env] += lr * (#informative groups for env this step)

Under this model an adaptive controller that maximises the informative-group
rate should learn fastest -- which is precisely the hypothesis our real
experiment tests. Run produces a JSON the same shape as a real training log so
the plotting code is shared.

Environments are made HETEROGENEOUS (different difficulty slopes and learning
speeds) so that a uniform sampler wastes rollouts on already-saturated envs
while the learning-progress sampler can reallocate them -- letting us see the
sampler's contribution, not just the controller's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict

from rlve.curriculum import Curriculum
from rlve.stats import group_by_env


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def env_params(env_names):
    """Deterministic per-env heterogeneity: difficulty slope + learning speed."""
    params = {}
    for i, n in enumerate(sorted(env_names)):
        # slope in [0.7, 1.3], learn-rate multiplier in [0.4, 1.6]
        slope = 0.7 + 0.6 * (i % 3) / 2.0
        lr_mult = 0.4 + 1.2 * ((i * 2 + 1) % 5) / 4.0
        params[n] = {"slope": slope, "lr_mult": lr_mult}
    return params


def run(controller: str, sampler: str, steps: int, batch: int, gens: int,
        lr: float, seed: int, controller_kw: dict, sampler_kw: dict) -> dict:
    rng = random.Random(seed)
    curr = Curriculum(controller_kind=controller, sampler_kind=sampler,
                      controller_kw=controller_kw, sampler_kw=sampler_kw)
    ability: Dict[str, float] = {n: 0.0 for n in curr.env_names}
    params = env_params(curr.env_names)

    for step in range(steps):
        samples = [curr.next_problem(rng) for _ in range(batch)]
        for i, s in enumerate(samples):
            key = f"{step}-{i}"
            slope = params[s.env_name]["slope"]
            p = sigmoid(slope * (ability[s.env_name] - s.difficulty))
            for _ in range(gens):
                correct = rng.random() < p
                curr.record(key, s.env_name, s.difficulty, correct)
        # learning happens BEFORE step_update drains the bus
        groups = list(curr.bus._pending.values())
        informative = {n: 0 for n in curr.env_names}
        for (e, d, n, c) in groups:
            if 0 < c < n:
                informative[e] += 1
        for n in curr.env_names:
            ability[n] += lr * params[n]["lr_mult"] * informative[n]
        m = curr.step_update()
        m["mean_ability"] = round(sum(ability.values()) / len(ability), 4)

    final_ability = {n: round(ability[n], 3) for n in curr.env_names}
    return {
        "controller": controller, "sampler": sampler,
        "config": {"steps": steps, "batch": batch, "gens": gens, "lr": lr,
                   "seed": seed, "controller_kw": controller_kw,
                   "sampler_kw": sampler_kw},
        "final_ability": final_ability,
        "mean_final_ability": round(sum(final_ability.values()) / len(final_ability), 4),
        "history": curr.history,
    }


def avg(hist, key, last=None):
    vals = [h[key] for h in hist if key in h]
    if last:
        vals = vals[-last:]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/sim")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    conditions = [
        ("static-easy", "static", "uniform", {"static_level": 2}, {}),
        ("static-hard", "static", "uniform", {"static_level": 9}, {}),
        ("static-easy+LP", "static", "lp", {"static_level": 2}, {}),
        ("threshold(RLVE)", "threshold", "uniform", {}, {}),
        ("threshold+LP", "threshold", "lp", {}, {}),
        ("STAD(ours)", "stad", "uniform", {}, {}),
        ("STAD+LP(ours-full)", "stad", "lp", {}, {}),
    ]

    print(f"{'condition':22s} {'meanAbility':>11s} {'effRatio(all)':>13s} "
          f"{'effRatio(last50)':>16s} {'success(last50)':>15s}")
    print("-" * 82)
    summary = {}
    for label, ctrl, samp, ckw, skw in conditions:
        res = run(ctrl, samp, args.steps, args.batch, args.gens, args.lr,
                  args.seed, ckw, skw)
        h = res["history"]
        eff_all = avg(h, "effective_ratio")
        eff_last = avg(h, "effective_ratio", last=50)
        succ_last = avg(h, "success_rate", last=50)
        summary[label] = {
            "mean_final_ability": res["mean_final_ability"],
            "eff_ratio_all": round(eff_all, 4),
            "eff_ratio_last50": round(eff_last, 4),
            "success_last50": round(succ_last, 4),
            "final_ability": res["final_ability"],
        }
        with open(os.path.join(args.out, f"{label.replace('/', '_')}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"{label:22s} {res['mean_final_ability']:11.3f} {eff_all:13.3f} "
              f"{eff_last:16.3f} {succ_last:15.3f}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote per-condition logs + summary.json to {args.out}/")


if __name__ == "__main__":
    main()
