"""Build a FIXED evaluation set, identical for every model (fair comparison).

For each environment we sample ``n_per`` problems at each difficulty in
``EVAL_DIFFICULTIES``. The set is generated from a fixed seed and cached to JSON
so the base model and every trained checkpoint are scored on exactly the same
problems. We deliberately include high difficulties that are rarely reached
during training, to probe difficulty generalization, plus the held-out
environments to probe cross-environment generalization.
"""
from __future__ import annotations

import json
import os
import random
from typing import List

from rlve.data import build_prompt
from rlve.envs.registry import (heldout_env_names, make_env, train_env_names)

EVAL_DIFFICULTIES = [1, 3, 5, 7, 9]


def build_eval_set(n_per: int = 16, seed: int = 12345) -> List[dict]:
    rng = random.Random(seed)
    items = []
    names = [(n, "train") for n in train_env_names()] + \
            [(n, "heldout") for n in heldout_env_names()]
    for name, split in names:
        env = make_env(name)
        for d in EVAL_DIFFICULTIES:
            if d < env.min_difficulty or d > env.max_difficulty:
                continue
            for _ in range(n_per):
                prob = env.generate(d, rng)
                items.append({
                    "env": name, "split": split, "difficulty": d,
                    "question": prob.question, "answer": prob.answer,
                    "prompt": build_prompt(prob.question),
                })
    return items


def get_or_build(path: str, n_per: int = 16, seed: int = 12345) -> List[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    items = build_eval_set(n_per=n_per, seed=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f)
    return items


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_set.json")
    ap.add_argument("--n-per", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    items = get_or_build(args.out, n_per=args.n_per, seed=args.seed)
    print(f"Built {len(items)} eval problems -> {args.out}")
