"""Build the course-required experiment arms from a SCALER train file.

Course landing point: 3-5 verifiable envs; STATIC vs ADAPTIVE difficulty;
generalization on held-out envs. SCALER's adaptive in-env difficulty controller
is serialized per-environment inside the train JSON (a dict env_name -> config,
with config['params']['difficulty']['params'] = DifficultyControl kwargs).

This script, from e.g. SCALER-8.json, emits:
  * <out>/SCALER-train.json     : the first N envs (ADAPTIVE arm, untouched)
  * <out>/SCALER-heldout.json   : the remaining envs (held-out generalization)
  * <out>/SCALER-static-lo/mid/hi.json : same N train envs but with each env's
        controller FROZEN at a fixed difficulty (k=0, step_cap=0, state.d=D),
        i.e. the STATIC-difficulty baselines (easy / medium / hard).

Run the ADAPTIVE arm with SCALER-train.json and each STATIC arm with the
corresponding file (same recipe, identical compute) to get the static-vs-adaptive
comparison. Evaluate every checkpoint on SCALER-heldout.json envs + the held-out
benchmarks for the generalization study.

Usage:
  python tools/scaler_make_arms.py --in SCALER-8.json --out arms --n-train 5 \
      --lo 3 --mid 9 --hi 15
"""
from __future__ import annotations

import argparse
import copy
import json
import os


def freeze(env_cfg: dict, d_level: int) -> None:
    """Freeze this env's difficulty controller at level d_level (clamped to dmax).

    Robust to whichever kwargs the JSON object-hook actually honors:
      * dmin = dmax = D  -> the controller clamps d to D every update (primary);
      * k = 0, step_cap = 0 -> the per-step delta is 0 (secondary);
      * state.d = D      -> it also STARTS at D.
    """
    dparams = env_cfg["params"]["difficulty"].setdefault("params", {})
    orig_dmax = int(dparams.get("dmax", 22))
    d = float(max(0, min(d_level, orig_dmax)))
    dparams["dmin"] = d
    dparams["dmax"] = d         # pin via clamp regardless of update law
    dparams["k"] = 0.0
    dparams["step_cap"] = 0
    dparams["state"] = {
        "d": d, "t": 0, "last_step": -1,
        "distance_history": [], "correct_history": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="e.g. SCALER-8.json")
    ap.add_argument("--out", default="arms")
    ap.add_argument("--n-train", type=int, default=5, help="3-5 verifiable envs to train on")
    ap.add_argument("--lo", type=int, default=3)
    ap.add_argument("--mid", type=int, default=9)
    ap.add_argument("--hi", type=int, default=15)
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)
    names = list(data.keys())
    assert len(names) >= args.n_train, f"need >= {args.n_train} envs, got {len(names)}"
    train_names = names[: args.n_train]
    held_names = names[args.n_train:]

    os.makedirs(args.out, exist_ok=True)
    train = {n: data[n] for n in train_names}
    held = {n: data[n] for n in held_names}

    def dump(obj, name):
        p = os.path.join(args.out, name)
        with open(p, "w") as f:
            json.dump(obj, f, ensure_ascii=False)
        return p

    out = []
    out.append(dump(train, "SCALER-train.json"))      # ADAPTIVE arm (default controller)
    if held:
        out.append(dump(held, "SCALER-heldout.json"))  # held-out generalization envs
    for tag, level in [("lo", args.lo), ("mid", args.mid), ("hi", args.hi)]:
        s = copy.deepcopy(train)
        for n in s:
            freeze(s[n], level)
        out.append(dump(s, f"SCALER-static-{tag}.json"))

    print(f"train envs ({len(train_names)}): {train_names}")
    print(f"held-out envs ({len(held_names)}): {held_names}")
    print("wrote:")
    for p in out:
        print("  ", p)
    print("\nARMS:")
    print("  ADAPTIVE : TRAIN_FILE=arms/SCALER-train.json   (SCALER default controller)")
    print(f"  STATIC-lo : TRAIN_FILE=arms/SCALER-static-lo.json  (frozen d={args.lo})")
    print(f"  STATIC-mid: TRAIN_FILE=arms/SCALER-static-mid.json (frozen d={args.mid})")
    print(f"  STATIC-hi : TRAIN_FILE=arms/SCALER-static-hi.json  (frozen d={args.hi})")


if __name__ == "__main__":
    main()
