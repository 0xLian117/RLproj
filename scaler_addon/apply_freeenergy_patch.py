"""Idempotently patch SCALER so the difficulty controller can be switched to the
free-energy variant at runtime via the env var ``DIFFICULTY_MODE=freeenergy``.

What it does (two tiny, reversible edits to the SCALER checkout):
  1. copy ``freeenergy_difficulty.py`` into ``recipe/environment/``;
  2. in ``recipe/environment/dapo_ray_trainer.py`` replace each
     ``object_hook=DifficultyControl.json_object_hook`` with a call to a helper
     ``_difficulty_hook()`` that returns the free-energy hook when
     ``DIFFICULTY_MODE`` in {freeenergy, fe}, else the original SCALER hook.

So with no env var the behaviour is byte-for-byte SCALER; with the env var set the
env-difficulty controller becomes the free-energy / Gibbs one. Run from anywhere:

    python apply_freeenergy_patch.py --scaler /path/to/SCALER

Re-running is safe (it detects an already-patched file and only refreshes the
copied controller).
"""
from __future__ import annotations

import argparse
import os
import shutil

HELPER = '''
# ---- BEGIN free-energy difficulty hook (added by apply_freeenergy_patch.py) ----
import os as _os
try:
    from .freeenergy_difficulty import FreeEnergyDifficultyControl as _FEDC
except Exception:
    _FEDC = None
def _difficulty_hook():
    if _os.environ.get("DIFFICULTY_MODE", "").lower() in ("freeenergy", "fe") and _FEDC is not None:
        return _FEDC.json_object_hook
    return DifficultyControl.json_object_hook
# ---- END free-energy difficulty hook ----
'''

OLD = "object_hook=DifficultyControl.json_object_hook"
NEW = "object_hook=_difficulty_hook()"
ANCHOR = "from .difficulty_control import DifficultyControl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaler", required=True, help="path to the SCALER repo root")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(args.scaler, "recipe", "environment")
    trainer = os.path.join(env_dir, "dapo_ray_trainer.py")
    src_ctrl = os.path.join(here, "freeenergy_difficulty.py")
    dst_ctrl = os.path.join(env_dir, "freeenergy_difficulty.py")

    assert os.path.isfile(trainer), f"not found: {trainer}"
    assert os.path.isfile(src_ctrl), f"not found: {src_ctrl}"

    # 1) copy / refresh the controller
    shutil.copyfile(src_ctrl, dst_ctrl)
    print(f"[patch] copied controller -> {dst_ctrl}")

    # 2) patch the trainer
    with open(trainer, "r") as f:
        code = f.read()

    if "_difficulty_hook" not in code:
        if ANCHOR not in code:
            raise SystemExit(f"[patch] anchor import not found in {trainer}; "
                             "is this the expected SCALER version?")
        code = code.replace(ANCHOR, ANCHOR + "\n" + HELPER, 1)
        n = code.count(OLD)
        code = code.replace(OLD, NEW)
        with open(trainer, "w") as f:
            f.write(code)
        print(f"[patch] inserted hook + replaced {n} object_hook site(s) in {trainer}")
    else:
        print("[patch] trainer already patched; controller refreshed only.")

    print("\nDone. Usage:")
    print("  default arms (SCALER controller): run normally")
    print("  free-energy arm:                  DIFFICULTY_MODE=freeenergy <run cmd>")


if __name__ == "__main__":
    main()
