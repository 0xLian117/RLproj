"""Wire the free-energy difficulty manager into an RLVE checkout, env-var gated.

Two tiny, reversible changes:
  1. copy `freeenergy_manager.py` into the RLVE repo root (so it's importable);
  2. in `slime/ray/rollout_data_source.py`, replace the single line
        self.rlve_manager = RLVEManager(args, tokenizer)
     with a switch that uses FreeEnergyRLVEManager when DIFFICULTY_MODE=freeenergy,
     and the original RLVEManager otherwise.

So with no env var the run is byte-for-byte RLVE; with DIFFICULTY_MODE=freeenergy
the difficulty scheduler becomes our Gibbs/free-energy controller. Idempotent.

    python apply_patch.py --rlve /path/to/RLVE
"""
from __future__ import annotations

import argparse
import os
import shutil

OLD = "self.rlve_manager = RLVEManager(args, tokenizer)"
NEW = (
    "import os as _os\n"
    "            if _os.environ.get(\"DIFFICULTY_MODE\", \"\").lower() in (\"freeenergy\", \"fe\"):\n"
    "                from freeenergy_manager import FreeEnergyRLVEManager as _RLVEMgr\n"
    "            else:\n"
    "                _RLVEMgr = RLVEManager\n"
    "            self.rlve_manager = _RLVEMgr(args, tokenizer)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlve", required=True, help="path to the RLVE repo root")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "freeenergy_manager.py")
    dst = os.path.join(args.rlve, "freeenergy_manager.py")
    target = os.path.join(args.rlve, "slime", "ray", "rollout_data_source.py")
    assert os.path.isfile(src), f"not found: {src}"
    assert os.path.isfile(target), f"not found: {target} (is --rlve the RLVE root?)"

    shutil.copyfile(src, dst)
    print(f"[patch] copied manager -> {dst}")

    code = open(target).read()
    if "DIFFICULTY_MODE" in code:
        print("[patch] rollout_data_source.py already patched; manager refreshed only.")
        return
    if OLD not in code:
        raise SystemExit(f"[patch] anchor line not found in {target}; RLVE version changed?")
    code = code.replace(OLD, NEW, 1)
    open(target, "w").write(code)
    print(f"[patch] switched RLVEManager instantiation in {target}")
    print("\nDone. Usage:")
    print("  RLVE baseline:    run as usual")
    print("  free-energy arm:  DIFFICULTY_MODE=freeenergy <run cmd>")


if __name__ == "__main__":
    main()
