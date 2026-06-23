"""Wire the FEP-RLVE difficulty manager into an RLVE checkout, env-var gated.

Changes (small, reversible):
  1. copy `active_inference_controller.py` + `active_inference_manager.py` into the
     RLVE repo root (so they're importable);
  2. in `slime/ray/rollout_data_source.py`, replace the single line
        self.rlve_manager = RLVEManager(args, tokenizer)
     with a switch: DIFFICULTY_MODE in {fep, signal} → FEPRLVEManager, else the
     original RLVEManager.

So with no env var the run is byte-for-byte RLVE; with DIFFICULTY_MODE=fep (or
=signal for the ablation) the difficulty scheduler becomes our EFE controller.
Idempotent.

    python apply_patch.py --rlve /path/to/RLVE
"""
from __future__ import annotations

import argparse
import os
import shutil

OLD = "self.rlve_manager = RLVEManager(args, tokenizer)"
NEW = (
    "import os as _os\n"
    "            if _os.environ.get(\"DIFFICULTY_MODE\", \"\").lower() in (\"fep\", \"signal\"):\n"
    "                from active_inference_manager import FEPRLVEManager as _RLVEMgr\n"
    "            else:\n"
    "                _RLVEMgr = RLVEManager\n"
    "            self.rlve_manager = _RLVEMgr(args, tokenizer)"
)
FILES = ["active_inference_controller.py", "active_inference_manager.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlve", required=True, help="path to the RLVE repo root")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(args.rlve, "slime", "ray", "rollout_data_source.py")
    assert os.path.isfile(target), f"not found: {target} (is --rlve the RLVE root?)"

    for f in FILES:
        src = os.path.join(here, f)
        assert os.path.isfile(src), f"not found: {src}"
        shutil.copyfile(src, os.path.join(args.rlve, f))
        print(f"[patch] copied {f} -> {args.rlve}")

    code = open(target).read()
    if "DIFFICULTY_MODE" in code:
        print("[patch] rollout_data_source.py already patched; modules refreshed only.")
        return
    if OLD not in code:
        raise SystemExit(f"[patch] anchor line not found in {target}; RLVE version changed?")
    code = code.replace(OLD, NEW, 1)
    open(target, "w").write(code)
    print(f"[patch] switched RLVEManager instantiation in {target}")
    print("\nDone. Usage:")
    print("  RLVE-90 baseline:  run as usual")
    print("  Signal-RLVE:       DIFFICULTY_MODE=signal <run cmd>   (ablation, lambda_info=0)")
    print("  FEP-RLVE (ours):   DIFFICULTY_MODE=fep    <run cmd>")


if __name__ == "__main__":
    main()
