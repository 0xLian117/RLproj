"""Process-level diagnostics for the 3-arm RLVE run (not results-only).

Reads results_all_arms/metrics.csv (success, truncated per arm/step), parses
entropy_loss/pg_loss sequences and HELD-OUT eval points from the worker/driver
logs, and writes results_all_arms/figures/diagnostics.png — the figure that
backs the mechanism story in ANALYSIS.md:
  (a) held-out is flat & floored        -> no generalization for any arm
  (b) truncation pinned 0.7-0.9         -> groups go degenerate, signal dies
  (c) entropy collapses ~40%            -> policy sharpens (it IS moving)...
  (d) training success parked near 0.5  -> ...but that 0.5 is controller-made

Usage: python diagnostics.py --logs results_logs_all_arms \
       --metrics results_all_arms/metrics.csv --out results_all_arms/figures/diagnostics.png
"""
from __future__ import annotations
import argparse, glob, os, re, csv, collections

ARM_ORDER = ["adaptive", "signal", "fep"]
ARM_LABEL = {"adaptive": "RLVE-90 (acc≥0.9)",
             "signal": "Signal-RLVE (λ_info=0)",
             "fep": "FEP-RLVE (ours)"}
COL = {"adaptive": "#2c6e8f", "signal": "#e0a030", "fep": "#c0392b"}
ENT = re.compile(r"entropy_loss':\s*(-?[0-9.eE-]+)")
EVAL = re.compile(r"eval (\d+):\s*\{[^}]*'eval/HELD-OUT_ENVIRONMENTS_128':\s*(-?\d+\.?\d*)")


def seq(arm, logs, rx):
    vals = []
    for f in sorted(glob.glob(os.path.join(logs, arm, "*.log")) +
                    glob.glob(os.path.join(logs, arm, "*.out"))):
        txt = open(f, errors="ignore").read()
        hits = [float(m.group(1)) for m in rx.finditer(txt)]
        if len(hits) > len(vals):
            vals = hits
    return vals


def evals(arm, logs):
    ev = {}
    for f in glob.glob(os.path.join(logs, arm, "*.out")):
        for m in EVAL.finditer(open(f, errors="ignore").read()):
            ev[int(m.group(1))] = float(m.group(2))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results_logs_all_arms")
    ap.add_argument("--metrics", default="results_all_arms/metrics.csv")
    ap.add_argument("--out", default="results_all_arms/figures/diagnostics.png")
    a = ap.parse_args()

    succ = collections.defaultdict(list)
    trunc = collections.defaultdict(list)
    with open(a.metrics) as fh:
        for r in csv.DictReader(fh):
            succ[r["arm"]].append((int(r["step"]), float(r["success"])))
            if r["truncated"] not in ("", "None"):
                trunc[r["arm"]].append((int(r["step"]), float(r["truncated"])))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    # (a) held-out trajectory
    for arm in ARM_ORDER:
        ev = evals(arm, a.logs)
        if ev:
            xs = sorted(ev); ax[0, 0].plot(xs, [ev[x] for x in xs], "o-",
                                           color=COL[arm], label=ARM_LABEL[arm])
    ax[0, 0].set_title("(a) Held-out reward: flat & floored (≈ init competence)")
    ax[0, 0].set_xlabel("training step"); ax[0, 0].set_ylabel("held-out reward (128 envs)")
    ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=.3)

    # (b) truncation
    for arm in ARM_ORDER:
        t = trunc[arm]
        if t: ax[0, 1].plot([s for s, _ in t], [v for _, v in t], ".-",
                            color=COL[arm], label=ARM_LABEL[arm])
    ax[0, 1].axhspan(0.7, 0.9, color="grey", alpha=.12)
    ax[0, 1].set_title("(b) Truncated fraction pinned 0.7–0.9 → degenerate groups")
    ax[0, 1].set_xlabel("training step"); ax[0, 1].set_ylabel("truncated ratio")
    ax[0, 1].set_ylim(0, 1); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=.3)

    # (c) entropy collapse
    for arm in ARM_ORDER:
        e = seq(arm, a.logs, ENT)
        if e: ax[1, 0].plot(range(len(e)), e, "-", color=COL[arm], label=ARM_LABEL[arm])
    ax[1, 0].set_title("(c) Policy entropy collapses ~40% (it IS moving)")
    ax[1, 0].set_xlabel("training step"); ax[1, 0].set_ylabel("entropy_loss")
    ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=.3)

    # (d) training success vs set-point
    for arm in ARM_ORDER:
        s = succ[arm]
        if s: ax[1, 1].plot([x for x, _ in s], [v for _, v in s], ".-",
                            color=COL[arm], label=ARM_LABEL[arm])
    ax[1, 1].axhline(0.5, ls="--", c="k", lw=.9)
    ax[1, 1].set_title("(d) Training success parked ≈0.5 (controller-made, not learned)")
    ax[1, 1].set_xlabel("training step"); ax[1, 1].set_ylabel("group success p")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3)

    fig.suptitle("Process diagnostics — why the arms don't separate (30 steps, ProRL-1.5B-v2)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130); print("wrote", a.out)


if __name__ == "__main__":
    main()
