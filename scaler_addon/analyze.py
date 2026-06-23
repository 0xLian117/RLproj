"""Parse SCALER run logs (~/runs_out/<arm>.log or wandb output.log) into the
report's quantitative results: training dynamics + the *effective sample ratio*
that is the centrepiece of our analysis.

For each arm we extract, per training step:
  * reward mean            (critic/rewards/mean)
  * group success rate p   (= (reward_mean+1)/2, since reward in {-1,+1})
  * effective sample ratio = 1 - p^G - (1-p)^G   (informative-group probability)
  * mean difficulty        (the 'distance' values the controller proposed)
and the before/after held-out benchmark accuracies (val/... lines) if present.

The effective sample ratio operationalises the GRPO learning signal: it is the
fraction of rollout groups expected to be non-degenerate (not all-correct /
all-wrong), maximised at p=0.5. Plotting it across arms shows static difficulty
collapsing it (saturation/stall) while adaptive/free-energy keep it high.

Usage:
  python analyze.py --logs ~/runs_out --out results --G 8
Outputs:  results/metrics.csv, results/figures/*.png, results/REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

STEP_RE = re.compile(r"step:(\d+)\b")
KV_RE = re.compile(r"([A-Za-z0-9_./]+):(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
DIST_RE = re.compile(r"'distance':\s*(\d+)")
# val metric lines vary across verl versions; grab "<something>val<something>:NUM"
VAL_RE = re.compile(r"(val[^ :]*?(?:acc|score|reward|mean)[^ :]*|[A-Za-z0-9_./-]*?(?:MATH|AMC|AIME|MMLU|BBEH|GPQA)[A-Za-z0-9_./-]*)\D{0,3}(-?\d+\.\d+)", re.IGNORECASE)


def eff_ratio(p, G):
    p = min(1.0, max(0.0, p))
    return 1.0 - p ** G - (1.0 - p) ** G


def parse_log(path, G):
    steps = {}            # step -> dict(reward, eff_ratio, ...)
    val_rows = []         # (key, value) benchmark lines
    recent_dists = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            for m in DIST_RE.finditer(line):
                recent_dists.append(int(m.group(1)))
                recent_dists[:] = recent_dists[-512:]
            sm = STEP_RE.search(line)
            if sm and "critic/rewards/mean" in line:
                step = int(sm.group(1))
                kv = {k: float(v) for k, v in KV_RE.findall(line)}
                rmean = kv.get("critic/rewards/mean")
                if rmean is None:
                    continue
                p = (rmean + 1.0) / 2.0          # reward in {-1,+1} -> success prob
                row = {
                    "step": step,
                    "reward_mean": round(rmean, 4),
                    "success_rate": round(p, 4),
                    "eff_ratio": round(eff_ratio(p, G), 4),
                    "resp_len": kv.get("response_length/mean"),
                    "grad_norm": kv.get("actor/grad_norm"),
                    "mean_difficulty": round(sum(recent_dists) / len(recent_dists), 3)
                    if recent_dists else None,
                }
                steps[step] = row
            for vk, vv in VAL_RE.findall(line):
                if "val" in vk.lower() or re.search(r"MATH|AMC|AIME|MMLU|BBEH|GPQA", vk, re.I):
                    val_rows.append((vk, float(vv)))
    return [steps[k] for k in sorted(steps)], val_rows


def arm_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/runs_out"),
                    help="dir of <arm>.log files (or pass --glob)")
    ap.add_argument("--glob", default=None, help="explicit glob, e.g. '~/runs_out/*.log'")
    ap.add_argument("--out", default="results")
    ap.add_argument("--G", type=int, default=8, help="rollouts per prompt (group size)")
    args = ap.parse_args()

    pattern = os.path.expanduser(args.glob) if args.glob else os.path.join(
        os.path.expanduser(args.logs), "*.log")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if os.path.basename(f) != "all.log"]
    if not files:
        raise SystemExit(f"no logs matched {pattern}")

    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    arms = {}
    val_summary = {}
    for fp in files:
        name = arm_name(fp)
        rows, vals = parse_log(fp, args.G)
        arms[name] = rows
        val_summary[name] = vals
        print(f"{name}: {len(rows)} steps, {len(vals)} val numbers")

    # --- metrics.csv ---
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "reward_mean", "success_rate", "eff_ratio",
                    "mean_difficulty", "resp_len", "grad_norm"])
        for name, rows in arms.items():
            for r in rows:
                w.writerow([name, r["step"], r["reward_mean"], r["success_rate"],
                            r["eff_ratio"], r["mean_difficulty"], r["resp_len"],
                            r["grad_norm"]])
    print("wrote", csv_path)

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot(metric, ylabel, fname, hline=None):
            fig, ax = plt.subplots(figsize=(7, 4.3))
            for name, rows in sorted(arms.items()):
                xs = [r["step"] for r in rows if r.get(metric) is not None]
                ys = [r[metric] for r in rows if r.get(metric) is not None]
                if xs:
                    ax.plot(xs, ys, marker=".", label=name)
            if hline is not None:
                ax.axhline(hline, ls="--", c="grey", lw=0.8)
            ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " by difficulty strategy")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout(); p = os.path.join(figdir, fname)
            fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)

        plot("eff_ratio", "effective sample ratio", "effective_sample_ratio.png", hline=None)
        plot("success_rate", "group success rate", "success_rate.png", hline=0.5)
        plot("reward_mean", "mean reward", "reward.png")
        plot("mean_difficulty", "mean proposed difficulty", "difficulty.png")
    except Exception as e:  # pragma: no cover
        print("plotting skipped:", e)

    # --- REPORT.md ---
    rep = os.path.join(args.out, "REPORT.md")
    with open(rep, "w") as f:
        f.write("# SCALER difficulty-strategy results\n\n")
        f.write("## Final-window averages (last up-to-10 steps)\n\n")
        f.write("| arm | success rate | effective sample ratio | mean difficulty |\n")
        f.write("|---|---|---|---|\n")
        for name, rows in sorted(arms.items()):
            tail = rows[-10:]
            if not tail:
                f.write(f"| {name} | (no steps parsed) | | |\n"); continue
            sr = sum(r["success_rate"] for r in tail) / len(tail)
            er = sum(r["eff_ratio"] for r in tail) / len(tail)
            md = [r["mean_difficulty"] for r in tail if r["mean_difficulty"] is not None]
            md = sum(md) / len(md) if md else float("nan")
            f.write(f"| {name} | {sr:.3f} | {er:.3f} | {md:.2f} |\n")
        f.write("\n## Held-out benchmark numbers found in logs\n\n")
        for name, vals in val_summary.items():
            if vals:
                f.write(f"**{name}**: " + ", ".join(f"{k}={v}" for k, v in vals[-12:]) + "\n\n")
        f.write("\n_Effective sample ratio = 1 - p^G - (1-p)^G (G=%d): the expected "
                "fraction of informative GRPO groups; higher = more learning signal._\n" % args.G)
    print("wrote", rep)


if __name__ == "__main__":
    main()
