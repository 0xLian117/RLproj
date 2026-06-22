"""Aggregate logs into figures + a markdown report.

Reads (all optional, skipped if absent):
  <results>/runs/<condition>/curriculum_log.jsonl   training dynamics
  <results>/runs/<condition>/train_summary.json     end-of-run summary
  <results>/eval/<tag>.json                          evaluation breakdowns
  <results>/sim/*.json                               CPU-simulation logs

Writes:
  <results>/figures/*.png
  <results>/REPORT.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["base", "static", "static_easy", "static_hard", "threshold", "stad", "stad_lp"]


def _order_key(tag):
    t = tag.lower().replace("(", "").replace(")", "").replace("+", "_")
    for i, o in enumerate(ORDER):
        if o in t:
            return (i, tag)
    return (len(ORDER), tag)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_train_logs(results):
    logs = {}
    for d in sorted(glob.glob(os.path.join(results, "runs", "*"))):
        p = os.path.join(d, "curriculum_log.jsonl")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            logs[os.path.basename(d)] = read_jsonl(p)
    return logs


def _x(rows):
    return [r.get("global_step", r.get("step", i)) for i, r in enumerate(rows)]


def plot_training_curves(logs, figdir):
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in sorted(logs, key=_order_key):
        rows = logs[name]
        axes[0].plot(_x(rows), [r.get("effective_ratio", 0) for r in rows], label=name)
        axes[1].plot(_x(rows), [r.get("success_rate", 0) for r in rows], label=name)
    axes[0].set_title("Effective sample ratio (informative-group fraction)")
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("effective ratio")
    axes[0].axhline(0.5, ls="--", c="grey", lw=0.8)
    axes[1].set_title("Batch success rate")
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("success rate")
    axes[1].axhline(0.5, ls="--", c="grey", lw=0.8, label="signal-max band p*=0.5")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    out = os.path.join(figdir, "training_dynamics.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_difficulty(logs, figdir):
    if not logs:
        return None
    n = len(logs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for ax, name in zip(axes[0], sorted(logs, key=_order_key)):
        rows = logs[name]
        envs = sorted({k.split("/", 1)[1] for r in rows for k in r if k.startswith("diff/")})
        for e in envs:
            ax.plot(_x(rows), [r.get(f"diff/{e}", None) for r in rows], label=e)
        ax.set_title(f"difficulty level — {name}")
        ax.set_xlabel("step"); ax.set_ylabel("difficulty d")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figdir, "difficulty_trajectories.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def load_eval(results):
    evals = {}
    for p in sorted(glob.glob(os.path.join(results, "eval", "*.json"))):
        with open(p) as f:
            evals[os.path.splitext(os.path.basename(p))[0]] = json.load(f)
    return evals


def plot_eval_bars(evals, figdir):
    if not evals:
        return None
    tags = sorted(evals, key=_order_key)
    train = [evals[t]["train_avg"] for t in tags]
    held = [evals[t]["heldout_avg"] for t in tags]
    x = range(len(tags)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.4 * len(tags) + 2, 4.2))
    ax.bar([i - w / 2 for i in x], train, w, label="train envs")
    ax.bar([i + w / 2 for i in x], held, w, label="held-out envs")
    ax.set_xticks(list(x)); ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("greedy pass@1 accuracy")
    ax.set_title("Evaluation accuracy by condition")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(train, held)):
        ax.text(i - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    out = os.path.join(figdir, "eval_accuracy.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def load_sim(results):
    sp = os.path.join(results, "sim")
    logs = {}
    for p in sorted(glob.glob(os.path.join(sp, "*.json"))):
        base = os.path.splitext(os.path.basename(p))[0]
        if base == "summary":
            continue
        with open(p) as f:
            logs[base] = json.load(f)
    summary = None
    if os.path.exists(os.path.join(sp, "summary.json")):
        with open(os.path.join(sp, "summary.json")) as f:
            summary = json.load(f)
    return logs, summary


def plot_sim(sim_logs, figdir):
    if not sim_logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in sorted(sim_logs, key=_order_key):
        h = sim_logs[name]["history"]
        xs = [r["step"] for r in h]
        axes[0].plot(xs, [r.get("mean_ability", 0) for r in h], label=name)
        axes[1].plot(xs, [r.get("effective_ratio", 0) for r in h], label=name)
    axes[0].set_title("Synthetic study: learned ability vs compute")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("mean ability")
    axes[1].set_title("Synthetic study: effective sample ratio")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("effective ratio")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(figdir, "simulation.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def write_report(results, logs, evals, sim_summary, figs):
    lines = ["# RLVE-lite results report\n"]
    if evals:
        lines.append("## Evaluation (greedy pass@1)\n")
        lines.append("| condition | train envs | held-out envs | overall |")
        lines.append("|---|---|---|---|")
        for t in sorted(evals, key=_order_key):
            e = evals[t]
            lines.append(f"| {t} | {e['train_avg']:.3f} | {e['heldout_avg']:.3f} "
                         f"| {e['overall_avg']:.3f} |")
        base = evals.get("base")
        if base:
            lines.append("\n**Improvement over base model (held-out envs):**\n")
            for t in sorted(evals, key=_order_key):
                if t == "base":
                    continue
                d = evals[t]["heldout_avg"] - base["heldout_avg"]
                lines.append(f"- {t}: {d:+.3f} absolute")
        lines.append("")
    if logs:
        lines.append("## Training dynamics (final-50-step averages)\n")
        lines.append("| condition | success rate | effective ratio |")
        lines.append("|---|---|---|")
        for name in sorted(logs, key=_order_key):
            tail = logs[name][-50:]
            sr = sum(r.get("success_rate", 0) for r in tail) / max(1, len(tail))
            er = sum(r.get("effective_ratio", 0) for r in tail) / max(1, len(tail))
            lines.append(f"| {name} | {sr:.3f} | {er:.3f} |")
        lines.append("")
    if sim_summary:
        lines.append("## CPU simulation (mechanism validation)\n")
        lines.append("| condition | mean final ability | eff-ratio (all) | "
                     "eff-ratio (last50) | success (last50) |")
        lines.append("|---|---|---|---|---|")
        for t in sorted(sim_summary, key=_order_key):
            s = sim_summary[t]
            lines.append(f"| {t} | {s['mean_final_ability']:.3f} | "
                         f"{s['eff_ratio_all']:.3f} | {s['eff_ratio_last50']:.3f} | "
                         f"{s['success_last50']:.3f} |")
        lines.append("")
    if figs:
        lines.append("## Figures\n")
        for f in figs:
            if f:
                lines.append(f"![{os.path.basename(f)}]({os.path.relpath(f, results)})")
        lines.append("")
    out = os.path.join(results, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    figdir = os.path.join(args.results, "figures")
    os.makedirs(figdir, exist_ok=True)

    logs = load_train_logs(args.results)
    evals = load_eval(args.results)
    sim_logs, sim_summary = load_sim(args.results)

    figs = [
        plot_training_curves(logs, figdir),
        plot_difficulty(logs, figdir),
        plot_eval_bars(evals, figdir),
        plot_sim(sim_logs, figdir),
    ]
    report = write_report(args.results, logs, evals, sim_summary, figs)
    print(f"Wrote figures to {figdir}/ and report to {report}")
    for f in figs:
        if f:
            print("  -", f)


if __name__ == "__main__":
    main()
