"""Plot per-environment latent-competence trajectories for the FEP/Signal arms.

The competence belief metrics (FEP/<env>/competence_mean, competence_std,
expected_difficulty) are logged to **wandb**, NOT to the worker/driver .out logs.
Export them to CSV first, then run this:

  # in the wandb UI: Runs table -> export, OR via API:
  #   import wandb; wandb.Api().run("ENTITY/PROJECT/RUN_ID").history(
  #       keys=[k for k in run.summary if k.startswith("FEP/")]
  #   ).to_csv("fep_history.csv")
  python plot_competence.py --csv fep_history.csv --out figures/competence.png

The CSV is expected to have a step column (auto-detected: 'step'/'_step'/'rollout')
and one column per logged key, e.g. 'FEP/algebra/competence_mean'. Columns are
auto-discovered by regex, so any number of environments works.
"""
from __future__ import annotations
import argparse, csv, os, re

KEY = re.compile(r"FEP/(?P<env>.+)/(?P<metric>competence_mean|competence_std|expected_difficulty)$")


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    cols = rows[0].keys()
    step_col = next((c for c in ("step", "_step", "rollout", "global_step") if c in cols), None)
    if step_col is None:
        raise SystemExit(f"no step column found in {list(cols)}")
    # env -> metric -> list[(step, value)]
    data = {}
    for r in rows:
        try:
            st = float(r[step_col])
        except (TypeError, ValueError):
            continue
        for c, v in r.items():
            m = KEY.match(c)
            if not m or v in ("", None, "NaN", "nan"):
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            data.setdefault(m["env"], {}).setdefault(m["metric"], []).append((st, val))
    if not data:
        raise SystemExit("no FEP/<env>/<metric> columns matched — check the CSV keys")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="wandb-exported history CSV")
    ap.add_argument("--out", default="figures/competence.png")
    a = ap.parse_args()
    data = load(a.csv)
    envs = sorted(data)
    print(f"environments found: {envs}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap("tab10")

    # (left) competence mean ± std band per env
    for i, e in enumerate(envs):
        mu = sorted(data[e].get("competence_mean", []))
        sd = dict(data[e].get("competence_std", []))
        if not mu:
            continue
        xs = [s for s, _ in mu]; ys = [v for _, v in mu]
        ax[0].plot(xs, ys, "-", color=cmap(i % 10), label=e)
        band = [sd.get(s) for s in xs]
        if all(b is not None for b in band):
            ax[0].fill_between(xs, [y - b for y, b in zip(ys, band)],
                               [y + b for y, b in zip(ys, band)],
                               color=cmap(i % 10), alpha=.15)
    ax[0].set_title("Latent competence posterior  μ ± σ  (per env)")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel("competence  E[s]")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    # (right) expected difficulty chosen per env
    for i, e in enumerate(envs):
        ed = sorted(data[e].get("expected_difficulty", []))
        if ed:
            ax[1].plot([s for s, _ in ed], [v for _, v in ed], "-",
                       color=cmap(i % 10), label=e)
    ax[1].set_title("Expected difficulty  E_π[d]  (per env)")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("E[d]")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    fig.suptitle("FEP-RLVE belief: does the posterior shrink (explore→exploit) and difficulty track competence?")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130); print("wrote", a.out)


if __name__ == "__main__":
    main()
