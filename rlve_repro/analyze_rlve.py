"""Parse RLVE worker logs (results_logs/<arm>/*.out) into the report's figures.

Each MegatronTrainRayActor worker prints, per step:
    rollout N: {'rollout/raw_reward': r, 'rollout/truncated': t, 'rollout/response_lengths': L, ...}
Binary-ish reward r∈[-1,1] → success p=(r+1)/2 → effective sample ratio
    U_K(p)=1-p^K-(1-p)^K   (informative-group probability; K = n_samples_per_prompt).

Also grabs, if present in the dir, the RolloutController lines:
    eval N: {'eval/HELD-OUT_ENVIRONMENTS_128': v, ...}
    FEP/<env>/expected_difficulty  /  RLVE/<env>/difficulty

Outputs: results/metrics.csv, results/REPORT.md, results/figures/*.png

Usage:  python analyze_rlve.py --logs results_logs --out results --K 8
"""
from __future__ import annotations
import argparse, glob, os, re, csv

STEP = re.compile(r"rollout (\d+):\s*\{([^}]*)\}")
RAW  = re.compile(r"'rollout/raw_reward':\s*(-?\d+\.?\d*(?:e-?\d+)?)")
TRUN = re.compile(r"'rollout/truncated':\s*(-?\d+\.?\d*(?:e-?\d+)?)")
RLEN = re.compile(r"'rollout/response_lengths':\s*(-?\d+\.?\d*(?:e-?\d+)?)")
EVAL = re.compile(r"eval (\d+):\s*\{[^}]*'eval/HELD-OUT_ENVIRONMENTS_128':\s*(-?\d+\.?\d*)")

ARM_ORDER = ["static", "adaptive", "signal", "fep"]
ARM_LABEL = {"static":"Static (frozen d)", "adaptive":"RLVE-90 (acc≥0.9)",
             "signal":"Signal-RLVE (λ_info=0)", "fep":"FEP-RLVE (ours)"}


def U_K(p, K):
    p = min(1.0, max(0.0, p))
    return 1.0 - p**K - (1.0-p)**K


def parse_arm(d, K):
    # pick the file with the most reward lines
    best, bestn = None, -1
    for f in glob.glob(os.path.join(d, "*.out")):
        n = sum(1 for _ in re.finditer(r"rollout \d+:", open(f, errors="ignore").read()))
        if n > bestn: best, bestn = f, n
    steps = {}
    evals = {}
    if best:
        txt = open(best, errors="ignore").read()
        for m in STEP.finditer(txt):
            i = int(m.group(1)); body = m.group(2)
            r = RAW.search(body); t = TRUN.search(body); L = RLEN.search(body)
            if r:
                p = (float(r.group(1)) + 1.0) / 2.0
                steps[i] = {"step": i, "raw_reward": float(r.group(1)),
                            "success": round(p,4), "eff_ratio": round(U_K(p,K),4),
                            "truncated": float(t.group(1)) if t else None,
                            "resp_len": float(L.group(1)) if L else None}
    # eval lines may live in a sibling controller log
    for f in glob.glob(os.path.join(d, "*.out")):
        for m in EVAL.finditer(open(f, errors="ignore").read()):
            evals[int(m.group(1))] = float(m.group(2))
    rows = [steps[k] for k in sorted(steps)]
    return rows, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results_logs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures"); os.makedirs(figdir, exist_ok=True)

    arms = {}
    evalsd = {}
    for d in sorted(glob.glob(os.path.join(args.logs, "*"))):
        if not os.path.isdir(d): continue
        name = os.path.basename(d)
        rows, evals = parse_arm(d, args.K)
        if rows: arms[name] = rows; evalsd[name] = evals
        print(f"{name}: {len(rows)} steps, {len(evals)} eval points")

    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]

    # metrics.csv
    with open(os.path.join(args.out,"metrics.csv"),"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["arm","step","raw_reward","success","eff_ratio","truncated","resp_len"])
        for a in order:
            for r in arms[a]:
                w.writerow([a,r["step"],r["raw_reward"],r["success"],r["eff_ratio"],r["truncated"],r["resp_len"]])

    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    COL={"static":"#888","adaptive":"#2c6e8f","signal":"#e0a030","fep":"#c0392b"}
    def plot(metric,ylabel,fname,hline=None):
        fig,ax=plt.subplots(figsize=(7,4.3))
        for a in order:
            xs=[r["step"] for r in arms[a] if r.get(metric) is not None]
            ys=[r[metric] for r in arms[a] if r.get(metric) is not None]
            if xs: ax.plot(xs,ys,marker=".",color=COL.get(a,"#444"),label=ARM_LABEL.get(a,a))
        if hline is not None: ax.axhline(hline,ls="--",c="grey",lw=.8)
        ax.set_xlabel("training step (rollout)"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel+" — RLVE-90 vs Signal-RLVE vs FEP-RLVE")
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        p=os.path.join(figdir,fname); fig.savefig(p,dpi=130); plt.close(fig); print("wrote",p)
    plot("eff_ratio","effective sample ratio  U_K(p)","effective_sample_ratio.png")
    plot("success","group success rate  p","success_rate.png",hline=0.5)
    plot("raw_reward","mean group reward","reward.png",hline=0.0)
    if any(any(r.get("truncated") is not None for r in arms[a]) for a in order):
        plot("truncated","truncated ratio","truncated.png")

    # REPORT.md
    with open(os.path.join(args.out,"REPORT.md"),"w") as f:
        f.write("# RLVE difficulty-strategy results (ProRL-1.5B-v2, 4 envs)\n\n")
        f.write("| arm | steps | mean success | mean eff. ratio | last-5 success | held-out (final) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for a in order:
            rs=arms[a]; ms=sum(r["success"] for r in rs)/len(rs); me=sum(r["eff_ratio"] for r in rs)/len(rs)
            l5=rs[-5:]; l5s=sum(r["success"] for r in l5)/len(l5)
            ho=evalsd[a].get(max(evalsd[a]),None) if evalsd[a] else None
            f.write(f"| {ARM_LABEL.get(a,a)} | {len(rs)} | {ms:.3f} | {me:.3f} | {l5s:.3f} | {ho if ho is not None else '—'} |\n")
        f.write("\n_eff. ratio = 1 − p^K − (1−p)^K (K=%d): expected fraction of informative GRPO groups._\n"%args.K)
    print("wrote", os.path.join(args.out,"REPORT.md"))


if __name__=="__main__":
    main()
