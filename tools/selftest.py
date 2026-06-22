"""CPU-only self-test: validates environments, verifiers, controllers, samplers
and the data/reward integration WITHOUT a model or GPU. Run before any remote
job to confirm the non-GPU half of the pipeline is correct.

    python tools/selftest.py     # exits non-zero on any failure
"""
from __future__ import annotations

import random
import sys

from rlve.curriculum import Curriculum
from rlve.data import AdaptiveDataset
from rlve.difficulty import (LearningProgressSampler, STADController,
                             StaticController, ThresholdBumpController,
                             UniformSampler)
from rlve.envs.registry import make_heldout_envs, make_train_envs
from rlve.reward import CurriculumReward
from rlve.stats import GroupOutcome

FAILS = []


def check(cond, msg):
    if cond:
        print(f"  ok  {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)


def test_envs():
    print("[1] environments: generation + verification")
    envs = {**make_train_envs(), **make_heldout_envs()}
    rng = random.Random(0)
    for name, env in envs.items():
        ok_self = ok_reject = True
        for d in range(env.min_difficulty, env.max_difficulty + 1):
            p = env.generate(d, rng)
            if not env.verify(f"...\\boxed{{{p.answer}}}", p).correct:
                ok_self = False
            wrong = "\\boxed{-987654321}"
            if env.verify(wrong, p).correct:
                ok_reject = False
        check(ok_self, f"{name}: gold answer verifies correct at all difficulties")
        check(ok_reject, f"{name}: clearly wrong answer is rejected")


def test_static():
    print("[2] StaticController stays fixed")
    env = make_train_envs()["arithmetic"]
    c = StaticController(env, level=5)
    rng = random.Random(1)
    levels = {c.sample_difficulty(rng) for _ in range(50)}
    check(levels == {5}, f"static level constant: {levels}")


def test_threshold():
    print("[3] ThresholdBumpController only increases, respects window")
    env = make_train_envs()["arithmetic"]
    c = ThresholdBumpController(env, l=0, tau_acc=0.9, tau_num=16, d_delta=4)
    hs = [c.h]
    # feed all-correct groups at the upper bound -> h should ramp up
    for _ in range(40):
        c.update([GroupOutcome("arithmetic", c.h, 8, 8) for _ in range(4)])
        hs.append(c.h)
    check(all(b >= a for a, b in zip(hs, hs[1:])), "h is monotonically non-decreasing")
    check(c.h > 0, f"h increased under high accuracy (h={c.h})")
    check(c.h - c.l + 1 <= 4, f"window respected (l={c.l}, h={c.h})")
    # now feed all-wrong -> h must NOT decrease (one-directional)
    h_before = c.h
    for _ in range(10):
        c.update([GroupOutcome("arithmetic", c.h, 8, 0) for _ in range(4)])
    check(c.h == h_before, "h does not decrease when accuracy drops (one-directional)")


def test_stad():
    print("[4] STADController moves toward p* from both sides")
    env = make_train_envs()["arithmetic"]
    # too-easy signal (success=1.0) -> difficulty should rise
    c = STADController(env, init=4, p_star=0.5, kp=1.5)
    for _ in range(30):
        c.update([GroupOutcome("arithmetic", int(round(c.mu)), 8, 8) for _ in range(4)])
    check(c.mu > 4, f"difficulty rises when too easy (mu={c.mu:.2f})")
    # too-hard signal (success=0.0) -> difficulty should fall
    c2 = STADController(env, init=8, p_star=0.5, kp=1.5)
    for _ in range(30):
        c2.update([GroupOutcome("arithmetic", int(round(c2.mu)), 8, 0) for _ in range(4)])
    check(c2.mu < 8, f"difficulty falls when too hard (mu={c2.mu:.2f})")
    # at target band success~0.5 -> difficulty roughly stable
    c3 = STADController(env, init=5, p_star=0.5, kp=1.5)
    for _ in range(40):
        c3.update([GroupOutcome("arithmetic", int(round(c3.mu)), 8, 4) for _ in range(4)])
    check(abs(c3.mu - 5) < 1.5, f"difficulty stable near target band (mu={c3.mu:.2f})")


def test_sampler():
    print("[5] LearningProgressSampler shifts mass toward high-signal envs")
    names = ["a", "b", "c"]
    s = LearningProgressSampler(names, temp=0.3, eps=0.1)
    for _ in range(20):
        s.update({"a": 1.0, "b": 0.0, "c": 0.0})  # only 'a' is informative
    w = s.weights()
    check(w["a"] > w["b"] and w["a"] > w["c"], f"high-signal env up-weighted: {w}")
    check(min(w.values()) > 0, "exploration floor keeps all envs sampled")
    u = UniformSampler(names)
    check(abs(u.weights()["a"] - 1/3) < 1e-9, "uniform sampler is uniform")


def test_integration():
    print("[6] data + reward + curriculum integration (TRL access pattern)")
    G, P, STEPS = 8, 8, 15
    curr = Curriculum(controller_kind="stad", sampler_kind="lp")
    ds = AdaptiveDataset(curr, seed=2)
    reward = CurriculumReward(curr)
    rng = random.Random(0)
    eff_first = eff_last = succ_last = None
    stable = True
    for step in range(STEPS):
        ds.new_round()
        rows = []
        for i in range(P):
            first = ds[i]
            for _ in range(G):
                if ds[i]["uid"] != first["uid"]:
                    stable = False
                rows.append(ds[i])
        comps = []
        for r in rows:
            pc = max(0.05, 1.0 - 0.09 * r["difficulty"])
            body = r["answer"] if rng.random() < pc else "0"
            comps.append([{"role": "assistant", "content": f"x \\boxed{{{body}}}"}])
        cols = {k: [r[k] for r in rows] for k in ["env", "answer", "difficulty", "uid"]}
        rew = reward(prompts=[r["prompt"] for r in rows], completions=comps, **cols)
        if len(rew) != P * G:
            stable = False
        m = curr.step_update()
        if step == 0:
            eff_first = m["effective_ratio"]
        eff_last, succ_last = m["effective_ratio"], m["success_rate"]
    check(stable, "group prompts stable across num_generations repeats; reward length correct")
    check(eff_last >= eff_first, f"effective ratio improves under adaptation "
          f"({eff_first:.2f} -> {eff_last:.2f})")
    check(0.2 < succ_last < 0.95, f"success driven into informative band "
          f"(final={succ_last:.2f})")


def main():
    for t in (test_envs, test_static, test_threshold, test_stad,
              test_sampler, test_integration):
        t()
    print()
    if FAILS:
        print(f"SELFTEST FAILED: {len(FAILS)} check(s) failed")
        return 1
    print("SELFTEST PASSED: all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
