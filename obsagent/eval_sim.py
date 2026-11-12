"""Compare trained agents with the baselines on simulated hours the agent
never trained on.

    python -m obsagent.eval_sim --seeds dev  --agents agent.npz,try2.npz   # for tuning
    python -m obsagent.eval_sim --seeds test --agents agent.npz            # final, once

Two disjoint seed sets keep the comparison honest: tune only on "dev",
and report "test", which is not looked at while tuning.
"""
import argparse
import json
import os

import numpy as np

from . import calib
from .config import TICK_S
from .dqn import DQNAgent
from .policies import DQNPolicy, baselines
from .runner import run_episode, summarize
from .sim import SimEnv

SEED_BASE = {"dev": 8_000_000, "test": 9_000_000}
SUM_KEYS = ("samples", "spans", "incidents")


def aggregate(rows):
    return {k: float(np.nansum([r[k] for r in rows])) if k.startswith(SUM_KEYS)
            else float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def evaluate(policy, episodes, ticks, series, spr, noise, gap, seed_base, benign=True, randomize=False):
    rows = []
    for ep in range(episodes):
        env = SimEnv(series=series, spans_per_req=spr, noise=noise, fault_gap_s=gap, benign=benign, randomize=randomize)
        env.reset(seed=seed_base + ep, duration_s=ticks * TICK_S)
        recs, _ = run_episode(env, policy, ticks, series, spr)
        rows.append(summarize(recs))
    return aggregate(rows)


def line(name, agg, full):
    vol = 0.5 * agg["samples"] / full["samples"] + 0.5 * agg["spans"] / full["spans"]
    return (f"{name:22s} volume={vol:6.2%} covered={agg['incidents_covered']:.0f}/{agg['incidents']:.0f} "
            f"ready@start={agg['incidents_ready_at_start']:.0f} detect={agg['mean_detect_s']:.1f}s "
            f"to-diag={agg['mean_time_to_diagnostic_s']:.1f}s changes/h={agg['changes_per_service_hour']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seeds", choices=sorted(SEED_BASE), default="dev")
    ap.add_argument("--agents", default="agent.npz", help="comma-separated weight files in results/")
    ap.add_argument("--fault-gap", type=float, default=150.0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--randomize", action="store_true", help="robustness check: shifted simulated worlds")
    ap.add_argument("--out", default=None, help="json file in results/ (default sim_eval_<seeds>.json)")
    args = ap.parse_args()
    series, spr, _ = calib.load()
    noise = None if args.no_noise else calib.load_noise()
    ticks = int(args.minutes * 60 / TICK_S)
    policies = baselines()
    for entry in filter(None, args.agents.split(",")):
        if "=" in entry:   # "label=weights": one named agent with the guard rail
            label, fname = entry.split("=", 1)
            p = DQNPolicy(net=DQNAgent.load_policy_net(os.path.join(calib.RESULTS, fname)), dwell_down=6)
            p.name = label
            policies.append(p)
            continue
        net = DQNAgent.load_policy_net(os.path.join(calib.RESULTS, entry))
        label = "" if entry == "agent.npz" else f"[{entry.removesuffix('.npz')}]"
        for p in (DQNPolicy(net=net), DQNPolicy(net=net, dwell_down=6)):
            p.name = p.name + label
            policies.append(p)
    out = {}
    for p in policies:
        out[p.name] = evaluate(p, args.episodes, ticks, series, spr, noise, args.fault_gap, SEED_BASE[args.seeds],
                               benign=not args.no_benign, randomize=args.randomize)
        print(line(p.name, out[p.name], out["static-full"]), flush=True)
    path = os.path.join(calib.RESULTS, args.out or f"sim_eval_{args.seeds}.json")
    with open(path, "w") as f:
        json.dump({"episodes": args.episodes, "minutes": args.minutes, "seeds": args.seeds,
                   "fault_gap_s": args.fault_gap, "noise": noise is not None,
                   "benign": not args.no_benign, "randomize": args.randomize, "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
