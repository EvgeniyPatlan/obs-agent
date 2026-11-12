"""Run every policy on the real bench with the same load and fault schedule.

    python -m obsagent.run_bench --minutes 20

Writes results/bench_<policy>.csv (per tick, per service) and
results/bench_summary.json.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from . import calib
from .bench_env import BenchEnv
from .config import TICK_S
from .dqn import DQNAgent
from .policies import DQNPolicy, baselines
from .runner import run_episode, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--policies", default="static-full,static-lean,rules,dqn-guarded")
    ap.add_argument("--fault-gap", type=float, default=240.0, help="mean seconds between faults")
    ap.add_argument("--agent", default="agent.npz", help="weight file in results/")
    ap.add_argument("--out-dir", default="", help="sub-folder of results/; each policy then gets its own summary file")
    args = ap.parse_args()
    out_dir = os.path.join(calib.RESULTS, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    series, spr, _ = calib.load()
    net = DQNAgent.load_policy_net(os.path.join(calib.RESULTS, args.agent))
    pool = {p.name: p for p in baselines() + [DQNPolicy(net=net), DQNPolicy(net=net, dwell_down=6)]}
    ticks = int(args.minutes * 60 / TICK_S)
    summary_path = os.path.join(out_dir, "bench_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) and not args.out_dir else {}
    for name in args.policies.split(","):
        if "=" in name:   # "label=weights" runs a named agent, e.g. agent-lean=agents/a.npz+agents/i.npz
            name, weights = name.split("=", 1)
            # "...@0.15": take a random action 15% of the time, to collect varied experience
            weights, _, eps = weights.partition("@")
            policy = DQNPolicy(net=DQNAgent.load_policy_net(os.path.join(calib.RESULTS, weights)), dwell_down=6,
                               epsilon=float(eps or 0), rng=np.random.default_rng(args.seed))
            policy.name = name
        else:
            policy = pool[name]
        print(f"=== {name}: {args.minutes:.0f} min, seed {args.seed}", flush=True)
        env = BenchEnv(seed=args.seed, duration_s=ticks * TICK_S, fault_gap_s=args.fault_gap)
        env.start()

        def progress(tick, recs):
            if tick % 24 == 0:
                lv = " ".join(f"{r['service']}=L{r['level']}" for r in recs)
                inc = [r["service"] for r in recs if r["incident"]]
                print(f"  t={tick * TICK_S:5d}s {lv} incident={inc or '-'}", flush=True)

        t0 = time.time()
        records, transitions = run_episode(env, policy, ticks, series, spr, on_tick=progress, infer_actions=True)
        env.stop()
        totals = env.run_totals()
        s = summarize(records)
        s["prometheus_samples_total"] = sum(v["samples"] for v in totals.values())
        s["spans_total"] = sum(v["spans"] for v in totals.values())
        s["span_bytes_total"] = sum(v["span_bytes"] for v in totals.values())
        s["wall_s"] = time.time() - t0
        s.update(seed=args.seed, minutes=args.minutes, fault_gap_s=args.fault_gap, agent=args.agent)
        summary[name] = s
        with open(os.path.join(out_dir, f"bench_{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        # Real experience, for fine-tuning an agent on the bench later (obsagent.finetune).
        if transitions:
            S, A, R, S2, D = (np.array(x) for x in zip(*transitions))
            np.savez_compressed(os.path.join(out_dir, f"transitions_{name}.npz"), s=S, a=A, r=R, s2=S2, done=D)
        # Parallel runs share out_dir, so each policy writes its own file there.
        path = os.path.join(out_dir, f"summary_{name}.json") if args.out_dir else summary_path
        with open(path, "w") as f:
            json.dump({name: s} if args.out_dir else summary, f, indent=2)
        print(f"  samples={s['prometheus_samples_total']:,.0f} spans={s['spans_total']:,} "
              f"covered={s['incidents_covered']}/{s['incidents']} ready@start={s['incidents_ready_at_start']} "
              f"detect={s['mean_detect_s']:.1f}s changes/h={s['changes_per_service_hour']:.1f}", flush=True)


if __name__ == "__main__":
    main()
