"""Train the DQN telemetry agent from scratch in the simulator.

    python -m obsagent.train --episodes 200

Writes results/agent.npz (the learned weights) and results/training.csv.
"""
import argparse
import csv
import os
import time

import numpy as np

from . import calib
from .agent_core import STATE_DIM, RewardWeights
from .dqn import DQNAgent
from .policies import DQNPolicy
from .runner import run_episode, summarize
from .sim import SimEnv
from .config import TICK_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=60, help="simulated minutes per episode")
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-min", type=float, default=0.02)
    ap.add_argument("--eps-decay", type=float, default=0.98)
    ap.add_argument("--learn-every", type=int, default=1, help="gradient steps per transition batch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-noise", action="store_true", help="train without the measured reading errors")
    ap.add_argument("--no-benign", action="store_true", help="train without harmless spikes and scrape gaps")
    ap.add_argument("--fault-gap", default="120,300", help="mean seconds between faults, or a low,high range")
    ap.add_argument("--out", default="agent.npz")
    ap.add_argument("--double", action="store_true", help="Double DQN targets (less overestimation)")
    ap.add_argument("--randomize", action="store_true", help="a slightly different simulated world each episode")
    ap.add_argument("--floor", type=int, default=0, help="lowest level the agent may use (1 = never below lean)")
    for name, val in vars(RewardWeights()).items():
        ap.add_argument(f"--w-{name}", type=type(val), default=val)
    args = ap.parse_args()
    weights = RewardWeights(**{k: getattr(args, f"w_{k}") for k in vars(RewardWeights())})
    gap = tuple(float(x) for x in args.fault_gap.split(","))
    gap = gap[0] if len(gap) == 1 else gap
    noise = None if args.no_noise else calib.load_noise()

    series, spans_per_req, calibrated = calib.load()
    print(f"calibration: {'bench' if calibrated else 'defaults'} series={series} spans/req={spans_per_req:.2f}")
    print(f"reading errors: {'measured on the bench' if noise else 'none'}; "
          f"harmless events: {'off' if args.no_benign else 'on'}; fault gap {gap}; {weights}")
    env = SimEnv(series=series, spans_per_req=spans_per_req, noise=noise, fault_gap_s=gap,
                 benign=not args.no_benign, randomize=args.randomize)
    agent = DQNAgent(STATE_DIM, seed=args.seed, floor=args.floor, double=args.double)
    ticks = int(args.minutes * 60 / TICK_S)
    os.makedirs(calib.RESULTS, exist_ok=True)
    log_path = os.path.join(calib.RESULTS, args.out.replace(".npz", "") + "_training.csv"
                            if args.out != "agent.npz" else "training.csv")
    best = -1e9
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "epsilon", "mean_reward", "loss", "incident_tick_coverage", "samples", "spans"])
        eps = args.eps_start
        for ep in range(args.episodes):
            env.reset(seed=args.seed * 100_000 + ep, duration_s=ticks * TICK_S)
            policy = DQNPolicy(agent=agent, epsilon=eps)
            records, transitions = run_episode(env, policy, ticks, series, spans_per_req, weights=weights)
            losses = []
            for tr in transitions:
                agent.mem.add(*tr)
                for _ in range(args.learn_every):
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
            s = summarize(records)
            w.writerow([ep, round(eps, 4), round(s["mean_reward"], 4),
                        round(float(np.mean(losses)), 5) if losses else "",
                        round(s["incident_tick_coverage"], 4), round(s["samples"]), round(s["spans"])])
            f.flush()
            if ep >= args.episodes // 2 and s["mean_reward"] > best:
                best = s["mean_reward"]
            if ep % 10 == 0 or ep == args.episodes - 1:
                print(f"ep {ep:4d} eps={eps:.3f} reward={s['mean_reward']:+.3f} "
                      f"coverage={s['incident_tick_coverage']:.2f} ({time.time() - t0:.0f}s)", flush=True)
            eps = max(args.eps_min, eps * args.eps_decay)
    agent.save(os.path.join(calib.RESULTS, args.out))
    print(f"saved {calib.RESULTS}/{args.out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
