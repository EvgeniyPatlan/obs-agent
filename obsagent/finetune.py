"""Fine-tune trained agents on real-bench experience.

    python -m obsagent.finetune --bench bench_v3b --src v3 --out v5

Every bench run saves its decisions (transitions_<policy>.npz, written by
run_bench) next to its per-tick CSV. For each agent in results/<src>/ this:

1. recomputes the bench rewards with that agent's own reward weights
   (from tune.sh), using the CSV;
2. records fresh simulator experience with the agent itself (a little
   exploration), so it does not forget what the simulator taught it;
3. continues Q-learning from the trained weights on a mix of both, with a
   small learning rate and Double DQN targets.

Real bench experience is the only data that shows the bench's own reading
errors and timing. The simulator part keeps the rest of the behaviour.
"""
import argparse
import csv
import glob
import os
import re
import time

import numpy as np

from . import calib
from .agent_core import STATE_DIM, RewardWeights, reward
from .config import SERVICES, TICK_S
from .dqn import DQNAgent, ReplayBuffer
from .nn import Adam
from .policies import DQNPolicy
from .runner import PRE_INCIDENT_TICKS, run_episode
from .sim import SimEnv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def variant_weights(path=os.path.join(HERE, "tune.sh")):
    """Reward weights of each variant, read from tune.sh so there is one source."""
    out = {}
    for name, flags in re.findall(r'\[(\w+)\]="([^"]*)"', open(path).read()):
        kw = {k.replace("-", "_"): float(v) for k, v in re.findall(r"--w-([\w-]+) (\S+)", flags)}
        out[name] = RewardWeights(**kw)
    return out


def _bool(x):
    return x in ("True", "true", "1")


def bench_rewards(csv_path, w):
    """Reward per (tick, service) recomputed with weights w. Also returns the
    default-weight reward saved in the CSV, to check the alignment."""
    rows = list(csv.DictReader(open(csv_path)))
    by = {(r["service"], int(r["tick"])): r for r in rows}
    ticks = max(int(r["tick"]) for r in rows) + 1
    new, saved = [], []
    since = {s: 99 for s in SERVICES}
    for t in range(ticks):
        for s in SERVICES:
            r = by[(s, t)]
            inc = _bool(r["incident"])
            pre = not inc and any(_bool(by[(s, t + k)]["incident"])
                                  for k in range(1, PRE_INCIDENT_TICKS + 1) if (s, t + k) in by)
            ch = _bool(r["changed"])
            new.append(reward(int(r["level"]), float(r["cost_frac"]), inc, pre, ch, since[s], w))
            saved.append(float(r["reward"]))
            since[s] = 0 if ch else since[s] + 1
    return np.array(new), np.array(saved)


def load_bench(folders, w, skip=("static-",)):
    """All bench transitions, rewards recomputed with w. Files whose rows do
    not line up with their CSV are skipped (and reported)."""
    parts = []
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(calib.RESULTS, folder, "*", "transitions_*.npz"))):
            name = os.path.basename(f)[len("transitions_"):-4]
            if name.startswith(skip):
                continue
            d = np.load(f)
            new, saved = bench_rewards(os.path.join(os.path.dirname(f), f"bench_{name}.csv"), w)
            n = len(d["r"])
            if n > len(saved) or not np.allclose(d["r"], saved[:n], atol=1e-6):
                print(f"  skip {f}: rows do not match the CSV")
                continue
            parts.append((d["s"], d["a"], new[:n], d["s2"], d["done"]))
    if not parts:
        raise SystemExit(f"no bench transitions found in {folders}")
    return [np.concatenate(x) for x in zip(*parts)]


def sim_experience(net, floor, w, episodes, seed, series, spr, noise, eps=0.1):
    """Fresh simulator transitions from the agent itself, with a little exploration."""
    pol = DQNPolicy(net=net, epsilon=eps, dwell_down=6, floor=floor, rng=np.random.default_rng(seed))
    out = []
    for ep in range(episodes):
        env = SimEnv(series=series, spans_per_req=spr, noise=noise, fault_gap_s=(120.0, 300.0))
        env.reset(seed=7_000_000 + seed * 1000 + ep, duration_s=3600)
        _, tr = run_episode(env, pol, int(3600 / TICK_S), series, spr, weights=w)
        out += tr
    return [np.array(x, dtype=float) for x in zip(*out)]


def fill(buf, data):
    for row in zip(*data):
        buf.add(*row)


def finetune_one(src, dst, w, bench, args, series, spr, noise, seed):
    agent = DQNAgent(STATE_DIM, lr=args.lr, buffer=1, double=True, seed=seed)
    net = DQNAgent.load_policy_net(src)
    agent.floor = net.floor
    agent.q.copy_from(net)
    agent.target.copy_from(net)
    agent.opt = Adam(agent.q, lr=args.lr)
    sim = sim_experience(agent.q, agent.floor, w, args.sim_episodes, seed, series, spr, noise)
    bufs = []
    for data in (bench, sim):
        b = ReplayBuffer(len(data[0]), STATE_DIM, seed=seed)
        fill(b, data)
        bufs.append(b)
    nb = int(round(agent.batch * args.bench_share))
    losses = []
    for step in range(args.steps):
        parts = [bufs[0].sample(nb), bufs[1].sample(agent.batch - nb)]
        batch = [np.concatenate(x) for x in zip(*parts)]
        losses.append(agent.learn(batch))
    agent.save(dst)
    return len(bench[0]), len(sim[0]), float(np.mean(losses[:500])), float(np.mean(losses[-500:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench_v3b", help="comma-separated folders in results/")
    ap.add_argument("--src", default="v3", help="folder with the trained agents")
    ap.add_argument("--out", default="v5")
    ap.add_argument("--only", default="", help="comma-separated variants (default: all in tune.sh)")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bench-share", type=float, default=0.5, help="share of each minibatch from the bench")
    ap.add_argument("--sim-episodes", type=int, default=20)
    args = ap.parse_args()
    series, spr, _ = calib.load()
    noise = calib.load_noise()
    os.makedirs(os.path.join(calib.RESULTS, args.out), exist_ok=True)
    variants = variant_weights()
    names = args.only.split(",") if args.only else list(variants)
    for i, v in enumerate(names):
        t0 = time.time()
        bench = load_bench(args.bench.split(","), variants[v])
        nb, ns, l0, l1 = finetune_one(os.path.join(calib.RESULTS, args.src, f"{v}.npz"),
                                      os.path.join(calib.RESULTS, args.out, f"{v}.npz"),
                                      variants[v], bench, args, series, spr, noise, seed=100 + i)
        print(f"{v}: bench={nb} sim={ns} loss {l0:.4f} -> {l1:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
