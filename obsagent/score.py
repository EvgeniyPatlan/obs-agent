"""Score one policy on simulator seeds, without re-running the baselines.

    python -m obsagent.score --seeds dev  --name x --agents v4/a.npz [--randomize] --out v4/score_x.json
    python -m obsagent.score --seeds dev  --name rules                                   (a baseline)

The volume reference (collect everything) is cached per seed set and world.
"""
import argparse
import json
import os

from . import calib
from .config import TICK_S
from .dqn import DQNAgent
from .eval_sim import SEED_BASE, evaluate, line
from .policies import DQNPolicy, baselines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=sorted(SEED_BASE), default="dev")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--name", required=True)
    ap.add_argument("--agents", default="", help="weights in results/ ('+' joins a voting group)")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    series, spr, _ = calib.load()
    noise = calib.load_noise()
    ticks = int(60 * 60 / TICK_S)
    run = lambda p: evaluate(p, args.episodes, ticks, series, spr, noise, 150.0, SEED_BASE[args.seeds],
                             randomize=args.randomize)
    world = "random" if args.randomize else "normal"
    ref_path = os.path.join(calib.RESULTS, f"full_{args.seeds}_{world}_{args.episodes}.json")
    if os.path.exists(ref_path):
        full = json.load(open(ref_path))
    else:
        full = run(baselines()[0])
        json.dump(full, open(ref_path, "w"))
    if args.agents:
        policy = DQNPolicy(net=DQNAgent.load_policy_net(os.path.join(calib.RESULTS, args.agents)), dwell_down=6)
    else:
        policy = next(p for p in baselines() if p.name == args.name)
    res = run(policy)
    res["volume"] = 0.5 * res["samples"] / full["samples"] + 0.5 * res["spans"] / full["spans"]
    print(line(f"{args.name} [{world}]", res, full), flush=True)
    path = os.path.join(calib.RESULTS, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"name": args.name, "seeds": args.seeds, "world": world, "agents": args.agents, **res},
              open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
