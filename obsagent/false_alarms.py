"""Where do false alarms come from? Compare calm readings on the bench and
in the simulator, level by level.

    python -m obsagent.false_alarms results/bench_v2/5201/bench_agent-balanced.csv ...
    python -m obsagent.false_alarms --sim agent-balanced=v2/a.npz+...

For calm ticks (no incident or fault within 60 s) it reports, per level:
  over   share of fresh readings over the SLA although the truth is under it
  again  chance the next fresh reading is over too (are spikes sticky?)
  run    mean length of a run of over-SLA fresh readings
and for the policy: rises to L2+ per service-hour with no incident within 60 s.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from .config import SLA_P99_S

SETTLE = 12


def by_service(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["service"]].append(r)
    for rs in out.values():
        rs.sort(key=lambda r: int(r["tick"]))
    return out


def calm_mask(rs):
    busy = [r["incident"] in ("True", True) or bool(r["fault"]) for r in rs]
    n = len(rs)
    calm = [True] * n
    for i, b in enumerate(busy):
        if b:
            for j in range(max(0, i - SETTLE), min(n, i + SETTLE + 1)):
                calm[j] = False
    return calm


def stats(rows):
    per_level = defaultdict(lambda: {"fresh": 0, "over": 0, "pairs": 0, "again": 0, "runs": []})
    false_rises, hours = 0, 0.0
    for svc, rs in by_service(rows).items():
        calm = calm_mask(rs)
        hours += len(rs) * 5 / 3600
        prev_obs, prev_over, run = None, {}, 0
        for i, r in enumerate(rs):
            lv = int(r["level"])
            obs, true = float(r["obs_p99"]), float(r["true_p99"])
            if i and lv >= 2 and int(rs[i - 1]["level"]) < 2 and calm[i]:
                false_rises += 1
            fresh = obs != prev_obs
            prev_obs = obs
            if not (fresh and calm[i]) or true >= SLA_P99_S:
                continue
            d = per_level[lv]
            over = obs > SLA_P99_S
            d["fresh"] += 1
            d["over"] += over
            if lv in prev_over:
                d["pairs"] += prev_over[lv]
                d["again"] += prev_over[lv] and over
            if over:
                run += 1
            elif run:
                d["runs"].append(run)
                run = 0
            prev_over = {lv: over}
    return per_level, false_rises / max(hours, 1e-9)


def report(name, rows):
    per_level, fr = stats(rows)
    print(f"== {name}: false rises to L2+ = {fr:.1f} per service-hour")
    for lv in sorted(per_level):
        d = per_level[lv]
        if not d["fresh"]:
            continue
        again = d["again"] / d["pairs"] if d["pairs"] else float("nan")
        run = np.mean(d["runs"]) if d["runs"] else 0
        print(f"   L{lv}: fresh={d['fresh']:6d} over={d['over'] / d['fresh']:6.2%} "
              f"again={again:5.0%} run={run:.2f}")


def sim_rows(spec, episodes, minutes):
    from . import calib
    from .config import TICK_S
    from .dqn import DQNAgent
    from .policies import DQNPolicy, baselines
    from .runner import run_episode
    from .sim import SimEnv
    series, spr, _ = calib.load()
    noise = calib.load_noise()
    label, _, fname = spec.partition("=")
    if fname:
        pol = DQNPolicy(net=DQNAgent.load_policy_net(os.path.join(calib.RESULTS, fname)), dwell_down=6)
    else:
        pol = next(p for p in baselines() if p.name == label)
    ticks = int(minutes * 60 / TICK_S)
    rows = []
    for ep in range(episodes):
        env = SimEnv(series=series, spans_per_req=spr, noise=noise, fault_gap_s=150.0)
        env.reset(seed=8_000_000 + ep, duration_s=ticks * TICK_S)
        recs, _ = run_episode(env, pol, ticks, series, spr)
        for rec in recs:
            rows.append({**rec, "incident": str(rec["incident"]), "service": f"{ep}:{rec['service']}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*")
    ap.add_argument("--sim", default=None, help="policy name, or label=weights")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args()
    if args.csv:
        rows = []
        for i, p in enumerate(args.csv):
            for r in csv.DictReader(open(p)):
                r["service"] = f"{i}:{r['service']}"
                rows.append(r)
        report("bench", rows)
    if args.sim:
        report("sim", sim_rows(args.sim, args.episodes, args.minutes))


if __name__ == "__main__":
    main()
