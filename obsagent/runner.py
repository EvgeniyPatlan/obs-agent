"""The control loop: observe -> decide -> apply -> wait one tick.

Works with both environments (SimEnv and BenchEnv) because they share the
same three methods: observe(), apply(levels), advance().
"""
from .agent_core import Memory, RewardWeights, build_state, cost_fraction, reward, update_memory
from .config import DIAGNOSABLE, SERVICES, SLA_ERR, SLA_P99_S, TICK_S
from .dqn import ACTIONS

PRE_INCIDENT_TICKS = 6  # 30 s before an incident starts


def run_episode(env, policy, ticks, series, spans_per_req, on_tick=None, weights=RewardWeights(),
                infer_actions=False):
    """Run one episode. Returns per-tick records and (for training) the
    transitions with rewards filled in after the fact. With infer_actions,
    policies that pick a level directly (rules, static) get their action
    recovered from the level change, so real-bench runs of any policy can
    later be used as training data."""
    policy.reset()
    mem = {s: Memory() for s in SERVICES}
    env.apply({s: mem[s].level for s in SERVICES})
    records, steps = [], []
    for tick in range(ticks):
        obs = env.observe()
        worst_p99 = max(o["p99"] for o in obs.values())
        worst_err = max(o["err"] for o in obs.values())
        states, decided = {}, {}
        for s in SERVICES:
            states[s] = build_state(obs[s], mem[s], worst_p99, worst_err)
            decided[s] = policy.decide(s, states[s], obs[s], mem[s])
        changed, since_before = {}, {}
        for s in SERVICES:
            since_before[s] = mem[s].since_change
            before = mem[s].level
            changed[s] = update_memory(mem[s], obs[s], decided[s][0], tick)
            delta = mem[s].level - before
            if infer_actions and decided[s][1] is None and abs(delta) <= 2:
                decided[s] = (decided[s][0], ACTIONS.index(delta))
        env.apply({s: mem[s].level for s in SERVICES})
        info = env.advance()
        for s in SERVICES:
            rec = {"tick": tick, "t": tick * TICK_S, "service": s, "level": mem[s].level,
                   "changed": changed[s], "obs_p99": obs[s]["p99"], "obs_err": obs[s]["err"],
                   "staleness": obs[s]["staleness"], **info[s]}
            records.append(rec)
            steps.append((s, tick, states[s], decided[s][1], changed[s], since_before[s]))
        if on_tick:
            on_tick(tick, records[-len(SERVICES):])

    # Rewards need hindsight ("was an incident about to start?"), so they are
    # computed once the episode is over. The state never sees this.
    incident = {(r["service"], r["tick"]): r["incident"] for r in records}
    by_key = {(r["service"], r["tick"]): r for r in records}
    transitions = []
    for i, (s, tick, state, action, ch, since) in enumerate(steps):
        rec = by_key[(s, tick)]
        pre = any(incident.get((s, tick + k), False) for k in range(1, PRE_INCIDENT_TICKS + 1)) and not rec["incident"]
        cf = cost_fraction(rec["level"], rec["rps"], series, spans_per_req)
        rec["reward"] = reward(rec["level"], cf, rec["incident"], pre, ch, since, weights)
        rec["cost_frac"] = cf
        nxt = i + len(SERVICES)
        if action is not None and nxt < len(steps):
            transitions.append((state, action, rec["reward"], steps[nxt][2], 0.0))
    return records, transitions


def summarize(records):
    """Headline numbers for one run (one policy)."""
    n = len(records)
    inc = [r for r in records if r["incident"]]
    samples = sum(r["samples"] for r in records)
    spans = sum(r["spans"] for r in records)
    # group consecutive incident ticks per service into incidents
    episodes, open_ = [], {}
    for r in sorted(records, key=lambda r: (r["service"], r["tick"])):
        key = r["service"]
        if r["incident"]:
            if key not in open_:
                open_[key] = {"service": key, "start": r["tick"], "ticks": [], "detected": None}
            e = open_[key]
            e["ticks"].append(r)
            if e["detected"] is None and (r["obs_p99"] > SLA_P99_S or r["obs_err"] > SLA_ERR):
                e["detected"] = r["tick"] - e["start"]
        elif key in open_:
            episodes.append(open_.pop(key))
    episodes.extend(open_.values())
    real = [e for e in episodes if len(e["ticks"]) >= 3]   # ignore 1-2 tick blips
    covered = [e for e in real if any(t["level"] >= DIAGNOSABLE for t in e["ticks"])]
    ready_at_start = [e for e in real if e["ticks"][0]["level"] >= DIAGNOSABLE]
    detect = [e["detected"] * TICK_S for e in real if e["detected"] is not None]
    to_diag = []
    for e in real:
        lvls = [t["level"] for t in e["ticks"]]
        hit = next((i for i, lv in enumerate(lvls) if lv >= DIAGNOSABLE), None)
        if hit is not None:
            to_diag.append(hit * TICK_S)
    hours = n / len({r["service"] for r in records}) * TICK_S / 3600
    return {
        "samples": samples, "spans": spans,
        "incident_ticks": len(inc),
        "incident_tick_coverage": (sum(r["level"] >= DIAGNOSABLE for r in inc) / len(inc)) if inc else 1.0,
        "incidents": len(real),
        "incidents_covered": len(covered),
        "incidents_ready_at_start": len(ready_at_start),
        "mean_detect_s": (sum(detect) / len(detect)) if detect else float("nan"),
        "mean_time_to_diagnostic_s": (sum(to_diag) / len(to_diag)) if to_diag else float("nan"),
        "changes_per_service_hour": sum(r["changed"] for r in records) / max(hours, 1e-9) / 3,
        "mean_reward": sum(r.get("reward", 0.0) for r in records) / max(n, 1),
    }
