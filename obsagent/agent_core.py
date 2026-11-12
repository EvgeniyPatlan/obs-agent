"""What the agent sees (state), what it is paid (reward) and what telemetry
costs. Shared by the simulator, training and the real bench so there is a
single definition of each.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import DIAGNOSABLE, LEVELS, MAX_LEVEL, RPS_REF, SLA_ERR, SLA_P99_S

STATE_DIM = 15
FEATURES = ("p99/SLA", "p50/SLA", "errors/SLA", "rps", "rps change", "utilisation",
            "level", "ticks since change", "data staleness", "p99 anomaly",
            "worst p99 (all services)", "worst errors (all services)",
            "share over SLA, last 1 min", "share over SLA, last 5 min", "share over SLA, last 60 min")
# Readings over the SLA in a short and a long window, as in multi-window
# burn-rate alerting: one spike moves the short window only, a real incident
# moves both. Agents trained before these were added read the first 12.
WINDOWS = (12, 60, 720)   # ticks: 1, 5 and 60 min


@dataclass
class Memory:
    """Per-service memory the agent keeps between decisions."""
    level: int = 1
    since_change: int = 99
    p99_ewma: float = 0.05
    prev_rps: float = 0.0
    last_change_ticks: list = field(default_factory=list)
    over: deque = field(default_factory=lambda: deque(maxlen=WINDOWS[-1]))


def over_sla(obs):
    return obs["p99"] > SLA_P99_S or obs["err"] > SLA_ERR


def window_shares(mem, obs):
    """Share of readings over the SLA in each window, the current one included."""
    hist, now = list(mem.over), over_sla(obs)
    out = []
    for w in WINDOWS:
        recent = hist[-(w - 1):] if w > 1 else []
        out.append((sum(recent) + now) / (len(recent) + 1))
    return out


def build_state(obs, mem, worst_p99, worst_err):
    """obs: what the collected telemetry shows for this service right now."""
    p99, p50, err = obs["p99"], obs["p50"], obs["err"]
    anomaly = (p99 - mem.p99_ewma) / max(mem.p99_ewma, 0.02)
    return np.array([
        np.clip(p99 / SLA_P99_S, 0, 4),
        np.clip(p50 / SLA_P99_S, 0, 4),
        np.clip(err / SLA_ERR, 0, 4),
        np.clip(obs["rps"] / RPS_REF, 0, 4),
        np.clip((obs["rps"] - mem.prev_rps) / RPS_REF, -2, 2),
        np.clip(obs["util"], 0, 1.5),
        mem.level / MAX_LEVEL,
        min(mem.since_change, 20) / 20,
        np.clip(obs["staleness"] / 30.0, 0, 2),
        np.clip(anomaly, -2, 4) / 4,
        np.clip(worst_p99 / SLA_P99_S, 0, 4),
        np.clip(worst_err / SLA_ERR, 0, 4),
        *window_shares(mem, obs),
    ])


def update_memory(mem, obs, new_level, tick):
    changed = new_level != mem.level
    mem.p99_ewma = 0.9 * mem.p99_ewma + 0.1 * obs["p99"]
    mem.prev_rps = obs["rps"]
    mem.over.append(over_sla(obs))
    mem.since_change = 0 if changed else mem.since_change + 1
    mem.level = new_level
    if changed:
        mem.last_change_ticks.append(tick)
    return changed


def apply_action(level, delta):
    return int(np.clip(level + delta, 0, MAX_LEVEL))


# ---------------------------------------------------------------- cost
def telemetry_rates(level, rps, series, spans_per_req):
    """Samples/s and spans/s that a service produces at this level."""
    lv = LEVELS[level]
    samples = sum(series[g] for g in lv.groups) / lv.interval_s
    spans = rps * spans_per_req * lv.sampling
    return samples, spans


def cost_fraction(level, rps, series, spans_per_req, rps_nominal=60.0):
    """Cost relative to 'full' at nominal load; half metrics, half traces."""
    s, p = telemetry_rates(level, rps, series, spans_per_req)
    s_max, p_max = telemetry_rates(MAX_LEVEL, rps_nominal, series, spans_per_req)
    return 0.5 * s / s_max + 0.5 * p / p_max


# ---------------------------------------------------------------- reward
@dataclass(frozen=True)
class RewardWeights:
    cost: float = 1.0        # r_cost: pay for every sample and span
    blind: float = 5.0       # r_SLA: incident while we cannot diagnose it
    partial: float = 2.0     # incident with only standard detail
    pre: float = 1.0         # lean right before an incident (data for root cause is lost)
    flap: float = 1.0        # r_stability: changing settings again within the cooldown
    change: float = 0.15     # every reconfiguration has a small price (reloads, gaps)
    bonus: float = 0.3       # r_optimal: the right level for the situation
    cooldown: int = 6


def reward(level, cost_frac, incident, pre_incident, changed, since_change_before, w=RewardWeights()):
    """Ground truth (incident / pre_incident) is used only here, never in the state."""
    r = -w.cost * cost_frac
    if incident:
        if level < DIAGNOSABLE - 1:
            r -= w.blind
        elif level < DIAGNOSABLE:
            r -= w.partial
        else:
            r += w.bonus
    else:
        if pre_incident and level < DIAGNOSABLE - 1:
            r -= w.pre
        if level <= 1 and not pre_incident:
            r += w.bonus
    if changed:
        r -= w.change
        if since_change_before < w.cooldown:
            r -= w.flap
    return r
