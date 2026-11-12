"""Traffic and fault generators shared by the simulator and the real bench,
so both see the same kind of world (and, with the same seed, the same one).
"""
import math
from dataclasses import dataclass

import numpy as np


class LoadProcess:
    """Requests per second: a slow 'daily' wave plus heavy-tailed bursts.

    Burst sizes follow a Pareto law with alpha = 1.5, the same shape used in
    the dissertation's 100-node experiment.
    """

    def __init__(self, seed, base=40.0, period_s=1800.0, burst_every_s=240.0,
                 burst_cap=4.0, burst_decay_s=25.0):
        self.rng = np.random.default_rng(seed)
        self.base, self.period, self.every = base, period_s, burst_every_s
        self.cap, self.decay = burst_cap, burst_decay_s
        self.t = 0.0
        self.burst = 0.0

    def step(self, dt):
        self.t += dt
        if self.rng.random() < dt / self.every:
            self.burst += min(float(self.rng.pareto(1.5)), self.cap)
        self.burst *= math.exp(-dt / self.decay)
        wave = 1.0 + 0.35 * math.sin(2 * math.pi * self.t / self.period)
        return self.base * wave * (1.0 + self.burst)


@dataclass(frozen=True)
class Fault:
    start_s: float
    ramp_s: float      # slow build-up: the window where early detail pays off
    hold_s: float
    kind: str          # latency | errors | cpu
    target: str
    peak: float        # latency: added seconds; errors: error share; cpu: busy seconds per request

    def severity(self, t):
        """0..1 strength at time t: ramp up, hold, quick recovery."""
        x = t - self.start_s
        if x < 0:
            return 0.0
        if x < self.ramp_s:
            return x / self.ramp_s
        if x < self.ramp_s + self.hold_s:
            return 1.0
        rec = x - self.ramp_s - self.hold_s
        return max(0.0, 1.0 - rec / 10.0)

    @property
    def end_s(self):
        return self.start_s + self.ramp_s + self.hold_s + 10.0


def fault_schedule(seed, duration_s, services, first_s=60.0, mean_gap_s=240.0):
    rng = np.random.default_rng(seed + 7919)
    faults, t = [], first_s + rng.exponential(mean_gap_s / 2)
    while t < duration_s - 90:
        kind = str(rng.choice(["latency", "errors", "cpu"]))
        peak = {"latency": rng.uniform(0.25, 0.7),
                "errors": rng.uniform(0.04, 0.15),
                "cpu": rng.uniform(0.010, 0.025)}[kind]
        f = Fault(start_s=float(t), ramp_s=float(rng.uniform(15, 60)), hold_s=float(rng.uniform(40, 150)),
                  kind=kind, target=str(rng.choice(services)), peak=float(peak))
        faults.append(f)
        t = f.end_s + 30 + rng.exponential(mean_gap_s)
    return faults


def active_faults(faults, t):
    """{service: {kind: strength}} at time t."""
    out = {}
    for f in faults:
        s = f.severity(t)
        if s > 0:
            out.setdefault(f.target, {})[f.kind] = out.get(f.target, {}).get(f.kind, 0.0) + s * f.peak
    return out


# ---------------------------------------------------------------- harmless events
# Shapes that look alarming but are not incidents. Taken from a public
# comparison of alerting detectors (docs/related-work.md): single spikes,
# a cron or GC bump, a cold start after a rollout, a short burst, and a
# scrape gap. Simulator only. They teach the agent that a spike alone is no
# reason to buy detail.
@dataclass(frozen=True)
class Benign:
    start_s: float
    dur_s: float
    kind: str          # spike | coldstart | burst | gap | periodic
    target: str
    factor: float      # latency multiplier at the peak (1.0 for a gap)
    period_s: float = 0.0

    @property
    def end_s(self):
        return self.start_s + self.dur_s

    def factor_at(self, t):
        x = t - self.start_s
        if x < 0 or x >= self.dur_s:
            return 1.0
        if self.kind == "coldstart":           # jump, then decay back to normal
            return 1.0 + (self.factor - 1.0) * (1.0 - x / self.dur_s)
        if self.kind == "periodic":            # 30 s bump every period_s
            return self.factor if x % self.period_s < 30.0 else 1.0
        if self.kind == "gap":
            return 1.0
        return self.factor                     # spike, burst: flat top


def benign_schedule(seed, duration_s, services, mean_gap_s=180.0):
    rng = np.random.default_rng(seed + 104729)
    events = []
    if rng.random() < 0.5:   # some episodes have a cron/GC bump on one service
        events.append(Benign(0.0, duration_s, "periodic", str(rng.choice(services)),
                             float(rng.uniform(1.3, 1.8)), period_s=float(rng.choice([240.0, 300.0, 600.0]))))
    t = rng.exponential(mean_gap_s)
    while t < duration_s - 30:
        kind = str(rng.choice(["spike", "spike", "coldstart", "burst", "gap"]))
        dur, factor = {"spike": (float(rng.choice([5.0, 10.0])), rng.uniform(2.0, 4.0)),
                       "coldstart": (rng.uniform(60, 180), rng.uniform(2.0, 3.0)),
                       "burst": (rng.uniform(30, 120), rng.uniform(1.5, 3.0)),
                       "gap": (rng.uniform(60, 180), 1.0)}[kind]
        events.append(Benign(float(t), float(dur), kind, str(rng.choice(services)), float(factor)))
        t += dur + rng.exponential(mean_gap_s)
    return events


def benign_factor(events, svc, t):
    f = 1.0
    for e in events:
        if e.target == svc and e.kind != "gap":
            f *= e.factor_at(t)
    return f
