"""Fast simulator of the bench: three chained services under bursty load
with injected faults. It lets the agent train for hundreds of simulated
hours in minutes. Its parameters mirror the real bench (bench/service.py)
and results/calibration.json when present.

The key property it reproduces: lean telemetry makes the agent's own view
stale and smoothed, so turning detail down is not free.
"""
import math
from collections import deque

import numpy as np

from .config import DEFAULT_SERIES, DEFAULT_SPANS_PER_REQ, LEVELS, SERVICES, SLA_ERR, SLA_P99_S, TICK_S
from .workload import LoadProcess, active_faults, benign_factor, benign_schedule, fault_schedule

# Same numbers as docker-compose.yml: own work (seconds) and worker pool size.
SERVICE_PARAMS = {
    "gateway": {"base": 0.005, "pool": 64},
    "api":     {"base": 0.010, "pool": 32},
    "store":   {"base": 0.015, "pool": 6},
}
CHAIN = ("store", "api", "gateway")  # computed bottom-up


def _queue(rps, own_s, service_s, pool, cpu_s):
    """One service. A worker is held for its own work plus the downstream call
    (service_s), so a slow store fills the upstream pools too. The queueing
    wait and the tail are this service's own; the downstream part is added
    once, in _chain, not here."""
    util = max(rps * service_s / pool, rps * cpu_s)   # cpu work is bounded by one core (GIL)
    u = min(util, 0.97)
    own_mean = own_s + service_s * u ** 3 / (1.0 - u)
    own_p99 = own_mean * (2.5 + 6.0 * u ** 3)
    overload = max(0.0, 1.0 - 0.97 / util) if util > 0.97 else 0.0
    return util, own_mean, own_p99, overload


def _chain(own_mean, own_p99, down):
    """Latency of own work plus the downstream call. Means add up. Tails
    combine like independent parts: the slowest 1% of the total is shorter
    than the sum of each part's slowest 1%. (Adding the two p99s, as the
    first version did, made calm latency 2-3x too high upstream.)"""
    if not down:
        return own_mean, own_p99
    mean = own_mean + down["mean"]
    tail = math.hypot(own_p99 - own_mean, down["p99"] - down["mean"])
    return mean, mean + tail


class SimEnv:
    def __init__(self, series=None, spans_per_req=DEFAULT_SPANS_PER_REQ, base_rps=40.0,
                 noise=None, fault_gap_s=240.0, benign=True, randomize=False):
        self.series = dict(series or DEFAULT_SERIES)
        self.spans_per_req = spans_per_req
        self.base_rps = base_rps
        self.noise = noise              # results/obs_noise.json: real reading errors, replayed
        self.fault_gap_s = fault_gap_s  # a number, or a (low, high) range drawn per episode
        self.use_benign = benign        # harmless bumps and scrape gaps (see workload.benign_schedule)
        self.randomize = randomize      # draw a slightly different world each episode (see _draw_world)

    def reset(self, seed, duration_s):
        self.rng = np.random.default_rng(seed)
        self.world = self._draw_world(seed)
        self.load = LoadProcess(seed, base=self.base_rps * self.world["load"])
        gap = self.fault_gap_s
        if isinstance(gap, (tuple, list)):
            gap = float(self.rng.uniform(*gap))
        self.faults = fault_schedule(seed, duration_s, SERVICES, mean_gap_s=gap)
        self.benign = (benign_schedule(seed, duration_s, SERVICES, mean_gap_s=self.world["benign_gap_s"])
                       if self.use_benign else [])
        self.t = 0.0
        self.levels = {s: 1 for s in SERVICES}
        self.hist = {s: deque(maxlen=12) for s in SERVICES}
        self.last_scrape = {s: -999.0 for s in SERVICES}
        self.view = {s: {"p99": 0.05, "p50": 0.02, "err": 0.0, "rps": 0.0, "util": 0.0} for s in SERVICES}
        self._truth = self._simulate(0.0, self.base_rps)
        return self

    def _draw_world(self, seed):
        """Domain randomization. The simulator matches the bench only roughly,
        so each training episode gets its own latency, load, reading noise,
        fault strength and rate of harmless spikes. A policy that works in
        all of them is less tied to the simulator's exact numbers. Uses its
        own random stream, so with randomize=False every episode is unchanged."""
        if not self.randomize:
            return {"base": {s: 1.0 for s in SERVICES}, "load": 1.0, "noise": 1.0, "fault": 1.0,
                    "jitter": 0.08, "benign_gap_s": 180.0}
        r = np.random.default_rng(seed + 7919)
        return {"base": {s: float(r.uniform(0.8, 1.25)) for s in SERVICES},
                "load": float(r.uniform(0.8, 1.25)),
                "noise": float(r.uniform(0.6, 1.6)),     # scales the measured reading errors
                "fault": float(r.uniform(0.7, 1.4)),
                "jitter": float(r.uniform(0.04, 0.15)),  # tick-to-tick latency wobble
                "benign_gap_s": float(r.uniform(90, 360))}

    def _simulate(self, t, rps):
        faults = active_faults(self.faults, t)
        out, down = {}, None
        for svc in CHAIN:
            p = SERVICE_PARAMS[svc]
            f = {k: v * self.world["fault"] for k, v in faults.get(svc, {}).items()}
            own = p["base"] * self.world["base"][svc] + f.get("latency", 0.0) + f.get("cpu", 0.0)
            service_s = own + (down["mean"] if down else 0.0)
            util, own_mean, own_p99, overload = _queue(rps, own, service_s, p["pool"], f.get("cpu", 0.0))
            noise = math.exp(self.rng.normal(0, self.world["jitter"]))
            mean, p99 = _chain(own_mean * noise, own_p99 * noise, down)
            bump = benign_factor(self.benign, svc, t)
            mean, p99 = mean * bump, p99 * bump
            p50 = 0.8 * mean
            err_own = min(1.0, 0.002 + f.get("errors", 0.0) + overload)
            err = 1 - (1 - err_own) * (1 - (down["err"] if down else 0.0))
            n = max(1, int(rps * TICK_S))
            err = self.rng.binomial(n, min(err, 1.0)) / n
            out[svc] = {"p99": p99, "p50": p50, "err": err, "rps": rps, "util": min(util, 1.5),
                        "mean": mean, "fault": ",".join(sorted(f)) if f else ""}
            down = out[svc]
        return out

    # -------------------------------------------------- env interface
    def observe(self):
        """What the collected telemetry shows: refreshed only at scrape times,
        averaged over a rate window of 2x the scrape interval."""
        obs = {}
        for s in SERVICES:
            lv = LEVELS[self.levels[s]]
            gap = any(e.kind == "gap" and e.target == s and e.start_s <= self.t < e.end_s for e in self.benign)
            if not gap and self.t - self.last_scrape[s] >= lv.interval_s:
                self.last_scrape[s] = self.t
                w = max(1, math.ceil(2 * lv.interval_s / TICK_S))
                recent = list(self.hist[s])[-w:] or [self._truth[s]]
                self.view[s] = self._read(s, self.levels[s], {k: float(np.mean([h[k] for h in recent]))
                                                              for k in ("p99", "p50", "err", "rps", "util")})
            stale = lv.interval_s / 2 if lv.interval_s < TICK_S and not gap else self.t - self.last_scrape[s]
            obs[s] = {**self.view[s], "staleness": stale}
        return obs

    def _read(self, svc, level, view):
        """Add a real reading error, drawn from what the bench measured."""
        if not self.noise:
            return view
        pool = self.noise[svc]["long" if level == 0 else "short"] or self.noise[svc]["short"]
        log_ratio, err_diff = pool[int(self.rng.integers(len(pool)))]
        factor = math.exp(log_ratio * self.world["noise"])
        return {**view, "p99": view["p99"] * factor, "p50": view["p50"] * factor,
                "err": min(1.0, max(0.0, view["err"] + err_diff))}

    def apply(self, levels):
        self.levels = dict(levels)

    def advance(self):
        self.t += TICK_S
        rps = self.load.step(TICK_S)   # already scaled by world["load"] through its base
        self._truth = self._simulate(self.t, rps)
        info = {}
        for s in SERVICES:
            tr = self._truth[s]
            self.hist[s].append(tr)
            lv = LEVELS[self.levels[s]]
            samples = sum(self.series[g] for g in lv.groups) / lv.interval_s * TICK_S
            spans = rps * TICK_S * self.spans_per_req * lv.sampling
            info[s] = {"true_p99": tr["p99"], "true_err": tr["err"], "rps": rps,
                       "incident": tr["p99"] > SLA_P99_S or tr["err"] > SLA_ERR,
                       "samples": samples, "spans": spans, "fault": tr["fault"]}
        return info
