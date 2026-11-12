"""Measure how far the real bench's readings are from the truth, so the
simulator can replay the same errors.

    python -m obsagent.noise_fit

Reads every results/**/bench_*.csv, keeps calm ticks (no incident, no
fault, not just after one) at levels whose readings are clean (L0, L1, L4),
and stores per service:
  - p99 reading / true p99, as a log ratio
  - error-rate reading minus true error rate
Writes results/obs_noise.json. Reading errors include histogram bucket bias
(readings run 5-50% high) and heavy-tailed spikes that look like an SLA breach.
"""
import csv
import glob
import json
import math
import os

import numpy as np

from . import calib
from .config import SERVICES

CLEAN_LEVELS = (0, 1, 4)
SETTLE_TICKS = 12          # skip 60 s after an incident or fault ends
MAX_PER_KEY = 3000


def window_key(level):
    """L0 averages over 75 s and is much smoother than the other levels."""
    return "long" if level == 0 else "short"


def _calm_rows(path):
    rows = list(csv.DictReader(open(path)))
    by_svc = {}
    for r in rows:
        by_svc.setdefault(r["service"], []).append(r)
    for svc, rs in by_svc.items():
        rs.sort(key=lambda r: int(r["tick"]))
        quiet = SETTLE_TICKS
        for r in rs:
            busy = r["incident"] == "True" or bool(r["fault"])
            quiet = 0 if busy else quiet + 1
            if quiet > SETTLE_TICKS:
                yield svc, r


def fit(paths):
    pairs = {s: {"long": [], "short": []} for s in SERVICES}
    for path in paths:
        for svc, r in _calm_rows(path):
            level = int(r["level"])
            obs, true = float(r["obs_p99"]), float(r["true_p99"])
            if level not in CLEAN_LEVELS or obs <= 0 or true <= 0:
                continue
            de = float(r["obs_err"]) - float(r["true_err"])
            pairs[svc][window_key(level)].append([math.log(obs / true), de])
    rng = np.random.default_rng(0)
    out = {}
    for svc, by_key in pairs.items():
        out[svc] = {}
        for key, p in by_key.items():
            if len(p) > MAX_PER_KEY:
                p = [p[i] for i in rng.choice(len(p), MAX_PER_KEY, replace=False)]
            out[svc][key] = [[round(a, 4), round(b, 5)] for a, b in p]
    return out


def describe(model):
    for svc, by_key in model.items():
        for key, p in by_key.items():
            if not p:
                continue
            a = np.array(p)
            print(f"  {svc:8s} {key:5s} n={len(a):5d} bias x{math.exp(np.median(a[:, 0])):.2f} "
                  f"spread {a[:, 0].std():.2f} spikes>x3 {np.mean(a[:, 0] > math.log(3)):.1%} "
                  f"err+1% {np.mean(a[:, 1] > 0.01):.1%}")


def main():
    paths = sorted(glob.glob(os.path.join(calib.RESULTS, "**", "bench_*.csv"), recursive=True))
    model = fit(paths)
    print(f"fitted from {len(paths)} bench files")
    describe(model)
    with open(os.path.join(calib.RESULTS, "obs_noise.json"), "w") as f:
        json.dump(model, f)


if __name__ == "__main__":
    main()
