"""Shared constants and the telemetry levels the agent can choose from."""
from dataclasses import dataclass

SERVICES = ("gateway", "api", "store")
TICK_S = 5              # the agent decides every 5 seconds
SLA_P99_S = 0.300       # an incident: p99 latency above 300 ms ...
SLA_ERR = 0.02          # ... or more than 2% errors
RPS_REF = 100.0         # normalisation constant for request rate
DIAGNOSABLE = 3         # level at or above which an incident can be diagnosed


@dataclass(frozen=True)
class Level:
    name: str
    interval_s: int          # scrape interval
    groups: tuple            # metric groups collected
    sampling: float          # share of trace spans kept


# One knob per service. Each step up collects more, more often.
LEVELS = (
    Level("L0 minimal",    30, ("core",),                    0.005),
    Level("L1 lean",       15, ("core",),                    0.01),
    Level("L2 standard",   10, ("core", "detail"),           0.05),
    Level("L3 diagnostic",  5, ("core", "detail"),           0.25),
    Level("L4 full",        2, ("core", "detail", "debug"),  1.0),
)
MAX_LEVEL = len(LEVELS) - 1

# Series per metric group per service. calibrate.py measures the real values
# on the bench and writes results/calibration.json, which overrides these.
DEFAULT_SERIES = {"core": 30, "detail": 330, "debug": 1600}
DEFAULT_SPANS_PER_REQ = 1.0
