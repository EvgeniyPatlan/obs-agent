"""The real environment: the agent changes Prometheus scrape config (interval
and which metric groups are kept) and each service's span sampling, then
reads its state back from Prometheus, i.e. only from what it chose to collect.
"""
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import LEVELS, SERVICES, SLA_ERR, SLA_P99_S, TICK_S
from .workload import active_faults, fault_schedule

PROM = os.environ.get("PROM_URL", "http://prometheus:9090")
PROM_CONFIG = os.environ.get("PROM_CONFIG", "/prom-config/prometheus.yml")
SINK = os.environ.get("SINK_URL", "http://tracesink:8080")
LOADGEN = os.environ.get("LOADGEN_URL", "http://loadgen:8089")
ALL_GROUPS = ("core", "detail", "debug")


def http_json(url, body=None, timeout=5, attempts=3):
    """POST (with body) or GET JSON. Retries a dropped connection twice:
    one reset socket must not end an hour-long bench run."""
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return json.loads(raw) if raw else {}
        except (ConnectionError, TimeoutError, urllib.error.URLError) as e:
            if attempt == attempts - 1:
                raise
            print(f"  retry {url}: {e}", flush=True)
            time.sleep(0.5 * (attempt + 1))


def svc_url(s):
    return f"http://{s}:8080"


def render_prom_config(levels):
    out = ["global:", "  scrape_interval: 15s", "scrape_configs:",
           "  - job_name: prometheus-self", "    scrape_interval: 15s",
           "    static_configs: [{targets: ['prometheus:9090']}]"]
    for s in SERVICES:
        lv = LEVELS[levels[s]]
        dropped = [g for g in ALL_GROUPS if g not in lv.groups]
        out += [f"  - job_name: {s}", f"    scrape_interval: {lv.interval_s}s",
                f"    scrape_timeout: {min(lv.interval_s, 10)}s",
                f"    static_configs: [{{targets: ['{s}:8080']}}]"]
        if dropped:
            out += ["    metric_relabel_configs:", "      - source_labels: [__name__]",
                    f"        regex: 'app_({'|'.join(dropped)})_.*'", "        action: drop"]
    return "\n".join(out) + "\n"


def write_prom_config(levels):
    tmp = PROM_CONFIG + ".tmp"
    with open(tmp, "w") as f:
        f.write(render_prom_config(levels))
    os.replace(tmp, PROM_CONFIG)


def prom_query(q, at=None):
    params = {"query": q}
    if at is not None:
        params["time"] = f"{at:.3f}"
    res = http_json(f"{PROM}/api/v1/query?" + urllib.parse.urlencode(params))
    vals = res.get("data", {}).get("result", [])
    if not vals:
        return float("nan")
    return float(vals[0]["value"][1])


class FaultDriver(threading.Thread):
    """Pushes the current fault strength to each service once a second."""

    def __init__(self, faults, t0):
        super().__init__(daemon=True)
        self.faults, self.t0, self.stop = faults, t0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            act = active_faults(self.faults, time.time() - self.t0)
            for s in SERVICES:
                try:
                    http_json(svc_url(s) + "/fault", act.get(s, {}))
                except Exception as exc:
                    print(f"fault push to {s} failed: {exc}", flush=True)
            self.stop.wait(1.0)


class BenchEnv:
    def __init__(self, seed, duration_s, fault_gap_s=240.0):
        self.seed, self.duration = seed, duration_s
        self.faults = fault_schedule(seed, duration_s, SERVICES, mean_gap_s=fault_gap_s)
        self.levels = None
        self.last = {s: {"p99": 0.05, "p50": 0.02, "err": 0.0, "rps": 0.0, "util": 0.0} for s in SERVICES}

    def start(self, warmup_s=20):
        for s in SERVICES:
            http_json(svc_url(s) + "/fault", {})
        http_json(LOADGEN + "/reset", {"seed": self.seed})
        self.apply({s: 1 for s in SERVICES})
        time.sleep(warmup_s)
        for s in SERVICES:
            http_json(svc_url(s) + "/truth")          # drop warm-up truth
        self.spans0 = http_json(SINK + "/stats")
        self.t0 = time.time()
        self.next_tick = self.t0 + TICK_S
        self.driver = FaultDriver(self.faults, self.t0)
        self.driver.start()

    def stop(self):
        self.driver.stop.set()
        self.driver.join(timeout=3)
        for s in SERVICES:
            http_json(svc_url(s) + "/fault", {})
        self.t_end = time.time()

    # -------------------------------------------------- env interface
    def apply(self, levels):
        if levels != self.levels:
            write_prom_config(levels)
            http_json(PROM + "/-/reload", {})
        for s in SERVICES:
            if self.levels is None or levels[s] != self.levels[s]:
                http_json(svc_url(s) + "/control", {"sampling": LEVELS[levels[s]].sampling})
        self.levels = dict(levels)

    def observe(self):
        obs = {}
        for s in SERVICES:
            w = f"{max(int(2.5 * LEVELS[self.levels[s]].interval_s), 10)}s"
            sel = f'service="{s}"'
            q = {
                "p99": f"histogram_quantile(0.99, sum by (le) (rate(app_core_request_duration_seconds_bucket{{{sel}}}[{w}])))",
                "p50": f"histogram_quantile(0.50, sum by (le) (rate(app_core_request_duration_seconds_bucket{{{sel}}}[{w}])))",
                "err": f'(sum(rate(app_core_requests_total{{{sel},code="500"}}[{w}])) or vector(0)) / sum(rate(app_core_requests_total{{{sel}}}[{w}]))',
                "rps": f"sum(rate(app_core_requests_total{{{sel}}}[{w}]))",
                "util": f"sum(rate(app_core_busy_seconds_total{{{sel}}}[{w}])) / max(app_core_pool_size{{{sel}}})",
                "staleness": f"time() - max(timestamp(app_core_pool_size{{{sel}}}))",
            }
            o = {}
            for k, expr in q.items():
                try:
                    v = prom_query(expr)
                except Exception as exc:
                    print(f"query {k} for {s} failed: {exc}", flush=True)
                    v = float("nan")
                if math.isnan(v) or math.isinf(v):
                    v = 60.0 if k == "staleness" else self.last[s].get(k, 0.0)
                o[k] = v
            self.last[s] = o
            obs[s] = o
        return obs

    def advance(self):
        time.sleep(max(0.0, self.next_tick - time.time()))
        self.next_tick += TICK_S
        now = time.time()
        spans = http_json(SINK + "/stats")
        info = {}
        for s in SERVICES:
            tr = http_json(svc_url(s) + "/truth")
            try:
                samples = prom_query(f'sum_over_time(scrape_samples_post_metric_relabeling{{job="{s}"}}[{TICK_S}s])', at=now)
            except Exception:
                samples = float("nan")
            prev = getattr(self, "_spans_prev", self.spans0).get(s, {"spans": 0, "bytes": 0})
            cur = spans.get(s, {"spans": 0, "bytes": 0})
            act = active_faults(self.faults, now - self.t0).get(s, {})
            info[s] = {"true_p99": tr["p99"], "true_err": tr["err"], "rps": tr["n"] / TICK_S,
                       "incident": tr["n"] > 0 and (tr["p99"] > SLA_P99_S or tr["err"] > SLA_ERR),
                       "samples": 0.0 if math.isnan(samples) else samples,
                       "spans": cur["spans"] - prev["spans"], "span_bytes": cur["bytes"] - prev["bytes"],
                       "fault": ",".join(sorted(act))}
        self._spans_prev = spans
        return info

    def run_totals(self):
        """Exact totals over the whole run, straight from Prometheus and the sink."""
        dur = int(self.t_end - self.t0)
        out = {}
        spans = http_json(SINK + "/stats")
        for s in SERVICES:
            out[s] = {"samples": prom_query(f'sum_over_time(scrape_samples_post_metric_relabeling{{job="{s}"}}[{dur}s])', at=self.t_end),
                      "spans": spans.get(s, {}).get("spans", 0) - self.spans0.get(s, {}).get("spans", 0),
                      "span_bytes": spans.get(s, {}).get("bytes", 0) - self.spans0.get(s, {}).get("bytes", 0)}
        return out
