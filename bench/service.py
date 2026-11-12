"""A generic bench service (stdlib only). One file, three roles set by env:
gateway -> api -> store. Each holds a worker slot for the whole request
(including the downstream call), so a slow store backs up the chain.

Endpoints
  GET  /work     handle one request
  GET  /metrics  Prometheus text; three groups: app_core_*, app_detail_*, app_debug_*
  POST /control  {"sampling": 0.25}         trace span sampling (set by the agent)
  POST /fault    {"latency": s, "errors": p, "cpu": s}   fault injection (bench only)
  GET  /truth    true p50/p99/errors since the last call (evaluation only, not telemetry)
"""
import json
import os
import random
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = os.environ["SERVICE"]
DOWNSTREAM = os.environ.get("DOWNSTREAM", "")
POOL = int(os.environ.get("POOL", "8"))
BASE_S = float(os.environ.get("BASE_MS", "10")) / 1000
SINK = os.environ.get("TRACESINK", "http://tracesink:8080/spans")

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.5, 5.0)
ROUTES = [f"/v1/resource{i}" for i in range(20)]
TENANTS = [f"t{i:03d}" for i in range(400)]

lock = threading.Lock()
slots = threading.BoundedSemaphore(POOL)
control = {"sampling": 0.01}
fault = {}
core = {"req": {"200": 0, "500": 0}, "hist": [0] * len(BUCKETS), "sum": 0.0, "count": 0,
        "busy": 0.0, "inflight": 0}
detail = {r: {"hist": [0] * len(BUCKETS), "sum": 0.0, "count": 0, "200": 0, "500": 0} for r in ROUTES}
debug = {t: {"req": 0, "bytes": 0, "lat_sum": 0.0, "errors": 0} for t in TENANTS}
truth = []
spans = []


def _observe(route, tenant, dur, code):
    with lock:
        core["req"][code] += 1
        core["sum"] += dur
        core["count"] += 1
        d = detail[route]
        d[code] += 1
        d["sum"] += dur
        d["count"] += 1
        for i, b in enumerate(BUCKETS):
            if dur <= b:
                core["hist"][i] += 1
                d["hist"][i] += 1
        t = debug[tenant]
        t["req"] += 1
        t["bytes"] += random.randint(200, 4000)
        t["lat_sum"] += dur
        t["errors"] += code == "500"
        truth.append((dur, code == "500"))


def _burn(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def handle_work():
    route = random.choice(ROUTES)
    tenant = TENANTS[min(int(random.paretovariate(1.2)) - 1, len(TENANTS) - 1)]
    start = time.perf_counter()
    code = "200"
    with lock:
        core["inflight"] += 1
    got = slots.acquire(timeout=3.0)
    try:
        if not got:
            code = "500"
        else:
            busy0 = time.perf_counter()
            f = dict(fault)
            time.sleep(random.lognormvariate(0, 0.35) * BASE_S + f.get("latency", 0.0))
            if f.get("cpu"):
                _burn(f["cpu"])
            if random.random() < 0.002 + f.get("errors", 0.0):
                code = "500"
            elif DOWNSTREAM:
                try:
                    with urllib.request.urlopen(DOWNSTREAM + "/work", timeout=5) as r:
                        r.read()
                except Exception:
                    code = "500"
            with lock:
                core["busy"] += time.perf_counter() - busy0
    finally:
        if got:
            slots.release()
        with lock:
            core["inflight"] -= 1
    dur = time.perf_counter() - start
    _observe(route, tenant, dur, code)
    if random.random() < control["sampling"]:
        with lock:
            spans.append({"service": NAME, "name": "GET " + route, "tenant": tenant,
                          "trace_id": "%032x" % random.getrandbits(128), "span_id": "%016x" % random.getrandbits(64),
                          "start": start, "duration_s": round(dur, 6), "status": code,
                          "attributes": {"http.route": route, "http.method": "GET", "peer": DOWNSTREAM or "none"}})
    return code


def _hist_lines(name, labels, hist, total_sum, count):
    out = []
    for b, c in zip(BUCKETS, hist):
        out.append(f'{name}_bucket{{{labels},le="{b}"}} {c}')
    out.append(f'{name}_bucket{{{labels},le="+Inf"}} {count}')
    out.append(f"{name}_sum{{{labels}}} {total_sum}")
    out.append(f"{name}_count{{{labels}}} {count}")
    return out


def render_metrics():
    s = f'service="{NAME}"'
    with lock:
        lines = [f'app_core_requests_total{{{s},code="{c}"}} {v}' for c, v in core["req"].items()]
        lines += _hist_lines("app_core_request_duration_seconds", s, core["hist"], core["sum"], core["count"])
        lines += [f"app_core_busy_seconds_total{{{s}}} {core['busy']}",
                  f"app_core_inflight{{{s}}} {core['inflight']}",
                  f"app_core_pool_size{{{s}}} {POOL}"]
        for r, d in detail.items():
            lab = f'{s},route="{r}"'
            lines += _hist_lines("app_detail_route_duration_seconds", lab, d["hist"], d["sum"], d["count"])
            lines += [f'app_detail_route_requests_total{{{lab},code="200"}} {d["200"]}',
                      f'app_detail_route_requests_total{{{lab},code="500"}} {d["500"]}']
        for t, d in debug.items():
            lab = f'{s},tenant="{t}"'
            lines += [f"app_debug_tenant_requests_total{{{lab}}} {d['req']}",
                      f"app_debug_tenant_bytes_total{{{lab}}} {d['bytes']}",
                      f"app_debug_tenant_latency_seconds_sum{{{lab}}} {d['lat_sum']}",
                      f"app_debug_tenant_errors_total{{{lab}}} {d['errors']}"]
    return "\n".join(lines) + "\n"


def take_truth():
    with lock:
        data = truth[:]
        truth.clear()
    if not data:
        return {"n": 0, "p50": 0.0, "p99": 0.0, "err": 0.0}
    durs = sorted(d for d, _ in data)
    pick = lambda q: durs[min(len(durs) - 1, int(q * len(durs)))]
    return {"n": len(data), "p50": pick(0.5), "p99": pick(0.99), "err": sum(e for _, e in data) / len(data)}


def flush_spans():
    while True:
        time.sleep(1.0)
        with lock:
            batch = spans[:]
            spans.clear()
        if batch:
            body = json.dumps(batch).encode()
            try:
                req = urllib.request.Request(SINK, data=body, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as exc:
                print(f"span export failed: {exc}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/work":
            code = handle_work()
            self._send(int(code), '{"ok": %s}' % ("true" if code == "200" else "false"))
        elif self.path == "/metrics":
            self._send(200, render_metrics(), "text/plain; version=0.0.4")
        elif self.path == "/truth":
            self._send(200, json.dumps(take_truth()))
        else:
            self._send(404, "{}")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            self._send(400, '{"error": "bad json"}')
            return
        if self.path == "/control":
            control["sampling"] = min(1.0, max(0.0, float(body.get("sampling", control["sampling"]))))
            self._send(200, json.dumps(control))
        elif self.path == "/fault":
            fault.clear()
            fault.update({k: float(v) for k, v in body.items() if k in ("latency", "errors", "cpu") and float(v) > 0})
            self._send(200, json.dumps(fault))
        else:
            self._send(404, "{}")


if __name__ == "__main__":
    threading.Thread(target=flush_spans, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.daemon_threads = True
    server.request_queue_size = 512
    print(f"{NAME} up: pool={POOL} base={BASE_S * 1000:.0f}ms downstream={DOWNSTREAM or '-'}", flush=True)
    server.serve_forever()
