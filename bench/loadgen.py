"""Open-loop load generator using the same bursty LoadProcess as the simulator.
POST /reset {"seed": n} restarts the traffic pattern so every policy sees
the same load; GET / returns the current target rate."""
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from obsagent.workload import LoadProcess

TARGET = os.environ.get("TARGET", "http://gateway:8080/work")
BASE = float(os.environ.get("BASE_RPS", "40"))
state = {"load": LoadProcess(0, base=BASE), "rng": np.random.default_rng(0), "rate": 0.0, "sent": 0}
lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=400)


def hit():
    try:
        urllib.request.urlopen(TARGET, timeout=10).read()
    except Exception:
        pass  # errors are expected during faults; services count them


def loop():
    dt = 0.1
    nxt = time.monotonic()
    while True:
        with lock:
            rate = state["load"].step(dt)
            n = state["rng"].poisson(rate * dt)
            state["rate"] = rate
            state["sent"] += int(n)
        for _ in range(n):
            pool.submit(hit)
        nxt += dt
        time.sleep(max(0.0, nxt - time.monotonic()))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        seed = int(body.get("seed", 0))
        with lock:
            state["load"] = LoadProcess(seed, base=BASE)
            state["rng"] = np.random.default_rng(seed)
        self._send({"reset": seed})

    def do_GET(self):
        with lock:
            self._send({"rate": state["rate"], "sent": state["sent"]})


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8089), Handler)
    srv.daemon_threads = True
    srv.serve_forever()
