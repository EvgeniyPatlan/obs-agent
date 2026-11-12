"""Measure the bench so the simulator matches it: series per metric group,
spans per request, and latency/utilisation under the base load.

    python -m obsagent.calibrate
"""
import json
import os
import time
import urllib.request

from . import calib
from .bench_env import http_json, svc_url
from .config import SERVICES

GROUPS = ("core", "detail", "debug")


def count_series(service):
    with urllib.request.urlopen(svc_url(service) + "/metrics", timeout=5) as r:
        text = r.read().decode()
    counts = {g: 0 for g in GROUPS}
    for line in text.splitlines():
        for g in GROUPS:
            if line.startswith(f"app_{g}_"):
                counts[g] += 1
    return counts


def main():
    for s in SERVICES:
        http_json(svc_url(s) + "/control", {"sampling": 1.0})
        http_json(svc_url(s) + "/fault", {})
        http_json(svc_url(s) + "/truth")
    sink0 = http_json("http://tracesink:8080/stats")
    time.sleep(30)
    truth = {s: http_json(svc_url(s) + "/truth") for s in SERVICES}
    time.sleep(2)  # let the last span batch arrive
    sink1 = http_json("http://tracesink:8080/stats")
    per_service = {s: count_series(s) for s in SERVICES}
    series = {g: round(sum(c[g] for c in per_service.values()) / len(SERVICES)) for g in GROUPS}
    spr = []
    for s in SERVICES:
        sent = sink1.get(s, {}).get("spans", 0) - sink0.get(s, {}).get("spans", 0)
        if truth[s]["n"]:
            spr.append(sent / truth[s]["n"])
    for s in SERVICES:
        http_json(svc_url(s) + "/control", {"sampling": 0.01})
    out = {"series": series, "spans_per_req": sum(spr) / len(spr) if spr else 1.0,
           "per_service_series": per_service, "base_load_truth": truth}
    os.makedirs(calib.RESULTS, exist_ok=True)
    with open(os.path.join(calib.RESULTS, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
