"""Totals over every seed of a parallel bench run.

    python -m obsagent.bench_totals bench_v3

Volume is half samples, half spans, relative to collecting everything on the
same seed. Time to diagnostic detail is averaged over diagnosable incidents.
"""
import glob
import json
import os
import sys

from . import calib


def totals(folder):
    seeds = sorted(d for d in os.listdir(folder) if d.isdigit())
    names = sorted({os.path.basename(f)[8:-5] for f in glob.glob(os.path.join(folder, "*", "summary_*.json"))})
    out = {}
    for p in names:
        rows = []
        for s in seeds:
            f, full = (os.path.join(folder, s, f"summary_{n}.json") for n in (p, "static-full"))
            if os.path.exists(f) and os.path.exists(full):
                rows.append((json.load(open(f))[p], json.load(open(full))["static-full"]))
        if len(rows) < len(seeds):
            print(f"  {p}: only {len(rows)}/{len(seeds)} seeds", file=sys.stderr)
        cov = sum(r["incidents_covered"] for r, _ in rows)
        out[p] = {
            "seeds": len(rows),
            "volume": 0.5 * sum(r["samples"] for r, _ in rows) / sum(f["samples"] for _, f in rows)
                      + 0.5 * sum(r["spans"] for r, _ in rows) / sum(f["spans"] for _, f in rows),
            "incidents": sum(r["incidents"] for r, _ in rows),
            "covered": cov,
            "ready_at_start": sum(r["incidents_ready_at_start"] for r, _ in rows),
            "time_to_diagnostic_s": sum(r["incidents_covered"] * r["mean_time_to_diagnostic_s"]
                                        for r, _ in rows if r["incidents_covered"]) / max(cov, 1),
            "changes_per_service_hour": sum(r["changes_per_service_hour"] for r, _ in rows) / len(rows),
        }
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "bench"
    res = totals(os.path.join(calib.RESULTS, name))
    for p, t in res.items():
        print(f"{p:18s} seeds={t['seeds']} volume={t['volume']:6.2%} diagnosable={t['covered']}/{t['incidents']} "
              f"ready@start={t['ready_at_start']} to-diag={t['time_to_diagnostic_s']:.1f}s "
              f"changes/h={t['changes_per_service_hour']:.1f}")
    with open(os.path.join(calib.RESULTS, name, "totals.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
