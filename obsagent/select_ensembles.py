"""Fix the two voting agents before any test run, from dev seeds only.

    python -m obsagent.eval_sim --seeds dev --agents a=v2/a.npz,b=v2/b.npz,... --out v2/sim_eval_dev.json
    python -m obsagent.select_ensembles v2

Rule, decided before looking at test seeds:
  balanced = every trained agent votes
  lean     = the cheapest half (lowest telemetry volume on dev) votes
Writes results/<dir>/ensembles.txt with one NAME=weights line each.
"""
import json
import os
import sys

from . import calib


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "v2"
    dev = json.load(open(os.path.join(calib.RESULTS, folder, "sim_eval_dev.json")))["results"]
    full = dev["static-full"]
    agents = {k: v for k, v in dev.items() if os.path.exists(os.path.join(calib.RESULTS, folder, f"{k}.npz"))}
    if not agents:
        sys.exit(f"no trained agents from results/{folder} found in sim_eval_dev.json")

    def vol(r):
        return 0.5 * r["samples"] / full["samples"] + 0.5 * r["spans"] / full["spans"]

    ranked = sorted(agents, key=lambda k: vol(agents[k]))
    for k in ranked:
        r = agents[k]
        print(f"{k:4s} volume={vol(r):6.2%} covered={r['incidents_covered']:.0f}/{r['incidents']:.0f} "
              f"changes/h={r['changes_per_service_hour']:.1f}")
    lean = ranked[: len(ranked) // 2]
    lines = [f"BAL={folder}/" + "+".join(f"{k}.npz" for k in sorted(agents)),
             f"LEAN={folder}/" + "+".join(f"{k}.npz" for k in lean)]
    with open(os.path.join(calib.RESULTS, folder, "ensembles.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
