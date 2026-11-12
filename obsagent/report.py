"""Turn results into slide-ready charts and a results table.

    python -m obsagent.report
Writes results/charts/*.png and results/RESULTS.md.
"""
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import calib  # noqa: E402
from .config import LEVELS, TICK_S  # noqa: E402

# Colour follows the policy, always the same slot (validated reference palette).
COLOR = {"dqn": "#2a78d6", "dqn-guarded": "#2a78d6", "agent-balanced": "#2a78d6", "agent-lean": "#e87ba4",
         "agent-v1": "#008300", "rules": "#eb6834", "rules-burn": "#6250d6",
         "static-full": "#1baf7a", "static-lean": "#eda100",
         # round 3: same slots, so the validated palette and its adjacent pairs are unchanged
         "agent-v2-lean": "#008300", "agent-v3-balanced": "#2a78d6", "agent-v3-lean": "#e87ba4"}
LABEL = {"dqn": "DQN agent", "dqn-guarded": "DQN agent + guard rail", "agent-balanced": "Agent, balanced",
         "agent-lean": "Agent, lean", "agent-v1": "First agent (old simulator)",
         "rules": "Threshold rules", "rules-burn": "Burn-rate rules", "static-full": "Collect everything",
         "static-lean": "Always lean", "agent-v2-lean": "Agent, lean (round 2)",
         "agent-v3-balanced": "Agent, balanced", "agent-v3-lean": "Agent, lean"}
# Bar order keeps orange (rules) away from yellow and magenta: validated adjacent pairs.
ORDER = ["static-lean", "static-full", "rules", "rules-burn", "agent-v1", "dqn", "dqn-guarded",
         "agent-balanced", "agent-lean", "agent-v2-lean", "agent-v3-balanced", "agent-v3-lean"]
INK, INK2, GRID, SURFACE, INCIDENT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f6c9c9"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 12})


def _out(name):
    d = os.path.join(calib.RESULTS, "charts")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def training_chart():
    path = os.path.join(calib.RESULTS, "training.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    r = np.array([float(x["mean_reward"]) for x in rows])
    k = 10
    smooth = np.convolve(r, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(range(len(r)), r, color=COLOR["dqn"], alpha=0.25, linewidth=1)
    ax.plot(range(k - 1, len(r)), smooth, color=COLOR["dqn"], linewidth=2)
    ax.set_title("The agent learns from scratch: reward per simulated hour", loc="left")
    ax.set_xlabel("training episode (1 simulated hour each)")
    ax.set_ylabel("mean reward per decision")
    fig.tight_layout()
    fig.savefig(_out("training_curve.png"), dpi=160)
    plt.close(fig)
    return "training_curve.png"


def compare_chart(results, title, fname):
    pols = [p for p in ORDER if p in results]
    full = results["static-full"]
    samples_key = "prometheus_samples_total" if "prometheus_samples_total" in full else "samples"
    spans_key = "spans_total" if "spans_total" in full else "spans"
    vol = [100 * (0.5 * results[p][samples_key] / full[samples_key] + 0.5 * results[p][spans_key] / full[spans_key])
           for p in pols]
    ready = [100 * results[p]["incidents_ready_at_start"] / max(results[p]["incidents"], 1) for p in pols]
    covered = [100 * results[p]["incidents_covered"] / max(results[p]["incidents"], 1) for p in pols]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panels = [(vol, "Telemetry volume\n(% of collect-everything)"),
              (covered, "Incidents diagnosable\n(% with full detail at some point)"),
              (ready, "Detail ready at incident start\n(% of incidents)")]
    y = np.arange(len(pols))
    for ax, (vals, name) in zip(axes, panels):
        ax.barh(y, vals, color=[COLOR[p] for p in pols], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", color=INK, fontsize=11)
        ax.set_xlim(0, 118)
        ax.set_title(name, loc="left", fontsize=12)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [LABEL[p] for p in pols])
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(_out(fname), dpi=160)
    plt.close(fig)
    return fname


BENCH_DIR = os.environ.get("BENCH", "bench")   # results/<BENCH>/<seed>/: which parallel run to report


def load_bench_seeds():
    """results/<BENCH>/<seed>/summary_<policy>.json -> {seed: {policy: summary}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(calib.RESULTS, BENCH_DIR, "*", "summary_*.json"))):
        seed = os.path.basename(os.path.dirname(path))
        out.setdefault(seed, {}).update(json.load(open(path)))
    out = {k: v for k, v in out.items() if "static-full" in v}
    # Totals are only comparable over the same seeds: drop policies missing from any seed.
    every = set.intersection(*(set(v) for v in out.values())) if out else set()
    partial = sorted(set().union(*(set(v) for v in out.values())) - every) if out else []
    if partial:
        print(f"report: left out (not run on every seed): {', '.join(partial)}")
    return {k: {p: r for p, r in v.items() if p in every} for k, v in out.items()}


def combine(per_seed):
    """Add up counts across seeds; average times weighted by incidents."""
    total = {}
    for runs in per_seed.values():
        for p, r in runs.items():
            t = total.setdefault(p, {"_w": 0.0, "_detect": 0.0, "_diag": 0.0, "_ch": []})
            for k in ("prometheus_samples_total", "spans_total", "incidents", "incidents_covered",
                      "incidents_ready_at_start"):
                t[k] = t.get(k, 0) + r[k]
            n = r["incidents"]
            if not math.isnan(r["mean_detect_s"]):
                t["_detect"] += r["mean_detect_s"] * n
                t["_w"] += n
            if not math.isnan(r["mean_time_to_diagnostic_s"]):
                t["_diag"] += r["mean_time_to_diagnostic_s"] * n
            t["_ch"].append(r["changes_per_service_hour"])
    for t in total.values():
        t["mean_detect_s"] = t["_detect"] / t["_w"] if t["_w"] else float("nan")
        t["mean_time_to_diagnostic_s"] = t["_diag"] / t["_w"] if t["_w"] and t["_diag"] else float("nan")
        t["changes_per_service_hour"] = float(np.mean(t["_ch"]))
    return total


def volume(r, full):
    return 100 * (0.5 * r["prometheus_samples_total"] / full["prometheus_samples_total"]
                  + 0.5 * r["spans_total"] / full["spans_total"])


def per_seed_table(per_seed):
    pols = [p for p in ORDER if all(p in runs for runs in per_seed.values())]
    lines = ["| Seed | " + " | ".join(LABEL[p] for p in pols) + " |", "|---|" + "---:|" * len(pols)]
    for seed, runs in per_seed.items():
        cells = [f"{volume(runs[p], runs['static-full']):.1f}% · {runs[p]['incidents_covered']}/{runs[p]['incidents']}"
                 for p in pols]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timeline_chart(folder=None):
    folder = folder or calib.RESULTS
    agent = next((a for a in ("agent-balanced", "dqn-guarded", "dqn")
                  if os.path.exists(os.path.join(folder, f"bench_{a}.csv"))), "dqn")
    files = {p: os.path.join(folder, f"bench_{p}.csv") for p in ("rules", agent)}
    if not all(os.path.exists(f) for f in files.values()):
        return None
    data = {p: list(csv.DictReader(open(f))) for p, f in files.items()}
    services = sorted({r["service"] for r in data[agent]})
    fig, axes = plt.subplots(len(services), 1, figsize=(13, 2.2 * len(services)), sharex=True)
    for ax, svc in zip(axes, services):
        for p in ("rules", agent):
            rows = [r for r in data[p] if r["service"] == svc]
            t = np.array([float(r["t"]) for r in rows]) / 60
            lv = np.array([int(r["level"]) for r in rows])
            ax.step(t, lv, where="post", color=COLOR[p], linewidth=2, label=LABEL[p])
            if p == agent:
                inc = np.array([r["incident"] == "True" for r in rows])
                for i in np.where(inc)[0]:
                    ax.axvspan(t[i], t[i] + TICK_S / 60, color=INCIDENT, linewidth=0, zorder=0)
        ax.set_yticks(range(len(LEVELS)), [lv.name.split()[0] for lv in LEVELS])
        ax.set_ylim(-0.3, len(LEVELS) - 0.7)
        ax.set_title(svc, loc="left", fontsize=11, color=INK2)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("minutes (shaded: SLA broken)")
    fig.suptitle("Telemetry level chosen per service on the real bench, same load and faults", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(_out("bench_timeline.png"), dpi=160)
    plt.close(fig)
    return "bench_timeline.png"


def table(results, cost_keys):
    full = results["static-full"]
    lines = ["| Policy | Samples | Spans | Volume vs full | Incidents diagnosable | Ready at start | "
             "Detect (s) | To full detail (s) | Changes / service-hour |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sk, pk = cost_keys
    for p in [p for p in ORDER if p in results]:
        r = results[p]
        vol = 100 * (0.5 * r[sk] / full[sk] + 0.5 * r[pk] / full[pk])
        lines.append(f"| {LABEL[p]} | {r[sk]:,.0f} | {r[pk]:,.0f} | {vol:.1f}% | "
                     f"{r['incidents_covered']:.0f}/{r['incidents']:.0f} | {r['incidents_ready_at_start']:.0f} | "
                     f"{r['mean_detect_s']:.1f} | {r['mean_time_to_diagnostic_s']:.1f} | {r['changes_per_service_hour']:.1f} |")
    return "\n".join(lines)


def main():
    md = ["# Results", ""]
    ch = training_chart()
    if ch:
        md += ["## Training", f"![training](charts/{ch})", ""]
    sim = os.path.join(calib.RESULTS, os.environ.get("SIM", "sim_eval_test.json"))
    if not os.path.exists(sim):
        sim = os.path.join(calib.RESULTS, "sim_eval.json")
    if os.path.exists(sim):
        s = json.load(open(sim))
        noise = "with measured reading errors" if s.get("noise") else "without reading errors"
        md += [f"## Simulator: {s['episodes']} held-out episodes x {s['minutes']:.0f} min, {noise}", "",
               table(s["results"], ("samples", "spans")), "",
               f"![sim](charts/{compare_chart(s['results'], 'Simulator, unseen hours', 'sim_compare.png')})", ""]
    per_seed = load_bench_seeds()
    bench = os.path.join(calib.RESULTS, "bench_summary.json")
    if per_seed:
        b = combine(per_seed)
        minutes = next(iter(next(iter(per_seed.values())).values())).get("minutes", 0)
        md += [f"## Real bench: {len(per_seed)} seeds x {minutes:.0f} min per policy, each policy on its own copy "
               "of the bench, same load and faults", "",
               table(b, ("prometheus_samples_total", "spans_total")), "",
               f"![bench](charts/{compare_chart(b, f'Real bench, {len(per_seed)} seeds', 'bench_compare.png')})", "",
               "Per seed: volume vs collect-everything · incidents diagnosable", "", per_seed_table(per_seed), ""]
        tl = timeline_chart(os.path.join(calib.RESULTS, BENCH_DIR, next(iter(per_seed))))
        if tl:
            md += [f"![timeline](charts/{tl})", ""]
    elif os.path.exists(bench):
        b = json.load(open(bench))
        if "static-full" in b:
            md += ["## Real bench (Docker, Prometheus): same load and fault schedule for every policy", "",
                   table(b, ("prometheus_samples_total", "spans_total")), "",
                   f"![bench](charts/{compare_chart(b, 'Real bench', 'bench_compare.png')})", ""]
        tl = timeline_chart()
        if tl:
            md += [f"![timeline](charts/{tl})", ""]
    md += ["Definitions: an incident is 3+ consecutive 5 s ticks with true p99 > 300 ms or errors > 2% on a service.",
           "Diagnosable = level L3 or higher (detail metrics, 5 s scrape, 25% traces). Detect = time until the",
           "collected telemetry itself shows the breach. Samples = samples ingested by Prometheus after relabelling."]
    with open(os.path.join(calib.RESULTS, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
