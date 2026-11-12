# Related work and outside results

Notes on outside work that bears on the agent. In talk text, cite these
generically (no product or model names), to keep the talk vendor-neutral.

---

## Anomaly detection for alerting: statistics vs LLMs vs decision models

Source: https://anomaly-detection.coroot.com/ (read 2026-11-10). Published
by an observability vendor, so the conclusion favours the method they ship.
The data is embedded in the page, and we recomputed the scoreboard from it
with the page's own scoring logic. The numbers below match their claim.

### Setup

- Task: given the p99 latency of a web service, decide whether to page an
  on-call engineer.
- 31 hand-labelled scenarios: 13 should page, 18 should not. None are marked
  ambiguous.
- Each detector sees the trailing 60 min at 15 s resolution and decides every
  minute between minutes 60 and 150.
- A page fires only after 3 page decisions in a row. This is a 3 min hold,
  like a `for` clause on an alert rule.
- Scoring is per scenario: correct if it paged when a human would have, or
  stayed silent when a human would not have.

### Scoreboard (recomputed, 3 min hold)

| Detector | Kind | Correct | False pages | Missed |
|---|---|---:|---:|---:|
| Burn rate: >14.4% of samples over the SLO in both the last 5 min and the last 60 min | rule | **30/31** | 1 | 0 |
| Decision model, threshold tuned on this dataset | model | 29/31 | 1 | 1 |
| Frontier chat models (4 tested) | model | 27-28/31 | 2-4 | 0-1 |
| Fixed SLO threshold (80% of last 5 min over 1000 ms) | rule | 28/31 | 0 | 3 |
| Trend slope (line fit over 30 min, projected rise) | rule | 28/31 | 0 | 3 |
| Ratio of last 5 min to preceding 55 min > 2x | rule | 27/31 | 2 | 2 |
| Robust 3-sigma (median + 3 MAD), IQR fence, generalized ESD | rule | 26-27/31 | 0-2 | 3-5 |
| EWMA control chart | rule | 18/31 | 0 | 13 |
| Textbook CUSUM / any single point over 3-sigma | rule | 16/31 | 15 | 0 |

### What it shows

- A simple two-window burn rate beat every model, including large
  general-purpose ones.
- Detectors that react to single points (any-point 3-sigma, textbook CUSUM)
  fail in the same way our first agent did: a page on every spike.
- Detectors that compare against "the last hour" drift along with slow
  incidents and miss them (EWMA, trailing p99).

### Their "should not page" scenarios

These are the shapes our simulator lacks. Today it replays only random
reading errors. Paraphrased from the page:

- isolated single-sample spikes every 10-15 min on a noisy, low-traffic p99
- a 30 s bump of 1.5x every 5 min (cron job or GC cycle)
- a 10 min sawtooth at 1.6x (periodic cache refill)
- latency jumps to 2.5x after a rollout and decays within 3 min (cold start)
- 3 min of missing data with normal latency either side (scrape gap)
- one minute pinned at the timeout ceiling, then normal
- 2 min at up to 3x, then normal
- wide random jitter between 0.6x and 2x (percentile over few requests)
- slow wandering within ±20%, or a 15% drift over 90 min
- a step *down* after a deploy (a change, but an improvement)

Scenarios that should page include a sustained 5x step, a linear climb past
the SLO, an accelerating rise, and every sample at the timeout ceiling.

### How it applies to the telemetry agent (all four done, 2026-11-10)

1. **Train on harmless shapes.** Add the shapes above to the simulator as
   events that are not incidents. The agent should learn that a spike alone
   is no reason to buy detail. This targets the switching on the real bench
   (about 32 changes per service-hour, against 15 in the simulator).
2. **Two-window inputs.** Add "share of samples over the SLA in the last
   5 min" and "in the last 60 min" to the state. Today the state has one p99
   anomaly score against a moving average.
3. **A stronger baseline.** Add a burn-rate rule next to the current
   threshold rules. If the agent costs less than both, the claim is harder
   to argue with.
4. **Where it does not transfer.** Paging can wait 3 min. Collecting detail
   cannot, or the start of the incident is lost. Keep "up fast, down slow".
   The lesson applies to *when* the agent goes up, not to adding a hold.
5. **For the talk.** It is an outside data point that large general-purpose
   models did not beat a well-chosen simple rule on a closely related task.
   That supports building a small, task-specific agent and comparing it
   against strong simple baselines. Cite it as "a public comparison of 25
   alerting detectors".
