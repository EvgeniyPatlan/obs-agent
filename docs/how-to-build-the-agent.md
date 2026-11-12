# How to build and train your own telemetry agent

This guide walks through building an agent that decides **what telemetry to
collect, how often to scrape it, and how many traces to keep**. You train
it yourself, from zero, on your own data. It uses no pretrained or public
model and no ML framework: the network and the learning loop are about 200 lines of NumPy.

Everything runs in Docker. You need Docker and docker-compose, and nothing else.

```
make build            # one image for everything
make up               # bench: 3 services, load generator, trace sink, Prometheus
make calibrate        # measure the bench so the simulator matches it
make bench            # a first real-bench pass, to record real readings
make noise            # fit the real reading errors from that pass
make tune             # train 14 agents in the simulator, in parallel (~8 min)
make select           # score them on dev seeds, fix the two voting groups
make eval-test        # report these seeds, never used while tuning
make bench-overnight  # every policy on its own bench copy, 4 seeds x 60 min (~4 h)
make finetune         # continue training on what the bench recorded (step 9)
make report           # charts + RESULTS.md
```

---

## 1. Frame the problem as decisions

Reinforcement learning fits when something has to make **repeated decisions
whose effects show up later**. Telemetry fits well:

- Turn detail **down** and you save money. But when an incident starts you
  are blind, and the data you need for root cause was never collected.
- Turn detail **up** and you can see everything, but you pay for it all day.

Static settings pick one point on this trade-off forever. Threshold rules
react only *after* the problem is visible, which means after the start of
the incident is already lost. An agent can learn to turn detail up when the
early signs appear.

## 2. Define the knob: telemetry levels

Keep the action space small. We use one knob per service with five levels.
Each level bundles three settings (`obsagent/config.py`):

| Level | Scrape interval | Metric groups | Trace sampling |
|---|---|---|---|
| L0 minimal | 30 s | core | 0.5% |
| L1 lean | 15 s | core | 1% |
| L2 standard | 10 s | core + detail | 5% |
| L3 diagnostic | 5 s | core + detail | 25% |
| L4 full | 2 s | core + detail + debug | 100% |

- **core**: a few series the agent always needs (requests, errors, latency
  histogram, busy time).
- **detail**: per-route histograms (hundreds of series).
- **debug**: per-tenant, high-cardinality series (thousands).

The agent's actions are **-2, -1, 0, +1, +2 levels**. Five discrete actions
make a small Q-network enough. One shared network decides for every service,
using that service's own state, so adding services needs no retraining.

How the knob is applied (`obsagent/bench_env.py`):
- Scrape interval: per-job `scrape_interval` in the Prometheus config, then
  `POST /-/reload`.
- Which metrics: `metric_relabel_configs` with `action: drop` on group prefixes.
- Trace sampling: a small control endpoint on each service.

## 3. Define what the agent sees: the state

15 numbers per service, all computed **only from telemetry the agent itself
chose to collect** (`obsagent/agent_core.py`):

`p99/SLA, p50/SLA, errors/SLA, request rate, change in rate, utilisation,
current level, time since last change, data staleness, p99 anomaly vs. its
moving average, worst p99 and worst errors across all services, share of
readings over the SLA in the last 1, 5 and 60 minutes.`

This is the important part: at lean levels the agent's own view is stale
and smoothed. The agent learns to weigh that blindness, which is why
"data staleness" is an input.

Normalise every feature to roughly 0..1 and clip outliers. A small network
cannot cope with raw values like 40 000 requests/s next to 0.002.

## 4. Define what "good" means: the reward

The reward is where you write down your trade-off. Ours has the same
structure as a scaling agent (SLA + cost + stability + bonus):

```
r = - cost_fraction                         # every sample and span costs
    - 5   if incident and level < L2         # blind during an incident
    - 2   if incident and level == L2        # partially blind
    + 0.3 if incident and level >= L3        # right call
    - 1   if an incident starts within 30 s and level < L2   # root-cause data lost
    + 0.3 if calm and level <= L1            # right call
    - 0.15 for any change, -1 more if changed again within 30 s   # no flapping
```

**The reward may use hindsight, the state may not.** "An incident starts
within 30 s" is known only afterwards, so rewards are computed when an
episode ends (`obsagent/runner.py`). The agent never sees it as an input.
It has to learn which early signs come before incidents.

Tuning notes from our runs:
- With a flapping penalty of only 0.5, the agent changed level 48 times per
  service-hour. Adding a per-change cost and a 30 s cooldown cut that to about 23.
- Set the costs relative to each other, not as absolute values. Ask "how many
  hours of full telemetry would I pay to avoid one blind incident?"

## 5. Build the network yourself

`obsagent/nn.py` is a fully connected network written from scratch:
`15 -> 64 -> 32 -> 5` with ReLU, He initialisation, manual backpropagation,
and Adam with gradient clipping. Check your backprop numerically once:

```
max relative gradient error 3.3e-08   # analytic vs finite differences
```

`obsagent/dqn.py` adds the standard DQN parts:
- **Experience replay** (50 000 transitions) breaks the correlation between
  consecutive ticks.
- **A target network**, synced every 500 steps, keeps the learning target still.
- **Epsilon-greedy** exploration (0.5 → 0.02) tries random actions early, fewer later.
- **Huber TD loss**: one bad reward spike does not blow up training.
- **γ = 0.95**: with 5 s ticks the horizon is about 100 s, which is the time
  scale of an incident build-up.

## 6. Train in a simulator first, calibrated from the real system

Training on live systems is slow: one hour of experience takes one hour.
Instead:

1. Measure the real bench (`make calibrate`): series per metric group and
   spans per request.
2. Build a simulator with the same structure (`obsagent/sim.py`): chained
   services with worker pools, queueing (latency grows sharply near
   saturation), bursty heavy-tailed load (Pareto, α = 1.5), and slow-ramping
   faults (latency, errors, CPU).
3. Train 300 simulated hours in about 4 minutes on a CPU (`make train`).
4. Evaluate on **unseen** seeds (`make eval-sim`), then on the **real** bench
   (`make bench`). The real bench is the number you report.

The simulator also has to model *observation*, not just the system. Telemetry
is refreshed only at scrape times and averaged over a rate window. Without
that, the agent would learn that lean telemetry is free.

It also has to model **how wrong the readings are**. Our first simulator
assumed readings close to the truth. On the real bench, a p99 computed from
histogram buckets read 5-50% high, and 1-3% of calm readings spiked to three
times the true value, enough to look like an SLA breach. An agent trained
without that jumped to full detail on every spike. `make noise`
(`obsagent/noise_fit.py`) records the gap between each real reading and the
truth during calm periods, and the simulator replays those gaps. With them,
the simulator reproduced the real problem (59-72 level changes per
service-hour, against 49-54 on the bench), and retraining fixed it.

### Check the simulator against the bench, number by number

Our agent looked fine in the simulator and still came out slow on the bench:
about 26 s to reach diagnostic detail against 12 s for the rules. The cause
was a simulator bug, found by putting the same numbers side by side:

| | Real bench | First simulator | Fixed simulator |
|---|---:|---:|---:|
| Calm p99: gateway / api / store | 57 / 49 / 35 ms | 164 / 97 / 38 ms | 59 / 53 / 38 ms |
| p99 in the 30 s before an incident | flat at 0.2 of the SLA, then a jump | slow ramp from 0.4 to 0.5 | flat at 0.2, then a jump |
| Delay until a reading shows the breach, L0 / L1 | 25 / 15 s | 25 / 13 s | 28 / 16 s |

The simulator added each downstream p99 on top of a tail that already
contained it. Calm latency upstream came out 2-3 times too high, close
enough to the SLA that incidents showed a slow early ramp. The agent learned
to wait for that ramp, and on the bench, where incidents start from a low
baseline with no warning, it sat at the cheapest level and was caught there.
After the fix (means add up, tails combine as independent parts,
`sim._chain`), the old agent did in the simulator what it had done on the
bench: about 4% volume, 27 s to diagnostic detail, 34 changes per
service-hour.

Measure these four things on both sides before you trust a simulator:
calm latency per service, the shape just before incidents, the delay of a
reading at each level, and incidents per hour.

### Teach it what is not an incident

Add harmless shapes as well as faults: single spikes, a 30 s bump every few
minutes from a cron job or garbage collection, a cold start after a deploy,
a short burst, a gap in scraping (`workload.benign_schedule`, taken from a
public comparison of alerting detectors, see [related-work.md](related-work.md)).
Give the agent the share of readings over the SLA in the last 1, 5 and 60
minutes, as multi-window burn-rate alerting does. One spike moves only the
short window, a real incident moves all of them.

## 7. Tune on one set of seeds, report another

Two disjoint sets of simulator seeds: `dev` for tuning and `test` for the
number you report. Look at `test` once, after the choices are made.

Training luck matters as much as the reward weights: the same weights
trained from two random seeds landed at different points of the cost and
readiness trade-off. So train several agents and let them **vote** by
averaging their Q-values (`a.npz+b.npz+...` in `--agents`). The vote
switched levels about half as often as any single agent (15 against 20-30
changes per service-hour) and detected incidents sooner.

The vote also lets you choose where to sit on the trade-off. Fix the
choice before the test runs. Ours: every trained agent votes ("balanced"),
and the cheapest half on dev seeds votes ("lean", `obsagent/select_ensembles.py`).
On 40 held-out simulated hours, about 840 incidents per policy:

| Policy | Volume vs full | Diagnosable | Time to diagnostic detail |
|---|---:|---:|---:|
| Threshold rules | 7.9% | 98.6% | 9.4 s |
| Burn-rate rules | 6.8% | 95.9% | 20.2 s |
| Agent, balanced | 6.7% | 99.1% | 6.0 s |
| Agent, lean | 5.9% | 98.9% | 7.8 s |

These agents use a floor at L1 (`make tune FLOOR=1`), see "Where the cost
goes" below.

## 8. Compare against honest baselines

Always run the same load and the same fault schedule (same seed) for:
- **collect everything**: the coverage ceiling and the cost ceiling
- **always lean**: the cost floor
- **threshold rules**: "if the SLA breaks, go diagnostic for 2 minutes".
- **burn-rate rules**: the same, but only when at least 30% of the last
  minute was over the SLA too, so a single spike is ignored. This is the
  stronger baseline to beat.

Report cost (samples ingested and spans), incidents you could diagnose,
**detail ready at incident start**, time to detect, and how often settings
changed.

### Real bench, 12 seeds x 60 min (every policy on its own copy at once)

| Policy | Volume vs full | Diagnosable | Ready at start | Time to diagnostic detail | Changes / service-hour |
|---|---:|---:|---:|---:|---:|
| Threshold rules | 6.38% | 219/225 (97.3%) | 29% | 10.8 s | 9 |
| Burn-rate rules | 5.89% | 216/230 (93.9%) | 30% | 20.6 s | 9 |
| Agent, balanced | 6.82% | 233/236 (98.7%) | 52% | 6.4 s | 19 |
| Agent, lean | 5.93% | 236/239 (98.7%) | 38% | 7.4 s | 23 |

The lean agent beats threshold rules on every column (cheaper on 10 of 12
hours). Against burn-rate rules it costs the same (cheaper on 8 of 12
hours, 0.04 points more in total), misses a fifth as many incidents, and
reaches diagnostic detail almost three times sooner.

## 9. Where the cost goes, and what did and did not help

**Measure where the money goes before tuning.** On our bench, 95% of the
telemetry cost was time spent at L3. One tick costs about 0.002 at L0,
0.005 at L1, 0.04 at L2 and 0.13 at L3 (share of full collection). So the
question is never "how low can it go when calm", only "how long does it stay
at L3, and when does it go up".

What helped:
- **A floor at L1.** L0 saves about 1% of volume but adds about 10 s before
  a reading shows a breach, so the agent climbed late and stayed high
  longer. With the floor, the agent got cheaper *and* faster.
- **Retrying dropped connections** in the bench driver (`bench_env.http_json`).
  One reset socket ended an hour-long run.

What did not help, and why it was worth trying:
- **Double DQN** and **domain randomisation** (a slightly different
  simulated world each episode: latency, load, reading noise, fault
  strength, harmless spikes; `train --double --randomize`). The agents
  landed on the same trade-off, a bit faster and 5-10% dearer. The current
  agents already held up in randomised worlds, so the simulator could no
  longer reproduce what goes wrong on the bench.
- **More episodes.** Reward stops improving at about episode 150, when
  exploration reaches its minimum.

**The remaining gap is bench-only:** about twice as many level changes on
the bench as in the simulator, mostly dropping and climbing back near
incidents. So we trained on the bench's own experience: every bench run
saves each decision of every policy (`transitions_<policy>.npz`), and
`make finetune` continues training the agents on that, mixed with fresh
simulator experience. Rewards are recomputed from the per-tick CSV with each
agent's own reward weights. To get varied data, run two extra copies of the
agent that take a random action 15% of the time (`label=weights@0.15`).

Result, 4 new bench hours: fine-tuning cut level changes by about 35% (21 to
13-14 per service-hour), but the agent was not cheaper (5.74-5.85% against
5.61% without it). After the L1 floor, switching no longer drove the cost.
Keep the simpler agent, and know that a calmer one is a fine-tune away if
reloads are expensive in your system.

**Lesson about the pipeline itself:** the bench runs code baked into the
image. One 3-hour run used an image built the night before, without the
floor. Rebuild before every bench run (`make bench-overnight` does).

## Adapting it to your system

1. Split your metrics into groups by cost (cheap core / detail / high-cardinality debug).
2. Define 3–6 levels as bundles of interval + groups + sampling.
3. Keep the state to 10–15 normalised numbers taken from core telemetry.
4. Write the reward as your own cost/blindness trade-off.
5. Calibrate a simulator from a day of your own metrics, train, then shadow-run:
   log the agent's decisions next to your current settings before letting it act.

Things to watch:
- **Sim-to-real gap.** Always validate on the real system. In our first real run
  the agent used 27% less telemetry than threshold rules and kept every incident
  diagnosable. But it switched level 54 times per service-hour, against 23 in
  the simulator, because real readings are noisier than simulated ones. In the
  second run that switching made it 45% more expensive than the rules. Replaying
  the measured reading errors in the simulator (step 6) cut the switching. A
  4-hour run then showed the agent 38% cheaper than the rules and slow: 26 s
  to diagnostic detail against 12 s. The latency bug in step 6 explained that.
- **Run the real bench in parallel.** One copy of the bench per policy, all
  started on the same seed (`bench_parallel.sh`). Four seeds of 60 minutes
  take about 4 hours instead of 16, and every policy sees the same machine load.
- **Guard rails.** Keep a floor level that is never turned off (core metrics).
  Use "up fast, down slow": the agent may always raise detail, but may lower it
  only after holding a level for 30 s (`DQNPolicy(dwell_down=6)`). Detail is
  never delayed. On its own this did *not* stop the switching (49 changes per
  service-hour): the agent then jumped up on noise instead. Training on real
  reading errors did.
- **Explainability.** Log Q-values with every decision, so "why did it turn
  tracing up?" has an answer.

Outside results that shaped the next steps are in
[related-work.md](related-work.md).
