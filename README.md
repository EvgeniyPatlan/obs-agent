# obs-agent: a self-trained agent that decides what telemetry to collect

A deep Q-network agent that sets, per service and every 5 seconds:
- which metric groups are collected (core / detail / high-cardinality debug),
- the scrape interval,
- the trace sampling rate.

It collects as little as possible while things are calm and switches to full
detail when an incident starts.

- Built and trained from scratch: about 200 lines of NumPy for the network and
  the learning loop. It uses no ML framework and no pretrained or public model.
- Everything runs in Docker. You need Docker and docker-compose, and nothing else.
- It is a research prototype. It shows the idea works on a small bench; it is
  not a production tool.

## Results: 12 bench hours, every policy on its own copy of the bench

| Policy | Volume vs collect-everything | Incidents diagnosable | Time to diagnostic detail | Changes / service-hour |
|---|---:|---:|---:|---:|
| Threshold rules | 6.38% | 219/225 (97.3%) | 10.8 s | 9.1 |
| Burn-rate rules | 5.89% | 216/230 (93.9%) | 20.6 s | 9.3 |
| **Agent, lean** | **5.93%** | **236/239 (98.7%)** | **7.4 s** | 22.6 |

The lean agent costs the same as the best rule set. It misses a fifth as many
incidents (1.3% against 6.1%) and reaches diagnostic detail almost 3 times
sooner. It is cheaper than threshold rules on 10 of 12 hours and better on every
column. It also changes settings more often than the rules do.

Full tables, per-hour numbers and charts: [`results/RESULTS.md`](results/RESULTS.md).
What worked, what did not, and why: [`docs/how-to-build-the-agent.md`](docs/how-to-build-the-agent.md), section 9.

## Run it

```
make build            # one image for everything
make up               # bench: 3 services, load generator, trace sink, Prometheus
make calibrate        # measure the bench so the simulator matches it
make bench            # a first real-bench pass, to record real readings
make noise            # fit the real reading errors from that pass
make tune             # train 14 agents in the simulator, in parallel (~9 min)
make select           # score them on dev seeds, fix the two voting groups
make eval-test        # report on seeds never used while tuning
make bench-overnight  # every policy on its own bench copy, 4 seeds x 60 min (~4 h)
make totals           # per-seed totals
make finetune         # optional: continue training on what the bench recorded
make report           # results/RESULTS.md + results/charts/*.png
make down
```

`make all` runs the whole pipeline. The trained round-3 agents are in
`results/v3/`, so you can skip training and run `make bench-overnight` against them.

## What is in the repository

| Path | What |
|---|---|
| `obsagent/nn.py` | neural network and Adam optimiser, hand-written |
| `obsagent/dqn.py` | DQN: replay buffer, target network, Double DQN, epsilon-greedy |
| `obsagent/config.py` | telemetry levels L0-L4, SLA |
| `obsagent/agent_core.py` | state vector, reward, cost model |
| `obsagent/sim.py` | simulator used for training, calibrated from the bench |
| `obsagent/bench_env.py` | real environment: scrape config and reload, sampling, faults |
| `obsagent/runner.py` | control loop and metrics |
| `obsagent/policies.py` | the agent and the rule baselines |
| `obsagent/finetune.py` | continue training on recorded bench experience |
| `bench/` | services, load generator, trace sink |
| `results/` | calibration, trained agents (`v3/`), per-seed bench totals, charts |
| `docs/how-to-build-the-agent.md` | step-by-step guide to build and train your own |
| `docs/related-work.md` | notes on related work |

Raw per-tick bench data (12 seeds, about 45 MB) is attached to the first release.

## Known limits

- The bench is small: 3 services at about 40-200 requests per second. The
  numbers show feasibility; they are not production claims.
- The agent is trained in a simulator calibrated from the bench, then run on the real bench.
- Fine-tuning on recorded bench data made the agent calmer (about 35% fewer
  changes) but not cheaper.

## License

MIT, see [LICENSE](LICENSE).
