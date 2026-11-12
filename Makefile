LAB = HOST_UID=$$(id -u) HOST_GID=$$(id -g) docker-compose run --rm lab
VARIANTS = a b c d e f g h i j a2 c2 g2 h2
ROUND ?= v3
FLOOR ?= 1

.PHONY: build up down calibrate noise train tune select eval-dev eval-test bench bench-overnight totals finetune report all

build:
	docker build -t obs-agent:dev .

up: build
	docker-compose up -d

down:
	docker-compose down -v

calibrate:
	$(LAB) python -m obsagent.calibrate

# Fit the real reading errors from earlier bench runs (needs results/**/bench_*.csv).
noise:
	$(LAB) python -m obsagent.noise_fit

# One agent, default reward weights.
train:
	$(LAB) python -m obsagent.train --episodes 300 --floor $(FLOOR)

# 14 agents with different reward weights and training seeds, in parallel (~9 min).
# The floor keeps them at L1 or above, like the rules: L0 saves ~1% and costs ~10 s of detection.
tune:
	EXTRA="--floor $(FLOOR)" ./tune.sh $(ROUND)

# Score each on dev seeds, then fix the two voting groups (results/$(ROUND)/ensembles.txt).
select:
	$(LAB) python -m obsagent.eval_sim --seeds dev --episodes 20 \
	  --agents "$(shell for v in $(VARIANTS); do printf '%s=$(ROUND)/%s.npz,' $$v $$v; done)" \
	  --out $(ROUND)/sim_eval_dev.json
	$(LAB) python -m obsagent.select_ensembles $(ROUND)

# Tune against these seeds ...
eval-dev:
	. results/$(ROUND)/ensembles.txt; $(LAB) python -m obsagent.eval_sim --seeds dev --episodes 20 \
	  --agents "agent-balanced=$$BAL,agent-lean=$$LEAN"

# ... and report these, which are never used while tuning.
eval-test:
	. results/$(ROUND)/ensembles.txt; $(LAB) python -m obsagent.eval_sim --seeds test --episodes 40 \
	  --agents "agent-balanced=$$BAL,agent-lean=$$LEAN" --out $(ROUND)/sim_eval_test.json

# One quick real-bench pass, one policy after another.
bench:
	$(LAB) python -m obsagent.run_bench --minutes 20

# Every policy on its own copy of the bench at once, 4 seeds x 60 min (~4 h).
bench-overnight: build down
	. results/$(ROUND)/ensembles.txt; OUT=bench_$(ROUND) \
	  POLICIES="static-full static-lean rules rules-burn agent-balanced=$$BAL agent-lean=$$LEAN" \
	  ./bench_parallel.sh "5301 5302 5303 5304" 60

totals:
	$(LAB) python -m obsagent.bench_totals bench_$(ROUND)

# Continue training the $(ROUND) agents on what the bench recorded (transitions_*.npz),
# mixed with fresh simulator experience. Writes results/$(ROUND)ft/.
finetune:
	$(LAB) python -m obsagent.finetune --bench $(or $(FT_BENCH),bench_$(ROUND)) --src $(ROUND) --out $(ROUND)ft

report:
	$(LAB) env BENCH=bench_$(ROUND) SIM=$(ROUND)/sim_eval_test.json python -m obsagent.report

all: up calibrate bench noise tune select eval-test bench-overnight totals report
