#!/bin/sh
# Run every policy on its own copy of the bench at the same time, one seed
# after another. Each copy is a separate compose project with its own
# network, Prometheus and services, so the policies do not share state.
#
#   ./bench_parallel.sh "5101 5102 5103 5104" 60
set -eu
SEEDS=${1:-"5101 5102 5103 5104"}
MINUTES=${2:-60}
GAP=${GAP:-150}
AGENT=${AGENT:-agent.npz}
POLICIES=${POLICIES:-"static-full static-lean rules dqn-guarded"}
OUT=${OUT:-bench}            # results/<OUT>/<seed>/
export HOST_UID=$(id -u) HOST_GID=$(id -g)

mkdir -p "results/$OUT"
i=0
for p in $POLICIES; do
  PROM_PORT=$((9100 + i)) docker-compose -p obsb$i up -d >/dev/null 2>&1
  i=$((i + 1))
done
sleep 20

for seed in $SEEDS; do
  echo "=== seed $seed: $MINUTES min, all policies in parallel ($(date +%H:%M))"
  i=0
  for p in $POLICIES; do
    PROM_PORT=$((9100 + i)) docker-compose -p obsb$i run --rm lab python -m obsagent.run_bench \
      --policies "$p" --seed "$seed" --minutes "$MINUTES" --fault-gap "$GAP" \
      --agent "$AGENT" --out-dir "$OUT/$seed" > "results/$OUT/log_${seed}_${p%%=*}.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  tail -n 1 results/$OUT/log_${seed}_*.log
done

i=0
for p in $POLICIES; do
  docker-compose -p obsb$i down -v >/dev/null 2>&1
  i=$((i + 1))
done
echo "done $(date +%H:%M)"
