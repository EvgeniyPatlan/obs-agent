#!/usr/bin/env bash
# Train the 14 agents that vote together, one container each, in parallel.
#
#   ./tune.sh v2            # writes results/v2/<variant>.npz and logs
#
# Ten reward-weight variants (a-j) from training seed 1, and four of them
# again from seed 2, because training luck matters as much as the weights.
set -euo pipefail
OUT=${1:-v2}
EPISODES=${EPISODES:-300}
EXTRA=${EXTRA:-}          # e.g. EXTRA="--floor 1 --double --randomize"
DOCKER_OPTS=${DOCKER_OPTS:-}  # e.g. "--cpus 1 --cpu-shares 128" while a bench runs
mkdir -p "results/$OUT"

declare -A VARIANTS=(
  [a]="--seed 1"
  [b]="--seed 1 --w-change 0.4"
  [c]="--seed 1 --w-change 0.4 --w-flap 2"
  [d]="--seed 1 --w-change 0.4 --w-cost 2"
  [e]="--seed 1 --w-cost 2"
  [f]="--seed 1 --w-change 0.3 --w-cost 3"
  [g]="--seed 1 --w-change 0.4 --w-flap 2 --w-cost 1.5"
  [h]="--seed 1 --w-change 0.25 --w-flap 2"
  [i]="--seed 1 --w-flap 2"
  [j]="--seed 1 --w-change 0.4 --w-flap 2 --w-cost 2"
  [a2]="--seed 2"
  [c2]="--seed 2 --w-change 0.4 --w-flap 2"
  [g2]="--seed 2 --w-change 0.4 --w-flap 2 --w-cost 1.5"
  [h2]="--seed 2 --w-change 0.25 --w-flap 2"
)

for v in "${!VARIANTS[@]}"; do
  # shellcheck disable=SC2086  # word splitting of DOCKER_OPTS, EXTRA and the variant flags is intended
  docker run --rm $DOCKER_OPTS -u "$(id -u):$(id -g)" -v "$PWD:/work" -w /work obs-agent:dev \
    python -m obsagent.train --episodes "$EPISODES" --out "$OUT/$v.npz" ${VARIANTS[$v]} $EXTRA \
    > "results/$OUT/$v.log" 2>&1 &
done
wait
echo "trained: $(ls results/$OUT/*.npz | wc -l) agents in results/$OUT"
