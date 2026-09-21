#!/usr/bin/env bash
# Usage:   bash run_sweep.sh
# Testing: EPOCHS=2 DEVICE=cpu bash run_sweep.sh
# Credit to Claude for writing this script

set -o pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-1000}
DEVICE=${DEVICE:-cuda}
mkdir -p logs

for seed in 0 1 2 3 4; do
  for layers in 1 2 3; do
    for width in 32 64 128 256; do
      name="h${layers}_w${width}_seed${seed}"
      echo "=== ${name} ($(date +%T)) ==="
      python3 neural_network.py \
        --hidden_layers "$layers" --neurons_per_hidden_layer "$width" \
        --learning_rate 1e-3 --batch_size 10000 --epochs "$EPOCHS" \
        --seed "$seed" --device "$DEVICE" \
        2>&1 | tee -a "logs/${name}.log" \
        || echo "!!! ${name} skipped or failed (see logs/${name}.log)"
    done
  done
done
