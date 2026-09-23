#!/usr/bin/env bash
# Usage:   bash run_dropout.sh
# Testing: EPOCHS=2 DEVICE=cpu bash run_dropout.sh
set -o pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-1000}
DEVICE=${DEVICE:-cuda}
DROPOUT=${DROPOUT:-0.1}
mkdir -p logs

# run <experiment> <hidden_layers> <neurons> <seed>
run () {
  name="$1_d${DROPOUT}_h$2_w$3_seed$4"
  echo "=== ${name} ($(date +%T)) ==="
  python3 neural_network.py \
    --experiment "$1" --hidden_layers "$2" --neurons_per_hidden_layer "$3" --dropout "$DROPOUT" \
    --learning_rate 1e-3 --batch_size 10000 --epochs "$EPOCHS" \
    --seed "$4" --device "$DEVICE" \
    2>&1 | tee -a "logs/${name}.log" \
    || echo "!!! ${name} skipped or failed (see logs/${name}.log)"
}

for seed in 0 1 2 3 4; do
  run dropout_standard 3 256 "$seed"

  run dropout_capacity 1 256 "$seed"
  run dropout_capacity 2 256 "$seed"
  run dropout_capacity 3 256 "$seed"

  run dropout_capacity 1 128 "$seed"
  run dropout_capacity 2 128 "$seed"
  run dropout_capacity 3 128 "$seed"

  run dropout_capacity 1 64 "$seed"
  run dropout_capacity 2 64 "$seed"
  run dropout_capacity 3 64 "$seed"
done
