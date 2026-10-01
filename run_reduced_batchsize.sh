#!/usr/bin/env bash
# Usage:   bash run_reduced_batchsize.sh
# Testing: EPOCHS=2 DEVICE=cpu bash run_reduced_batchsize.sh
set -o pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-1000}
DEVICE=${DEVICE:-cuda}
mkdir -p logs

# run <hidden_layers> <neurons> <seed>
run () {
  name="reduced_batchsize_bn_h$1_w$2_seed$3"
  echo "=== ${name} ($(date +%T)) ==="
  python3 neural_network.py \
    --experiment reduced_batchsize --hidden_layers "$1" --neurons_per_hidden_layer "$2" \
    --learning_rate 1e-3 --batch_size 1000 --epochs "$EPOCHS" \
    --seed "$3" --device "$DEVICE" \
    2>&1 | tee -a "logs/${name}.log" \
    || echo "!!! ${name} skipped or failed (see logs/${name}.log)"
}

for seed in 0 1 2 3 4; do
  run 1 256 "$seed"
  run 2 256 "$seed"
  run 3 256 "$seed"

  run 1 128 "$seed"
  run 2 128 "$seed"
  run 3 128 "$seed"

  run 1 64 "$seed"
  run 2 64 "$seed"
  run 3 64 "$seed"
done
