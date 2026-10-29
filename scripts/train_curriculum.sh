#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for method in HILEPP SAC; do
  python -m hilepp.train_circuit --method "$method" --track IMS --steps 12000 \
    --speed-ceiling 34 --seed 42 --output checkpoints/pretraining
  python -m hilepp.train_circuit --method "$method" --track IMS --steps 6000 \
    --speed-ceiling 28 --seed 43 --seed-offset 5000 \
    --initialize checkpoints/pretraining --output checkpoints/circuit
done
