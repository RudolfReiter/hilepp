#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for method in HILEPP SAC; do
  python -m hilepp.train_circuit --method "$method" --steps 12000 \
    --tracks Oval IMS --seed 52 --seed-offset 1000 --output checkpoints/deepsets
done
