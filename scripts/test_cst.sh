#!/bin/bash
# scripts/test_cst.sh — Evaluate on CST-Anti-UAV (no-TTA and TTA comparison)

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " CST-Anti-UAV Evaluation (No-TTA vs TTA)"
echo "============================================"

python tools/cst_test.py \
    --config configs/base.yaml \
    --split test \
    --compare \
    "$@"
