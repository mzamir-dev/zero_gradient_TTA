#!/bin/bash
# scripts/test_antiuav.sh — Test on Anti-UAV and Anti-UAV410 test sets

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " Testing on Anti-UAV test set"
echo "============================================"
python tools/test.py \
    --config configs/base.yaml \
    --dataset antiuav \
    --split test \
    "$@"

echo ""
echo "============================================"
echo " Testing on Anti-UAV410 test set"
echo "============================================"
python tools/test.py \
    --config configs/base.yaml \
    --dataset antiuav410 \
    --split test \
    "$@"
