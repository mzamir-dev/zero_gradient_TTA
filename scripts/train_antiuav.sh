#!/bin/bash
# scripts/train_antiuav.sh — Train on Anti-UAV + Anti-UAV410

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " Training CTTA-OWOD on Anti-UAV + Anti-UAV410"
echo "============================================"

python tools/train.py \
    --config configs/base.yaml \
    "$@"
