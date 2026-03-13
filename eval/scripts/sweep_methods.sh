#!/usr/bin/env bash
# Sweep registration methods, keep everything else from JSON config.
# Usage: bash eval/scripts/sweep_methods.sh [config]
set -e
cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/NCLT.json}
OUT=results/sweep_methods

for METHOD in teaser mac kiss; do
    echo "=== reg=$METHOD ==="
    python eval/test.py --config "$CONFIG" --reg "$METHOD" --out_dir "$OUT"
done
