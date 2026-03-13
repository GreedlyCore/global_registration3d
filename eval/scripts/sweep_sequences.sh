#!/usr/bin/env bash
# Sweep sequences, keep everything else from JSON config.
# Usage: bash eval/scripts/sweep_sequences.sh [config]
set -e
cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/NCLT.json}

for SEQ in 2012-01-08 2012-02-04 2012-11-17; do
    echo "=== seq=$SEQ ==="
    python eval/test.py --config "$CONFIG" --seq "$SEQ"
done
