#!/usr/bin/env bash
# Sweep voxel sizes, keep everything else from JSON config.
# Usage: bash eval/scripts/sweep_voxel.sh [config]
set -e
cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/NCLT.json}

for VS in 0.1 0.3 0.5 1.0; do
    echo "=== voxel_size=$VS ==="
    python eval/test.py --config "$CONFIG" --voxel_size "$VS"
done
