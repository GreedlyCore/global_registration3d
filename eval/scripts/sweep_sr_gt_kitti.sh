#!/usr/bin/env bash
# Sweep KITTI (01/04) for SR vs GT-distance analysis.
# Usage:
#   bash eval/scripts/sweep_sr_gt_kitti.sh [config]
# Notes:
#   - Uses random pairing with GT-distance filtering in eval/test.py.
#   - Pair generation cycles temporal offsets in [1..20], then keeps pairs
#     whose GT distance is within [dist_min, dist_max].
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/KITTI.json}
BASE_OUT=${BASE_OUT:-results/sr_gt_sweep}
TEST_COUNT=${TEST_COUNT:-100}
SEED=${SEED:-42}

SEQS=(01 04)
FEATS=(FasterPFH)
REGS=(mac teaser kiss)
# Explicit GT-distance bins (m) for three-bin evaluation.
DIST_BINS=("0-50" "50-100" "100-200")
VOXELS=(0.4 0.5 0.6 0.7 0.8 0.9)

echo "[sweep] config=$CONFIG"
echo "[sweep] out_root=$BASE_OUT"
echo "[sweep] seqs=${SEQS[*]} feats=${FEATS[*]} regs=${REGS[*]} dist_bins=${DIST_BINS[*]} voxels=${VOXELS[*]}"

for SEQ in "${SEQS[@]}"; do
  for FEAT in "${FEATS[@]}"; do
    for REG in "${REGS[@]}"; do
      for BIN in "${DIST_BINS[@]}"; do
        DIST_MIN="${BIN%-*}"
        DIST_MAX="${BIN#*-}"
        for VOX in "${VOXELS[@]}"; do
          OUT_DIR="$BASE_OUT/dist_${DIST_MIN}_${DIST_MAX}/voxel_${VOX}"
          echo "=== seq=$SEQ feat=$FEAT reg=$REG dist_range=[${DIST_MIN},${DIST_MAX}]m voxel_size=$VOX test_count=$TEST_COUNT ==="
          python eval/test.py \
            --config "$CONFIG" \
            --dataset KITTI \
            --seq "$SEQ" \
            --feat "$FEAT" \
            --reg "$REG" \
            --voxel_size "$VOX" \
            --test_type random \
            --test_count "$TEST_COUNT" \
            --dist_min "$DIST_MIN" \
            --dist_max "$DIST_MAX" \
            --seed "$SEED" \
            --out_dir "$OUT_DIR"
        done
      done
    done
  done
done

echo "[done] Sweep finished."
echo "Next: python eval/analyze_sr_vs_gt.py --results_root $BASE_OUT"
