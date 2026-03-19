#!/usr/bin/env bash
# Sweep QUATRO params (noise_bound, cbar2) and report SR across GT-distance bins.
#
# Usage:
#   bash eval/scripts/sweep_quatro_sr_gt.sh [config]
#
# Defaults:
#   config      = eval/config/KITTI.json
#   BASE_OUT    = results/quatro_sr_gt_sweep
#   TEST_COUNT  = 100
#   SEED        = 42
#
# Output:
#   - Per-run CSVs under: $BASE_OUT/runs/
#   - Summary CSV:         $BASE_OUT/quatro_sr_gt_summary.csv

set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/KITTI.json}
BASE_OUT=${BASE_OUT:-results/quatro_sr_gt_sweep}
TEST_COUNT=${TEST_COUNT:-100}
SEED=${SEED:-42}

NOISE_BOUNDS=(0.1 0.2 0.3 0.4 0.5 0.6)
CBAR2S=(0.5 0.8 1.0 1.2 1.5)

DIST_MINS=(2 6 10)
DIST_MAXS=(6 10 12)
DIST_TAGS=("2_6" "6_10" "10_12")

mkdir -p "$BASE_OUT/tmp_configs" "$BASE_OUT/runs"

SUMMARY_CSV="$BASE_OUT/quatro_sr_gt_summary.csv"
echo "noise_bound,cbar2,SR_2_6,SR_6_10,SR_10_12" > "$SUMMARY_CSV"

echo "============================================================"
echo " QUATRO SR(GT) Sweep"
echo "  config      : $CONFIG"
echo "  out_root    : $BASE_OUT"
echo "  test_count  : $TEST_COUNT"
echo "  seed        : $SEED"
echo "  noise_bound : ${NOISE_BOUNDS[*]}"
echo "  cbar2       : ${CBAR2S[*]}"
echo "  bins (m)    : [2,6], [6,10], [10,12]"
echo "============================================================"
echo

extract_summary_sr() {
    local csv_path=$1
    python3 - "$csv_path" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
try:
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('pair_id') == 'SUMMARY':
                val = row.get('success', 'nan')
                print(val if val != '' else 'nan')
                raise SystemExit(0)
except FileNotFoundError:
    pass

print('nan')
PY
}

for NB in "${NOISE_BOUNDS[@]}"; do
    for CB in "${CBAR2S[@]}"; do
        NB_TAG=${NB/./p}
        CB_TAG=${CB/./p}

        SR_2_6="nan"
        SR_6_10="nan"
        SR_10_12="nan"

        for IDX in 0 1 2; do
            DMIN=${DIST_MINS[$IDX]}
            DMAX=${DIST_MAXS[$IDX]}
            DTAG=${DIST_TAGS[$IDX]}

            TMP_CFG="$BASE_OUT/tmp_configs/kitti_quatro_nb${NB_TAG}_cb${CB_TAG}.json"
            RUN_OUT="$BASE_OUT/runs/nb${NB_TAG}_cb${CB_TAG}_d${DTAG}"
            mkdir -p "$RUN_OUT"

            python3 - "$CONFIG" "$TMP_CFG" "$NB" "$CB" <<'PY'
import json
import sys

src_cfg, dst_cfg, noise_bound, cbar2 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

with open(src_cfg) as f:
    cfg = json.load(f)

cfg['reg'] = 'quatro'
if 'quatro' not in cfg or not isinstance(cfg['quatro'], dict):
    cfg['quatro'] = {}
cfg['quatro']['noise_bound'] = noise_bound
cfg['quatro']['cbar2'] = cbar2

with open(dst_cfg, 'w') as f:
    json.dump(cfg, f, indent=2)
PY

            echo "--- nb=$NB cbar2=$CB dist=[$DMIN,$DMAX]m ---"
            python3 eval/test.py \
                --config "$TMP_CFG" \
                --test_type random \
                --dist_min "$DMIN" \
                --dist_max "$DMAX" \
                --test_count "$TEST_COUNT" \
                --seed "$SEED" \
                --out_dir "$RUN_OUT"

            CSV_PATH=$(find "$RUN_OUT" -type f -name '*_random_*_quatro.csv' | head -n 1 || true)
            SR=$(extract_summary_sr "$CSV_PATH")

            case "$IDX" in
                0) SR_2_6="$SR" ;;
                1) SR_6_10="$SR" ;;
                2) SR_10_12="$SR" ;;
            esac
        done

        echo "$NB,$CB,$SR_2_6,$SR_6_10,$SR_10_12" >> "$SUMMARY_CSV"
        echo "row => nb=$NB cbar2=$CB | SR=[2,6]:$SR_2_6  [6,10]:$SR_6_10  [10,12]:$SR_10_12"
        echo
    done
done

echo "[done] Summary: $SUMMARY_CSV"
