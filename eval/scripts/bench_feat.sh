#!/usr/bin/env bash
# Benchmark feature extractions across teaser/mac solvers.

# ============================================
#  Feature Extraction Benchmark Summary
# ============================================
# FEAT           REG        avg_feat_time_s
# -------------- ---------- ---------------
# FPFH_PCL       teaser     0.218
# FPFH_PCL       mac        0.2279
# FPFH           teaser     0.2439
# FPFH           mac        0.2423
# FasterPFH      teaser     0.086
# FasterPFH      mac        0.0846
# ============================================

set -e
cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/KITTI.json}
OUT=results/bench_feat

FEATS=(FPFH_PCL FPFH FasterPFH)
REGS=(teaser mac)

declare -A FEAT_TIMES

for FEAT in "${FEATS[@]}"; do
    for REG in "${REGS[@]}"; do
        echo "=== feat=$FEAT reg=$REG ==="
        python eval/test.py --config "$CONFIG" --feat "$FEAT" --reg "$REG" --out_dir "$OUT"

        # Extract mean feat_time_s from SUMMARY row of the CSV
        DATASET=$(python -c "import json; c=json.load(open('$CONFIG')); print(c['dataset'].lower())")
        SEQ=$(python -c "import json; c=json.load(open('$CONFIG')); print(str(c.get('seq','')).zfill(2))")
        TEST_TYPE=$(python -c "import json; c=json.load(open('$CONFIG')); print(c.get('test_type','random'))")
        CSV="$OUT/${DATASET}_${SEQ}_${TEST_TYPE}_${FEAT}_${REG}.csv"

        AVG=$(python -c "
import csv
with open('$CSV') as f:
    for row in csv.DictReader(f):
        if row['pair_id'] == 'SUMMARY':
            print(row['feat_time_s'])
            break
")
        FEAT_TIMES["${FEAT}__${REG}"]=$AVG
        echo "  -> avg feat_time_s = $AVG"
        echo
    done
done

echo "============================================"
echo " Feature Extraction Benchmark Summary"
echo "============================================"
printf "%-14s %-10s %s\n" "FEAT" "REG" "avg_feat_time_s"
printf "%-14s %-10s %s\n" "--------------" "----------" "---------------"
for FEAT in "${FEATS[@]}"; do
    for REG in "${REGS[@]}"; do
        printf "%-14s %-10s %s\n" "$FEAT" "$REG" "${FEAT_TIMES[${FEAT}__${REG}]}"
    done
done
echo "============================================"
