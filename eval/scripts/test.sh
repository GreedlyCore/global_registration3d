#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export EVAL_ENABLE_CSV_OUTPUT=0
export EVAL_ENABLE_FILE_LOGGING=0

CONFIG=${CONFIG:-eval/config/KITTI.json}
SEQ=${SEQ:-01}
TEST_COUNT=${TEST_COUNT:-10}
SEED=${SEED:-42}

# TEST IF specified methods are build proprely
# SETUP: Specify methods here
METHODS=(macpp mac teaser quatro kiss gmor trde)
# FasterPFH // FPFH // SHOT_PCL // STD

all_ok=1
for method in "${METHODS[@]}"; do
  echo "METHOD=$method running..."
  if python3 eval/test.py \
      --config "$CONFIG" \
      --dataset KITTI \
      --seq "$SEQ" \
      --feat FPFH \ 
      --reg "$method" \
      --voxel_size 0.5 \
      --test_type random \
      --dist_min 6 \
      --dist_max 12 \
      --test_count "$TEST_COUNT" \
      --seed "$SEED" >/dev/null; then
    echo "METHOD=$method PASS"
  else
    echo "METHOD=$method FAIL"
    all_ok=0
  fi
done

if [[ "$all_ok" -eq 1 ]]; then
  echo "ALL METHODS PASS: built and inference succesfully"
  exit 0
fi

echo "SOME METHODS FAILED: check specific errors above."
exit 1