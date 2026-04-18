#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export EVAL_ENABLE_CSV_OUTPUT=0
export EVAL_ENABLE_FILE_LOGGING=0

CONFIG=${CONFIG:-eval/config/KITTI.json}
SEQ=${SEQ:-01}
TEST_COUNT=${TEST_COUNT:-10}
SEED=${SEED:-42}
TEST_TYPE=${TEST_TYPE:-scan2map} # scan2scan | scan2map | random
MAP_PREV_SCANS=${MAP_PREV_SCANS:-5}
DIST_MIN=${DIST_MIN:-6}
DIST_MAX=${DIST_MAX:-12}

# TEST IF specified methods are build proprely
# SETUP: Specify methods here
METHODS=(macpp mac teaser quatro kiss gmor trde)
# FasterPFH // FPFH // SHOT_PCL // STD

RUN_CONFIG="$CONFIG"
TMP_GEN_JSON=""
TMP_CFG=""

cleanup() {
  if [[ -n "$TMP_GEN_JSON" && -f "$TMP_GEN_JSON" ]]; then
    rm -f "$TMP_GEN_JSON"
  fi
  if [[ -n "$TMP_CFG" && -f "$TMP_CFG" ]]; then
    rm -f "$TMP_CFG"
  fi
}
trap cleanup EXIT

if [[ "$TEST_TYPE" == "scan2scan" || "$TEST_TYPE" == "scan2map" ]]; then
  TMP_GEN_JSON=$(mktemp)
  TMP_CFG=$(mktemp --suffix .json)

  EMIT_SCAN2MAP_FLAG=()
  if [[ "$TEST_TYPE" == "scan2map" ]]; then
    EMIT_SCAN2MAP_FLAG=(--emit_scan2map)
  fi

  python3 eval/scripts/generate_scan2scan_pairs.py \
    --out_json "$TMP_GEN_JSON" \
    --test_count "$TEST_COUNT" \
    --seed "$SEED" \
    --kitti_seqs "$SEQ" \
    --dist_mins "$DIST_MIN" \
    --dist_maxs "$DIST_MAX" \
    --dist_tags smoke \
    "${EMIT_SCAN2MAP_FLAG[@]}" \
    --map_prev_scans "$MAP_PREV_SCANS"

  python3 eval/test_helper.py \
    --base_cfg "$CONFIG" \
    --generated_json "$TMP_GEN_JSON" \
    --dataset KITTI \
    --scene "$SEQ" \
    --dist_tag smoke \
    --out_cfg "$TMP_CFG" \
    --mode "$TEST_TYPE" \
    --map_prev_scans "$MAP_PREV_SCANS"

  RUN_CONFIG="$TMP_CFG"
fi

all_ok=1
for method in "${METHODS[@]}"; do
  echo "METHOD=$method running..."
  if python3 eval/test.py \
      --config "$RUN_CONFIG" \
      --dataset KITTI \
      --seq "$SEQ" \
      --feat FPFH \
      --reg "$method" \
      --voxel_size 0.5 \
      --test_type "$TEST_TYPE" \
      --dist_min "$DIST_MIN" \
      --dist_max "$DIST_MAX" \
      --test_count "$TEST_COUNT" \
      --map_prev_scans "$MAP_PREV_SCANS" \
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