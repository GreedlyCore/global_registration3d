#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_OUT=${BASE_OUT:-results/sr_gt_methods_kitti_mulran/$(date +%H-%M-%S)}
TEST_COUNT=${TEST_COUNT:-50}
SEED=${SEED:-42}

KITTI_CONFIG=${KITTI_CONFIG:-eval/config/KITTI.json}
MULRAN_CONFIG=${MULRAN_CONFIG:-eval/config/MulRan.json}

METHODS=(quatro mac teaser kiss)
DIST_MINS=(2 6 10)
DIST_MAXS=(6 10 12)
DIST_TAGS=("2_6" "6_10" "10_12")

# KITTI_SEQS=(00 01 02 03 04 05 06 07 08 09 10)
KITTI_SEQS=(01 04)
MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)

## Baseline for now but i'm not sure ...

KITTI_VOXEL=0.3
KITTI_RNORMAL=0.5
KITTI_RFPFH=0.65

KITTI_BASE_VOXEL=0.5
KITTI_BASE_RNORMAL=1.5
KITTI_BASE_RFPFH=2.25

MULRAN_VOXEL=0.6
MULRAN_RNORMAL=1.5
MULRAN_RFPFH=2.25

MULRAN_BASE_VOXEL=0.5
MULRAN_BASE_RNORMAL=1.0
MULRAN_BASE_RFPFH=2.5

## Experiments ...

# KITTI_VOXEL=0.5
# KITTI_RNORMAL=1.8
# KITTI_RFPFH=2.5

# KITTI_BASE_VOXEL=0.5
# KITTI_BASE_RNORMAL=1.8
# KITTI_BASE_RFPFH=2.5

# MULRAN_VOXEL=0.6
# MULRAN_RNORMAL=1.9
# MULRAN_RFPFH=3.00

# MULRAN_BASE_VOXEL=0.5
# MULRAN_BASE_RNORMAL=1.8
# MULRAN_BASE_RFPFH=2.5


mkdir -p "$BASE_OUT/runs"

SUMMARY_CSV="$BASE_OUT/sr_gt_summary.csv"
echo "dataset,scene,method,dist_min,dist_max,dist_tag,test_type,test_count,seed,voxel_size,rnormal,rFPFH,sr_percent,csv_path" > "$SUMMARY_CSV"

echo "============================================================"
echo " SR(GT) Sweep: quatro / kiss / mac"
echo "  out_root      : $BASE_OUT"
echo "  test_type     : random"
echo "  test_count    : $TEST_COUNT"
echo "  seed          : $SEED"
echo "  bins (m)      : [2,6], [6,10], [10,12]"
echo "  KITTI scenes  : ${KITTI_SEQS[*]}"
echo "  MulRan scenes : ${MULRAN_SEQS[*]}"
echo "============================================================"
echo

extract_summary_sr() {
  local csv_path=$1
  if [[ -z "$csv_path" || ! -f "$csv_path" ]]; then
    echo "nan"
    return 0
  fi
  awk -F, 'NR>1 && $1=="SUMMARY" {print $4; found=1; exit} END{if(!found) print "nan"}' "$csv_path"
}

run_one() {
  local dataset=$1
  local scene=$2
  local method=$3
  local dmin=$4
  local dmax=$5
  local dtag=$6
  local base_cfg=$7

  local run_out="$BASE_OUT/runs/${dataset,,}_${scene,,}/${method}/d${dtag}"
  mkdir -p "$run_out"

  local voxel
  local rnormal
  local rFPFH

  if [[ "$method" == "kiss" || "$method" == "mac" ]]; then
    if [[ "$dataset" == "KITTI" ]]; then
      voxel="$KITTI_VOXEL"
      rnormal="$KITTI_RNORMAL"
      rFPFH="$KITTI_RFPFH"
    else
      voxel="$MULRAN_VOXEL"
      rnormal="$MULRAN_RNORMAL"
      rFPFH="$MULRAN_RFPFH"
    fi
  elif [[ "$dataset" == "KITTI" ]]; then
    voxel="$KITTI_BASE_VOXEL"
    rnormal="$KITTI_BASE_RNORMAL"
    rFPFH="$KITTI_BASE_RFPFH"
  else
    voxel="$MULRAN_BASE_VOXEL"
    rnormal="$MULRAN_BASE_RNORMAL"
    rFPFH="$MULRAN_BASE_RFPFH"
  fi

  echo "--- dataset=$dataset scene=$scene method=$method dist=[$dmin,$dmax] test_count=$TEST_COUNT ---"

  local -a cmd=(
    python3 eval/test.py
    --config "$base_cfg"
    --dataset "$dataset"
    --seq "$scene"
    --reg "$method"
    --test_type random
    --dist_min "$dmin"
    --dist_max "$dmax"
    --test_count "$TEST_COUNT"
    --seed "$SEED"
    --out_dir "$run_out"
  )

  cmd+=(--voxel_size "$voxel" --rnormal "$rnormal" --rFPFH "$rFPFH")

  "${cmd[@]}"

  local method_lc="${method,,}"
  local csv_path
  csv_path=$(find "$run_out" -type f -name "*_random_*_${method_lc}.csv" | head -n 1 || true)
  local sr
  sr=$(extract_summary_sr "$csv_path")

  echo "$dataset,$scene,$method,$dmin,$dmax,$dtag,random,$TEST_COUNT,$SEED,$voxel,$rnormal,$rFPFH,$sr,$csv_path" >> "$SUMMARY_CSV"
}

for scene in "${KITTI_SEQS[@]}"; do
  for method in "${METHODS[@]}"; do
    for i in 0 1 2; do
      run_one "KITTI" "$scene" "$method" "${DIST_MINS[$i]}" "${DIST_MAXS[$i]}" "${DIST_TAGS[$i]}" "$KITTI_CONFIG"
    done
  done
done

for scene in "${MULRAN_SEQS[@]}"; do
  for method in "${METHODS[@]}"; do
    for i in 0 1 2; do
      run_one "MulRan" "$scene" "$method" "${DIST_MINS[$i]}" "${DIST_MAXS[$i]}" "${DIST_TAGS[$i]}" "$MULRAN_CONFIG"
    done
  done
done

echo
echo "[done] Summary CSV: $SUMMARY_CSV"
