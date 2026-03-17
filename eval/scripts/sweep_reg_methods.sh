#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE_OUT=${BASE_OUT:-results/feat_research/$(date +%H-%M-%S)}
TEST_COUNT=${TEST_COUNT:-5}
SEED=${SEED:-42}

KITTI_CONFIG=${KITTI_CONFIG:-eval/config/KITTI.json}
MULRAN_CONFIG=${MULRAN_CONFIG:-eval/config/MulRan.json}

for cfg in "$KITTI_CONFIG" "$MULRAN_CONFIG"; do
  if [[ ! -r "$cfg" ]]; then
    echo "[error] Config is not readable: $cfg" >&2
    exit 1
  fi
done

# 4 * 2 * 3 * 5 * 5 * 2 * 3 = 7200 runs in full version ... careful!
# METHODS=(mac teaser quatro kiss)
# FEATS=(FasterPFH FPFH SHOT_PCL)
# VOXELS=(0.1 0.3 0.5)
# ALPHAS=(2.0 2.5 3.0 3.5 4.0)
# BETAS=(4.0 5.0 6.0 7.0 8.0)

# KITTI_SEQS=(01 04)
# MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)


# 1 * 1 * 4 * 5 * 5 * 2 * 3 = 600 runs 
METHODS=(quatro)
FEATS=(FasterPFH)
VOXELS=(0.1 0.3 0.5 0.7)
ALPHAS=(2.0 2.5 3.0 3.5 4.0)
BETAS=(4.0 5.0 6.0 7.0 8.0)
# ALPHAS=(2.0 2.5 )
# BETAS=(4.0 5.0 )
# KITTI_SEQS=( )
# MULRAN_SEQS=( RIVERSIDE02 )

# N = N_{fail} + N_{succ} 
# N = TEST_COUNT * ( len(KITTI_SEQS) + len(MULRAN_SEQS) )
KITTI_SEQS=(01 04)
MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)

# Doing a single GT bin 
DIST_MINS=(10)
DIST_MAXS=(12)
DIST_TAGS=("10_12")

RUNS_DIR="$BASE_OUT/runs"
OVERALL_DIR="$BASE_OUT/overall"
mkdir -p "$RUNS_DIR" "$OVERALL_DIR"

DETAIL_CSV="$BASE_OUT/overall_detail.csv"
echo "dataset,scene,method,feat,dist_min,dist_max,dist_tag,test_count,seed,voxel_size,alpha,beta,rnormal,rFPFH,sr_percent,time_s,csv_path" > "$DETAIL_CSV"

extract_summary_values() {
  local csv_path=$1
  if [[ -z "$csv_path" || ! -f "$csv_path" ]]; then
    echo "nan,nan"
    return 0
  fi

  awk -F, '
    NR==1 {
      for (i=1; i<=NF; ++i) {
        gsub(/\r/, "", $i)
        if ($i == "pair_id") pair_idx=i
        else if ($i == "success") sr_idx=i
        else if ($i == "total_time_s") time_idx=i
      }
      next
    }
    pair_idx && $pair_idx == "SUMMARY" {
      sr = (sr_idx ? $sr_idx : "nan")
      tm = (time_idx ? $time_idx : "nan")
      gsub(/\r/, "", sr)
      gsub(/\r/, "", tm)
      print sr "," tm
      found=1
      exit
    }
    END {
      if (!found) print "nan,nan"
    }
  ' "$csv_path"
}

run_one() {
  local dataset=$1
  local scene=$2
  local method=$3
  local feat=$4
  local dmin=$5
  local dmax=$6
  local dtag=$7
  local voxel=$8
  local alpha=$9
  local beta=${10}
  local base_cfg=${11}

  local rnormal
  local rFPFH
  rnormal=$(awk -v a="$alpha" -v v="$voxel" 'BEGIN{printf "%.6f", a*v}')
  rFPFH=$(awk -v b="$beta" -v v="$voxel" 'BEGIN{printf "%.6f", b*v}')

  local run_out="$RUNS_DIR/${method}/${feat}/v${voxel}_a${alpha}_b${beta}/${dataset,,}_${scene,,}/d${dtag}"
  mkdir -p "$run_out"

  echo "--- dataset=$dataset scene=$scene method=$method feat=$feat dist=[$dmin,$dmax] voxel=$voxel alpha=$alpha beta=$beta ---"

  python3 eval/test.py \
    --config "$base_cfg" \
    --dataset "$dataset" \
    --seq "$scene" \
    --feat "$feat" \
    --reg "$method" \
    --test_type random \
    --dist_min "$dmin" \
    --dist_max "$dmax" \
    --test_count "$TEST_COUNT" \
    --seed "$SEED" \
    --voxel_size "$voxel" \
    --rnormal "$rnormal" \
    --rFPFH "$rFPFH" \
    --out_dir "$run_out"

  local method_lc="${method,,}"
  local csv_path
  csv_path=$(find "$run_out" -type f -name "*_random_*_${method_lc}.csv" | head -n 1 || true)

  local stats
  local sr
  local tm
  stats=$(extract_summary_values "$csv_path")
  sr=${stats%,*}
  tm=${stats#*,}

  echo "$dataset,$scene,$method,$feat,$dmin,$dmax,$dtag,$TEST_COUNT,$SEED,$voxel,$alpha,$beta,$rnormal,$rFPFH,$sr,$tm,$csv_path" >> "$DETAIL_CSV"
}

for method in "${METHODS[@]}"; do
  for feat in "${FEATS[@]}"; do
    for voxel in "${VOXELS[@]}"; do
      for alpha in "${ALPHAS[@]}"; do
        for beta in "${BETAS[@]}"; do
          for i in "${!DIST_MINS[@]}"; do
            for scene in "${KITTI_SEQS[@]}"; do
              run_one \
                "KITTI" \
                "$scene" \
                "$method" \
                "$feat" \
                "${DIST_MINS[$i]}" \
                "${DIST_MAXS[$i]}" \
                "${DIST_TAGS[$i]}" \
                "$voxel" \
                "$alpha" \
                "$beta" \
                "$KITTI_CONFIG"
            done
            for scene in "${MULRAN_SEQS[@]}"; do
              run_one \
                "MulRan" \
                "$scene" \
                "$method" \
                "$feat" \
                "${DIST_MINS[$i]}" \
                "${DIST_MAXS[$i]}" \
                "${DIST_TAGS[$i]}" \
                "$voxel" \
                "$alpha" \
                "$beta" \
                "$MULRAN_CONFIG"
            done
          done
        done
      done
    done
  done
done

python3 "$(dirname "$0")/aggregate_detail.py" "$DETAIL_CSV" "$OVERALL_DIR"

echo
echo "[done] detailed: $DETAIL_CSV"
echo "[done] overall : $OVERALL_DIR"
