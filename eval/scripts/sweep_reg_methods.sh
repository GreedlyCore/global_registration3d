#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$PWD/.venv2/bin/activate" ]]; then
  source "$PWD/.venv2/bin/activate"
fi

if [[ "${EUID}" -eq 0 ]]; then
  echo "[error] Do not run this script with sudo." >&2
  echo "[hint] Run: bash eval/scripts/sweep_reg_methods.sh" >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-"$PWD/.venv2/bin/python3"}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] Python executable not found: $PYTHON_BIN" >&2
  echo "[hint] Create/activate venv first or set PYTHON_BIN." >&2
  exit 1
fi

BASE_OUT=${BASE_OUT:-results/feat_research/$(date +%H-%M-%S.%N | sed 's/[0-9]\{6\}$//')}
TEST_COUNT=${TEST_COUNT:-5}
SEED=${SEED:-42}

if ! mkdir -p "$BASE_OUT" 2>/dev/null; then
  FALLBACK_OUT="results/feat_research_user/$(date +%H-%M-%S.%N | sed 's/[0-9]\{6\}$//')"
  echo "[warn] BASE_OUT is not writable: $BASE_OUT" >&2
  echo "[warn] Falling back to: $FALLBACK_OUT" >&2
  BASE_OUT="$FALLBACK_OUT"
  mkdir -p "$BASE_OUT"
fi

KITTI_CONFIG=${KITTI_CONFIG:-eval/config/KITTI.json}
MULRAN_CONFIG=${MULRAN_CONFIG:-eval/config/MulRan.json}

for cfg in "$KITTI_CONFIG" "$MULRAN_CONFIG"; do
  if [[ ! -r "$cfg" ]]; then
    echo "[error] Config is not readable: $cfg" >&2
    exit 1
  fi
done



METHODS=(mac teaser quatro kiss)
# METHODS=(mac)
# METHODS=(kiss)

# FEATS=(FasterPFH)
FEATS=(FasterPFH FPFH)

# FEATS=(SHOT_PCL)

# VOXELS=(0.1 0.3 0.5 0.7)
# ALPHAS=(2.0 2.5 3.0 3.5 4.0)
# BETAS=(4.0 5.0 6.0 7.0 8.0)

# VOXELS=(0.1 0.3 0.5 0.7)
VOXELS=(0.3 0.5 0.7) # for speed up, too few points in case of 
# VOXELS=(0.7) # for speed speed up
ALPHA_MULTI=${ALPHA_MULTI:-3.5} # 2 // 3.5
BETA_MULTI=${BETA_MULTI:-5.0}  # 5 // 5

ALPHAS=()
BETAS=()

for v in "${VOXELS[@]}"; do
    alpha=$(echo "$v * $ALPHA_MULTI" | bc -l)
    beta=$(echo "$v * $BETA_MULTI" | bc -l)
    ALPHAS+=("$alpha")
    BETAS+=("$beta")
done

# Format with 2 decimal places for nicer output
# echo "Configuration:"
# echo "  Alpha multiplier: $ALPHA_MULTI"
# echo "  Beta multiplier:  $BETA_MULTI"
# echo
# echo "Results:"
# for i in "${!VOXELS[@]}"; do
#     printf "voxel=%.1f → alpha=%.2f, beta=%.2f\n" \
#            "${VOXELS[$i]}" "${ALPHAS[$i]}" "${BETAS[$i]}"
# done


KITTI_SEQS=(01 02 03 04 05 06 07 08 09 10)
MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)
# OXFORD_SEQS = () # TODO
# VIRAL_SEQS = () # TODO
# NCLT -- ? think twice


# 1 * 1 * 4 * 5 * 5 * 2 * 3 = 600 runs 
# METHODS=(quatro)
# FEATS=(FasterPFH)
# VOXELS=(0.1 0.3 0.5 0.7)
# ALPHAS=(2.0 2.5 3.0 3.5 4.0)
# BETAS=(4.0 5.0 6.0 7.0 8.0)
# ALPHAS=(2.0 2.5 )
# BETAS=(4.0 5.0 )
# KITTI_SEQS=( )
# MULRAN_SEQS=( RIVERSIDE02 )

# N = N_{fail} + N_{succ} 
# N = TEST_COUNT * ( len(KITTI_SEQS) + len(MULRAN_SEQS) )
# KITTI_SEQS=(01 04)
# MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)

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

  "$PYTHON_BIN" eval/test.py \
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

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_detail.py" "$DETAIL_CSV" "$OVERALL_DIR"

echo
echo "[done] detailed: $DETAIL_CSV"
echo "[done] overall : $OVERALL_DIR"
