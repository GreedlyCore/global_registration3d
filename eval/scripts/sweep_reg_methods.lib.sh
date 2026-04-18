#!/usr/bin/env bash

declare -a DATASETS=(KITTI MULRAN OXFORD)
declare -A DATASET_LABELS=(
  [KITTI]="KITTI"
  [MULRAN]="MulRan"
  [OXFORD]="OXFORD"
)
declare -A DATASET_CONFIGS=(
  [KITTI]="$KITTI_CONFIG"
  [MULRAN]="$MULRAN_CONFIG"
  [OXFORD]="$OXFORD_CONFIG"
)
declare -A DATASET_SEQ_VARS=(
  [KITTI]="KITTI_SEQS"
  [MULRAN]="MULRAN_SEQS"
  [OXFORD]="OXFORD_SEQS"
)

RUNS_DIR="$BASE_OUT/runs"
OVERALL_DIR="$BASE_OUT/overall"
GENERATED_SEQ_JSON="$BASE_OUT/generated_sequences.json"
SCAN2SCAN_CFG_DIR="$BASE_OUT/generated_scan2scan_configs"
DETAIL_CSV="$BASE_OUT/overall_detail.csv"
TEST_TYPE=${TEST_TYPE:-scan2scan}
MAP_PREV_SCANS=${MAP_PREV_SCANS:-5}
TOTAL_RUNS=0
COMPLETED_RUNS=0

validate_env() {
  if [[ "$TEST_TYPE" != "scan2scan" && "$TEST_TYPE" != "scan2map" ]]; then
    echo "[error] TEST_TYPE must be one of: scan2scan, scan2map" >&2
    exit 1
  fi

  if [[ -z "${VIRTUAL_ENV:-}" && -f "$PWD/.venv2/bin/activate" ]]; then
    source "$PWD/.venv2/bin/activate"
  fi

  if [[ "${EUID}" -eq 0 ]]; then
    echo "[error] Do not run this script with sudo." >&2
    echo "[hint] Run: bash eval/scripts/sweep_reg_methods.sh" >&2
    exit 1
  fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] Python executable not found: $PYTHON_BIN" >&2
    echo "[hint] Create/activate venv first or set PYTHON_BIN." >&2
    exit 1
  fi

  if ! mkdir -p "$BASE_OUT" 2>/dev/null; then
    FALLBACK_OUT="results/feat_research_user/$(date +%H-%M-%S.%N | sed 's/[0-9]\{6\}$//')"
    echo "[warn] BASE_OUT is not writable: $BASE_OUT" >&2
    echo "[warn] Falling back to: $FALLBACK_OUT" >&2
    BASE_OUT="$FALLBACK_OUT"
    RUNS_DIR="$BASE_OUT/runs"
    OVERALL_DIR="$BASE_OUT/overall"
    GENERATED_SEQ_JSON="$BASE_OUT/generated_sequences.json"
    SCAN2SCAN_CFG_DIR="$BASE_OUT/generated_scan2scan_configs"
    DETAIL_CSV="$BASE_OUT/overall_detail.csv"
    mkdir -p "$BASE_OUT"
  fi

  for cfg in "$KITTI_CONFIG" "$MULRAN_CONFIG" "$OXFORD_CONFIG"; do
    if [[ ! -r "$cfg" ]]; then
      echo "[error] Config is not readable: $cfg" >&2
      exit 1
    fi
  done

  mkdir -p "$RUNS_DIR" "$OVERALL_DIR" "$SCAN2SCAN_CFG_DIR"
  echo "dataset,scene,method,feat,dist_min,dist_max,dist_tag,test_count,seed,voxel_size,alpha,beta,rnormal,rFPFH,sr_percent,time_s,csv_path" > "$DETAIL_CSV"
}

load_matrix() {
  if [[ "$PARAM_SWEEP_MODE" != "pairwise" && "$PARAM_SWEEP_MODE" != "grid" ]]; then
    echo "[error] PARAM_SWEEP_MODE must be one of: pairwise, grid" >&2
    exit 1
  fi

  if [[ "$PARAM_SWEEP_MODE" == "pairwise" ]]; then
    if [[ ${#PAIRWISE_ALPHA_RATIOS[@]} -ne ${#PAIRWISE_BETA_RATIOS[@]} ]]; then
      echo "[error] pairwise mode requires equal lengths: PAIRWISE_ALPHA_RATIOS, PAIRWISE_BETA_RATIOS" >&2
      exit 1
    fi
    if [[ ${#PAIRWISE_ALPHA_RATIOS[@]} -eq 0 ]]; then
      echo "[error] pairwise mode requires at least one scheme" >&2
      exit 1
    fi
  fi

  if [[ "$PARAM_SWEEP_MODE" == "grid" ]]; then
    if [[ ${#GRID_ALPHA_RATIOS[@]} -eq 0 || ${#GRID_BETA_RATIOS[@]} -eq 0 ]]; then
      echo "[error] grid mode requires non-empty GRID_ALPHA_RATIOS and GRID_BETA_RATIOS" >&2
      exit 1
    fi
  fi
}

compute_total_runs() {
  local dataset_scene_total=0
  local dataset seq_var_name

  for dataset in "${DATASETS[@]}"; do
    seq_var_name=${DATASET_SEQ_VARS[$dataset]}
    local -n seqs_ref="$seq_var_name"
    dataset_scene_total=$((dataset_scene_total + ${#seqs_ref[@]}))
  done

  local sweep_combo_count=0
  if [[ "$PARAM_SWEEP_MODE" == "pairwise" ]]; then
    sweep_combo_count=${#PAIRWISE_ALPHA_RATIOS[@]}
  else
    sweep_combo_count=$(( ${#GRID_ALPHA_RATIOS[@]} * ${#GRID_BETA_RATIOS[@]} ))
  fi

  TOTAL_RUNS=$(( ${#METHODS[@]} * ${#FEATS[@]} * ${#VOXELS[@]} * ${#DIST_MINS[@]} * dataset_scene_total * sweep_combo_count ))
}

progress_tick() {
  COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
  printf '[%d/%d]\n' "$COMPLETED_RUNS" "$TOTAL_RUNS"
}

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

build_scan2scan_config() {
  local base_cfg=$1
  local dataset=$2
  local scene=$3
  local dtag=$4
  local out_cfg=$5

  "$PYTHON_BIN" eval/test_helper.py \
    --base_cfg "$base_cfg" \
    --generated_json "$GENERATED_SEQ_JSON" \
    --dataset "$dataset" \
    --scene "$scene" \
    --dist_tag "$dtag" \
    --out_cfg "$out_cfg" \
    --mode "$TEST_TYPE" \
    --map_prev_scans "$MAP_PREV_SCANS"
}

generate_pairs() {
  local -a kitti_seqs=()
  local -a mulran_seqs=()
  local -a oxford_seqs=()
  local dataset seq_var_name

  for dataset in "${DATASETS[@]}"; do
    seq_var_name=${DATASET_SEQ_VARS[$dataset]}
    local -n seqs_ref="$seq_var_name"
    case "$dataset" in
      KITTI)
        kitti_seqs=("${seqs_ref[@]}")
        ;;
      MULRAN)
        mulran_seqs=("${seqs_ref[@]}")
        ;;
      OXFORD)
        oxford_seqs=("${seqs_ref[@]}")
        ;;
    esac
  done

  "$PYTHON_BIN" "$SCRIPT_DIR/generate_scan2scan_pairs.py" \
    --out_json "$GENERATED_SEQ_JSON" \
    --test_count "$TEST_COUNT" \
    --seed "$SEED" \
    --kitti_seqs "${kitti_seqs[@]}" \
    --mulran_seqs "${mulran_seqs[@]}" \
    --oxford_seqs "${oxford_seqs[@]}" \
    --dist_mins "${DIST_MINS[@]}" \
    --dist_maxs "${DIST_MAXS[@]}" \
    --dist_tags "${DIST_TAGS[@]}"
}

prepare_configs() {
  local dtag dataset seq_var_name config_path scene out_cfg

  for dtag in "${DIST_TAGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      seq_var_name=${DATASET_SEQ_VARS[$dataset]}
      config_path=${DATASET_CONFIGS[$dataset]}
      local -n seqs_ref="$seq_var_name"

      for scene in "${seqs_ref[@]}"; do
        out_cfg="$SCAN2SCAN_CFG_DIR/${dataset,,}_${scene,,}_d${dtag}.json"
        if [[ ! -f "$out_cfg" ]]; then
          build_scan2scan_config "$config_path" "${DATASET_LABELS[$dataset]}" "$scene" "$dtag" "$out_cfg"
        fi
      done
    done
  done
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
  local run_cfg
  rnormal=$(awk -v a="$alpha" -v v="$voxel" 'BEGIN{printf "%.6f", a*v}')
  rFPFH=$(awk -v b="$beta" -v v="$voxel" 'BEGIN{printf "%.6f", b*v}')
  run_cfg="$SCAN2SCAN_CFG_DIR/${dataset,,}_${scene,,}_d${dtag}.json"
  if [[ ! -f "$run_cfg" ]]; then
    build_scan2scan_config "$base_cfg" "$dataset" "$scene" "$dtag" "$run_cfg"
  fi

  local run_out="$RUNS_DIR/${method}/${feat}/v${voxel}_a${alpha}_b${beta}/${dataset,,}_${scene,,}/d${dtag}"
  mkdir -p "$run_out"

  # SHOT emits many per-point warnings from PCL internals; allow optional stderr filtering.
  local shot_warn_filter_regex='^\[pcl::SHOTEstimation::(createBinDistanceShape|computeFeature)\]'
  if [[ "$feat" == "SHOT_PCL" && "${QUIET_SHOT_WARNINGS:-0}" == "1" ]]; then
    set +e
    "$PYTHON_BIN" eval/test.py \
      --config "$run_cfg" \
      --dataset "$dataset" \
      --seq "$scene" \
      --feat "$feat" \
      --reg "$method" \
      --test_type "$TEST_TYPE" \
      --map_prev_scans "$MAP_PREV_SCANS" \
      --seed "$SEED" \
      --voxel_size "$voxel" \
      --rnormal "$rnormal" \
      --rFPFH "$rFPFH" \
      --out_dir "$run_out" \
      2> >(grep -Ev "$shot_warn_filter_regex" >&2)
    local run_rc=$?
    set -e
    if [[ "$run_rc" -ne 0 ]]; then
      return "$run_rc"
    fi
  else
    "$PYTHON_BIN" eval/test.py \
      --config "$run_cfg" \
      --dataset "$dataset" \
      --seq "$scene" \
      --feat "$feat" \
      --reg "$method" \
      --test_type "$TEST_TYPE" \
      --map_prev_scans "$MAP_PREV_SCANS" \
      --seed "$SEED" \
      --voxel_size "$voxel" \
      --rnormal "$rnormal" \
      --rFPFH "$rFPFH" \
      --out_dir "$run_out"
  fi

  local method_lc="${method,,}"
  local csv_path
  csv_path=$(find "$run_out" -type f -name "*_${TEST_TYPE}_*_${method_lc}.csv" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)

  local stats
  local sr
  local tm
  stats=$(extract_summary_values "$csv_path")
  sr=${stats%,*}
  tm=${stats#*,}

  echo "$dataset,$scene,$method,$feat,$dmin,$dmax,$dtag,$TEST_COUNT,$SEED,$voxel,$alpha,$beta,$rnormal,$rFPFH,$sr,$tm,$csv_path" >> "$DETAIL_CSV"
  progress_tick
}

run_dataset_scenes() {
  local dataset=$1
  local method=$2
  local feat=$3
  local dmin=$4
  local dmax=$5
  local dtag=$6
  local voxel=$7
  local alpha=$8
  local beta=$9

  local seq_var_name=${DATASET_SEQ_VARS[$dataset]}
  local config_path=${DATASET_CONFIGS[$dataset]}
  local dataset_label=${DATASET_LABELS[$dataset]}
  local -n seqs_ref="$seq_var_name"
  local scene

  for scene in "${seqs_ref[@]}"; do
    run_one \
      "$dataset_label" \
      "$scene" \
      "$method" \
      "$feat" \
      "$dmin" \
      "$dmax" \
      "$dtag" \
      "$voxel" \
      "$alpha" \
      "$beta" \
      "$config_path"
  done
}

run_matrix() {
  local method feat voxel alpha beta dmin dmax dtag vi scheme_idx i dataset

  for method in "${METHODS[@]}"; do
    for feat in "${FEATS[@]}"; do
      if [[ "$PARAM_SWEEP_MODE" == "pairwise" ]]; then
        for vi in "${!VOXELS[@]}"; do
          voxel="${VOXELS[$vi]}"
          for scheme_idx in "${!PAIRWISE_ALPHA_RATIOS[@]}"; do
            alpha="${PAIRWISE_ALPHA_RATIOS[$scheme_idx]}"
            beta="${PAIRWISE_BETA_RATIOS[$scheme_idx]}"
            for i in "${!DIST_MINS[@]}"; do
              dmin="${DIST_MINS[$i]}"
              dmax="${DIST_MAXS[$i]}"
              dtag="${DIST_TAGS[$i]}"
              for dataset in "${DATASETS[@]}"; do
                run_dataset_scenes \
                  "$dataset" \
                  "$method" \
                  "$feat" \
                  "$dmin" \
                  "$dmax" \
                  "$dtag" \
                  "$voxel" \
                  "$alpha" \
                  "$beta"
              done
            done
          done
        done
      else
        for voxel in "${VOXELS[@]}"; do
          for alpha in "${GRID_ALPHA_RATIOS[@]}"; do
            for beta in "${GRID_BETA_RATIOS[@]}"; do
              for i in "${!DIST_MINS[@]}"; do
                dmin="${DIST_MINS[$i]}"
                dmax="${DIST_MAXS[$i]}"
                dtag="${DIST_TAGS[$i]}"
                for dataset in "${DATASETS[@]}"; do
                  run_dataset_scenes \
                    "$dataset" \
                    "$method" \
                    "$feat" \
                    "$dmin" \
                    "$dmax" \
                    "$dtag" \
                    "$voxel" \
                    "$alpha" \
                    "$beta"
                done
              done
            done
          done
        done
      fi
    done
  done
}

aggregate() {
  "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_detail.py" "$DETAIL_CSV" "$OVERALL_DIR"
}

main() {
  validate_env
  load_matrix
  compute_total_runs
  echo "[info] planned runs: $TOTAL_RUNS"
  generate_pairs
  prepare_configs
  run_matrix
  aggregate
}