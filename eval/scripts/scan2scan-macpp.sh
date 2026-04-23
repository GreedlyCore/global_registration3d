#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$PWD/.venv2/bin/python3"}
 
RUN_NAME=${1:-}
if [[ -z "$RUN_NAME" ]]; then
  echo "[error] Missing run name." >&2
  exit 1
fi

BASE_OUT=${BASE_OUT:-"results/feat_research/$RUN_NAME"}
TEST_COUNT=${TEST_COUNT:-1}
SEED=${SEED:-42}
TEST_TYPE=${TEST_TYPE:-scan2scan}
MAP_PREV_SCANS=${MAP_PREV_SCANS:-5}

KITTI_CONFIG=${KITTI_CONFIG:-eval/config/KITTI.json}
MULRAN_CONFIG=${MULRAN_CONFIG:-eval/config/MulRan.json}
if [[ -r "eval/config/OXFORD.json" ]]; then
  OXFORD_CONFIG=${OXFORD_CONFIG:-eval/config/OXFORD.json}
else
  OXFORD_CONFIG=${OXFORD_CONFIG:-eval/config/KITTI.json}
fi


METHODS=(macpp)

FEATS=(FasterPFH)

VOXELS=(0.7)

# Keep descriptor params fixed while sweeping Quatro params.
FIXED_DESCRIPTOR_ALPHA_RATIO=2.0
FIXED_DESCRIPTOR_BETA_RATIO=5.0
FIXED_RNORMAL=$(awk -v a="$FIXED_DESCRIPTOR_ALPHA_RATIO" -v v="${VOXELS[0]}" 'BEGIN{printf "%.6f", a*v}')
FIXED_RFPFH=$(awk -v b="$FIXED_DESCRIPTOR_BETA_RATIO" -v v="${VOXELS[0]}" 'BEGIN{printf "%.6f", b*v}')

# Quatro sweep params.
NU=0.7
GRID_ALPHA_RATIOS=(0.2 0.5 1.0)
NOISE_BOUND_NU_SCALES=(0.2 0.4 0.6)
GRID_BETA_RATIOS=()
for s in "${NOISE_BOUND_NU_SCALES[@]}"; do
  GRID_BETA_RATIOS+=("$(awk -v nu="$NU" -v k="$s" 'BEGIN{printf "%.6f", nu*k}')")
done

# Optional pairwise mode (alpha=noise_bound_coeff, beta=noise_bound)
PAIRWISE_ALPHA_RATIOS=(0.2 0.5 1.0)
PAIRWISE_BETA_RATIOS=(0.14 0.28 0.42)

# Select: Grid for creating a heatmap plot, but cost a lost of runs
# Pairwise for anything else
# pairwise | grid
PARAM_SWEEP_MODE=${PARAM_SWEEP_MODE:-grid}


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


KITTI_SEQS=(01 02 03 04 05 06 07 08 09 10)
MULRAN_SEQS=(DCC02 RIVERSIDE02 KAIST02)
# OXFORD_SEQS=(2024-03-18-christ-church-01 2024-03-18-christ-church-02 2024-03-20-christ-church-06)
OXFORD_SEQS=()

# VIRAL_SEQS = () # TODO
# NCLRT_SEQS = () # TODO

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


DIST_MINS=(15)
DIST_MAXS=(20)
DIST_TAGS=("15_20")


RUNS_DIR="$BASE_OUT/runs"
OVERALL_DIR="$BASE_OUT/overall"
GENERATED_SEQ_JSON="$BASE_OUT/generated_sequences.json"
SCAN2SCAN_CFG_DIR="$BASE_OUT/generated_scan2map_configs"
mkdir -p "$RUNS_DIR" "$OVERALL_DIR"
mkdir -p "$SCAN2SCAN_CFG_DIR"

DETAIL_CSV="$BASE_OUT/overall_detail.csv"

source "$SCRIPT_DIR/sweep_reg_methods.lib.sh"
main "$@"

exit 0
