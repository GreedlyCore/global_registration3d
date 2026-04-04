#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-"$PWD/.venv2/bin/python3"}

BASE_OUT=${BASE_OUT:-results/feat_research/$(date +%H-%M-%S.%N | sed 's/[0-9]\{6\}$//')}
TEST_COUNT=${TEST_COUNT:-5}
SEED=${SEED:-42}

KITTI_CONFIG=${KITTI_CONFIG:-eval/config/KITTI.json}
MULRAN_CONFIG=${MULRAN_CONFIG:-eval/config/MulRan.json}
if [[ -r "eval/config/OXFORD.json" ]]; then
  OXFORD_CONFIG=${OXFORD_CONFIG:-eval/config/OXFORD.json}
else
  OXFORD_CONFIG=${OXFORD_CONFIG:-eval/config/KITTI.json}
fi

# METHODS=(kiss)
METHODS=(macpp quatro kiss)
# METHODS=(mac)
# METHODS=(trde gmor)

# FEATS=(FasterPFH)
FEATS=(FasterPFH FPFH)
# FEATS=(SHOT_PCL)

# VOXELS=(0.1 0.3 0.5 0.7)
VOXELS=(0.3 0.5 0.7 1.0)

# Pairwise mode runs 
PAIRWISE_ALPHA_RATIOS=(2.0 3.5)
PAIRWISE_BETA_RATIOS=(5.0 5.0)
GRID_ALPHA_RATIOS=(${GRID_ALPHA_RATIOS:-2.0 3.5})
GRID_BETA_RATIOS=(${GRID_BETA_RATIOS:-5.0})
# Select: Grid for creating a heatmap plot, but cost a lost of runs
# Pairwise for anything else
PARAM_SWEEP_MODE=${PARAM_SWEEP_MODE:-grid}  # pairwise | grid


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
OXFORD_SEQS=()
# OXFORD_SEQS=(2024-03-18-christ-church-01 2024-03-18-christ-church-02 2024-03-20-christ-church-06)
# VIRAL_SEQS = () # TODO
# NCLT -- ? think twice

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

DIST_MINS=(2 6 10 15)
DIST_MAXS=(6 10 12 20)
DIST_TAGS=("2_6" "6_10" "10_12" "15_20")

RUNS_DIR="$BASE_OUT/runs"
OVERALL_DIR="$BASE_OUT/overall"
GENERATED_SEQ_JSON="$BASE_OUT/generated_sequences.json"
SCAN2SCAN_CFG_DIR="$BASE_OUT/generated_scan2scan_configs"
mkdir -p "$RUNS_DIR" "$OVERALL_DIR"
mkdir -p "$SCAN2SCAN_CFG_DIR"

DETAIL_CSV="$BASE_OUT/overall_detail.csv"

source "$SCRIPT_DIR/sweep_reg_methods.lib.sh"
main "$@"

exit 0
