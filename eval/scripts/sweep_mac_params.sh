#!/usr/bin/env bash
# Sweep MAC parameters (tcmp × dcmp) across all three distance bins.
#
# Usage:
#   ./eval/scripts/sweep_mac_params.sh [config] [out_dir] [max_pairs]
#
# Defaults:
#   config    = eval/config/KITTI.json
#   out_dir   = results/sweep_mac
#   max_pairs = 200           (pairs per bin; cached after first build)
#
# Output:
#   One CSV per (tcmp, dcmp, bin) in out_dir/
#   Summary table printed to stdout at the end.

set -e
cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/config/KITTI.json}
OUT=${2:-results/sweep_mac}
MAX_PAIRS=${3:-200}

TCMPS=(0.90 0.95 0.97 0.99 0.999)
DCMPS=(0.5 1.0 2.0 5.0 10.0)
BINS=(0 1 2)
BIN_NAMES=('[2-6m)' '[6-10m)' '[10-12m]')

TOTAL=$(( ${#TCMPS[@]} * ${#DCMPS[@]} * ${#BINS[@]} ))
RUN=0

echo "========================================================"
echo " MAC Parameter Sweep"
echo "  config    : $CONFIG"
echo "  out_dir   : $OUT"
echo "  max_pairs : $MAX_PAIRS"
echo "  grid      : ${#TCMPS[@]} tcmp × ${#DCMPS[@]} dcmp × ${#BINS[@]} bins = $TOTAL runs"
echo "========================================================"
echo

# Helper: read the SUMMARY row value of a named column from a CSV
_summary() {
    local csv=$1 col=$2
    python3 -c "
import csv, sys
with open('$csv') as f:
    for row in csv.DictReader(f):
        if row.get('pair_id') == 'SUMMARY':
            v = row.get('$col', '')
            print(v if v != '' else 'nan')
            sys.exit(0)
print('nan')
" 2>/dev/null
}

# Associative array: key="tcmp__dcmp__bin" → SR value
declare -A SR_TABLE SEP_TABLE FPURE_TABLE RSTAR_TABLE

for BIN in "${BINS[@]}"; do
    for TCMP in "${TCMPS[@]}"; do
        for DCMP in "${DCMPS[@]}"; do
            RUN=$(( RUN + 1 ))
            echo "--- [$RUN/$TOTAL]  tcmp=$TCMP  dcmp=$DCMP  bin=$BIN ${BIN_NAMES[$BIN]} ---"

            python eval/sweep_mac.py \
                --config   "$CONFIG"  \
                --tcmp     "$TCMP"    \
                --dcmp     "$DCMP"    \
                --dist_bin "$BIN"     \
                --out_dir  "$OUT"     \
                --cache_dir "$OUT/cache" \
                --max_pairs "$MAX_PAIRS"

            # Derive CSV filename (mirrors sweep_mac.py naming logic)
            DATASET=$(python3 -c "import json; print(json.load(open('$CONFIG'))['dataset'].lower())")
            SEQ=$(python3     -c "import json; print(str(json.load(open('$CONFIG')).get('seq','')).zfill(2))")
            TCMP_TAG=$(echo "$TCMP" | tr '.' 'p')
            DCMP_TAG=$(echo "$DCMP" | tr '.' 'p')
            CSV="$OUT/${DATASET}_${SEQ}_bin${BIN}_tcmp${TCMP_TAG}_dcmp${DCMP_TAG}.csv"

            if [ -f "$CSV" ]; then
                SR=$(_summary   "$CSV" success)
                SEP=$(_summary  "$CSV" sep)
                FP=$(_summary   "$CSV" f_pure)
                RS=$(_summary   "$CSV" r_star)
            else
                SR=nan; SEP=nan; FP=nan; RS=nan
            fi

            KEY="${TCMP}__${DCMP}__${BIN}"
            SR_TABLE["$KEY"]=$SR
            SEP_TABLE["$KEY"]=$SEP
            FPURE_TABLE["$KEY"]=$FP
            RSTAR_TABLE["$KEY"]=$RS

            echo "    SR=${SR}%  sep=${SEP}  f_pure=${FP}  r_star=${RS}"
            echo
        done
    done
done

# ------------------------------------------------------------------ #
# Summary table
# ------------------------------------------------------------------ #
echo
echo "========================================================================"
echo " Sweep Results Summary"
echo "========================================================================"
printf "%-6s %-5s %-4s  %8s  %8s  %8s  %8s\n" \
    "tcmp" "dcmp" "bin" "SR(%)" "sep" "f_pure" "r_star"
printf "%-6s %-5s %-4s  %8s  %8s  %8s  %8s\n" \
    "------" "-----" "----" "--------" "--------" "--------" "--------"
for TCMP in "${TCMPS[@]}"; do
    for DCMP in "${DCMPS[@]}"; do
        for BIN in "${BINS[@]}"; do
            KEY="${TCMP}__${DCMP}__${BIN}"
            printf "%-6s %-5s %-4s  %8s  %8s  %8s  %8s\n" \
                "$TCMP" "$DCMP" "$BIN" \
                "${SR_TABLE[$KEY]}"    \
                "${SEP_TABLE[$KEY]}"   \
                "${FPURE_TABLE[$KEY]}" \
                "${RSTAR_TABLE[$KEY]}"
        done
    done
done
echo "========================================================================"
