#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_PYTHON="${BUILD_PYTHON:-ON}"

#   ./build.sh                    # build demo + python module
#   BUILD_PYTHON=OFF ./build.sh   # build demo only


cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_PYTHON="$BUILD_PYTHON"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "[done] Built: $BUILD_DIR/adaptive_bootstrap"
if [[ "$BUILD_PYTHON" == "ON" ]]; then
  echo "[done] Built Python module: $SCRIPT_DIR/../eval/adaptive_bootstrap*.so"
fi


