#!/usr/bin/env bash

# TODO: resolve copy operations later

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python3.12"
BUILD_DIR="/tmp/km_build"

rm -rf "$BUILD_DIR" && mkdir "$BUILD_DIR"

cmake "$REPO_ROOT/KISS-Matcher/python" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -B "$BUILD_DIR"

cmake --build "$BUILD_DIR" -j"$(nproc)"

cp "$BUILD_DIR/_kiss_matcher.cpython-312-x86_64-linux-gnu.so" \
   "$REPO_ROOT/.venv2/lib/python3.12/site-packages/_kiss_matcher.cpython-312-x86_64-linux-gnu.so"

cp "$BUILD_DIR/_kiss_matcher.cpython-312-x86_64-linux-gnu.so" \
   "$REPO_ROOT/KISS-Matcher/python/kiss_matcher/_kiss_matcher.cpython-312-x86_64-linux-gnu.so"

"$VENV_PYTHON" -c "from kiss_matcher._kiss_matcher import FPFH, FasterPFH, voxelgrid_sampling; print('kiss_matcher bindings OK')"
