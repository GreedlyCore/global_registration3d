#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python3.12"
BUILD_DIR="/tmp/km_build"

EXT_SUFFIX="$($VENV_PYTHON - <<'PY'
import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')
PY
)"
MODULE_FILENAME="_kiss_matcher${EXT_SUFFIX}"

SITE_PACKAGES_DIR="$($VENV_PYTHON - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

PKG_DST_DIR="$SITE_PACKAGES_DIR/kiss_matcher"
SRC_DST_DIR="$REPO_ROOT/KISS-Matcher/python/kiss_matcher"

rm -rf "$BUILD_DIR" && mkdir "$BUILD_DIR"

cmake "$REPO_ROOT/KISS-Matcher/python" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -B "$BUILD_DIR"

cmake --build "$BUILD_DIR" -j"$(nproc)"

cp "$BUILD_DIR/$MODULE_FILENAME" \
   "$SITE_PACKAGES_DIR/$MODULE_FILENAME"

mkdir -p "$PKG_DST_DIR" "$SRC_DST_DIR"

cp "$BUILD_DIR/$MODULE_FILENAME" \
   "$PKG_DST_DIR/$MODULE_FILENAME"

cp "$BUILD_DIR/$MODULE_FILENAME" \
   "$SRC_DST_DIR/$MODULE_FILENAME"

"$VENV_PYTHON" -c "from kiss_matcher._kiss_matcher import FPFH, SHOT, FasterPFH, voxelgrid_sampling; print('kiss_matcher bindings OK')"
