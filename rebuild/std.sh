#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STD_SRC_DIR="$REPO_ROOT/STD_REG"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python"

if [[ ! -d "$STD_SRC_DIR" ]]; then
  echo "[error] STD_REG source directory not found: $STD_SRC_DIR"
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] venv python not found at $VENV_PYTHON; falling back to python3"
  VENV_PYTHON="$(command -v python3)"
fi

BUILD_DIR="$STD_SRC_DIR/build_pybind"

echo "Cleaning stale STD_REG build directory..."
rm -rf "$BUILD_DIR"

echo "Rebuilding STD_REG Python bindings..."
cmake -S "$STD_SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev -DPython3_EXECUTABLE="$VENV_PYTHON"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Done!"