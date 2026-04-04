#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAC_SRC_DIR="$REPO_ROOT/MAC/src"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python"

if [[ ! -d "$MAC_SRC_DIR" ]]; then
  echo "[error] MAC source directory not found: $MAC_SRC_DIR"
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] venv python not found at $VENV_PYTHON; falling back to python3"
  VENV_PYTHON="$(command -v python3)"
fi

echo "Clearing stale MAC CMake build directories..."
rm -rf "$MAC_SRC_DIR/build" "$MAC_SRC_DIR/build_pybind"

echo "Removing stale MAC Python modules from eval/..."
rm -f "$REPO_ROOT/eval/mac_solver"*.so

echo "Rebuilding MAC C++ binary..."
cmake -S "$MAC_SRC_DIR" -B "$MAC_SRC_DIR/build" -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build "$MAC_SRC_DIR/build" -j"$(nproc)"

echo "Rebuilding MAC Python bindings..."
cmake -S "$MAC_SRC_DIR/pybind" -B "$MAC_SRC_DIR/build_pybind" -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build "$MAC_SRC_DIR/build_pybind" -j"$(nproc)"

echo "Verifying mac_solver import from eval/..."
PYTHONPATH="$REPO_ROOT/eval:${PYTHONPATH:-}" \
"$VENV_PYTHON" -c "import mac_solver; print('mac_solver import OK')"

echo "Done!"
