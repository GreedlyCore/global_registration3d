#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GMOR_DIR="$REPO_ROOT/GMOR"
GMOR_BUILD_DIR="$GMOR_DIR/build"
GMOR_PYBIND_DIR="$GMOR_DIR/pybind"
GMOR_PYBIND_BUILD_DIR="$GMOR_DIR/build_pybind"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    echo "[error] Do not run this script with sudo/root."
    exit 1
fi

if [[ ! -d "$GMOR_DIR" ]]; then
    echo "[error] GMOR directory not found: $GMOR_DIR"
    echo "[hint] Run init.sh to clone GMOR automatically."
    exit 1
fi

if [[ ! -d "$GMOR_PYBIND_DIR" ]]; then
    echo "[error] GMOR pybind directory not found: $GMOR_PYBIND_DIR"
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[warn] venv python not found at $VENV_PYTHON, using system python3."
    VENV_PYTHON="python3"
fi

echo "Clearing stale GMOR build directories..."
rm -rf "$GMOR_BUILD_DIR" "$GMOR_PYBIND_BUILD_DIR"

echo "Configuring GMOR project..."
cmake -S "$GMOR_DIR" -B "$GMOR_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev

echo "Building GMOR project..."
cmake --build "$GMOR_BUILD_DIR" -j"$(nproc)"

echo "Configuring GMOR/TRDE Python binding..."
cmake -S "$GMOR_PYBIND_DIR" -B "$GMOR_PYBIND_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
    -Wno-dev

echo "Building GMOR/TRDE Python binding..."
cmake --build "$GMOR_PYBIND_BUILD_DIR" -j"$(nproc)"

echo "[done] GMOR build dir: $GMOR_BUILD_DIR"
# echo "[done] gmor_trde_solver module should be in: $REPO_ROOT/eval"