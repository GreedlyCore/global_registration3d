#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MACPP_DIR="$REPO_ROOT/MAC-PLUS-PLUS"
MACPP_SRC_DIR="$MACPP_DIR/src"
MACPP_BUILD_DIR="$MACPP_SRC_DIR/build"
MACPP_PYBIND_DIR="$MACPP_SRC_DIR/pybind"
MACPP_PYBIND_BUILD_DIR="$MACPP_SRC_DIR/build_pybind"

if [[ ! -d "$MACPP_DIR" ]]; then
  echo "[error] MAC++ directory not found: $MACPP_DIR"
  exit 1
fi

if [[ ! -d "$MACPP_SRC_DIR" ]]; then
  echo "[error] MAC++ src directory not found: $MACPP_SRC_DIR"
  exit 1
fi

echo "Clearing stale MAC++ build directories..."
rm -rf "$MACPP_BUILD_DIR" "$MACPP_PYBIND_BUILD_DIR"

echo "Configuring MAC++ (expects igraph already installed, e.g. via init.sh)..."
if ! cmake -S "$MACPP_SRC_DIR" -B "$MACPP_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -Wno-dev; then
  echo "[error] CMake configure failed. Ensure igraph 0.10.6 is installed and discoverable by CMake."
  echo "[hint] Your init.sh already installs igraph from third_party/igraph."
  exit 1
fi

echo "Building MAC++ binaries..."
cmake --build "$MACPP_BUILD_DIR" -j"$(nproc)"

echo "Building MAC++ Python binding..."
cmake -S "$MACPP_PYBIND_DIR" -B "$MACPP_PYBIND_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -Wno-dev
cmake --build "$MACPP_PYBIND_BUILD_DIR" -j"$(nproc)"

echo "[done] MAC++ binaries are in: $MACPP_BUILD_DIR"
echo "[done] macpp_solver module is built into eval/"