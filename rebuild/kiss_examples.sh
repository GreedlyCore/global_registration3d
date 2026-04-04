#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/KISS-Matcher/cpp/examples"

echo "Clearing stale CMake cache..."
rm -rf "$EXAMPLES_DIR/build"

echo "Building KISS-Matcher C++ examples..."
mkdir -p "$EXAMPLES_DIR/build"
cmake -S "$EXAMPLES_DIR" -B "$EXAMPLES_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$EXAMPLES_DIR/build" -j"$(nproc)"

echo "Done! Binaries in $EXAMPLES_DIR/build/"
