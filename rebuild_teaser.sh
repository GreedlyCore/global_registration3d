#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Clearing stale CMake cache..."
rm -rf "$REPO_ROOT/TEASER-plusplus/build"

echo "Rebuilding TEASER++..."
cd "$REPO_ROOT/TEASER-plusplus"
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TEASER_FPFH=ON \
    -DTEASERPP_PYTHON_VERSION=3.12
make teaserpp_python -j"$(nproc)"

echo "Reinstalling TEASER++ Python bindings..."
cd "$REPO_ROOT/TEASER-plusplus"
pip install .

echo "Done!"
