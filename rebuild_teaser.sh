#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python3"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    echo "[error] Do not run this script with sudo/root."
    exit 1
fi

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
    -DTEASERPP_PYTHON_VERSION="$PYTHON_MM" \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -DPYTHON_EXECUTABLE="$VENV_PYTHON"
make teaserpp_python -j"$(nproc)"

echo "Reinstalling TEASER++ Python bindings..."
cd "$REPO_ROOT/TEASER-plusplus"
"$VENV_PYTHON" -m pip install .

echo "Done!"
