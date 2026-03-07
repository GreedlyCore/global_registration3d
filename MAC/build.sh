#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"

# --- standalone binary ---
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. -Wno-dev
make -j$(nproc)
cd ..

# --- python binding ---
cmake -S pybind -B build_pybind -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build_pybind -j$(nproc)
