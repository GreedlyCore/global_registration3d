#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 1. submodules
echo "Initializing git submodules..."
git -C "$REPO_ROOT" submodule update --init --recursive

# 2. system deps needed by igraph build
echo "Installing system dependencies..."
sudo apt-get install -y bison flex

# 3. igraph - required by MAC
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON is needed because mac_solver is a shared lib (.so)
echo "Building and installing igraph 0.10.6..."
cd "$REPO_ROOT/third_party/igraph"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build . -j"$(nproc)"
sudo cmake --install .
cd "$REPO_ROOT"

# 4. MAC python binding
echo "Building MAC..."
bash "$REPO_ROOT/MAC/build.sh"

# 5. TEASER++ clone + python binding
echo "Cloning TEASER++..."
if [ ! -d "$REPO_ROOT/TEASER-plusplus/.git" ]; then
    git clone https://github.com/MIT-SPARK/TEASER-plusplus.git "$REPO_ROOT/TEASER-plusplus"
fi
echo "Building TEASER++ python bindings..."
cd "$REPO_ROOT/TEASER-plusplus"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TEASER_FPFH=ON -DTEASERPP_PYTHON_VERSION=3.12
make teaserpp_python -j"$(nproc)"
cd python && pip install .
cd "$REPO_ROOT"

echo "Done ! ! !"
