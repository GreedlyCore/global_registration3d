#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 1. submodules
echo "Initializing git submodules..."
git -C "$REPO_ROOT" submodule update --init --recursive

# 2. system deps needed by igraph build
echo "Installing system dependencies..."
sudo apt-get install -y bison flex python3.12-venv

# 2b. create and activate virtualenv
echo "Creating virtual environment .venv2..."
python3 -m venv "$REPO_ROOT/.venv2"
source "$REPO_ROOT/.venv2/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ROOT/req.txt"

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
cd "$REPO_ROOT/TEASER-plusplus" && pip install .
cd "$REPO_ROOT"

# 6. KISS-Matcher C++ build + Python bindings
echo "Building KISS-Matcher..."
cd "$REPO_ROOT/KISS-Matcher"
make deps
# make cppinstall                                           
make cppinstall_matcher_only # robin already installed case
echo "Installing KISS-Matcher Python bindings..."
pip install --upgrade setuptools wheel scikit-build-core ninja cmake build
pip install -e python/
cd "$REPO_ROOT"

echo "Done ! ! !"
