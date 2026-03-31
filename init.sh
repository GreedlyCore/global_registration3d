#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 1. submodules
echo "Initializing git submodules..."
git -C "$REPO_ROOT" submodule set-url Quatro https://github.com/GreedlyCore/Quatro.git 2>/dev/null || true
git -C "$REPO_ROOT" submodule sync --recursive
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

# MAC++ binaries + python binding
echo "Building MAC++..."
bash "$REPO_ROOT/rebuild_macpp.sh"

# GMOR / TRDE clone + build + python binding
echo "Preparing GMOR..."
if [ ! -d "$REPO_ROOT/GMOR" ]; then
    git clone https://github.com/Bitzhaozheng/GMOR.git "$REPO_ROOT/GMOR"
fi
echo "Building GMOR + TRDE Python binding..."
bash "$REPO_ROOT/rebuild_gmor_trde.sh"

# 5. TEASER++ clone + python binding
echo "Cloning TEASER++..."
if [ ! -d "$REPO_ROOT/TEASER-plusplus/.git" ]; then
    git clone https://github.com/MIT-SPARK/TEASER-plusplus.git "$REPO_ROOT/TEASER-plusplus"
fi
echo "Building TEASER++ python bindings..."
cd "$REPO_ROOT/TEASER-plusplus"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MARCH_NATIVE=ON -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TEASER_FPFH=ON -DTEASERPP_PYTHON_VERSION=3.12
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

# 7. Quatro ROS2_core / Node build (Python module is optional)
echo "Building Quatro (ROS2 Kilted)..."
cd "$REPO_ROOT/Quatro"
export PATH=/usr/bin:$PATH
source /opt/ros/kilted/setup.bash
colcon build --cmake-args \
    -DPython3_EXECUTABLE="$REPO_ROOT/.venv2/bin/python" \
    -DPYTHON_EXECUTABLE="$REPO_ROOT/.venv2/bin/python" \
    -DQUATRO_BUILD_PYTHON=ON \
    -DQUATRO_BUILD_ROS2_NODE=ON
cd "$REPO_ROOT"

# 8. Sanity check: verify quatro_solver is importable with Quatro local libs
echo "Checking Quatro Python binding import..."
QUATRO_LIB_DIR="$REPO_ROOT/Quatro/install/quatro_ros2/lib"
QUATRO_PMC_LIB_DIR="$REPO_ROOT/Quatro/build/quatro_ros2/pmc/lib"
LD_LIBRARY_PATH="$QUATRO_LIB_DIR:$QUATRO_PMC_LIB_DIR:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$QUATRO_LIB_DIR:${PYTHONPATH:-}" 

echo "Done ! ! !"
