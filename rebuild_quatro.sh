#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
QUATRO_DIR="$REPO_ROOT/Quatro"
VENV_PYTHON="$REPO_ROOT/.venv2/bin/python"

if [[ ! -d "$QUATRO_DIR" ]]; then
  echo "[error] Quatro directory not found: $QUATRO_DIR"
  exit 1
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "[error] colcon is not installed or not in PATH"
  exit 1
fi

if [[ ! -f "/opt/ros/kilted/setup.bash" ]]; then
  echo "[error] ROS setup not found at /opt/ros/kilted/setup.bash"
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[warn] venv python not found at $VENV_PYTHON; falling back to python3"
  VENV_PYTHON="$(command -v python3)"
fi

echo "Clearing stale Quatro build/install/log directories..."
rm -rf "$QUATRO_DIR/build" "$QUATRO_DIR/install" "$QUATRO_DIR/log"

echo "Rebuilding Quatro C++ core + Python bindings..."
cd "$QUATRO_DIR"
export PATH=/usr/bin:$PATH
# ROS setup scripts may reference unset variables; temporarily relax nounset.
set +u
source /opt/ros/kilted/setup.bash
set -u

colcon build --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VENV_PYTHON" \
  -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
  -DQUATRO_BUILD_PYTHON=ON \
  -DQUATRO_BUILD_ROS2_NODE=ON

echo "Verifying quatro_solver import..."
QUATRO_LIB_DIR="$QUATRO_DIR/install/quatro_ros2/lib"
QUATRO_PMC_LIB_DIR="$QUATRO_DIR/build/quatro_ros2/pmc/lib"

LD_LIBRARY_PATH="$QUATRO_LIB_DIR:$QUATRO_PMC_LIB_DIR:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$QUATRO_LIB_DIR:${PYTHONPATH:-}" \
"$VENV_PYTHON" -c "import quatro_solver; print('quatro_solver import OK')"