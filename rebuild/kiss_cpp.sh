#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Clearing stale CMake cache..."
rm -rf "$REPO_ROOT/KISS-Matcher/cpp/kiss_matcher/build"

echo "Rebuilding KISS-Matcher C++ library..."
cd "$REPO_ROOT/KISS-Matcher"
make cppinstall_matcher_only

echo "Done!"
