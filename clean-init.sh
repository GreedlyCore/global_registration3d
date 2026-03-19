#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

say() {
    printf '\n==> %s\n' "$*"
}

remove_path() {
    local path="$1"
    if [ -e "$path" ] || [ -L "$path" ]; then
        printf '  - removing %s\n' "${path#"$REPO_ROOT"/}"
        rm -rf "$path" 2>/dev/null || {
            printf '    permission denied, retrying with sudo...\n'
            sudo rm -rf "$path"
        }
    fi
}

remove_glob() {
    local pattern="$1"
    shopt -s nullglob
    local matches=( $pattern )
    shopt -u nullglob
    local path
    for path in "${matches[@]}"; do
        remove_path "$path"
    done
}

clean_submodule() {
    local rel="$1"
    say "Cleaning submodule $rel"
    git -C "$REPO_ROOT" submodule deinit -f -- "$rel" >/dev/null 2>&1 || true
    remove_path "$REPO_ROOT/$rel"
    remove_path "$REPO_ROOT/.git/modules/$rel"
}

say "Removing repo-local build artifacts while keeping .venv2"

# Local build outputs.
remove_path "$REPO_ROOT/MAC/src/build"
remove_path "$REPO_ROOT/MAC/src/build_pybind"
remove_path "$REPO_ROOT/TEASER-plusplus/build"
remove_path "$REPO_ROOT/Quatro/build"
remove_path "$REPO_ROOT/Quatro/install"
remove_path "$REPO_ROOT/Quatro/log"
remove_path "$REPO_ROOT/KISS-Matcher/build"
remove_path "$REPO_ROOT/KISS-Matcher/build_pybind"
remove_path "$REPO_ROOT/KISS-Matcher/python/build"
remove_path "$REPO_ROOT/KISS-Matcher/python/dist"
remove_path "$REPO_ROOT/third_party/igraph/build"

# Common editable-install leftovers inside the repo.
remove_glob "$REPO_ROOT/KISS-Matcher/python/*.egg-info"
remove_glob "$REPO_ROOT/TEASER-plusplus/*.egg-info"
remove_glob "$REPO_ROOT/TEASER-plusplus/python/*.egg-info"
remove_glob "$REPO_ROOT/MAC/src/pybind/*.egg-info"

# Reset fetched dependency trees so init.sh can recreate them from scratch.
clean_submodule "KISS-Matcher"
clean_submodule "Quatro"
clean_submodule "third_party/igraph"

# TEASER-plusplus is cloned by init.sh, not tracked by the main repo.
if [ -d "$REPO_ROOT/TEASER-plusplus/.git" ]; then
    say "Removing TEASER-plusplus clone"
    remove_path "$REPO_ROOT/TEASER-plusplus"
fi

say "Cleanup finished"
