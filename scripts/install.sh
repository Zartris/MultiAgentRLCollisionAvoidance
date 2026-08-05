#!/usr/bin/env bash
#
# Reproducible native install for GIANT.
#
# Creates a project-local .venv, installs the pinned Python dependencies, and
# builds the Python-RVO2 baseline. RVO2 needs `--no-build-isolation` because its
# setup.py imports Cython at build time without declaring it as a build
# requirement; in an isolated build env that fails with "No module named
# 'Cython'". Installing it against the venv (which already has Cython + cmake
# from requirements.txt) is what makes the build work.
#
# Usage:
#   ./scripts/install.sh              # full install incl. RVO2 baseline
#   SKIP_RVO2=1 ./scripts/install.sh  # core only (no RVO baseline / C++ build)
#
# Requirements on the host: a C++ compiler (g++) for the RVO2 build, and either
# `uv` (preferred) or a python with a working `venv`/`ensurepip`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# mit-acl fork: has setAgentCollabCoeff (NH-ORCA), which rvo_policy.py needs.
RVO2_GIT="git+https://github.com/mit-acl/Python-RVO2.git"

cd "$REPO_ROOT"

# 1. Create the virtualenv (prefer uv; fall back to the stdlib venv).
if command -v uv >/dev/null 2>&1; then
    echo ">>> Creating venv with uv (python $PYTHON_VERSION)"
    uv venv "$VENV" --python "$PYTHON_VERSION"
    PIP=(uv pip install --python "$VENV/bin/python")
else
    echo ">>> Creating venv with python -m venv"
    "python$PYTHON_VERSION" -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    PIP=("$VENV/bin/python" -m pip install)
fi

# 2. Install pinned runtime dependencies + test/build tooling.
echo ">>> Installing requirements.txt"
"${PIP[@]}" -r requirements.txt
echo ">>> Installing test + build tooling (pytest, wheel)"
"${PIP[@]}" pytest wheel

# 3. Build the Python-RVO2 baseline (optional via SKIP_RVO2=1).
if [ "${SKIP_RVO2:-0}" = "1" ]; then
    echo ">>> Skipping Python-RVO2 build (SKIP_RVO2=1)"
else
    echo ">>> Building Python-RVO2 (needs system g++ + cmake)"
    "${PIP[@]}" --no-build-isolation "$RVO2_GIT"
fi

cat <<EOF

>>> Done. Next steps:
      source .venv/bin/activate
      # vmas imports pyglet at load, so headless shells need xvfb:
      xvfb-run -a python -m pytest tests/ -q
      xvfb-run -a python -m train.LidarSingleStep --config configs/smoke.yaml
EOF
