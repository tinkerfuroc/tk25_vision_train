#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_venv.sh
#   VENV_DIR=.venv310 PYTHON_BIN=python3.10 bash scripts/setup_venv.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found in PATH."
  echo "Set PYTHON_BIN to a valid Python 3.10 executable."
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

echo "Creating virtual environment in: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

cat <<EOF

Virtual environment is ready.

Activate it with:
  source $VENV_DIR/bin/activate

Run CLI help:
  python -m yolo_tuning.vision_tuning.cli -h

EOF
