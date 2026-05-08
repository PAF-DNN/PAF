#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-"$ROOT_DIR/.venv"}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-"$ROOT_DIR/.cache/matplotlib"}"
mkdir -p "$MPLCONFIGDIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "Python version: $("$PYTHON" --version)"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
"$PYTHON" -m pip install --no-build-isolation -e "$ROOT_DIR"

if ! command -v dot >/dev/null 2>&1; then
  echo
  echo "Note: the Python graphviz package is installed, but the Graphviz 'dot' executable was not found."
  echo "On macOS, install it with: brew install graphviz"
fi

echo
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"
echo
echo "Run the main module with:"
echo "  PYTHONPATH=src python -m Evaluation.paf_main"
