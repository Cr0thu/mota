#!/bin/zsh
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONDONTWRITEBYTECODE=1

choose_python() {
  local candidates=(
    "$MOTA_VISUALIZER_PYTHON"
    "/Users/cr0/anaconda3/envs/Cr0/bin/python"
    "/Users/cr0/anaconda3/envs/drl_hw2_mac/bin/python"
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/venv/bin/python"
    "$(command -v python3 2>/dev/null || true)"
    "$(command -v python 2>/dev/null || true)"
  )

  local py
  for py in "${candidates[@]}"; do
    [[ -n "$py" && -x "$py" ]] || continue
    if "$py" - <<'PY' >/dev/null 2>&1
import tkinter
import torch
import PIL
import pandas
PY
    then
      echo "$py"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[visualizer] No usable Python found."
  echo "[visualizer] Need an environment with tkinter, torch, Pillow and pandas."
  echo "[visualizer] You can set MOTA_VISUALIZER_PYTHON=/path/to/python and rerun."
  read -k 1 "?Press any key to close..."
  exit 1
fi

echo "[visualizer] using python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, torch
print(f"[visualizer] python: {sys.version.split()[0]}")
print(f"[visualizer] torch: {torch.__version__}")
PY

exec "$PYTHON_BIN" tools/visualizer/run_this.py
