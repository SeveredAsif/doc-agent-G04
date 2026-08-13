#!/usr/bin/env bash
set -euo pipefail

DRIVE_URL="https://drive.google.com/drive/folders/1hanjTrUN_sVL52UpnkqYGDm8ERjMewMP"
EXPECTED_IMAGES=704

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$REPO_DIR/data/raw"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: Python is required." >&2
  exit 1
}

"$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || {
import gdown
PY
  echo "ERROR: Python package 'gdown' is required." >&2
  echo "Install it with: $PYTHON_BIN -m pip install gdown" >&2
  exit 1
}

count_images() {
  find "$RAW_DIR" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
    -iname '*.tif' -o -iname '*.tiff' \
  \) | wc -l
}

draw_progress() {
  local current percent filled empty bar_width
  bar_width=30
  current="$(count_images | tr -d ' ')"
  if (( current > EXPECTED_IMAGES )); then
    current="$EXPECTED_IMAGES"
  fi
  percent=$(( current * 100 / EXPECTED_IMAGES ))
  filled=$(( current * bar_width / EXPECTED_IMAGES ))
  empty=$(( bar_width - filled ))
  printf '\rDownloading images: ['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%% (%d/%d)' "$percent" "$current" "$EXPECTED_IMAGES"
}

echo "Preparing raw data directory: $RAW_DIR"
mkdir -p "$RAW_DIR"
find "$RAW_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
echo "data/raw is now empty."

echo "Downloading Google Drive folder into data/raw..."
(
  while true; do
    draw_progress
    sleep 0.1
  done
) &
PROGRESS_PID=$!

cleanup() {
  kill "$PROGRESS_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$PYTHON_BIN" -m gdown --folder "$DRIVE_URL" -O "$RAW_DIR" >/tmp/doc_agent_gdown.log 2>&1 || {
  cleanup
  echo
  echo "ERROR: download failed. gdown output:" >&2
  cat /tmp/doc_agent_gdown.log >&2
  exit 1
}

cleanup
trap - EXIT
draw_progress
echo

downloaded="$(count_images | tr -d ' ')"
echo "Downloaded image files: $downloaded"

if (( downloaded != EXPECTED_IMAGES )); then
  echo "WARNING: expected $EXPECTED_IMAGES images, but found $downloaded." >&2
  echo "Check whether the Drive folder is public and whether gdown skipped files." >&2
else
  echo "Done: all $EXPECTED_IMAGES images are in data/raw."
fi
