#!/usr/bin/env bash
set -euo pipefail

DRIVE_FILE_ID="13MHQMp08MfCwuV16SLtdB6KWcBBFrUZy"
EXPECTED_IMAGES=704

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$REPO_DIR/data/raw"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
ZIP_PATH="$RAW_DIR/corpus.zip"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: Python is required." >&2
  exit 1
}

command -v unzip >/dev/null 2>&1 || {
  echo "ERROR: unzip is required." >&2
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
  \) | wc -l | tr -d ' '
}

echo "Preparing raw data directory: $RAW_DIR"
mkdir -p "$RAW_DIR"
find "$RAW_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
echo "data/raw is now empty."

echo "Downloading corpus zip from Google Drive..."
"$PYTHON_BIN" -m gdown "https://drive.google.com/uc?id=$DRIVE_FILE_ID" -O "$ZIP_PATH"

echo "Extracting corpus zip into data/raw..."
unzip -q "$ZIP_PATH" -d "$RAW_DIR"
rm -f "$ZIP_PATH"

if [[ -d "$RAW_DIR/ocr_images" ]]; then
  echo "Flattening extracted ocr_images directory..."
  find "$RAW_DIR/ocr_images" -maxdepth 1 -type f -exec mv -t "$RAW_DIR" {} +
  rmdir "$RAW_DIR/ocr_images"
fi

downloaded="$(count_images)"
echo "Image files found: $downloaded"

if (( downloaded != EXPECTED_IMAGES )); then
  echo "ERROR: expected $EXPECTED_IMAGES images, but found $downloaded in data/raw." >&2
  exit 1
fi

echo "Done: all $EXPECTED_IMAGES images are in data/raw."
