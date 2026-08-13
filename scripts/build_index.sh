#!/usr/bin/env bash
# A2 - one-command, reproducible knowledge-base build (Stages 1-4).
# Run from any POSIX shell: macOS/Linux Terminal, WSL, or Git Bash on Windows.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

install_tesseract() {
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || {
        echo "Homebrew is required to install Tesseract automatically: https://brew.sh" >&2
        return 1
      }
      brew install tesseract
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr curl
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y tesseract curl
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --needed tesseract curl
      else
        echo "Install Tesseract with your Linux distribution's package manager, then rerun." >&2
        return 1
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      command -v winget.exe >/dev/null 2>&1 || {
        echo "Windows Package Manager (winget) is required; install Tesseract manually instead." >&2
        return 1
      }
      winget.exe install --id UB-Mannheim.TesseractOCR --exact --silent \
        --accept-package-agreements --accept-source-agreements
      export PATH="$PATH:/c/Program Files/Tesseract-OCR"
      ;;
    *)
      echo "Unsupported OS. Install Tesseract, ensure 'tesseract' is on PATH, then rerun." >&2
      return 1
      ;;
  esac
}

# The standard Windows installer does not always update Git Bash's PATH.
if ! command -v tesseract >/dev/null 2>&1 && [ -x "/c/Program Files/Tesseract-OCR/tesseract.exe" ]; then
  export PATH="$PATH:/c/Program Files/Tesseract-OCR"
fi
if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract is not installed; attempting platform-specific installation..."
  install_tesseract
fi
command -v tesseract >/dev/null 2>&1 || {
  echo "Tesseract installation did not put the executable on PATH." >&2
  exit 1
}

TESSDATA_DIR="data/interim/tessdata"
mkdir -p "$TESSDATA_DIR"
for language in ben eng; do
  target="$TESSDATA_DIR/$language.traineddata"
  if [ ! -s "$target" ]; then
    echo "Downloading Tesseract $language language data..."
    curl --fail --location --retry 3 \
      "https://github.com/tesseract-ocr/tessdata_best/raw/main/$language.traineddata" \
      --output "$target"
  fi
done

# This invokes load -> preprocess -> layout -> OCR -> chunk -> embed -> store once.
PYTHONUTF8=1 python scripts/run_index.py
