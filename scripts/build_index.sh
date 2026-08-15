#!/usr/bin/env bash
# A2 - one-command, reproducible knowledge-base build (Stages 1-4).
# Run from macOS/Linux Terminal, WSL, or Git Bash on Windows.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# When this repository is opened through WSL but its dependencies were
# installed in the Windows .venv, use that interpreter and its Windows
# Tesseract installation rather than asking WSL to install a second stack.
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
WINDOWS_TESSERACT_AVAILABLE=false
if [ -n "${WSL_INTEROP:-}" ] && [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
  if [ -x "/mnt/c/Program Files/Tesseract-OCR/tesseract.exe" ]; then
    WINDOWS_TESSERACT_AVAILABLE=true
  fi
fi

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

if ! command -v tesseract >/dev/null 2>&1 && [ -x "/c/Program Files/Tesseract-OCR/tesseract.exe" ]; then
  export PATH="$PATH:/c/Program Files/Tesseract-OCR"
fi
if ! $WINDOWS_TESSERACT_AVAILABLE && ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract is not installed; attempting platform-specific installation..."
  install_tesseract
fi
if ! $WINDOWS_TESSERACT_AVAILABLE && ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract installation did not put the executable on PATH." >&2
  exit 1
fi

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

export HF_HOME="${HF_HOME:-$ROOT_DIR/data/interim/huggingface}"

PYTHONUTF8=1 PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/run_index.py
