# doc-agent — start here

**New here? Read `SUBMISSION.md` first (how to submit), then `handbook/01-START-HERE.pdf`.**

- **Submit via GitHub:** this folder *is* your repo. Create a **public** GitHub repo, `git push`, and each
  milestone `git tag aN-submit && git push --tags`. Full steps: **`SUBMISSION.md`**. A private repo = no submission.
- **Read in order:** `handbook/01-START-HERE` → `02-How-To-Submit` → `03-Project-Specification` →
  `04-Project-Walkthrough` → `05-Codebase-Guide` → `06`-Group-Assignment-Workbook (domains, specialities, NFRs, sources, build buckets).
- **Fill each milestone's form** in `forms/AN_form.docx` and commit it.

---

# doc-agent — regulated starter repo (scanned-document Agentic-RAG)

A fixed skeleton. **You choose models & parameters (in `configs/`). You do NOT choose where code goes.**
Implement only inside functions marked `# IMPLEMENT`. Do not move, rename, or add top-level modules.
CI rejects a repo whose structure or interfaces drift (`tests/test_structure.py`).

## Phase → file map
| Phase | Where |
|---|---|
| 0 Problem/config | `configs/task.yaml`, `configs/config.yaml` |
| 1 Ingestion | `src/doc_agent/ingest/loader.py`, `preprocess.py` |
| 1 Enhancement (VAE/diffusion) | `src/doc_agent/ingest/enhance.py` |
| 2 Layout detection | `src/doc_agent/vision/layout.py` |
| 3 OCR / HTR | `src/doc_agent/vision/ocr.py` |
| 4 Index (chunk/embed/store) | `src/doc_agent/index/` |
| 5 Retrieval | `src/doc_agent/retrieval/` |
| 6 Agent (query→answer) | `src/doc_agent/agent/agent.py`, `tools.py` |
| 6 HITL | `src/doc_agent/agent/hitl.py` |
| 6 Guardrails/security | `src/doc_agent/agent/guardrails.py` |
| 7 RL policy + RLVR | `src/doc_agent/rl/` |
| Training | `src/doc_agent/training/` |
| 8 Serving | `src/doc_agent/serve/`, `Dockerfile` |
| 9 Validation/eval | `src/doc_agent/eval/`, `tests/` |
| MLOps | `src/doc_agent/mlops/` |
| CI/CD | `.github/workflows/` |
| Eval tasks | `grading_kit/tasks.jsonl`, `grading_kit/success_check.py` |
| Pipeline (fixed order) | `src/doc_agent/pipeline.py` |

## Run
```
make setup        # uv sync (pinned lockfile)
make seed         # deterministic seeds
make ingest index # build the KB
make eval         # metrics on tasks.jsonl
make serve        # FastAPI + Gradio
```
See `STRUCTURE.md` for the rules CI enforces.

## Embedding model

Stage 4 uses `sentence-transformers/all-MiniLM-L6-v2` by default. Install the
declared dependency with `uv sync` (or `python -m pip install sentence-transformers`)
before building the index; the first run downloads the model weights.

## Bangla OCR with Tesseract

This project uses Tesseract as its printed Bangla/English OCR baseline. It
receives one line crop at a time from `vision/layout.py`, using the `ben+eng`
language setting in `configs/config.yaml`. The engine is a machine dependency;
the language files are project-local so a fresh clone can reproduce the OCR
configuration.

### One-command A2 build

After the corpus has been placed in `data/raw/` and Python dependencies have
been installed, run the following from a POSIX shell:

```bash
bash scripts/build_index.sh
```

If Bash reports an error ending in `\r` or `$'\r'`, your Git client checked
out the shell script with Windows CRLF line endings. Run this once in the
repository, then save `scripts/build_index.sh` with LF line endings (in VS
Code, click `CRLF` in the status bar and select `LF`):

```bash
git config core.autocrlf input
```

Use `bash scripts/build_index.sh`, rather than running the `.sh` file directly
from PowerShell.

### Five-page build and retrieval demo

For a quick local demonstration, run this in PowerShell. It overrides the
development page limit only in memory: five pages are processed, indexed, then
the same persisted index is queried. Change the Bangla query to test another
topic.

```powershell
$env:PYTHONUTF8 = "1"
@'
from doc_agent import config, pipeline
from doc_agent.retrieval.retriever import Retriever

cfg = config.load()
cfg["preprocess"]["max_pages"] = 5

# Stages 1-4: pages -> preprocessing -> layout -> OCR -> chunks -> embeddings -> FAISS.
pipeline.build_knowledge_base(cfg)

# Stage 5: load the just-written index and retrieve its top three passages.
for rank, hit in enumerate(Retriever(cfg).retrieve("বাস্তব সংখ্যা ও সেট", k=3), start=1):
    print(f"\\n#{rank} score={hit.score:.3f} page={hit.page_ids[0]}")
    print(hit.text)
'@ | .\.venv\Scripts\python.exe -
```

The resulting index is stored in `data/processed/index/` as `index.faiss`,
`chunks.jsonl`, and `metadata.json`. For a full A2 corpus build, set
`preprocess.max_pages` to the desired count (or remove it) and use
`bash scripts/build_index.sh`.

This is the required A2 entrypoint declared by `grading_kit/manifest.yaml`. It
installs Tesseract if it is missing (Homebrew on macOS, apt/dnf/pacman on Linux,
or winget from Git Bash on Windows), downloads the exact Bangla and English
language files, then runs Stages 1--4 exactly once:

```text
load -> preprocess -> layout -> Tesseract OCR -> chunk -> embed -> FAISS store
```

On Windows, run the command in Git Bash or WSL. The automatic install may ask
for an administrator password on Linux or a Windows package-manager prompt.
The script needs `curl` (it is normally present; apt/dnf/pacman install it when
they install Tesseract). It deliberately does not download the corpus: the
A1 `scripts/get_data.sh` / `data/raw/` data contract owns that step.

If you open this Windows checkout in WSL, the script detects the existing
`.venv/Scripts/python.exe` and Windows Tesseract installation and reuses them;
it does not require a second WSL virtual environment or OCR installation.

### Manual installation

Install the engine first if you prefer not to let the script do it:

```bash
# macOS (Homebrew)
brew install tesseract

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr curl

# Fedora
sudo dnf install -y tesseract curl

# Arch
sudo pacman -Sy --needed tesseract curl

# Windows PowerShell
winget install --id UB-Mannheim.TesseractOCR --exact
```

Download the official high-accuracy Tesseract language files from
`tesseract-ocr/tessdata_best` and put them here (the build script does this
automatically):

```text
data/interim/tessdata/ben.traineddata
data/interim/tessdata/eng.traineddata
```

Manual download command:

```bash
mkdir -p data/interim/tessdata
curl -fL https://github.com/tesseract-ocr/tessdata_best/raw/main/ben.traineddata \
  -o data/interim/tessdata/ben.traineddata
curl -fL https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata \
  -o data/interim/tessdata/eng.traineddata
```

The repo deliberately references this directory through
`ocr.tessdata_dir`; do not copy the files into a global Tesseract installation.
If Tesseract is installed outside `PATH`, set `ocr.tesseract_cmd` in
`configs/config.yaml` to the full executable path. Use `PYTHONUTF8=1` in a
shell that cannot print Bangla text correctly.
