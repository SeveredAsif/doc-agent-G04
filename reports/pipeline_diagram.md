# Knowledge-base pipeline diagram (A2)

```mermaid
flowchart LR
    A["data/raw/*.png\n704 scanned pages\n(higher_math, math)"] --> B["Stage 1: Ingest\ningest/loader.py\nload_pages()"]
    B --> C["Stage 1: Preprocess\ningest/preprocess.py\ndeskew, denoise,\nadaptive threshold"]
    C -.->|"bonus, disabled"| C2["Stage 1: Enhance\ningest/enhance.py\nVAE/diffusion denoiser"]
    C --> D["Stage 2: Layout\nvision/layout.py\nprojection-profile\nline segmentation"]
    D --> E["Stage 3: OCR\nvision/ocr.py\nTesseract (ben+eng, psm 7)\n[compared vs TrOCR baseline\n+ Bangla-finetuned TrOCR]"]
    E --> F["Stage 4: Chunk\nindex/chunk.py\nsemantic split on\ntextbook boundary markers"]
    F --> G["Stage 4: Embed\nindex/embed.py\nsentence-transformers\nall-MiniLM-L6-v2 (384-dim)"]
    G --> H["Stage 4: Store\nindex/store.py\nFAISS IndexFlatIP\n+ chunks.jsonl + metadata.json"]
    H --> I["data/processed/index/\nindex.faiss, chunks.jsonl,\nmetadata.json"]
```

**Stages built for A2:** ingest → preprocess → layout → OCR → chunk → embed → store (Stages 0–4 of the spec).
**Changed from the default:** enhancement (VAE/diffusion) is bonus-tier and left disabled; OCR runs Tesseract
as the primary/reproduced baseline after comparing it against two TrOCR checkpoints (English-only and a
Bangla-finetuned model) on the same pages — Tesseract-on-lines gave the strongest output. Chunking uses a
semantic splitter keyed to Bangla textbook structure (chapters/examples/proofs/exercises) instead of plain
fixed-size windows. See `configs/design_choices.md` for the full per-stage justification.
