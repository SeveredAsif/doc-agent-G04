"""FIXED end-to-end order (Stages 0-9) + cross-cutting seams.
Do not reorder stages or remove hooks.run()/register_all() calls."""
from __future__ import annotations
from . import config, hooks, wiring  # noqa: F401
from .ingest import loader, preprocess, enhance
from .vision import layout, ocr
from .index import chunk, embed, store
from .retrieval import retriever
from .agent import agent

def build_knowledge_base(cfg: dict) -> None:
    wiring.register_all(cfg)                        # wire cross-cutting features
    print("starting")
    pages = loader.load_pages(cfg)
    print("Pages loaded")
    pages = preprocess.run(pages, cfg)
    print("Pages pre processed")
    #pages = enhance.run(pages, cfg)                 # Stage 1 - enhancement (VAE/diffusion)
    print("Pages enhanced")
    hooks.run(hooks.AFTER_INGEST, {"pages": pages})
    regions = layout.detect(pages, cfg)             # Stage 2
    print("Regions detected")
    text = ocr.transcribe(regions, cfg)             # Stage 3
    print(text[3].text)
    hooks.run(hooks.AFTER_OCR, {"chunks": text})    # e.g. PII redaction on extracted text
    chunks = chunk.split(text, cfg)                 # Stage 4
    hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})
    vectors = embed.encode(chunks, cfg)
    store.build(chunks, vectors, cfg)

def answer(query_text: str, cfg: dict):
    wiring.register_all(cfg)
    r = retriever.Retriever(cfg)                    # Stage 5
    return agent.Agent(cfg, r).run(query_text)      # Stage 6 (seams run inside the loop)
