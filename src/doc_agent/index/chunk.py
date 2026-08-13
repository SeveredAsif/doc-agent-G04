"""Stage 4 — chunk text"""
from __future__ import annotations

import re

from ..contracts import *  # noqa


_BOUNDARY_MARKERS = (
    "অধ্যায়",
    "উদাহরণ",
    "সমাধান",
    "অনুশীলনী",
    "উপপাদ্য",
    "প্রমাণ",
    "সূত্র",
    "সংজ্ঞা",
    "কাজ",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _sentences(text: str) -> list[str]:
    pieces = re.findall(r"[^।.!?\n]+[।.!?]?", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _semantic_units(text: str) -> list[str]:
    """Split on textbook cues first, then sentence boundaries inside each block."""
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        starts_new_unit = any(line.startswith(marker) for marker in _BOUNDARY_MARKERS)
        if starts_new_unit and current:
            blocks.append(" ".join(current))
            current = []
        current.append(line)

    if current:
        blocks.append(" ".join(current))

    units: list[str] = []
    for block in blocks or [text]:
        units.extend(_sentences(block) or [block.strip()])
    return [unit for unit in units if unit]


def _window_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    toks = _tokens(text)
    if len(toks) <= max_tokens:
        return [text.strip()] if text.strip() else []

    step = max(1, max_tokens - overlap)
    windows: list[str] = []
    for start in range(0, len(toks), step):
        window = toks[start : start + max_tokens]
        if window:
            windows.append(" ".join(window))
        if start + max_tokens >= len(toks):
            break
    return windows


def _merge_small_units(units: list[str], max_tokens: int, min_tokens: int) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(_tokens(unit))
        if current and current_len + unit_len > max_tokens:
            merged.append(" ".join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += unit_len

    if current:
        tail = " ".join(current)
        if merged and len(_tokens(tail)) < min_tokens:
            merged[-1] = f"{merged[-1]} {tail}"
        else:
            merged.append(tail)
    return merged



def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk OCR text into retrieval-sized pieces.

    Default mode is semantic: respect textbook markers such as examples,
    proofs, formulas, and exercises, then apply token windows only when needed.
    """
    index_cfg = cfg.get("index", {})
    max_tokens = int(index_cfg.get("chunk_tokens", 256))
    overlap = int(index_cfg.get("overlap", 32))
    mode = index_cfg.get("chunking", "semantic")
    min_tokens = int(index_cfg.get("min_chunk_tokens", 24))

    if max_tokens <= 0:
        raise ValueError("index.chunk_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("index.overlap must be >= 0 and smaller than index.chunk_tokens")

    rechunked: list[Chunk] = []
    for source in chunks:
        text = " ".join(source.text.split())
        if not text:
            continue

        if mode == "fixed":
            pieces = _window_by_tokens(text, max_tokens, overlap)
        elif mode == "semantic":
            units = _semantic_units(source.text)
            merged = _merge_small_units(units, max_tokens, min_tokens)
            pieces = []
            for unit in merged:
                pieces.extend(_window_by_tokens(unit, max_tokens, overlap))
        else:
            raise ValueError(f"Unknown index.chunking mode: {mode}")

        for piece_index, piece in enumerate(pieces):
            rechunked.append(
                Chunk(
                    id=f"{source.id}__chunk_{piece_index:04d}",
                    doc_id=source.doc_id,
                    text=piece,
                    page_ids=source.page_ids,
                )
            )

    return rechunked


### ======== ALI ASIF ER CHUNKING LIES HERE ===========

# def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
#     """Re-chunk to cfg['index'] size/overlap.

#     Concatenates incoming (region-level) chunks per doc_id into a single
#     token stream, then re-windows it into fixed-size, overlapping chunks
#     of ``chunk_tokens`` whitespace tokens with ``overlap`` tokens shared
#     between consecutive windows. ``page_ids`` on the output chunk is the
#     de-duplicated, order-preserving union of source pages contributing
#     tokens to that window.
#     """
#     index_cfg = cfg["index"]
#     chunk_tokens = int(index_cfg["chunk_tokens"])
#     overlap = int(index_cfg["overlap"])
#     if chunk_tokens <= 0:
#         raise ValueError("index.chunk_tokens must be positive")
#     if overlap < 0 or overlap >= chunk_tokens:
#         raise ValueError("index.overlap must be in [0, chunk_tokens)")

#     by_doc: dict[str, list[Chunk]] = {}
#     for c in chunks:
#         by_doc.setdefault(c.doc_id, []).append(c)

#     step = chunk_tokens - overlap
#     out: list[Chunk] = []
#     for doc_id, doc_chunks in by_doc.items():
#         tokens: list[str] = []
#         token_pages: list[list[str]] = []
#         for c in doc_chunks:
#             for word in c.text.split():
#                 tokens.append(word)
#                 token_pages.append(c.page_ids)

#         n = len(tokens)
#         if n == 0:
#             continue

#         chunk_num = 0
#         start = 0
#         while start < n:
#             end = min(start + chunk_tokens, n)
#             window_pages = token_pages[start:end]
#             page_ids = list(dict.fromkeys(p for pages in window_pages for p in pages))
#             out.append(
#                 Chunk(
#                     id=f"{doc_id}__chunk_{chunk_num:04d}",
#                     doc_id=doc_id,
#                     text=" ".join(tokens[start:end]),
#                     page_ids=page_ids,
#                 )
#             )
#             chunk_num += 1
#             if end >= n:
#                 break
#             start += step

#     return out

