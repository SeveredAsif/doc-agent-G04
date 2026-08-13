"""Stage 4 — chunk text"""
from __future__ import annotations
from ..contracts import *  # noqa


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap.

    Concatenates incoming (region-level) chunks per doc_id into a single
    token stream, then re-windows it into fixed-size, overlapping chunks
    of ``chunk_tokens`` whitespace tokens with ``overlap`` tokens shared
    between consecutive windows. ``page_ids`` on the output chunk is the
    de-duplicated, order-preserving union of source pages contributing
    tokens to that window.
    """
    index_cfg = cfg["index"]
    chunk_tokens = int(index_cfg["chunk_tokens"])
    overlap = int(index_cfg["overlap"])
    if chunk_tokens <= 0:
        raise ValueError("index.chunk_tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("index.overlap must be in [0, chunk_tokens)")

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    step = chunk_tokens - overlap
    out: list[Chunk] = []
    for doc_id, doc_chunks in by_doc.items():
        tokens: list[str] = []
        token_pages: list[list[str]] = []
        for c in doc_chunks:
            for word in c.text.split():
                tokens.append(word)
                token_pages.append(c.page_ids)

        n = len(tokens)
        if n == 0:
            continue

        chunk_num = 0
        start = 0
        while start < n:
            end = min(start + chunk_tokens, n)
            window_pages = token_pages[start:end]
            page_ids = list(dict.fromkeys(p for pages in window_pages for p in pages))
            out.append(
                Chunk(
                    id=f"{doc_id}__chunk_{chunk_num:04d}",
                    doc_id=doc_id,
                    text=" ".join(tokens[start:end]),
                    page_ids=page_ids,
                )
            )
            chunk_num += 1
            if end >= n:
                break
            start += step

    return out
