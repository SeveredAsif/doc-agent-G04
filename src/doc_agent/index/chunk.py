"""Stage 4 — chunk text"""
from __future__ import annotations

import re

from ..contracts import Chunk
from .embed import Encoder

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


def _semantic_units(text: str) -> list[tuple[str, bool]]:
    """Return sentences, marking boundaries introduced by textbook structure."""
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    block_starts_at_marker = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append((" ".join(current), block_starts_at_marker))
                current = []
                block_starts_at_marker = False
            continue

        starts_new_unit = any(line.startswith(marker) for marker in _BOUNDARY_MARKERS)
        if starts_new_unit and current:
            blocks.append((" ".join(current), block_starts_at_marker))
            current = []
            block_starts_at_marker = True
        elif not current:
            block_starts_at_marker = starts_new_unit
        current.append(line)

    if current:
        blocks.append((" ".join(current), block_starts_at_marker))

    units: list[tuple[str, bool]] = []
    for block, starts_at_marker in blocks or [(text, False)]:
        sentences = _sentences(block) or [block.strip()]
        units.extend(
            (sentence, index == 0 and starts_at_marker)
            for index, sentence in enumerate(sentences)
        )
    return [(unit, hard_break) for unit, hard_break in units if unit]


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


def _semantic_split(
    text: str,
    encoder: Encoder,
    max_tokens: int,
    min_tokens: int,
    similarity_threshold: float,
) -> list[str]:
    """Group adjacent sentences until a topic shift, heading, or size limit."""
    units = _semantic_units(text)
    if not units:
        return []

    sentences = [sentence for sentence, _ in units]
    vectors = encoder.encode_texts(sentences)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for index, (sentence, hard_break) in enumerate(units):
        sentence_len = len(_tokens(sentence))
        previous_similarity = (
            float(vectors[index - 1] @ vectors[index]) if index else 1.0
        )
        starts_new_chunk = bool(current) and (
            hard_break
            or current_len + sentence_len > max_tokens
            or (current_len >= min_tokens and previous_similarity < similarity_threshold)
        )
        if starts_new_chunk:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        if sentence_len > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(_window_by_tokens(sentence, max_tokens, 0))
            continue

        current.append(sentence)
        current_len += sentence_len

    if current:
        tail = " ".join(current)
        if chunks and len(_tokens(tail)) < min_tokens:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    return chunks


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk OCR text into retrieval-sized pieces.

    Semantic mode embeds each sentence and starts a new chunk where adjacent
    sentence similarity falls below the configured threshold. Textbook markers
    (for example, "উদাহরণ" and "প্রমাণ") remain hard boundaries.
    """
    index_cfg = cfg.get("index", {})
    max_tokens = int(index_cfg.get("chunk_tokens", 256))
    overlap = int(index_cfg.get("overlap", 32))
    mode = index_cfg.get("chunking", "semantic")
    min_tokens = int(index_cfg.get("min_chunk_tokens", 24))
    similarity_threshold = float(index_cfg.get("semantic_similarity_threshold", 0.35))

    if max_tokens <= 0:
        raise ValueError("index.chunk_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("index.overlap must be >= 0 and smaller than index.chunk_tokens")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("index.semantic_similarity_threshold must be between -1 and 1")

    rechunked: list[Chunk] = []
    encoder = Encoder(cfg) if mode == "semantic" else None
    for source in chunks:
        text = " ".join(source.text.split())
        if not text:
            continue

        if mode == "fixed":
            pieces = _window_by_tokens(text, max_tokens, overlap)
        elif mode == "semantic":
            pieces = _semantic_split(
                source.text,
                encoder,
                max_tokens,
                min_tokens,
                similarity_threshold,
            )
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
