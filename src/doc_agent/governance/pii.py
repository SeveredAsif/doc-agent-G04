"""Governance — PII detection + redaction (mandatory)"""
from __future__ import annotations

import re

from ..contracts import *  # noqa


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    # Bangladesh +880 format and common local 01XXXXXXXXX format
    "phone": re.compile(
        r"(?<!\d)(?:\+?880|0)1[3-9]\d{8}(?!\d)"
    ),

    # Bangladesh NID numbers: 10, 13, or 17 digits
    "nid": re.compile(
        r"(?<!\d)(?:\d{10}|\d{13}|\d{17})(?!\d)"
    ),

    # Common card-number format: 13–19 digits, optionally separated
    # by spaces or hyphens.
    "card": re.compile(
        r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
    ),
}


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans."""

    spans: list[tuple[int, int, str]] = []

    for pii_type, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append(
                (match.start(), match.end(), pii_type)
            )

    # Sort by location in the original text.
    spans.sort(key=lambda span: (span[0], span[1]))

    # Remove overlapping spans.
    result: list[tuple[int, int, str]] = []

    for span in spans:
        start, end, pii_type = span

        if result and start < result[-1][1]:
            # Keep the earlier/longer span.
            prev_start, prev_end, prev_type = result[-1]

            if end > prev_end:
                result[-1] = span

            continue

        result.append(span)

    return result


def redact(text: str) -> str:
    """Replace detected PII with [REDACTED]."""

    spans = detect(text)

    if not spans:
        return text

    parts: list[str] = []
    cursor = 0

    for start, end, _pii_type in spans:
        parts.append(text[cursor:start])
        parts.append("[REDACTED]")
        cursor = end

    parts.append(text[cursor:])

    return "".join(parts)


def register(hooks) -> None:
    """Wire PII redaction into the pipeline."""

    def _scrub(ctx: dict) -> dict:
        """Redact PII from text-like fields in the hook context."""

        # OCR provides {"chunks": [...]}
        if "chunks" in ctx:
            chunks = ctx["chunks"]

            if isinstance(chunks, list):
                for chunk in chunks:
                    if hasattr(chunk, "text"):
                        chunk.text = redact(chunk.text)

                    elif isinstance(chunk, dict) and "text" in chunk:
                        chunk["text"] = redact(chunk["text"])

            elif isinstance(chunks, str):
                ctx["chunks"] = redact(chunks)

        # Answer text
        if "answer" in ctx and isinstance(ctx["answer"], str):
            ctx["answer"] = redact(ctx["answer"])

        # Log text
        if "message" in ctx and isinstance(ctx["message"], str):
            ctx["message"] = redact(ctx["message"])

        if "text" in ctx and isinstance(ctx["text"], str):
            ctx["text"] = redact(ctx["text"])

        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)
    hooks.register(hooks.BEFORE_ANSWER, _scrub)
    hooks.register(hooks.ON_LOG, _scrub)