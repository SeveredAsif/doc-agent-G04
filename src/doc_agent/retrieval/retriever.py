"""Stage 5 — dense retrieval"""
from __future__ import annotations

import numpy as np

from ..contracts import *  # noqa
from ..index import embed, store

class Retriever:
    def __init__(self, cfg: dict) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["retrieve"]
        self.index, self.chunks, self.metadata = store.load(cfg)
        self.encoder = embed.Encoder(cfg)

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Top-k dense retrieval. Set chunk.score (relevance) on every result so decide() can judge
        whether the evidence is weak. IMPLEMENT."""
        top_k = int(k or self.cfg.get("k", 10))
        if top_k <= 0:
            raise ValueError("k must be positive")

        query_vector = self.encoder.encode_texts([query])
        scores, indices = self.index.search(np.asarray(query_vector, dtype=np.float32), top_k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            source = self.chunks[int(idx)]
            results.append(source.model_copy(update={"score": float(score)}))
        return results


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)

def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]

def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
