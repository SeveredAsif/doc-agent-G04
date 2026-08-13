"""Stage 4 — embed chunks"""
from __future__ import annotations

import hashlib

import numpy as np

from ..contracts import *  # noqa


class Encoder:
    """Small wrapper around the configured embedding model."""

    def __init__(self, cfg: dict) -> None:
        embed_cfg = cfg.get("embed", {})
        self.model_name = embed_cfg.get("model", "local:hashing")
        self.dim = int(embed_cfg.get("dim", 384))
        self.batch_size = int(embed_cfg.get("batch_size", 32))
        self.allow_hash_fallback = bool(embed_cfg.get("allow_hash_fallback", True))
        self._model = None

        if not self.model_name.startswith("local:"):
            try:
                from sentence_transformers import SentenceTransformer

                device = cfg.get("device", "cpu")
                self._model = SentenceTransformer(self.model_name, device=device)
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception as exc:
                if not self.allow_hash_fallback:
                    raise RuntimeError(
                        f"Could not load embedding model {self.model_name!r}. "
                        "Set embed.allow_hash_fallback=true for offline smoke tests."
                    ) from exc
                self.model_name = f"local:hashing-fallback:{self.model_name}"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._model is not None:
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype(np.float32)

        vectors = np.vstack([self._hash_embed(text) for text in texts]).astype(np.float32)
        return _normalize(vectors)

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def encode_texts(texts: list[str], cfg: dict) -> np.ndarray:
    return Encoder(cfg).encode_texts(texts)


def encode(chunks: list[Chunk], cfg: dict):
    """Embed with cfg['embed']['model']; vectors are normalized for cosine/IP search."""
    return encode_texts([chunk.text for chunk in chunks], cfg)
