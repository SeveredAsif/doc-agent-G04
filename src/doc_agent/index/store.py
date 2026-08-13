"""Stage 4 — vector store"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts import *  # noqa


def _index_dir(cfg: dict) -> Path:
    return Path(cfg.get("index", {}).get("output_dir", "data/processed/index"))


def _chunk_to_json(chunk: Chunk) -> dict:
    return chunk.model_dump()


def _chunk_from_json(row: dict) -> Chunk:
    return Chunk(**row)


def _faiss_index(vectors: np.ndarray, index_type: str):
    import faiss

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    dim = vectors.shape[1]
    normalized = np.ascontiguousarray(vectors.astype(np.float32))
    if index_type in {"faiss:flat", "faiss:flat-ip", "flat"}:
        index = faiss.IndexFlatIP(dim)
    elif index_type == "faiss:hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 80
    else:
        raise ValueError(f"Unsupported index.type: {index_type}")

    index.add(normalized)
    return index


def build(chunks, vectors, cfg: dict) -> None:
    """Persist a FAISS index plus chunk and build metadata sidecars."""
    import faiss
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
    if len(chunks) == 0:
        raise ValueError("Cannot build an index with zero chunks")

    index_cfg = cfg.get("index", {})
    output_dir = _index_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_type = index_cfg.get("type", "faiss:flat")
    index = _faiss_index(vectors, index_type)
    faiss.write_index(index, str(output_dir / "index.faiss"))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(_chunk_to_json(chunk), ensure_ascii=False) + "\n")

    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "embedding_model": cfg.get("embed", {}).get("model"),
        "index_type": index_type,
        "normalized_vectors": True,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load(cfg: dict):
    """Load index artifacts; returns (faiss_index, chunks, metadata)."""
    import faiss

    output_dir = _index_dir(cfg)
    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"
    metadata_path = output_dir / "metadata.json"

    if not index_path.is_file():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Missing chunk sidecar: {chunks_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing index metadata: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with chunks_path.open(encoding="utf-8") as f:
        chunks = [_chunk_from_json(json.loads(line)) for line in f if line.strip()]
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(chunks):
        raise ValueError(f"Index/chunk mismatch: {index.ntotal} vectors vs {len(chunks)} chunks")
    return index, chunks, metadata
