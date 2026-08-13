"""Unit tests for Stage 4 indexing and minimal dense retrieval."""
import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk, embed, store
from doc_agent.retrieval.retriever import Retriever


def _cfg(tmp_path):
    return {
        "device": "cpu",
        "embed": {"model": "local:hashing", "dim": 64},
        "index": {
            "type": "faiss:flat",
            "chunking": "semantic",
            "chunk_tokens": 18,
            "overlap": 4,
            "output_dir": str(tmp_path / "index"),
        },
        "retrieve": {"k": 2, "k_step": 2, "k_max": 4, "weak_threshold": 0.2},
    }


def test_semantic_chunking_respects_textbook_boundaries(tmp_path):
    cfg = _cfg(tmp_path)
    source = Chunk(
        id="math__p0163__region_0000",
        doc_id="math",
        page_ids=["math__p0163"],
        text=(
            "উপপাদ্য ২০। বৃত্তের একই চাপের উপর দণ্ডায়মান কেন্দ্রস্থ কোণ দ্বিগুণ।\n"
            "প্রমাণ: ধাপ ১। OA = OB.\n"
            "উদাহরণ ১। কেন্দ্রস্থ কোণ নির্ণয় কর।"
        ),
    )

    chunks = chunk.split([source], cfg)

    assert len(chunks) >= 2
    assert all(c.id.startswith(source.id) for c in chunks)
    assert any("উপপাদ্য" in c.text for c in chunks)
    assert any("উদাহরণ" in c.text for c in chunks)


def test_store_load_and_retrieve_round_trip(tmp_path):
    pytest.importorskip("faiss")
    cfg = _cfg(tmp_path)
    chunks = [
        Chunk(id="c1", doc_id="math", page_ids=["p1"], text="কেন্দ্রস্থ কোণ বৃত্তের কেন্দ্রে থাকে"),
        Chunk(id="c2", doc_id="math", page_ids=["p2"], text="সমান্তর ধারা পদের সমষ্টি নির্ণয়"),
        Chunk(id="c3", doc_id="math", page_ids=["p3"], text="ত্রিভুজের কোণের সমষ্টি দুই সমকোণ"),
    ]

    vectors = embed.encode(chunks, cfg)
    store.build(chunks, vectors, cfg)
    index, loaded_chunks, metadata = store.load(cfg)
    results = Retriever(cfg).retrieve("কেন্দ্রস্থ কোণ", k=2)

    assert index.ntotal == 3
    assert len(loaded_chunks) == 3
    assert metadata["embedding_dim"] == 64
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].id == "c1"
