"""
tests/test_vector_store.py
--------------------------
Unit tests for VectorStore: adding, searching, MMR, persistence, removal.
"""

import tempfile
import numpy as np
import pytest

from core.vector_store import VectorStore, RetrievalResult, _calibrate_confidence, _mmr
from core.chunker import StructureAwareChunker

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

DIM = 32   # small dimension for fast tests

def _random_unit_vec(dim: int = DIM) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_chunks(n: int, doc_id: str = "test_doc"):
    chunker = StructureAwareChunker()
    text    = "\n\n".join([f"This is paragraph {i} of the document." for i in range(n)])
    return chunker.chunk(text, doc_id)


def _make_store_with_chunks(n: int = 20) -> tuple[VectorStore, list, np.ndarray]:
    store   = VectorStore(dimension=DIM)
    chunks  = _make_chunks(n)
    vectors = np.stack([_random_unit_vec() for _ in range(len(chunks))])
    store.add_chunks(chunks, vectors)
    return store, chunks, vectors


# ──────────────────────────────────────────────────────────────
# Confidence calibration
# ──────────────────────────────────────────────────────────────

def test_confidence_high():
    assert _calibrate_confidence(0.90) >= 80

def test_confidence_low():
    assert _calibrate_confidence(0.30) <= 25

def test_confidence_mid():
    score = _calibrate_confidence(0.65)
    assert 40 <= score <= 70

def test_confidence_bounds():
    for score in np.linspace(-1, 1, 50):
        pct = _calibrate_confidence(float(score))
        assert 0 <= pct <= 100


# ──────────────────────────────────────────────────────────────
# VectorStore: basic operations
# ──────────────────────────────────────────────────────────────

def test_add_and_count():
    store, chunks, vecs = _make_store_with_chunks(10)
    assert store.num_chunks == len(chunks)


def test_search_returns_results():
    store, chunks, vecs = _make_store_with_chunks(20)
    query = _random_unit_vec()
    results = store.search(query, top_k=5, mmr_k=3, min_score=-1.0)
    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_search_sorted_by_score():
    store, _, _ = _make_store_with_chunks(20)
    query       = _random_unit_vec()
    results     = store.search(query, top_k=10, mmr_k=5, min_score=-1.0)
    scores      = [r.cosine_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_exact_top_result():
    """The nearest vector to a query should be retrieved as top result."""
    store   = VectorStore(dimension=DIM)
    chunks  = _make_chunks(10)
    vectors = np.stack([_random_unit_vec() for _ in range(len(chunks))])
    store.add_chunks(chunks, vectors)

    # Query is identical to vectors[3]
    query   = vectors[3].copy()
    results = store.search(query, top_k=10, mmr_k=1, min_score=-1.0)
    assert len(results) == 1
    assert results[0].cosine_score == pytest.approx(1.0, abs=0.01)


def test_min_score_filter():
    store, _, _ = _make_store_with_chunks(20)
    query       = _random_unit_vec()
    results     = store.search(query, top_k=10, mmr_k=5, min_score=2.0)  # impossible threshold
    assert results == []


def test_doc_filter():
    store   = VectorStore(dimension=DIM)
    chunks_a = _make_chunks(5, "doc_A")
    chunks_b = _make_chunks(5, "doc_B")
    vecs_a   = np.stack([_random_unit_vec() for _ in range(5)])
    vecs_b   = np.stack([_random_unit_vec() for _ in range(5)])
    store.add_chunks(chunks_a, vecs_a)
    store.add_chunks(chunks_b, vecs_b)

    query   = _random_unit_vec()
    results = store.search(query, top_k=10, mmr_k=5, min_score=-1.0, doc_filter="doc_A")
    assert all(r.doc_id == "doc_A" for r in results)


def test_remove_doc():
    store, _, _ = _make_store_with_chunks(10)
    removed     = store.remove_doc("test_doc")
    assert removed == 10
    assert store.num_chunks == 0


def test_remove_nonexistent_doc():
    store, _, _ = _make_store_with_chunks(5)
    removed     = store.remove_doc("does_not_exist")
    assert removed == 0
    assert store.num_chunks == 5


def test_doc_ids_list():
    store = VectorStore(dimension=DIM)
    for i in range(3):
        chunks = _make_chunks(3, f"doc_{i}")
        vecs   = np.stack([_random_unit_vec() for _ in range(len(chunks))])
        store.add_chunks(chunks, vecs)
    assert set(store.doc_ids) == {"doc_0", "doc_1", "doc_2"}


# ──────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────

def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store, chunks, vecs = _make_store_with_chunks(15)
        store.save(tmpdir)

        loaded = VectorStore.load(tmpdir)
        assert loaded.num_chunks == store.num_chunks
        assert loaded.dimension  == store.dimension

        query   = _random_unit_vec()
        results = loaded.search(query, top_k=5, mmr_k=3, min_score=-1.0)
        assert len(results) == 3


# ──────────────────────────────────────────────────────────────
# MMR
# ──────────────────────────────────────────────────────────────

def test_mmr_reduces_redundancy():
    """
    If we have 2 near-duplicate vectors and 1 diverse vector,
    MMR with λ<0.5 should select the diverse one over the duplicate.
    """
    base    = _random_unit_vec()
    near    = (base + 0.01 * _random_unit_vec())
    near    = near / np.linalg.norm(near)
    diverse = _random_unit_vec()

    vecs   = np.stack([base, near, diverse])
    chunks = _make_chunks(3)

    results_mmr = _mmr(
        query_vec    = base,
        candidate_vecs = vecs,
        candidates   = chunks,
        k            = 2,
        lambda_mmr   = 0.3,   # diversity-heavy
    )
    # With heavy diversity weight, the diverse vector should be selected
    assert len(results_mmr) == 2
