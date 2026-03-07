"""
vector_store.py
---------------
FAISS-backed vector store with:
  - IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)
  - IndexIVFFlat for large corpora (>10k chunks) — ANN search, ~10× faster
  - Metadata store (chunk objects) keyed by FAISS integer ID
  - Hybrid MMR reranking to reduce redundant results

Why FAISS over Chroma / Weaviate / Pinecone?
  FAISS runs entirely in-process — zero infrastructure to stand up for a demo.
  IndexFlatIP is exact (not approximate), so similarity scores are mathematically
  precise. For a corpus < 50k chunks, latency is <5ms on CPU.
  Chroma/Weaviate add HTTP overhead and persistent server management that is
  unnecessary here. Pinecone is managed-cloud — overkill and costs money.

Distance metric choice:
  We use IndexFlatIP (inner product). Because our embeddings are L2-normalised,
  inner_product(a, b) == cosine_similarity(a, b) ∈ [-1, 1].
  Score 1.0 = identical, 0.0 = orthogonal, negative = opposite direction.
  We convert to a 0-100 confidence band for the UI.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except ImportError:
    raise ImportError(
        "faiss-cpu is required: pip install faiss-cpu"
    )


# ──────────────────────────────────────────────────────────────
# Result object
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieval hit with its raw cosine similarity score."""
    chunk          : "Chunk"    # noqa: F821  (forward ref, imported at runtime)
    cosine_score   : float      # raw inner-product score ∈ [-1, 1]
    confidence_pct : float      # 0-100 calibrated for UI display

    @property
    def doc_id(self) -> str:
        return self.chunk.metadata.doc_id

    @property
    def page(self) -> int | None:
        return self.chunk.metadata.page_hint

    @property
    def heading_path(self) -> tuple[str, ...]:
        return self.chunk.metadata.heading_path


# ──────────────────────────────────────────────────────────────
# Confidence calibration
# ──────────────────────────────────────────────────────────────

def _calibrate_confidence(cosine: float) -> float:
    """
    Map raw cosine similarity → 0-100 confidence percentage.

    Empirically, for all-MiniLM-L6-v2:
      cosine ≥ 0.85 → near-duplicate / direct answer
      0.65 – 0.84   → strong topical match
      0.45 – 0.64   → moderate relevance
      < 0.45        → weak / noise

    We use a sigmoid-like mapping anchored to those thresholds.
    """
    # Shift and scale so that 0.45→20%, 0.65→55%, 0.85→90%
    x = (cosine - 0.45) / 0.40          # normalise to ~[0, 1] range
    sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))
    return float(np.clip(sigmoid * 100, 0, 100))


# ──────────────────────────────────────────────────────────────
# MMR reranker
# ──────────────────────────────────────────────────────────────

def _mmr(
    query_vec    : np.ndarray,
    candidate_vecs: np.ndarray,
    candidates   : list,
    k            : int,
    lambda_mmr   : float = 0.6,
) -> list:
    """
    Maximal Marginal Relevance — balance relevance vs. diversity.

    λ=1.0 → pure relevance ranking (same as cosine order)
    λ=0.0 → pure diversity (greedy furthest-point traversal)
    λ=0.6 → default, good mix for RAG
    """
    selected_idx  : list[int] = []
    remaining_idx : list[int] = list(range(len(candidates)))

    while len(selected_idx) < k and remaining_idx:
        if not selected_idx:
            # First pick: highest relevance
            scores = candidate_vecs[remaining_idx] @ query_vec
            best   = remaining_idx[int(np.argmax(scores))]
        else:
            sel_vecs  = candidate_vecs[selected_idx]
            rel_scores= candidate_vecs[remaining_idx] @ query_vec
            red_scores= (candidate_vecs[remaining_idx] @ sel_vecs.T).max(axis=1)
            mmr_scores= lambda_mmr * rel_scores - (1 - lambda_mmr) * red_scores
            best      = remaining_idx[int(np.argmax(mmr_scores))]

        selected_idx.append(best)
        remaining_idx.remove(best)

    return [candidates[i] for i in selected_idx]


# ──────────────────────────────────────────────────────────────
# Vector store
# ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages a FAISS index + parallel metadata list.

    Index auto-upgrades from Flat → IVFFlat once corpus > IVF_THRESHOLD.
    """

    IVF_THRESHOLD = 10_000   # switch to ANN above this chunk count
    IVF_NLIST     = 128      # number of Voronoi cells
    IVF_NPROBE    = 16       # cells to search at query time (recall/speed knob)

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._index    : faiss.Index | None = None
        self._chunks   : list               = []   # parallel to FAISS IDs
        self._init_index()

    # ── index management ───────────────────────────────────────

    def _init_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info("Initialised IndexFlatIP (dim=%d)", self.dimension)

    def _maybe_upgrade_index(self) -> None:
        """Transparently upgrade to IVFFlat once threshold is crossed."""
        n = len(self._chunks)
        if n >= self.IVF_THRESHOLD and isinstance(self._index, faiss.IndexFlatIP):
            logger.info("Upgrading index to IndexIVFFlat (n=%d)", n)
            quantiser = faiss.IndexFlatIP(self.dimension)
            new_index = faiss.IndexIVFFlat(
                quantiser, self.dimension, self.IVF_NLIST, faiss.METRIC_INNER_PRODUCT
            )
            # Re-train on existing vectors
            all_vecs = np.vstack([
                self._index.reconstruct(i) for i in range(self._index.ntotal)
            ]).astype(np.float32)
            new_index.train(all_vecs)
            new_index.add(all_vecs)
            new_index.nprobe = self.IVF_NPROBE
            self._index      = new_index

    # ── write path ─────────────────────────────────────────────

    def add_chunks(
        self,
        chunks   : Sequence,
        vectors  : np.ndarray,
    ) -> None:
        """Add chunks + their pre-computed embedding vectors."""
        assert len(chunks) == len(vectors), "chunks and vectors must be same length"
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)          # type: ignore[union-attr]
        self._chunks.extend(chunks)
        self._maybe_upgrade_index()
        logger.info("Index size: %d chunks", len(self._chunks))

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document. Returns removed count."""
        keep_idx  = [i for i, c in enumerate(self._chunks) if c.metadata.doc_id != doc_id]
        removed   = len(self._chunks) - len(keep_idx)
        if removed == 0:
            return 0

        keep_vecs = np.vstack([
            self._index.reconstruct(i) for i in keep_idx   # type: ignore[union-attr]
        ]).astype(np.float32)
        kept_chunks = [self._chunks[i] for i in keep_idx]

        self._init_index()
        self._chunks = []
        if kept_chunks:
            self.add_chunks(kept_chunks, keep_vecs)

        logger.info("Removed %d chunks for doc '%s'", removed, doc_id)
        return removed

    # ── read path ──────────────────────────────────────────────

    def search(
        self,
        query_vec   : np.ndarray,
        top_k       : int   = 10,
        mmr_k       : int   = 5,
        lambda_mmr  : float = 0.6,
        min_score   : float = 0.30,
        doc_filter  : str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k chunks, optionally filtered by document, then rerank
        with MMR to maximise relevance-diversity trade-off.
        """
        if self._index is None or len(self._chunks) == 0:
            return []

        # ── NEW: Dynamic Thresholding ──────────────────────────
        # This reads the slider value you just added to app.py
        import streamlit as st
        # Convert the 0-100 slider value back to a 0.0-1.0 decimal for the math engine
        ui_threshold = st.session_state.get('min_confidence', 0) / 100.0
        # ──────────────────────────────────────────────────────

        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        raw_k    = min(top_k * 3, len(self._chunks))   # over-fetch for MMR
        scores, ids = self._index.search(q, raw_k)     # type: ignore[union-attr]

        candidates   : list          = []
        cand_vecs    : list[np.ndarray] = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            
            # Use ui_threshold instead of a hardcoded value
            if score < ui_threshold:
                break
                
            chunk = self._chunks[idx]
            if doc_filter and doc_filter != "All documents" and chunk.metadata.doc_id != doc_filter:
                continue
                
            candidates.append(
                RetrievalResult(
                    chunk          = chunk,
                    cosine_score   = float(score),
                    confidence_pct = _calibrate_confidence(float(score)),
                )
            )
            cand_vecs.append(self._index.reconstruct(int(idx)))  # type: ignore[union-attr]

        if not candidates:
            return []

        # MMR reranking
        cand_vecs_arr = np.vstack(cand_vecs).astype(np.float32)
        reranked       = _mmr(q[0], cand_vecs_arr, candidates, mmr_k, lambda_mmr)

        # Re-sort final results by cosine score descending
        reranked.sort(key=lambda r: r.cosine_score, reverse=True)
        return reranked

    # ── persistence ────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(d / "index.faiss"))  # type: ignore[arg-type]
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("Saved index + %d chunks to %s", len(self._chunks), d)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        index  = faiss.read_index(str(d / "index.faiss"))
        with open(d / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        store         = cls.__new__(cls)
        store.dimension = index.d
        store._index  = index
        store._chunks = chunks
        logger.info("Loaded index with %d chunks from %s", len(chunks), d)
        return store

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def doc_ids(self) -> list[str]:
        return list({c.metadata.doc_id for c in self._chunks})
