"""
embedder.py
-----------
Custom embedding pipeline with:
  - Batch processing with configurable size
  - Disk-level SHA-256 caching (avoid re-embedding unchanged docs)
  - L2-normalized vectors ready for cosine similarity via inner product
  - Pluggable model backend (sentence-transformers default)

Why sentence-transformers/all-MiniLM-L6-v2?
  384-dim vectors → small FAISS index, fast retrieval, good multilingual recall.
  For production, swap to text-embedding-3-small (OpenAI) or
  mixedbread-ai/mxbai-embed-large-v1 for higher accuracy at the cost of latency.

The EmbeddingCache stores (text_hash → numpy array) in a shelve database.
This is intentionally simple — swap for Redis or a vector-store built-in
cache in a distributed deployment.
"""

from __future__ import annotations

import hashlib
import logging
import shelve
import time
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lazy model loader — avoids slow import at module level
# ──────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}


def _load_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model: %s", model_name)
        t0 = time.perf_counter()
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Model loaded in %.2fs", time.perf_counter() - t0)
    return _model_cache[model_name]


# ──────────────────────────────────────────────────────────────
# Disk cache
# ──────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Persistent key-value store mapping text hashes to embedding vectors.
    Uses Python's shelve (backed by sqlite3/dbm) for zero-dependency storage.
    """

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self._path = str(Path(cache_dir) / "embeddings")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        with shelve.open(self._path) as db:
            vec = db.get(self._key(text))
        return vec  # type: ignore[return-value]

    def put(self, text: str, vector: np.ndarray) -> None:
        with shelve.open(self._path) as db:
            db[self._key(text)] = vector

    def clear(self) -> None:
        with shelve.open(self._path) as db:
            db.clear()


# ──────────────────────────────────────────────────────────────
# Core embedder
# ──────────────────────────────────────────────────────────────

class Embedder:
    """
    Converts text into L2-normalised embedding vectors.

    Parameters
    ----------
    model_name : str
        Sentence-Transformers model identifier.
    batch_size : int
        How many texts to embed per forward pass.
    use_cache  : bool
        Whether to skip re-embedding texts seen before.
    cache_dir  : str | Path
        Where to persist the embedding cache.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name : str            = DEFAULT_MODEL,
        batch_size : int            = 64,
        use_cache  : bool           = True,
        cache_dir  : str | Path     = ".cache",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache     = EmbeddingCache(cache_dir) if use_cache else None

    # ── public API ─────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return embedding dimension (used when initialising FAISS index)."""
        model = _load_model(self.model_name)
        return model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a list of strings. Returns shape (N, D) float32 array,
        each row L2-normalised so inner product == cosine similarity.
        """
        texts     = list(texts)
        n         = len(texts)
        result    = np.zeros((n, self.dimension), dtype=np.float32)
        to_embed  : list[tuple[int, str]] = []   # (original_index, text)

        # Cache hit pass
        for i, text in enumerate(texts):
            if self._cache is not None:
                cached = self._cache.get(text)
                if cached is not None:
                    result[i] = cached
                    continue
            to_embed.append((i, text))

        logger.info(
            "Embedding %d/%d texts (cache hits: %d)",
            len(to_embed), n, n - len(to_embed),
        )

        # Batch forward pass
        if to_embed:
            indices, raw_texts = zip(*to_embed)
            model = _load_model(self.model_name)
            embeddings = model.encode(  # type: ignore[union-attr]
                list(raw_texts),
                batch_size          = self.batch_size,
                show_progress_bar   = False,
                normalize_embeddings= True,   # L2 norm → cosine via inner product
                convert_to_numpy    = True,
            ).astype(np.float32)

            for idx, vec, text in zip(indices, embeddings, raw_texts):
                result[idx] = vec
                if self._cache is not None:
                    self._cache.put(text, vec)

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (D,) float32.
        Prefix the query with "query:" — many bi-encoders are trained
        with asymmetric prompts and perform better with this hint.
        """
        prefixed = f"query: {query}"
        vecs     = self.embed_texts([prefixed])
        return vecs[0]
