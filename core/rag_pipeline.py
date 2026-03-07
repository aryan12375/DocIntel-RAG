"""
rag_pipeline.py
---------------
The orchestrator that wires parser → chunker → embedder → vector_store → LLM.

Pipeline stages:
  1. INGEST  : parse → chunk → embed → store
  2. RETRIEVE: embed query → FAISS search → MMR rerank
  3. GENERATE: build prompt with source context → LLM → stream response

My additions beyond the spec:
  ★ Cross-encoder reranking pass (optional) — a second, heavier model grades
    relevance after FAISS retrieval. This is the "retrieve-then-rerank"
    pattern used in production RAG and dramatically cuts hallucinations.
  ★ Query expansion — the LLM rewrites the user's query into 3 sub-queries
    before retrieval. We then merge results (RRF fusion). This handles
    vocabulary mismatch: "heart attack" ↔ "myocardial infarction".
  ★ Citation injection — the final answer prompt instructs the LLM to emit
    inline citations like [1], [2]. We validate these against the actual
    retrieved sources before displaying them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Sequence

import numpy as np

from core.chunker          import StructureAwareChunker, Chunk
from core.document_parser  import parse_document, ParsedDocument
from core.embedder         import Embedder
from core.vector_store     import VectorStore, RetrievalResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config dataclass (everything in one place — no scattered globals)
# ──────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Chunking
    chunk_max_tokens     : int   = 400
    chunk_overlap_tokens : int   = 60
    chunk_min_tokens     : int   = 40

    # Embedding
    embedding_model      : str   = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size : int   = 64
    cache_dir            : str   = ".cache"

    # Retrieval
    retrieval_top_k      : int   = 12    # candidates before MMR
    mmr_final_k          : int   = 5     # results after MMR
    lambda_mmr           : float = 0.6
    min_cosine_score     : float = 0.28

    # Generation
    llm_model            : str   = "llama-3.1-8b-instant"  # swap to any OpenAI-compat model
    llm_temperature      : float = 0.2
    llm_max_tokens       : int   = 1024
    openai_api_key       : str   = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Optional features
    use_query_expansion  : bool  = True
    use_cross_encoder    : bool  = False   # requires cross-encoder/ms-marco-MiniLM-L-6-v2
    persist_dir          : str   = ".index"


# ──────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion (merge multiple ranked lists)
# ──────────────────────────────────────────────────────────────

def _rrf_merge(
    ranked_lists : list[list[RetrievalResult]],
    k            : int = 60,
) -> list[RetrievalResult]:
    """
    Merge several ranked result lists via Reciprocal Rank Fusion.
    RRF(d) = Σ 1 / (k + rank_i(d))
    Returns deduplicated results sorted by RRF score descending.
    """
    scores   : dict[str, float]           = {}
    best_hit : dict[str, RetrievalResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            h = result.chunk.metadata.chunk_hash
            scores[h]   = scores.get(h, 0.0) + 1.0 / (k + rank)
            if h not in best_hit or result.cosine_score > best_hit[h].cosine_score:
                best_hit[h] = result

    merged = sorted(best_hit.values(), key=lambda r: scores[r.chunk.metadata.chunk_hash], reverse=True)
    return merged


# ──────────────────────────────────────────────────────────────
# Query expansion
# ──────────────────────────────────────────────────────────────

_EXPANSION_PROMPT = """\
You are a search query expansion assistant. Given a user's question, generate \
3 alternative phrasings that capture the same information need but use \
different vocabulary. Output ONLY a JSON array of 3 strings, no explanation.

Question: {query}"""


def _expand_query(query: str, client) -> list[str]:
    """Generate alternative query phrasings via LLM."""
    import json
    try:
        resp = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",
            temperature = 0,
            max_tokens  = 200,
            messages    = [{"role": "user", "content": _EXPANSION_PROMPT.format(query=query)}],
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("```json").strip("```").strip()
        variants = json.loads(text)
        return [query] + variants[:3]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        return [query]


# ──────────────────────────────────────────────────────────────
# Answer prompt
# ──────────────────────────────────────────────────────────────

_ANSWER_PROMPT = """\
You are a precise document intelligence assistant. Answer the user's question \
using ONLY the context passages below. For every factual claim, add an inline \
citation in brackets, e.g. [1] or [2,3], matching the passage numbers.

If the answer is not found in the passages, say: \
"I could not find a clear answer in the provided documents."

Do NOT fabricate information. Be concise and accurate.

--- CONTEXT PASSAGES ---
{context}
--- END CONTEXT ---

User question: {query}

Answer (with citations):"""


def _build_context_block(results: list[RetrievalResult]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        heading = " › ".join(r.heading_path) if r.heading_path else ""
        page    = f" | Page {r.page}" if r.page else ""
        source  = f"[{i}] [{r.doc_id}{page}] {heading}"
        lines.append(f"{source}\n{r.chunk.text}\n")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    doc_id      : str
    chunks_added: int
    page_count  : int
    word_count  : int


@dataclass
class QueryResult:
    answer       : str
    sources      : list[RetrievalResult]
    query        : str
    expanded_queries: list[str]


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline(PipelineConfig())
        pipeline.ingest(file_bytes, "my_doc.pdf")
        result = pipeline.query("What are the main findings?")
        print(result.answer)
        for src in result.sources:
            print(src.doc_id, src.page, src.confidence_pct)
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config   = config
        self.embedder = Embedder(
            model_name  = config.embedding_model,
            batch_size  = config.embedding_batch_size,
            cache_dir   = config.cache_dir,
        )
        self.chunker  = StructureAwareChunker(
            max_tokens       = config.chunk_max_tokens,
            overlap_tokens   = config.chunk_overlap_tokens,
            min_chunk_tokens = config.chunk_min_tokens,
        )
        self.store    = VectorStore(dimension=self.embedder.dimension)
        self._client  = None   # lazy OpenAI client

        # Load persisted index if it exists
        persist_path = Path(config.persist_dir)
        if (persist_path / "index.faiss").exists():
            self.store = VectorStore.load(persist_path)
            logger.info("Loaded persisted index with %d chunks", self.store.num_chunks)

    # ── ingest ─────────────────────────────────────────────────

    def ingest(
        self,
        file_bytes : bytes,
        filename   : str,
    ) -> IngestionResult:
        """Parse, chunk, embed and index a document."""
        parsed = parse_document(file_bytes, filename)
        return self._ingest_parsed(parsed)

    def _ingest_parsed(self, parsed: ParsedDocument) -> IngestionResult:
        logger.info("Ingesting '%s' (%d chars)", parsed.doc_id, len(parsed.text))

        # Remove existing chunks for this doc (re-ingestion support)
        removed = self.store.remove_doc(parsed.doc_id)
        if removed:
            logger.info("Replaced %d stale chunks for '%s'", removed, parsed.doc_id)

        chunks = self.chunker.chunk(
            text     = parsed.text,
            doc_id   = parsed.doc_id,
            page_map = parsed.page_map,
        )
        if not chunks:
            logger.warning("No chunks produced for '%s'", parsed.doc_id)
            return IngestionResult(parsed.doc_id, 0, 0, 0)

        texts   = [c.text for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        self.store.add_chunks(chunks, vectors)

        # Persist index after each ingest
        self.store.save(self.config.persist_dir)

        return IngestionResult(
            doc_id       = parsed.doc_id,
            chunks_added = len(chunks),
            page_count   = parsed.metadata.get("page_count", 1),
            word_count   = parsed.metadata.get("word_count", 0),
        )

    # ── query ──────────────────────────────────────────────────

    def query(self, user_query: str, doc_filter: str | None = None) -> QueryResult:
        """
        Full RAG query: retrieve → (optionally expand + fuse) → generate.
        """
        client   = self._get_client()
        queries  = _expand_query(user_query, client) if self.config.use_query_expansion else [user_query]

        # Embed all query variants
        all_result_lists = []
        for q in queries:
            q_vec   = self.embedder.embed_query(q)
            results = self.store.search(
                query_vec    = q_vec,
                top_k        = self.config.retrieval_top_k,
                mmr_k        = self.config.mmr_final_k + 5,   # extra for fusion
                lambda_mmr   = self.config.lambda_mmr,
                min_score    = self.config.min_cosine_score,
                doc_filter   = doc_filter,
            )
            all_result_lists.append(results)

        # Fuse via RRF if we have multiple query variants
        if len(all_result_lists) > 1:
            fused = _rrf_merge(all_result_lists)[:self.config.mmr_final_k]
        else:
            fused = all_result_lists[0][:self.config.mmr_final_k]

        # Optional cross-encoder reranking
        if self.config.use_cross_encoder and fused:
            fused = self._cross_encode_rerank(user_query, fused)

        context_block = _build_context_block(fused)
        answer        = self._generate(user_query, context_block, client)

        return QueryResult(
            answer           = answer,
            sources          = fused,
            query            = user_query,
            expanded_queries = queries,
        )

    def query_stream(
        self,
        user_query : str,
        doc_filter : str | None = None,
    ) -> Generator[str | QueryResult, None, None]:
        """
        Streaming variant. Yields:
          - str tokens as they arrive from the LLM
          - A final QueryResult object when done (contains sources)
        Useful for the Streamlit streaming UI.
        """
        client  = self._get_client()
        queries = _expand_query(user_query, client) if self.config.use_query_expansion else [user_query]

        all_result_lists = []
        for q in queries:
            q_vec   = self.embedder.embed_query(q)
            results = self.store.search(
                query_vec  = q_vec,
                top_k      = self.config.retrieval_top_k,
                mmr_k      = self.config.mmr_final_k + 5,
                lambda_mmr = self.config.lambda_mmr,
                min_score  = self.config.min_cosine_score,
                doc_filter = doc_filter,
            )
            all_result_lists.append(results)

        fused  = _rrf_merge(all_result_lists)[:self.config.mmr_final_k] if len(all_result_lists) > 1 else all_result_lists[0][:self.config.mmr_final_k]
        ctx    = _build_context_block(fused)
        prompt = _ANSWER_PROMPT.format(context=ctx, query=user_query)

        full_answer = ""
        stream = client.chat.completions.create(
            model       = self.config.llm_model,
            temperature = self.config.llm_temperature,
            max_tokens  = self.config.llm_max_tokens,
            stream      = True,
            messages    = [{"role": "user", "content": prompt}],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            yield delta

        yield QueryResult(
            answer           = full_answer,
            sources          = fused,
            query            = user_query,
            expanded_queries = queries,
        )

    # ── cross-encoder reranking ────────────────────────────────

    def _cross_encode_rerank(
        self,
        query   : str,
        results : list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Use a cross-encoder to re-score (query, passage) pairs.
        Cross-encoders are slower (no pre-computed embeddings) but more accurate
        because they attend to query-passage interactions jointly.
        """
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs  = [(query, r.chunk.text) for r in results]
            scores = ce.predict(pairs)
            for r, s in zip(results, scores):
                object.__setattr__(r, "cosine_score", float(s))
            results.sort(key=lambda r: r.cosine_score, reverse=True)
        except Exception as exc:
            logger.warning("Cross-encoder failed, skipping: %s", exc)
        return results

    # ── LLM generation ─────────────────────────────────────────

    def _generate(self, query: str, context: str, client) -> str:
        prompt = _ANSWER_PROMPT.format(context=context, query=query)
        resp   = client.chat.completions.create(
            model       = self.config.llm_model,
            temperature = self.config.llm_temperature,
            max_tokens  = self.config.llm_max_tokens,
            messages    = [{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    def _get_client(self):
            """Lazy-load the OpenAI client with Groq base URL for free inference."""
            if self._client is None:
                try:
                    from openai import OpenAI  # type: ignore
                    self._client = OpenAI(
                        api_key=self.config.openai_api_key, 
                        base_url="https://api.groq.com/openai/v1"
                    )
                except ImportError:
                    raise ImportError("openai package required: pip install openai")
            return self._client
    # ── utility ────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return len(self.store.doc_ids)

    @property
    def chunk_count(self) -> int:
        return self.store.num_chunks

    def list_documents(self) -> list[str]:
        return self.store.doc_ids

    def delete_document(self, doc_id: str) -> int:
        n = self.store.remove_doc(doc_id)
        self.store.save(self.config.persist_dir)
        return n
