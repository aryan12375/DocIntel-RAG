"""
app.py
------
Streamlit frontend for the RAG Document Intelligence System.

Run:  streamlit run app.py

Features:
  - Drag-and-drop multi-document upload
  - Real-time streaming answers
  - Source highlighting with page + heading breadcrumb
  - Live confidence gauge per source chunk
  - Query expansion toggle
  - Document management (list / delete)
  - Dark-mode UI with custom CSS matching the portfolio card
"""

import sys
import os
import logging
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title = "RAG Document Intelligence",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

  /* ── Cards ── */
  .rag-card {
    background: linear-gradient(135deg, #161b22 0%, #1a2035 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .source-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 4px solid #238636;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
  }
  .source-card.high   { border-left-color: #2ea043; }
  .source-card.medium { border-left-color: #d29922; }
  .source-card.low    { border-left-color: #f85149; }

  /* ── Confidence badge ── */
  .conf-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }
  .conf-high   { background: #1a3d2b; color: #2ea043; border: 1px solid #2ea043; }
  .conf-medium { background: #3d2e0a; color: #d29922; border: 1px solid #d29922; }
  .conf-low    { background: #3d1010; color: #f85149; border: 1px solid #f85149; }

  /* ── Heading path ── */
  .heading-path {
    color: #58a6ff;
    font-size: 0.80rem;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Answer box ── */
  .answer-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    font-size: 1.0rem;
    line-height: 1.75;
    white-space: pre-wrap;
  }

  /* ── Chunk highlight ── */
  .highlight {
    background: rgba(35, 134, 54, 0.15);
    border-radius: 3px;
    padding: 1px 3px;
  }

  /* ── Metric tiles ── */
  .metric-tile {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 20px;
    text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-lbl { font-size: 0.78rem; color: #8b949e; margin-top: 2px; }

  /* ── Buttons ── */
  .stButton>button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all .2s ease;
  }
  .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(31,111,235,.5); }

  /* ── Tags ── */
  .tag {
    display: inline-block;
    background: #1f2d3d;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    margin: 2px;
  }

  /* ── Section headers ── */
  .section-hdr {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 10px;
    margin-top: 24px;
  }

  /* ── Divider ── */
  hr { border-color: #21262d !important; }

  /* ── Input ── */
  .stTextInput>div>div>input, .stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────────

def _resolve_api_key() -> str:
    """
    Resolve OpenAI API key using a safe priority chain:
      1. Streamlit Cloud secrets  (set via dashboard — never committed to git)
      2. Environment variable     (set in shell or via .env loaded externally)

    NEVER hardcode the key here. NEVER commit a .env file.
    Streamlit Cloud: App Settings → Secrets → add:
        OPENAI_API_KEY = "sk-..."
    Local dev: export OPENAI_API_KEY=sk-... in your shell.
    """
    key = st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key


def _check_api_key():
    """Show a prominent error if the API key is missing — fail fast, fail clearly."""
    if not _resolve_api_key():
        st.error(
            "**OpenAI API key not found.**\n\n"
            "**Local dev:** `export OPENAI_API_KEY=sk-...` before running Streamlit.\n\n"
            "**Streamlit Cloud:** Go to **Settings → Secrets** and add:\n"
            "```toml\nOPENAI_API_KEY = \"sk-...\"\n```\n"
            "Never commit your key to GitHub.",
            icon="🔑",
        )
        st.stop()


@st.cache_resource(show_spinner="Initialising pipeline…")
def get_pipeline():
    from core.rag_pipeline import RAGPipeline, PipelineConfig
    cfg = PipelineConfig(openai_api_key=_resolve_api_key())
    return RAGPipeline(cfg)


def init_state():
    defaults = dict(
        query_result      = None,
        answer_text       = "",
        processing        = False,
        ingested_docs     = [],    # list of IngestionResult
        last_query        = "",
        expanded_queries  = [],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def confidence_class(pct: float) -> str:
    if pct >= 65: return "high"
    if pct >= 40: return "medium"
    return "low"


def confidence_badge(pct: float) -> str:
    cls = confidence_class(pct)
    return f'<span class="conf-badge conf-{cls}">{pct:.0f}%</span>'


def render_source_card(result, index: int) -> str:
    cls      = confidence_class(result.confidence_pct)
    badge    = confidence_badge(result.confidence_pct)
    heading  = " › ".join(result.heading_path) if result.heading_path else "—"
    page     = f"p.{result.page}" if result.page else "—"
    # Truncate chunk text for display
    display_text = result.chunk.text[:400] + ("…" if len(result.chunk.text) > 400 else "")

    return f"""
    <div class="source-card {cls}">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:0.8rem; color:#8b949e; font-weight:600;">
          [{index}] 📄 {result.doc_id}  ·  {page}
        </span>
        {badge}
      </div>
      <div class="heading-path">📍 {heading}</div>
      <div style="color:#c9d1d9;">{display_text}</div>
      <div style="margin-top:8px; font-size:0.75rem; color:#6e7681;">
        cosine = {result.cosine_score:.4f}
        · hash = <code style="color:#58a6ff">{result.chunk.metadata.chunk_hash}</code>
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 20px;">
      <div style="font-size:2rem;">🔍</div>
      <div style="font-size:1.15rem; font-weight:700; color:#e6edf3;">DocIntel RAG</div>
      <div style="font-size:0.75rem; color:#8b949e; margin-top:4px;">
        Structure-aware · MMR · Explainable
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, or TXT files",
        type    = ["pdf", "docx", "txt", "md"],
        accept_multiple_files = True,
        label_visibility      = "collapsed",
    )

    if uploaded_files:
        if st.button("🚀 Ingest All Documents", use_container_width=True):
            pipeline = get_pipeline()
            progress = st.progress(0, text="Starting…")
            for i, f in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Ingesting {f.name}…")
                try:
                    result = pipeline.ingest(f.read(), f.name)
                    if 'ingested_docs' not in st.session_state:
                        st.session_state.ingested_docs = []
                    st.session_state.ingested_docs.append(result)
                    st.success(f"✅ {f.name} — {result.chunks_added} chunks")
                except Exception as exc:
                    st.error(f"❌ {f.name}: {exc}")
            progress.progress(1.0, text="Done!")
            st.rerun()

    st.markdown('<div class="section-hdr">Indexed Documents</div>', unsafe_allow_html=True)

    doc_ids = []
    try:
        pipeline = get_pipeline()
        doc_ids  = pipeline.list_documents()
        if doc_ids:
            for did in doc_ids:
                cols = st.columns([4, 1])
                cols[0].markdown(f"📄 `{did}`")
                if cols[1].button("🗑", key=f"del_{did}", help=f"Delete {did}"):
                    pipeline.delete_document(did)
                    st.rerun()
        else:
            st.caption("No documents indexed yet.")
    except Exception:
        st.caption("Pipeline not ready.")

    st.markdown("---")

    st.markdown('<div class="section-hdr">Settings</div>', unsafe_allow_html=True)
    
    st.session_state.min_confidence = st.slider(
        "Minimum Confidence (%)",
        min_value=0,
        max_value=100,
        value=0, 
        help="Lower this if the AI says it can't find an answer."
    )

    use_expansion = st.toggle("Query Expansion (RRF)", value=True)
    
    doc_filter = st.selectbox(
        "Filter to document",
        options = ["All documents"] + doc_ids,
    )
    
    mmr_k = st.slider("Max sources", 1, 8, 5)

    st.markdown("---")

    # ── NEW: OPTIMIZATION BUTTON ──────────────────────────────────
    st.markdown('<div class="section-hdr">System Maintenance</div>', unsafe_allow_html=True)
    if st.button("🧹 Reset & Optimize Index", use_container_width=True):
        import shutil
        import os
        # Path from your PipelineConfig persist_dir
        index_path = ".index" 
        cache_path = ".cache"
        
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            
        st.cache_resource.clear()
        st.success("Wiped! Re-ingest now for 400-token optimization.")
        st.rerun()
    # ──────────────────────────────────────────────────────────────

    st.markdown("---")
    
    st.markdown("""
    <div style="font-size:0.72rem; color:#6e7681; text-align:center; line-height:1.8;">
      Stack: FAISS · sentence-transformers<br>
      all-MiniLM-L6-v2 · Llama-3.1-8B (Groq)<br>
      IndexFlatIP cosine similarity
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────

# Validate API key before rendering anything interactive
_check_api_key()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="rag-card" style="margin-bottom:28px;">
  <h1 style="margin:0 0 6px; font-size:1.8rem; font-weight:700;">
    🔍 RAG Document Intelligence System
  </h1>
  <p style="margin:0; color:#8b949e; font-size:0.9rem;">
    Structure-aware chunking · MMR retrieval · Real cosine confidence ·
    Query expansion via RRF · Explainable citations
  </p>
</div>
""", unsafe_allow_html=True)


# ── Metrics row ───────────────────────────────────────────────
try:
    pipeline = get_pipeline()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    for col, val, lbl in [
        (m_col1, pipeline.document_count,    "Documents"),
        (m_col2, pipeline.chunk_count,       "Chunks"),
        (m_col3, pipeline.embedder.dimension,"Embed Dim"),
        (m_col4, "FlatIP", "FAISS Index"),
    ]:
        col.markdown(f"""
        <div class="metric-tile">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
except Exception:
    pass

st.markdown("")

# ── Query input ───────────────────────────────────────────────
st.markdown('<div class="section-hdr">Ask a question about your documents</div>', unsafe_allow_html=True)

query_col, btn_col = st.columns([6, 1])
user_query = query_col.text_input(
    "Query",
    placeholder = "e.g. What are the key risk factors mentioned in the report?",
    label_visibility = "collapsed",
)
ask_btn = btn_col.button("Ask →", use_container_width=True)

# ── Run query ─────────────────────────────────────────────────
if ask_btn and user_query.strip():
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Pipeline init failed: {exc}")
        st.stop()

    if pipeline.chunk_count == 0:
        st.warning("⚠️ No documents indexed. Upload and ingest documents first.")
        st.stop()

    # Apply UI settings to pipeline
    pipeline.config.use_query_expansion = use_expansion
    pipeline.config.mmr_final_k         = mmr_k
    _filter = None if doc_filter == "All documents" else doc_filter

    # ── Streaming answer ──────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-hdr">Answer</div>', unsafe_allow_html=True)
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_answer = ""
        sources     = []

        with st.spinner("Retrieving & generating…"):
            for token_or_result in pipeline.query_stream(user_query, doc_filter=_filter):
                if isinstance(token_or_result, str):
                    full_answer += token_or_result
                    answer_placeholder.markdown(
                        f'<div class="answer-box">{full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Final QueryResult
                    sources = token_or_result.sources
                    st.session_state.query_result    = token_or_result
                    st.session_state.expanded_queries= token_or_result.expanded_queries

        # Final answer (no cursor)
        answer_placeholder.markdown(
            f'<div class="answer-box">{full_answer}</div>',
            unsafe_allow_html=True,
        )

        # ── Expanded queries used ─────────────────────────────
        if st.session_state.expanded_queries and len(st.session_state.expanded_queries) > 1:
            st.markdown('<div class="section-hdr">Query expansion (RRF fused)</div>', unsafe_allow_html=True)
            tags = "".join(f'<span class="tag">{q}</span>' for q in st.session_state.expanded_queries)
            st.markdown(f'<div>{tags}</div>', unsafe_allow_html=True)

        # ── Sources ───────────────────────────────────────────
        st.markdown('<div class="section-hdr">Source passages (confidence scored)</div>', unsafe_allow_html=True)

        if sources:
            # Confidence distribution bar
            scores = [r.confidence_pct for r in sources]
            avg    = sum(scores) / len(scores)
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
                        padding:12px 16px; margin-bottom:16px; font-family:'JetBrains Mono', monospace; font-size:0.82rem;">
              <span style="color:#8b949e;">Retrieved {len(sources)} passages  ·  </span>
              <span style="color:#58a6ff;">avg confidence: {avg:.0f}%  ·  </span>
              <span style="color:#8b949e;">top cosine: {sources[0].cosine_score:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

            for i, result in enumerate(sources, start=1):
                st.markdown(render_source_card(result, i), unsafe_allow_html=True)
        else:
            st.info("No relevant passages found above the confidence threshold.")


# ── Show last result if re-rendering ──────────────────────────
elif st.session_state.query_result and not ask_btn:
    qr = st.session_state.query_result
    st.markdown('<div class="section-hdr">Last answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{qr.answer}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Source passages</div>', unsafe_allow_html=True)
    for i, result in enumerate(qr.sources, start=1):
        st.markdown(render_source_card(result, i), unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#6e7681;">
      <div style="font-size:3rem; margin-bottom:16px;">📄</div>
      <div style="font-size:1.1rem; font-weight:600; color:#8b949e; margin-bottom:8px;">
        Upload documents, then ask anything
      </div>
      <div style="font-size:0.85rem; line-height:1.7;">
        The system will chunk them intelligently, embed with<br>
        all-MiniLM-L6-v2, retrieve via FAISS + MMR, and generate<br>
        a cited answer using GPT-4o-mini.
      </div>
    </div>
    """, unsafe_allow_html=True)
