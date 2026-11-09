import sys
import torch.nn as nn

sys.modules["nn"] = nn

import html
import os
import streamlit as st
import rag_core

# The questions are Indonesian because the documents and the users are Indonesian.
EXAMPLE_QUESTIONS = [
    "Nomor berapa yang digunakan untuk layanan panggilan darurat?",
    "Berapa lama masa berlaku Sertifikat SMPI?",
    "Anak umur berapa yang akunnya harus dinonaktifkan di media sosial?",
    "Berapa jam sehari Sistem Pusat Panggilan Darurat beroperasi?",
]

CARD_CSS = """
<style>
.source-card {
    background-color: rgba(120, 120, 120, 0.08);
    border-left: 3px solid #4a90d9;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.source-card-ocr { border-left: 3px solid #d9a14a; }
.relevance-badge {
    display: inline-block;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 10px;
    background-color: rgba(74, 144, 217, 0.15);
    margin-left: 6px;
}
</style>
"""

@st.cache_resource(show_spinner="Loading the regulations (scanned pages are OCR'd on the first run, please wait)...")
def get_index():
    return rag_core.build_or_load_index()


def source_from_node(node):
    """Flatten a retrieved node into a plain dict (safe to keep in the chat history)."""
    return {
        "file": node.metadata.get("file_name", "Unknown"),
        "page": node.metadata.get("page"),
        "ocr": node.metadata.get("ocr", False),
        "score": node.score or 0.0,
        "text": node.text[:350] + "...",
    }


def render_sources(sources):
    """Show the retrieved pages as small cards."""
    for i, source in enumerate(sources, start=1):
        card_class = "source-card source-card-ocr" if source["ocr"] else "source-card"
        page_label = f", p. {source['page']}" if source["page"] else ""
        ocr_label = " 🔍 <i>(OCR)</i>" if source["ocr"] else ""
        snippet = html.escape(" ".join(source["text"].split()))  # OCR text is full of blank lines
        st.markdown(
            f"""
            <div class="{card_class}">
                <b>Source {i}</b> — <i>{html.escape(str(source['file']))}</i>{page_label}{ocr_label}
                <span class="relevance-badge">relevance: {source['score']:.3f}</span>
                <p style="margin-top:8px; font-size:0.9rem; opacity:0.85;">{snippet}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Indonesian Regulation Q&A", page_icon="🏛️", layout="centered")

if not rag_core.GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and fill it in.")
    st.stop()

st.markdown(CARD_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏛️ Indonesian Regulation Q&A")
    st.caption("Ask questions about Indonesian regulations, answered from the source documents.")
    st.divider()

    if os.path.exists(rag_core.DOCS_DIR):
        pdf_files = [f for f in os.listdir(rag_core.DOCS_DIR) if f.lower().endswith(".pdf")]
        st.markdown(f"**📄 Documents loaded:** {len(pdf_files)}")
        with st.expander("Show document list"):
            for name in pdf_files:
                st.write(f"- {name}")
    else:
        st.warning(f"Folder '{rag_core.DOCS_DIR}' not found")

    st.divider()
    st.markdown("**ℹ️ How it works**")
    st.caption(
        "Every page is checked: if its text is empty or unreadable (for example a scan), "
        "that page is run through OCR before indexing."
    )
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        f"LLM: {rag_core.GEMINI_MODEL} · Embeddings: {rag_core.EMBED_MODEL.split('/')[-1]} · OCR: Tesseract"
    )

index, _collection = get_index()
retriever = index.as_retriever(similarity_top_k=rag_core.TOP_K)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🏛️ Indonesian Regulation Q&A")
st.caption("Answers are grounded in the loaded documents and cite their source pages.")

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
example_query = None
if not st.session_state.messages:
    st.markdown("**💡 Example questions:**")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[i % 2].button(example, use_container_width=True, key=f"example_{i}"):
            example_query = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View sources"):
                render_sources(message["sources"])

query = st.chat_input("Ask a question (Indonesian works best)...") or example_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            nodes = retriever.retrieve(query)
            # Interactive use: retry briefly rather than wait minutes for a rate limit to clear.
            answer, ok = rag_core.call_gemini(rag_core.build_prompt(query, nodes), max_attempts=3, retry_wait=2)
            if not ok:
                print(f"Gemini call failed: {answer}", flush=True)  # reason goes to the terminal
                answer = "⚠️ Gemini is busy or not responding. Please try again in a moment."
            st.markdown(answer)

            sources = [source_from_node(node) for node in nodes]
            with st.expander("📚 View sources"):
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
