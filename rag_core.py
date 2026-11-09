"""Core RAG logic shared by the Streamlit app (app.py) and the evaluation script (evaluate.py).

Responsibilities:
  * configuration from environment variables / .env
  * reading PDFs page by page, with an OCR fallback for scanned pages
  * building or loading the Chroma vector index (rebuilt automatically when the PDFs
    or the embedding model change)
  * prompt construction and the Gemini REST call

This module deliberately does not import Streamlit.
"""
import hashlib
import json
import os
import re
import time

import chromadb
import pytesseract
import requests
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from pdf2image import convert_from_path
from pypdf import PdfReader

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DOCS_DIR = "docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "regulations"
MANIFEST_PATH = "./index_manifest.json"
TOP_K = 3                 # chunks retrieved per question
# Chroma searches with an approximate index (HNSW) whose default beam width (search_ef=10) can miss true nearest
# neighbours. With search_ef above the number of chunks the search is effectively exact and reproducible.
# Keep it above the chunk count if the corpus grows.
HNSW_SEARCH_EF = 512
MIN_PAGE_CHARS = 100      # pages with less extracted text than this are treated as scans

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")        # empty = use the system PATH
POPPLER_PATH = os.getenv("POPPLER_PATH") or None    # empty = use the system PATH

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# The assistant answers in Indonesian, so the prompt and the refusal sentence stay in Indonesian.
# The evaluation detects refusals by this exact sentence: change it and old results are not comparable.
REFUSAL_SENTENCE = "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia."


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
def build_embed_model(name):
    """Create the local embedding model. e5 models need 'query: ' / 'passage: ' prefixes."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # heavy import, kept lazy

    if "e5" in name.lower():
        return HuggingFaceEmbedding(model_name=name, query_instruction="query: ", text_instruction="passage: ")
    return HuggingFaceEmbedding(model_name=name)


def configure_llama():
    """Register the embedding model globally and disable LlamaIndex's own LLM (Gemini is called directly)."""
    Settings.embed_model = build_embed_model(EMBED_MODEL)
    Settings.llm = None


# --------------------------------------------------------------------------- #
# Reading PDFs (page by page, OCR only where needed)
# --------------------------------------------------------------------------- #
def is_garbled(text):
    """True when extracted text looks like garbage (typical for scanned PDFs)."""
    words = text.split()
    if not words:
        return True
    readable = [w for w in words if sum(c.isalpha() for c in w) / len(w) > 0.6]
    return len(readable) / len(words) < 0.4


def ocr_page(pdf_path, page_number):
    """OCR a single page (1-based) at 300 dpi so digits and small print stay legible."""
    image = convert_from_path(
        pdf_path, dpi=300, first_page=page_number, last_page=page_number, poppler_path=POPPLER_PATH
    )[0]
    return pytesseract.image_to_string(image, lang="ind")


def clean_text(text):
    """Remove recurring noise from the source site."""
    return text.replace("www.peraturan.go.id", "").strip()


def load_pages(docs_dir):
    """Read every PDF page by page and return one LlamaIndex Document per page.

    The OCR decision is made per page, so a PDF with native-text pages plus a scanned
    appendix keeps its appendix. Metadata (file_name, page, ocr) is stored on each page.
    """
    documents = []
    for file_name in sorted(os.listdir(docs_dir)):
        if not file_name.lower().endswith(".pdf"):
            continue
        path = os.path.join(docs_dir, file_name)
        for page_number, page in enumerate(PdfReader(path).pages, start=1):
            text = (page.extract_text() or "").strip()
            used_ocr = False
            if len(text) < MIN_PAGE_CHARS or is_garbled(text):
                print(f"[OCR] {file_name} page {page_number}", flush=True)
                text = ocr_page(path, page_number)
                used_ocr = True
            text = clean_text(text)
            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={"file_name": file_name, "page": page_number, "ocr": used_ocr},
                        # page/OCR flags are bookkeeping: keep them out of embeddings and prompts
                        excluded_embed_metadata_keys=["page", "ocr"],
                        excluded_llm_metadata_keys=["page", "ocr"],
                    )
                )
    return documents


# --------------------------------------------------------------------------- #
# Index (Chroma) with automatic rebuild
# --------------------------------------------------------------------------- #
def docs_signature(docs_dir):
    """Fingerprint of everything the index depends on: embedding model + PDF contents."""
    signature = {"_embed_model": EMBED_MODEL, "_hnsw_search_ef": HNSW_SEARCH_EF}
    for file_name in sorted(os.listdir(docs_dir)):
        if file_name.lower().endswith(".pdf"):
            with open(os.path.join(docs_dir, file_name), "rb") as f:
                signature[file_name] = hashlib.md5(f.read()).hexdigest()
    return signature


def index_is_current(docs_dir):
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f) == docs_signature(docs_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def manifest_embed_model():
    """Embedding model the current index was built with (None if unknown)."""
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f).get("_embed_model")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=chromadb.Settings(anonymized_telemetry=False))


def build_or_load_index():
    """Load the persisted index, or (re)build it when the PDFs or the embedding model changed."""
    configure_llama()
    client = _chroma_client()
    if not index_is_current(DOCS_DIR):
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # the collection did not exist yet
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:search_ef": HNSW_SEARCH_EF})
    store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=store)

    if collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store=store, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_documents(load_pages(DOCS_DIR), storage_context=storage_context)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(docs_signature(DOCS_DIR), f)
    return index, collection


def open_existing_index():
    """Open the persisted index read-only for evaluation. Never rebuilds; fails loudly on a mismatch."""
    built_with = manifest_embed_model()
    if built_with != EMBED_MODEL:
        raise RuntimeError(
            f"The index was built with embedding model '{built_with}' but EMBED_MODEL is '{EMBED_MODEL}'. "
            "Run the app once (open the page until it finishes loading) to rebuild the index."
        )
    configure_llama()
    try:
        collection = _chroma_client().get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError("No index found. Run the app once so it can build the index.") from error
    if collection.count() == 0:
        raise RuntimeError("The index is empty. Run the app once so it can build the index.")
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(store), collection


# --------------------------------------------------------------------------- #
# Prompt and Gemini
# --------------------------------------------------------------------------- #
def build_prompt(question, nodes):
    """Prompt sent to Gemini: retrieved pages as context, answer only from them."""
    context = "\n\n".join(
        f"[{n.metadata.get('file_name', 'Unknown')}, hlm. {n.metadata.get('page', '?')}]\n{n.text}"
        for n in nodes
    )
    return f"""Kamu asisten tanya jawab regulasi Indonesia.
Jawab HANYA berdasarkan dokumen di bawah. Jika jawabannya tidak ada di dokumen,
jawab persis: "{REFUSAL_SENTENCE}" Jangan menebak.
Sebutkan nama dokumen dan pasal atau halaman yang menjadi dasar jawaban.

Dokumen:
{context}

Pertanyaan: {question}

Jawab dalam Bahasa Indonesia yang jelas dan ringkas."""


def is_refusal(answer):
    """True when the answer is the standard 'not found' sentence."""
    normalize = lambda s: " ".join(s.lower().split())
    return normalize(REFUSAL_SENTENCE.rstrip(".")) in normalize(answer)


def call_gemini(prompt, api_key=None, model=None, post=None, sleep=time.sleep, max_attempts=4, retry_wait=10):
    """Call the Gemini REST API. Returns (text, ok).

    * 429 per-minute and 5xx errors are retried with a growing wait (scaled by retry_wait)
    * errors that retrying cannot fix (bad key, unknown model, daily quota) return immediately
    * the error text includes the quota id and suggested wait time when Google provides them
    """
    api_key = api_key or GEMINI_API_KEY
    model = model or GEMINI_MODEL
    post = post or requests.post
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last_error = "[ERROR: no attempt made]"

    for attempt in range(1, max_attempts + 1):
        try:
            response = post(
                url,
                headers={"x-goog-api-key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60,
            )
        except requests.exceptions.RequestException as error:
            last_error = f"[ERROR network: {type(error).__name__}]"
            sleep(retry_wait / 2 * attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), True
            except (KeyError, IndexError, ValueError):
                return "[ERROR: empty or blocked response]", False

        body = " ".join(response.text.split())[:300]
        quota_ids = sorted(set(re.findall(r'"quotaId":\s*"([^"]+)"', response.text)))
        retry_delay = re.search(r'"retryDelay":\s*"([^"]+)"', response.text)
        detail = (" | quota: " + ", ".join(quota_ids) if quota_ids else "") + (
            f" | retry in {retry_delay.group(1)}" if retry_delay else ""
        )
        last_error = f"[ERROR HTTP {response.status_code}]{detail} {body}"

        if response.status_code == 429 and "PerDay" not in response.text:
            sleep(2 * retry_wait * attempt)  # per-minute limit: wait, then retry
            continue
        if response.status_code in (500, 503):
            sleep(retry_wait * attempt)
            continue
        return last_error, False  # 400/403/404 or daily quota: retrying is pointless
    return last_error, False