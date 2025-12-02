import sqlite3
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import openai

from config import (
    DB_PATH,
    DOCS_DIR,
    EMBED_MODEL_NAME,
    TOP_K,
    OPENAI_API_KEY,
    logger,
)

# ---- Model setup ----
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
openai.api_key = OPENAI_API_KEY

# Simple in-memory cache for query embeddings
_query_embedding_cache: dict[str, np.ndarray] = {}
MAX_QUERY_CACHE_SIZE = 100


# =========================
# Database utilities
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            chunk_text TEXT,
            embedding BLOB
        )
        """
    )
    conn.commit()
    conn.close()


def clear_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(doc_name: str, chunk_text: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    emb_json = json.dumps(embedding.tolist())
    cur.execute(
        """
        INSERT INTO chunks (doc_name, chunk_text, embedding)
        VALUES (?, ?, ?)
        """,
        (doc_name, chunk_text, emb_json),
    )
    conn.commit()
    conn.close()


def load_all_chunks() -> List[Tuple[int, str, str, np.ndarray]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_text, embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    chunks = []
    for _id, doc_name, chunk_text, emb_json in rows:
        emb = np.array(json.loads(emb_json), dtype=np.float32)
        chunks.append((_id, doc_name, chunk_text, emb))
    return chunks


# =========================
# Embeddings + retrieval
# =========================
def embed_text(texts: List[str]) -> np.ndarray:
    return np.array(embed_model.encode(texts, normalize_embeddings=True))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def get_query_embedding(query: str) -> np.ndarray:
    q = query.strip()
    if q in _query_embedding_cache:
        return _query_embedding_cache[q]

    emb = embed_text([q])[0]

    if len(_query_embedding_cache) >= MAX_QUERY_CACHE_SIZE:
        _query_embedding_cache.pop(next(iter(_query_embedding_cache)))

    _query_embedding_cache[q] = emb
    return emb


def retrieve_top_k(query: str, k: int = TOP_K):
    query_vec = get_query_embedding(query)
    chunks = load_all_chunks()
    scored = []
    for _id, doc_name, chunk_text, emb in chunks:
        score = cosine_sim(query_vec, emb)
        scored.append((score, doc_name, chunk_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# =========================
# Document indexing
# =========================
def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def index_documents():
    init_db()
    clear_db()

    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        logger.warning("Docs folder not found. Create a 'docs/' folder with .txt files.")
        return

    for path in docs_path.glob("*"):
        if not path.is_file():
            continue
        if path.suffix not in [".txt", ".md"]:
            continue

        doc_name = path.name
        text = path.read_text(encoding="utf-8")
        raw_chunks = chunk_text(text)

        chunks_with_header = [f"[DOC: {doc_name}]\n{c}" for c in raw_chunks]

        logger.info(f"Indexing {doc_name} with {len(chunks_with_header)} chunks...")
        embeddings = embed_text(chunks_with_header)
        for chunk_text_str, emb_vec in zip(chunks_with_header, embeddings):
            insert_chunk(doc_name, chunk_text_str, emb_vec)
    logger.info("Indexing done.")


# =========================
# LLM Call
# =========================
def call_llm(context: str, question: str) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the given context. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message["content"].strip()


# =========================
# Shared help + RAG pipeline
# =========================
def get_help_text() -> str:
    return (
        "Hi! I am a simple RAG bot.\n\n"
        "Commands:\n"
        "/ask <your question> - Ask me something about the docs.\n"
        "/image - (Not enabled in this variant, I only handle text.)\n"
        "/help - Show this message.\n"
        "/summarize - Summarize your recent interactions.\n"
    )


def run_rag_pipeline(query: str) -> tuple[str, list[str]]:
    """
    Shared RAG pipeline used by both Telegram and Gradio.
    Returns (answer, sources_lines).
    sources_lines is a list of human-readable strings:
      "- doc_name (score=...) → \"snippet\""
    """
    top_chunks = retrieve_top_k(query, k=TOP_K)
    if not top_chunks:
        return "I have no documents indexed yet. Check your docs/ folder.", []

    context_str = ""
    sources_lines: list[str] = []
    for score, doc_name, chunk_text in top_chunks:
        context_str += f"[From {doc_name}, score={score:.3f}]\n{chunk_text}\n\n"

        snippet = chunk_text.replace("\n", " ")
        snippet = (snippet[:120] + "...") if len(snippet) > 120 else snippet
        sources_lines.append(f"- {doc_name} (score={score:.3f}) → \"{snippet}\"")

    try:
        answer = call_llm(context_str, query)
    except Exception as e:
        logger.exception("Error calling LLM in RAG pipeline: %s", e)
        return "Error calling LLM. Check server logs/config.", []

    return answer, sources_lines


def rag_answer(query: str):
    """
    RAG pipeline for local UI (Gradio):
    - Handles some 'commands' locally (/start, /help, image)
    - Otherwise uses run_rag_pipeline(query).
    Returns: (answer, human_readable_sources)
    """
    query = query.strip()
    if not query:
        return "Please enter a question.", ""

    if query.lower() in ["/start", "/help"]:
        return get_help_text(), "Local help message — no retrieval done."

    if query.lower() in ["image", "/image"]:
        msg = (
            "This bot is running the Mini-RAG (text) variant.\n"
            "Image mode is not enabled here; I only handle text questions "
            "over the stored documents."
        )
        return msg, "Image mode disabled in this variant."

    answer, sources_lines = run_rag_pipeline(query)

    if sources_lines:
        human_sources = "Sources used:\n" + "\n".join(sources_lines)
    else:
        human_sources = "Sources used: (none)"

    return answer, human_sources

