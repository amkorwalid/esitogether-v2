"""
Document ingestion pipeline:
  1. Read markdown file
  2. Chunk by tokens with overlap
  3. Embed via Ollama (Qwen3)
  4. Upsert into ChromaDB
"""

import os
from pathlib import Path
from typing import List
from uuid import uuid4

import chromadb
import httpx
import tiktoken

# ----------------------------
# Config (read from env or defaults)
# ----------------------------

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "esitogether_documents")

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "3000"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.10"))
OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * OVERLAP_RATIO)
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "4000"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "20000"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# ----------------------------
# ChromaDB helpers
# ----------------------------


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


# ----------------------------
# Tokenization helpers
# ----------------------------


def _get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


# ----------------------------
# Chunking
# ----------------------------


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[str]:
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    n = len(tokens)
    chunks: List[str] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break

        chunk_text = enc.decode(chunk_tokens)
        if len(chunk_text) > MAX_CHARS_PER_CHUNK:
            chunk_text = chunk_text[:MAX_CHARS_PER_CHUNK]

        chunks.append(chunk_text)

        if end == n:
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ----------------------------
# Embedding via Ollama
# ----------------------------


def embed_chunk(chunk: str) -> List[float]:
    if len(chunk) > MAX_CHARS_PER_CHUNK:
        chunk = chunk[:MAX_CHARS_PER_CHUNK]

    enc = _get_tokenizer()
    num_tokens = len(enc.encode(chunk))
    if num_tokens > MAX_EMBED_TOKENS:
        raise ValueError(
            f"Chunk too long: {num_tokens} tokens (max {MAX_EMBED_TOKENS})"
        )

    with httpx.Client(timeout=None) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": chunk},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama embedding error {resp.status_code}: {resp.text[:400]}"
            ) from exc

        data = resp.json()

    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError(f"No 'embedding' field in Ollama response: {data}")
    return embedding


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for idx, chunk in enumerate(chunks):
        print(f"  Embedding chunk {idx + 1}/{len(chunks)} (chars={len(chunk)})...")
        embeddings.append(embed_chunk(chunk))
    return embeddings


# ----------------------------
# Embed query (for retrieval)
# ----------------------------


def embed_query(question: str) -> List[float]:
    with httpx.Client(timeout=None) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": question},
        )
        resp.raise_for_status()
        data = resp.json()
    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError(f"No 'embedding' in Ollama response: {data}")
    return embedding


# ----------------------------
# Ingest a markdown file
# ----------------------------


def ingest_markdown_file(md_path: Path, original_filename: str, upload_time: str) -> dict:
    """
    Read a markdown file, chunk it, embed it, and upsert into ChromaDB.

    Returns:
        dict with doc_id, filename, chunk_count
    """
    text = md_path.read_text(encoding="utf-8")
    chunks = chunk_text_by_tokens(text)

    if not chunks:
        raise ValueError(f"No content could be extracted from {md_path.name}")

    print(f"[ingest] '{md_path.name}': {len(chunks)} chunks — embedding...")
    embeddings = embed_chunks(chunks)

    doc_id = str(uuid4())
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": original_filename,
            "chunk_index": i,
            "upload_time": upload_time,
        }
        for i in range(len(chunks))
    ]

    collection = get_chroma_collection()
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[ingest] Stored {len(chunks)} chunks for doc_id={doc_id}")
    return {"doc_id": doc_id, "filename": original_filename, "chunk_count": len(chunks)}


# ----------------------------
# List documents in ChromaDB
# ----------------------------


def list_documents() -> List[dict]:
    """
    Return a deduplicated list of documents stored in ChromaDB.
    Each entry: {doc_id, filename, chunk_count, upload_time}
    """
    collection = get_chroma_collection()
    total = collection.count()
    if total == 0:
        return []

    results = collection.get(include=["metadatas"], limit=total)
    metadatas = results.get("metadatas") or []

    seen: dict = {}
    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "upload_time": meta.get("upload_time", ""),
                "chunk_count": 0,
            }
        seen[doc_id]["chunk_count"] += 1

    return list(seen.values())


# ----------------------------
# Delete a document from ChromaDB
# ----------------------------


def delete_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to doc_id. Returns number of chunks deleted.
    """
    collection = get_chroma_collection()
    total = collection.count()
    if total == 0:
        return 0

    results = collection.get(
        where={"doc_id": doc_id},
        include=["metadatas"],
    )
    ids_to_delete = results.get("ids") or []

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"[ingest] Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")

    return len(ids_to_delete)


# ----------------------------
# RAG retrieval
# ----------------------------


def retrieve_context(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]
