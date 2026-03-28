import json
import os
from pathlib import Path
from typing import List
from uuid import uuid4

import httpx
import tiktoken
import chromadb
from chromadb.config import Settings

# ----------------------------
# Chroma setup
# ----------------------------

COLLECTION_NAME = "esitogether_documents"

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' ready for ingestion (existing data kept).")

# ----------------------------
# Configuration
# ----------------------------

MARKDOWN_DIR = "./data"              # folder with your .md files
PROCESSED_JSON_PATH = "./processed_files.json"

CHUNK_SIZE_TOKENS = 3000             # target token size
OVERLAP_RATIO = 0.10                 # 10% overlap
OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * OVERLAP_RATIO)

MAX_EMBED_TOKENS = 4000              # safety cap for tokens
MAX_CHARS_PER_CHUNK = 30000          # hard cap for characters (your request)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"

# ----------------------------
# Tokenization helpers
# ----------------------------

def get_tokenizer():
    """Use tiktoken's cl100k_base as a practical approximation."""
    return tiktoken.get_encoding("cl100k_base")

def tokenize_text(text: str, encoding) -> List[int]:
    return encoding.encode(text)

def detokenize(tokens: List[int], encoding) -> str:
    return encoding.decode(tokens)

# ----------------------------
# Chunking logic
# ----------------------------

def chunk_text_by_tokens(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[str]:
    """
    Chunk text into segments of up to `chunk_size` tokens,
    with `overlap` tokens repeated between consecutive chunks.
    Returns a list of text chunks (strings).
    """
    encoding = get_tokenizer()
    tokens = tokenize_text(text, encoding)
    n = len(tokens)

    chunks: List[str] = []
    start = 0
    idx = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break

        chunk_text = detokenize(chunk_tokens, encoding)

        # Extra safety: enforce a 30k-char limit at chunk level
        if len(chunk_text) > MAX_CHARS_PER_CHUNK:
            chunk_text = chunk_text[:MAX_CHARS_PER_CHUNK]

        idx += 1

        chunks.append(chunk_text)

        if end == n:
            break

        # Slide window with overlap
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

# ----------------------------
# Ollama Qwen3 embedding (with guards)
# ----------------------------

def embed_chunk_ollama_qwen(chunk: str) -> List[float]:
    """
    Call Ollama's /api/embeddings endpoint with qwen3-embedding:0.6b
    for a single chunk. Returns the embedding vector.
    Enforces both token and char limits.
    """
    # Hard char limit first
    if len(chunk) > MAX_CHARS_PER_CHUNK:
        # You can choose to skip instead:
        # raise ValueError(f"Chunk exceeds {MAX_CHARS_PER_CHUNK} chars; skipping.")
        chunk = chunk[:MAX_CHARS_PER_CHUNK]

    encoding = get_tokenizer()
    num_tokens = len(encoding.encode(chunk))
    if num_tokens > MAX_EMBED_TOKENS:
        raise ValueError(
            f"Chunk too long for embedding: {num_tokens} tokens "
            f"(max {MAX_EMBED_TOKENS}). Consider reducing CHUNK_SIZE_TOKENS "
            f"or handling this file specially."
        )

    with httpx.Client(timeout=None) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": chunk,
            },
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama /api/embeddings error {resp.status_code}: "
                f"{resp.text[:500]} "
                f"(chunk tokens={num_tokens}, chars={len(chunk)})"
            ) from e

        data = resp.json()

    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError(f"No 'embedding' field in Ollama response: {data}")
    return embedding

def embed_chunks_ollama_qwen(chunks: List[str]) -> List[List[float]]:
    """
    Embed many chunks sequentially using Qwen3 embeddings via Ollama.
    """
    embeddings: List[List[float]] = []
    for idx, chunk in enumerate(chunks):
        print(f"  Embedding chunk {idx + 1}/{len(chunks)} "
              f"(chars={len(chunk)})...")
        emb = embed_chunk_ollama_qwen(chunk)
        embeddings.append(emb)
    return embeddings

# ----------------------------
# File discovery
# ----------------------------

def load_markdown_files(folder: str) -> List[Path]:
    """
    Return all .md files in folder (recursive).
    """
    base = Path(folder)
    return list(base.rglob("*.md"))

# ----------------------------
# Processed files JSON helpers
# ----------------------------

def load_processed_map(path: str) -> dict:
    """Load JSON mapping of processed files -> metadata."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_processed_map(path: str, data: dict) -> None:
    """Save processed files map to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def file_needs_embedding(md_path: Path, processed_map: dict) -> bool:
    """
    Decide whether this file needs (re)embedding, based on:
    - If it's not in the processed_map, or
    - If last_modified has changed since last embedding.
    """
    abs_path = str(md_path.resolve())
    current_mtime = md_path.stat().st_mtime

    entry = processed_map.get(abs_path)
    if entry is None:
        return True  # never processed
    last_mtime = entry.get("last_modified")
    return last_mtime is None or float(last_mtime) != float(current_mtime)

def mark_file_as_processed(md_path: Path, processed_map: dict) -> None:
    """Update processed_map with the current last_modified for this file."""
    abs_path = str(md_path.resolve())
    current_mtime = md_path.stat().st_mtime
    processed_map[abs_path] = {
        "last_modified": current_mtime
    }

# ----------------------------
# Per-file processing & upsert
# ----------------------------

def process_markdown_file(md_path: Path):
    """
    For a single markdown file:
      1) Read content
      2) Chunk into token-based chunks with overlap
      3) Enforce 30k-char max per chunk
      4) Embed via Qwen3 embeddings
      5) Add chunks directly to Chroma collection
    """
    print(f"\nProcessing file: {md_path}")
    text = md_path.read_text(encoding="utf-8")

    # 1) Chunk
    chunks = chunk_text_by_tokens(text)
    print(f"  Created {len(chunks)} chunks from {md_path.name}")

    if not chunks:
        print("  No chunks produced, skipping.")
        return

    # 2) Embed
    embeddings = embed_chunks_ollama_qwen(chunks)
    print(f"  Got {len(embeddings)} embeddings")

    # 3) Add to Chroma (per file)
    doc_id = str(uuid4())
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": str(md_path),
            "chunk_index": i,
            "doc_id": doc_id,
            "filename": md_path.name,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  Upserted {len(chunks)} chunks into collection.")

# ----------------------------
# Main script
# ----------------------------

def main():
    md_files = load_markdown_files(MARKDOWN_DIR)
    if not md_files:
        print(f"No markdown files found in {MARKDOWN_DIR}")
        return

    print(f"Found {len(md_files)} markdown files in {MARKDOWN_DIR}")

    processed_map = load_processed_map(PROCESSED_JSON_PATH)
    print(f"Loaded processed map with {len(processed_map)} entries.")

    processed_count = 0
    skipped_count = 0

    for md_path in md_files:
        abs_path = str(md_path.resolve())

        if file_needs_embedding(md_path, processed_map):
            print(f"\n[EMBED] {abs_path}")
            process_markdown_file(md_path)
            mark_file_as_processed(md_path, processed_map)
            processed_count += 1
            save_processed_map(PROCESSED_JSON_PATH, processed_map)
        else:
            print(f"\n[SKIP]  {abs_path} (already embedded, unchanged)")
            skipped_count += 1

    print(f"\nDone. Processed {processed_count} file(s), skipped {skipped_count} file(s).")
    print(f"Processed files map saved to {PROCESSED_JSON_PATH}.")

if __name__ == "__main__":
    main()
