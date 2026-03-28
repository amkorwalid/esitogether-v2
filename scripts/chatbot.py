import json
import sys
import os
import textwrap
from typing import List
from urllib import response

import chromadb
import httpx
import tiktoken
from openai import OpenAI
import load_dotenv


load_dotenv.load_dotenv()

# ----------------------------
# Config
# ----------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "esitogether_documents"

# Embedding model (same as ingestion)
EMBED_MODEL = "qwen3-embedding:0.6b"

OLLAMA_URL = "http://localhost:11434"

TOP_K = 5


# ----------------------------
# Tokenizer
# ----------------------------

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

# ----------------------------
# Embedding: question -> vector
# ----------------------------

def embed_query(question: str) -> List[float]:
    with httpx.Client(timeout=None) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBED_MODEL,
                "prompt": question,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    embedding = data.get("embedding")
    if embedding is None:
        raise ValueError(f"No 'embedding' in Ollama embedding response: {data}")
    return embedding


# ----------------------------
# Chroma: retrieve top-k chunks
# ----------------------------

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

def retrieve_context(query_embedding: List[float], top_k: int = TOP_K):
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contexts = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        contexts.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return contexts


# ----------------------------
# Prompt building
# ----------------------------

def build_prompt(question: str, contexts: List[dict]) -> str:
    pieces = []
    total = 0
    for c in contexts:
        txt = c["text"]
        pieces.append(txt)
        total += len(txt)

    context_block = "\n\n---\n\n".join(pieces)

    system_instructions = (
        "You are a helpful assistant that answers questions based ONLY on the provided context.\n"
        "Answer in the same language as the user's question."
    )

    prompt = f"""{system_instructions}

Context:
{context_block}

Question:
{question}

Answer:"""

    return prompt


# ----------------------------
# Call Ollama LLM with streaming
# ----------------------------

def call_llm_stream(prompt: str):
    """
    Stream the response from Ollama and print tokens as they arrive.
    Also returns the full answer text (if you want to log it).
    """
    response = client.responses.create(
        model="gpt-5.4-nano",
        input=prompt
    )
    print(response.output_text)

    return response.output_text


# ----------------------------
# Main RAG function (no printing here)
# ----------------------------

def rag_answer_stream(question: str):
    # 1) Embed query
    q_emb = embed_query(question)

    # 2) Retrieve from Chroma
    contexts = retrieve_context(q_emb, top_k=TOP_K)

    if not contexts:
        print("Je ne trouve aucune information pertinente dans la base de connaissances.")
        return

    # 3) Build prompt
    prompt = build_prompt(question, contexts)

    # 4) Call LLM in streaming mode (printing as we go)
    full_answer = call_llm_stream(prompt)
    return full_answer


# ----------------------------
# CLI loop
# ----------------------------

def main():
    print("RAG Chatbot (Ollama + Chroma) - streaming mode")
    print(f"- Collection: {COLLECTION_NAME} (path: {CHROMA_PATH})")
    print(f"- Embed model: {EMBED_MODEL}")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        print("Assistant:\n")
        try:
            _ = rag_answer_stream(question)
        except Exception as e:
            print(f"\n[Error during RAG call: {e}]")
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()