import os
from typing import List

import chromadb
import httpx
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars (OPENAI_API_KEY, etc.)
load_dotenv()

# ----------------------------
# Config
# ----------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "esitogether_documents"

EMBED_MODEL = "qwen3-embedding:0.6b"  # Ollama embedding model
OLLAMA_URL = "http://localhost:11434"

TOP_K = 5

# ----------------------------
# Tokenizer
# ----------------------------

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

# ----------------------------
# Embedding: question -> vector (Ollama)
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
    for c in contexts:
        txt = c["text"]
        pieces.append(txt)

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
# Call OpenAI LLM (non-stream for HTTP API)
# ----------------------------

def call_llm(prompt: str) -> str:
    # Using Responses API (like in your script)
    resp = openai_client.responses.create(
        model="gpt-5.4-nano",
        input=prompt,
    )
    return resp.output_text.strip()

# ----------------------------
# RAG function
# ----------------------------

def rag_answer(question: str) -> str:
    q_emb = embed_query(question)
    contexts = retrieve_context(q_emb, top_k=TOP_K)

    if not contexts:
        return "Je ne trouve aucune information pertinente dans la base de connaissances."

    prompt = build_prompt(question, contexts)
    answer = call_llm(prompt)
    return answer

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="ESI Together RAG API")

# CORS (allow front-end running on same or different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for HTML frontend) from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        answer = rag_answer(req.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}