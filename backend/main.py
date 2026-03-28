import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.ingest import (
    delete_document,
    embed_query,
    ingest_markdown_file,
    list_documents,
    retrieve_context,
)
from rag.converter import docs_to_markdown

# Load env vars (OPENAI_API_KEY, etc.)
load_dotenv()

# ----------------------------
# Config
# ----------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))

# Directories (relative to the working directory, i.e. backend/)
DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
TMP_DIR = DATA_DIR / "tmp"

ALLOWED_EXTENSIONS = {".pdf", ".PDF"}

# ----------------------------
# Prompt building
# ----------------------------


def build_prompt(question: str, contexts: List[dict]) -> str:
    pieces = [c["text"] for c in contexts]
    context_block = "\n\n---\n\n".join(pieces)

    system_instructions = (
        "You are a helpful school-administration assistant. "
        "Answer questions based ONLY on the provided context. "
        "If the context does not contain the answer, say so politely. "
        "Answer in the same language as the user's question."
    )

    return (
        f"{system_instructions}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )


# ----------------------------
# Call OpenAI LLM
# ----------------------------


def call_llm(prompt: str) -> str:
    resp = openai_client.responses.create(
        model=LLM_MODEL,
        input=prompt,
    )
    return resp.output_text.strip()


# ----------------------------
# RAG answer
# ----------------------------


def rag_answer(question: str) -> str:
    q_emb = embed_query(question)
    contexts = retrieve_context(q_emb, top_k=TOP_K)

    if not contexts:
        return "Je ne trouve aucune information pertinente dans la base de connaissances."

    prompt = build_prompt(question, contexts)
    return call_llm(prompt)


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="ESI Together RAG API", version="2.0", docs_url="/api-docs", redoc_url="/api-redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# Page routes
# ----------------------------


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/docs-page", include_in_schema=False)
def docs_page():
    return RedirectResponse(url="/static/docs.html")


# ----------------------------
# Health
# ----------------------------


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


# ----------------------------
# Chat endpoint
# ----------------------------


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    try:
        answer = rag_answer(req.question)
        return ChatResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ----------------------------
# Document management endpoints
# ----------------------------


@app.get("/docs", tags=["Documents"])
def get_docs():
    """List all documents stored in the knowledge base."""
    try:
        docs = list_documents()
        return {"documents": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/docs/upload", tags=["Documents"])
async def upload_doc(file: UploadFile = File(...)):
    """
    Upload a PDF document.
    Pipeline: PDF → Markdown (Docling) → chunk → embed (Qwen3/Ollama) → ChromaDB.
    """
    suffix = Path(file.filename).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are supported. Got: '{suffix}'",
        )

    # Ensure data directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Save uploaded file (use a timestamped name to avoid conflicts)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    # Sanitize the filename stem to prevent directory traversal or path injection
    safe_stem = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in Path(file.filename).stem
    )[:100]
    raw_filename = f"{safe_stem}_{timestamp}.pdf"
    raw_path = RAW_DIR / raw_filename

    try:
        with open(raw_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    # Convert PDF → Markdown
    ok = docs_to_markdown(raw_path, CLEAN_DIR, TMP_DIR)
    if not ok:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to convert PDF to Markdown. Check server logs.",
        )

    md_path = CLEAN_DIR / f"{raw_path.stem}.md"
    if not md_path.exists():
        raw_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Markdown output not found after conversion.",
        )

    # Ingest into ChromaDB
    upload_time = datetime.now(timezone.utc).isoformat()
    try:
        result = ingest_markdown_file(md_path, file.filename, upload_time)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "message": "Document uploaded and indexed successfully.",
        **result,
    }


@app.delete("/docs/{doc_id}", tags=["Documents"])
def remove_doc(doc_id: str):
    """Delete a document and all its chunks from the knowledge base."""
    try:
        deleted_count = delete_document(doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with doc_id='{doc_id}'",
        )

    return {"message": f"Deleted {deleted_count} chunk(s) for doc_id='{doc_id}'", "deleted": deleted_count}