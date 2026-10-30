"""Minimal HTTP adapter for the existing Self-Healing RAG LangGraph workflow.

Copy this file to the root of the existing self-healing-rag project, next to
rag_agent.py, ingest.py, and pyproject.toml.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# rag_agent.py compiles the LangGraph workflow as `app` and initializes the
# Chroma collection as `collection`. Importing it also validates GROQ_API_KEY.
from rag_agent import app as rag_workflow
from rag_agent import collection


api = FastAPI(
    title="Self-Healing RAG API",
    version="0.1.0",
    description="HTTP adapter for the existing LangGraph RAG workflow.",
)

# Development origins for the Vite frontend. Restrict these further when hosted.
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RetrievedChunk(BaseModel):
    id: str
    text: str


@api.get("/api/health")
def health() -> dict[str, Any]:
    """Report API availability and the current Chroma collection size."""
    try:
        chunk_count = collection.count()
    except Exception as exc:  # pragma: no cover - environment-specific failure
        raise HTTPException(
            status_code=503,
            detail="The Chroma knowledge base is unavailable.",
        ) from exc

    return {
        "status": "ok",
        "knowledge_base_chunks": chunk_count,
    }


@api.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Run the existing self-healing LangGraph workflow for one question."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    initial_state = {
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "answer": "",
        "grade": "",
        "retry_count": 0,
    }

    try:
        result = rag_workflow.invoke(initial_state)
    except Exception as exc:  # Keep provider/internal details out of API output.
        print(f"[API] RAG workflow failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="The RAG workflow could not complete. Check the backend terminal.",
        ) from exc

    documents = result.get("documents", []) or []
    chunks = [
        {"id": f"retrieved-{index + 1}", "text": str(document)}
        for index, document in enumerate(documents)
    ]
    grade = str(result.get("grade", "UNKNOWN")).upper()

    return {
        "question": question,
        "answer": result.get("answer", "No answer was generated."),
        "grade": grade,
        "retry_count": int(result.get("retry_count", 0)),
        "rewritten_question": result.get("rewritten_question", ""),
        "retrieved_chunks": chunks,
        "retrieved_count": len(chunks),
        "outcome": "grounded" if grade == "PASS" else "not_verified",
    }
