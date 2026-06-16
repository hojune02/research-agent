from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title="Soundable Research Agent",
    description="Local multi-user research automation agent with RAG, tool calling, memory, and local LLM serving.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Sounable Research Agent backend is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_llm": settings.MOCK_LLM,
        "llm_base_url": settings.LLM_BASE_URL,
        "llm_model": settings.LLM_MODEL,
        "upload_dir": settings.UPLOAD_DIR,
        "chroma_path": settings.CHROMA_PATH,
        "sqlite_path": settings.SQLITE_PATH,
        "top_k": settings.TOP_K,
    }