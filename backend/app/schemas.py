from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to send to the LLM")
    context: str = Field(default="", description="Optional retrieved context")


class LLMResponse(BaseModel):
    text: str
    backend: str
    model: str
    mock: bool
    generation_latency_ms: int
    estimated_tokens_per_second: float

class UploadResponse(BaseModel):
    status: str
    user_id: str
    project_id: str
    filename: str
    saved_path: str
    size_bytes: int

class ParsePDFRequest(BaseModel):
    user_id: str
    project_id: str
    filename: str


class DocumentChunk(BaseModel):
    chunk_id: str
    user_id: str
    project_id: str
    source: str
    page: int
    text: str
    char_count: int


class ParsePDFResponse(BaseModel):
    status: str
    user_id: str
    project_id: str
    filename: str
    pages_parsed: int
    chunks_created: int
    chunks: list[DocumentChunk]

class ParsePDFPreviewResponse(BaseModel):
    status: str
    user_id: str
    project_id: str
    filename: str
    pages_parsed: int
    chunks_created: int
    preview_chunks: list[DocumentChunk]

class IndexPDFRequest(BaseModel):
    user_id: str
    project_id: str
    filename: str


class IndexPDFResponse(BaseModel):
    status: str
    user_id: str
    project_id: str
    filename: str
    collection: str
    pages_parsed: int
    chunks_indexed: int
    embedding_model: str


class VectorDBStatsResponse(BaseModel):
    collection: str
    total_chunks: int
    chroma_path: str
    embedding_model: str

class SearchRequest(BaseModel):
    user_id: str
    project_id: str
    query: str
    top_k: int | None = None


class SearchResult(BaseModel):
    source: str
    page: int
    chunk_id: str
    text: str
    score: float | None = None


class SearchResponse(BaseModel):
    status: str
    user_id: str
    project_id: str
    query: str
    top_k: int
    results: list[SearchResult]
    retrieval_latency_ms: int


class AskRequest(BaseModel):
    user_id: str
    project_id: str
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    source: str
    page: int
    chunk_id: str
    text: str


class AskMetrics(BaseModel):
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    llm_backend: str
    llm_model: str
    mock: bool
    estimated_tokens_per_second: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    metrics: AskMetrics

class AgentDebugResponse(BaseModel):
    task_type: str
    retrieved_chunks: int
    citations: int
    memory_updates: list[str]
    warnings: list[str]
    answer: str
