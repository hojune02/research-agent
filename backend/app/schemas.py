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