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