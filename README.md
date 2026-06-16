# PaperOps Agent

PaperOps Agent is a local multi-user research automation MVP.

The project is designed to demonstrate:

- Local LLM serving with vLLM / llama.cpp
- OpenAI-compatible LLM backend abstraction
- Document-based RAG and vector search
- LangGraph agent workflows
- Tool calling
- Memory structure design
- Dockerized local deployment
- Research paper QA, comparison, and literature review generation

## Phase 1 Status

Implemented:

- FastAPI backend skeleton
- Environment configuration
- `/health` endpoint
- Basic project structure

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000

## Phase 2 Status

Implemented:

- OpenAI-compatible LLM client abstraction
- `MOCK_LLM=true` local mock mode
- `/debug/llm` endpoint
- Basic generation latency metric
- Graceful error handling when local LLM server is unavailable

The LLM client is designed so the app can later switch between:

- mock backend
- vLLM OpenAI-compatible server
- llama.cpp OpenAI-compatible server

without changing application logic.