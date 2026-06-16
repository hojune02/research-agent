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