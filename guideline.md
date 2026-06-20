# Research Agent

Research Agent is a local multi-user research automation system for uploaded academic papers.

It allows users to upload PDF papers, index them into a vector database, ask citation-grounded questions, compare papers, generate mini literature reviews, extract methods and limitations, and persist project-level memory. The system supports mock LLM mode for quick demos and real local LLM inference through a llama.cpp OpenAI-compatible server.

## Demo Features

* Upload PDF papers
* Parse and chunk PDFs
* Store document chunks in Chroma vector DB
* Perform user/project-isolated semantic search
* Ask citation-grounded questions over uploaded papers
* Generate paper comparisons
* Generate mini literature reviews
* Extract methods, datasets, limitations, and future work
* Run a LangGraph-based agent workflow
* Store persistent project memory in SQLite
* Use a local llama.cpp LLM server instead of mock generation
* Run backend and frontend with Docker Compose

## Architecture

```text
Gradio Frontend
    ↓
FastAPI Backend
    ↓
LangGraph Agent Workflow
    ↓
Tools
  - retrieve_context
  - compare_documents
  - generate_literature_review
  - extract_methods_and_limitations
  - save_memory
    ↓
Storage
  - Chroma vector DB for document chunks
  - SQLite for persistent project memory
    ↓
LLM Backend
  - MOCK_LLM=true for quick demo
  - llama.cpp OpenAI-compatible local server
  - vLLM-compatible backend through OpenAI-compatible API
```

## Agent Workflow

```text
START
  → planner_node
  → retrieve_node
  → synthesize_node
  → citation_check_node
  → memory_update_node
  → END
```

The planner classifies the user request into one of:

* `qa`
* `compare`
* `lit_review`
* `unknown`

The retrieval node calls the `retrieve_context` tool, which searches Chroma using strict `user_id` and `project_id` metadata filtering. The synthesis node sends retrieved chunks to the configured LLM backend. The citation checker attaches source/page/chunk citations. The memory update node stores useful project-level memory in SQLite.

## Tech Stack

| Layer             | Technology                      |
| ----------------- | ------------------------------- |
| Backend API       | FastAPI                         |
| Frontend          | Gradio                          |
| Agent workflow    | LangGraph                       |
| Vector database   | Chroma                          |
| Embeddings        | sentence-transformers           |
| PDF parsing       | pypdf                           |
| Memory DB         | SQLite                          |
| LLM client        | OpenAI-compatible Python client |
| Local LLM serving | llama.cpp / llama-cpp-python    |
| Containerization  | Docker, Docker Compose          |

## JD Requirement Mapping

| JD Requirement                                          | How Research Agent Satisfies It                                                                                                                                                                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM serving: vLLM, llama.cpp                            | Implemented real local LLM serving with llama.cpp using `llama-cpp-python` OpenAI-compatible server. The backend uses an OpenAI-compatible `LLM_BASE_URL`, so it can also connect to a vLLM server by changing environment variables. |
| LLM application frameworks: LangChain, LlamaIndex, etc. | Implemented a LangGraph-based agent workflow. The project uses modular RAG components and tool-style functions for retrieval, comparison, literature review generation, and memory saving.                                            |
| Local LLM deployment and serving                        | Downloaded a GGUF model, served it locally with llama.cpp, and connected Research to it through `MOCK_LLM=false`, `LLM_BASE_URL=http://localhost:8002/v1`, and `LLM_MODEL=local-model`.                                               |
| Agent workflow, tool calling, memory structure design   | Implemented planner, retrieval, synthesis, citation-checking, and memory-update nodes. Tools include `retrieve_context`, `compare_documents`, `generate_literature_review`, `extract_methods_and_limitations`, and `save_memory`.     |
| Document-based QA and search system                     | Built PDF upload, parsing, chunking, embedding, Chroma indexing, `/search`, and citation-grounded `/ask` endpoints.                                                                                                                   |
| Linux, Docker, GPU inference optimization               | Dockerized FastAPI backend and Gradio frontend with Docker Compose. Added llama.cpp local serving path and vLLM-compatible serving path. Tracks retrieval/generation latency and estimated tokens/sec.                                |
| Research automation tool development                    | Added paper comparison, literature review generation, method/limitation extraction, citation output, and persistent project memory.                                                                                                   |

## Core API Endpoints

### Health and Debug

```http
GET  /health
POST /debug/llm
POST /debug/agent
```

### Document Pipeline

```http
POST /upload/raw
POST /parse/pdf/preview
POST /index/pdf
GET  /vector/stats
```

### Search and RAG

```http
POST /search
POST /ask
```

### Research Automation Tools

```http
GET  /tools
POST /compare
POST /lit-review
POST /extract-insights
```

### Memory and Metrics

```http
GET    /memory/{user_id}/{project_id}
POST   /memory
DELETE /memory/{memory_id}
GET    /metrics/latest
```

## Local Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install backend and frontend dependencies:

```bash
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

Run backend:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Run frontend in another terminal:

```bash
source .venv/bin/activate
python frontend/app.py
```

Open:

```text
Frontend: http://localhost:7860
Backend docs: http://localhost:8000/docs
```

## Docker Setup

Run the full backend/frontend app:

```bash
docker compose up --build
```

Open:

```text
Frontend: http://localhost:7860
Backend docs: http://localhost:8000/docs
```

By default, use:

```env
MOCK_LLM=true
```

for a quick demo without a local model.

## Local llama.cpp Serving Mode

Research can use a local llama.cpp server through the OpenAI-compatible API provided by `llama-cpp-python`.

### Install llama.cpp server dependencies

For Mac Apple Silicon with Metal acceleration:

```bash
python -m pip install -r requirements-llama-metal.txt
```

For generic CPU mode:

```bash
python -m pip install -r requirements-llama-cpu.txt
```

### Download a GGUF model

```bash
mkdir -p models

hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir models
```

### Start llama.cpp server

```bash
python -m llama_cpp.server \
  --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --model_alias local-model \
  --host 0.0.0.0 \
  --port 8002 \
  --n_ctx 4096 \
  --n_gpu_layers -1
```

### Configure Research for real local LLM inference

In `.env`:

```env
MOCK_LLM=false
LLM_BASE_URL=http://localhost:8002/v1
LLM_API_KEY=local-key
LLM_MODEL=local-model
```

Restart the backend:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Test:

```bash
curl -X POST "http://localhost:8000/debug/llm" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Research Agent in two sentences.",
    "context": ""
  }'
```

If the response contains:

```json
"mock": false
```

then Research is using the local llama.cpp server.

## vLLM-Compatible Serving Mode

The backend is designed to work with any OpenAI-compatible LLM server. To use vLLM on a CUDA Linux GPU machine, start a vLLM server and configure:

```env
MOCK_LLM=false
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=local-key
LLM_MODEL=local-model
```

Example vLLM command:

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8001:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name local-model \
  --host 0.0.0.0 \
  --port 8000
```

The current implementation has been tested with llama.cpp local serving. vLLM can be connected through the same OpenAI-compatible client abstraction.

## Demo Script

1. Start llama.cpp server.
2. Set `MOCK_LLM=false` in `.env`.
3. Start FastAPI backend.
4. Start Gradio frontend.
5. Open `http://localhost:7860`.
6. Enter:

   * `user_id`: `hojune`
   * `project_id`: `test-project`
7. Upload a PDF.
8. Copy the returned filename.
9. Index the PDF into Chroma.
10. Ask: `What is the main contribution of this paper?`
11. Check answer, citations, and metrics.
12. Try `Compare Papers`.
13. Try `Generate Literature Review`.
14. Open `Memory & Stats`.
15. Check Chroma vector DB stats, latest metrics, and SQLite memory.

## Example Environment

```env
MOCK_LLM=false

LLM_BASE_URL=http://localhost:8002/v1
LLM_API_KEY=local-key
LLM_MODEL=local-model

TOP_K=5
CHUNK_SIZE=900
CHUNK_OVERLAP=150

CHROMA_COLLECTION=Research_documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## What This Project Demonstrates

* Built a multi-user RAG system over uploaded PDFs
* Designed metadata-filtered vector search using Chroma
* Implemented citation-grounded document QA
* Built a LangGraph agent workflow with planner, retriever, synthesizer, citation checker, and memory updater
* Added research automation tools for comparison, literature review generation, and method/limitation extraction
* Integrated persistent SQLite memory
* Replaced mock generation with local llama.cpp LLM serving
* Dockerized backend and frontend for reproducible deployment
* Added latency and tokens/sec metrics for inference-awareness

## CV Bullet

Built Research Agent, a local multi-user research automation platform using FastAPI, Gradio, LangGraph, Chroma, SQLite, Docker, and llama.cpp; implemented PDF-based RAG with citations, user-isolated vector search, agentic tool-calling workflows, persistent project memory, research paper comparison/literature-review tools, OpenAI-compatible local LLM serving, and latency/tokens-per-second metrics.

## Limitations

* Image-only scanned PDFs are not supported.
* The MVP uses username/project fields instead of full authentication.
* Retrieval uses simple top-k semantic search without reranking.
* llama.cpp quality depends on the chosen GGUF model.
* vLLM support is implemented through an OpenAI-compatible backend abstraction, but requires a separate CUDA Linux GPU environment for full testing.
