# Soundable Research Agent

Soundable Research Agent is a local multi-user research automation MVP.

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
```

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

## Phase 3 Status

Implemented:

- Raw PDF upload endpoint
- User/project-based file organization
- Filename sanitization
- Basic PDF validation
- Local file persistence under `data/uploads/{user_id}/{project_id}/`

Endpoint:

```http
POST /upload/raw
```

## Phase 4 Status

Implemented:

- PDF text extraction with `pypdf`
- Page-level parsing
- Character-based chunking with overlap
- Metadata-rich document chunks
- `/parse/pdf` endpoint
- `/parse/pdf/preview` endpoint

Each chunk includes:

```json
{
  "chunk_id": "paper_p3_c1",
  "user_id": "hojune",
  "project_id": "test-project",
  "source": "paper.pdf",
  "page": 3,
  "text": "...",
  "char_count": 900
}
```

## Phase 5 Status

Implemented:

- Chroma persistent vector database
- Local sentence-transformer embedding model
- PDF chunk indexing
- Metadata-rich vector records
- User/project/source/page/chunk metadata
- `/index/pdf` endpoint
- `/vector/stats` endpoint

Vector DB record structure:

```json
{
  "id": "hojune::test-project::paper.pdf::paper_p3_c1",
  "document": "chunk text...",
  "metadata": {
    "user_id": "hojune",
    "project_id": "test-project",
    "source": "paper.pdf",
    "page": 3,
    "chunk_id": "paper_p3_c1",
    "char_count": 900
  },
  "embedding": "[384-dimensional vector]"
}
```

## Phase 6 Status

Implemented:

- Semantic vector search over Chroma
- User/project metadata filtering
- `/search` endpoint
- RAG QA endpoint `/ask`
- Context construction from retrieved chunks
- Citation objects
- No-document fallback to reduce hallucination
- Retrieval/generation/total latency metrics

Search flow:

```text
query
  -> embedding model
  -> Chroma top-k search
  -> metadata filter by user_id/project_id
  -> retrieved chunks
```

## Phase 7 Status

Implemented:

- LangGraph-based agent workflow
- Shared `AgentState` using `TypedDict`
- Planner node for task classification
- Retrieval node using Chroma search
- Synthesis node using LLM client
- Citation-check node
- Memory-stub node for future persistent memory
- `/debug/agent` endpoint for inspecting workflow behavior

Agent workflow:

```text
START
  -> planner_node
  -> retrieve_node
  -> synthesize_node
  -> citation_check_node
  -> memory_stub_node
  -> END
```

## Phase 8 Status

Implemented research automation tools:

- `retrieve_context`
- `compare_documents`
- `generate_literature_review`
- `extract_methods_and_limitations`
- `save_memory` stub

New endpoints:

```http
GET  /tools
POST /compare
POST /lit-review
POST /extract-insights
```

## Phase 9 Status

Implemented persistent SQLite memory:

- SQLite database initialization
- `memories` table
- `create_memory`
- `list_memories`
- `delete_memory`
- duplicate-aware `create_memory_if_new`
- persistent `save_memory` tool
- LangGraph `memory_update_node`
- memory API endpoints

New endpoints:

```http
GET    /memory/{user_id}/{project_id}
POST   /memory
DELETE /memory/{memory_id}
```

## Local llama.cpp Serving Mode

Soundable Research Agent can use a local llama.cpp server through the OpenAI-compatible API provided by `llama-cpp-python`.

### Install llama.cpp server dependencies

For Mac Apple Silicon with Metal acceleration:

```bash
python -m pip install -r requirements-llama-metal.txt
```

## Download a GGUF model

```bash
mkdir -p models

hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir models
```

## Start llama.cpp server

```bash
python -m llama_cpp.server \
  --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --model_alias local-model \
  --host 0.0.0.0 \
  --port 8002 \
  --n_ctx 4096 \
  --n_gpu_layers -1
```
