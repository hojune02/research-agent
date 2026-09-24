# Research Agent: project guide

Research Agent is a local application for asking questions about uploaded papers. It combines a FastAPI service, PDF extraction, Chroma retrieval, an OpenAI-compatible model client, and a small frontend. The useful engineering question in this repository is whether an answer can be traced back to the uploaded material, including when the material does not support an answer.

## Where to look

| Concern | Code or evidence |
| --- | --- |
| API and request handling | [`backend/app/main.py`](backend/app/main.py), [`backend/app/schemas.py`](backend/app/schemas.py) |
| PDF parsing and indexing | [`backend/app/rag/pdf_parser.py`](backend/app/rag/pdf_parser.py), [`backend/app/rag/indexer.py`](backend/app/rag/indexer.py) |
| Retrieval, reranking, and answers | [`backend/app/rag/vectorstore.py`](backend/app/rag/vectorstore.py), [`backend/app/rag/reranker.py`](backend/app/rag/reranker.py), [`backend/app/rag/qa.py`](backend/app/rag/qa.py) |
| Model integration | [`backend/app/llm/client.py`](backend/app/llm/client.py) |
| Agent workflow and memory | [`backend/app/agents/graph.py`](backend/app/agents/graph.py), [`backend/app/db/memory.py`](backend/app/db/memory.py) |
| Local application setup | [`docker-compose.yml`](docker-compose.yml), [`README.md`](README.md) |
| Checks and evaluation | [`backend/tests/test_smoke.py`](backend/tests/test_smoke.py), [`scripts/run_reference_benchmark.py`](scripts/run_reference_benchmark.py), [`evals/`](evals/) |

## How a question moves through the system

The service parses uploaded PDFs into page-linked chunks and indexes them for retrieval. A question searches the relevant user and project collection. The answer path builds context from retrieved chunks, calls the configured model, and returns an answer with citation objects and timing fields. If retrieval finds no chunks, the answer path returns an explicit lack-of-evidence response. The repository also contains a reranker path and a separate agent workflow; inspect the configuration and entry point before assuming either is active for a particular run.

## What the evidence supports

The smoke tests cover startup, the health endpoint, an empty-project refusal, and the tool listing in mock mode. The evaluation scripts and saved outputs show that answer quality, citation behavior, and latency were investigated on a paper-question set. Those artifacts are useful for reproducing and questioning the results; they are not a general accuracy guarantee. Some judging uses a model, so its scores should be read alongside the questions, generated answers, and judge setup.

## Current limits

This is a local MVP, not a hosted service claim. The README documents local setup, while model availability, embedding downloads, stored PDFs, and machine resources affect a fresh run. The smoke tests do not establish behavior across arbitrary papers or concurrent users. Citation objects describe retrieved source material; a reader should still check whether each answer statement is supported by its cited excerpt.

For a quick review, start with the API, follow the retrieval and answer path, then compare the evaluation input with the saved outputs. That path shows both the implemented system and the boundary of the evidence without relying on a feature list.
