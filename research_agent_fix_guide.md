# Research Agent — Fix Implementation Guide

Follow the fixes **in order**. Each fix is one commit. Every fix has: why, exact changes, and a verification step. Where I reference code I could not fully see (e.g., `prompts.py`, `schemas.py`), the instruction says ADAPT — match the pattern to your local code.

Run everything with `MOCK_LLM=true` first to verify plumbing, then re-verify with the real model where noted.

---

## Fix 0 — Score-scale bug: reranker logits vs distance-derived similarity

**Why.** `search_chunks` sets `score = 1/(1+distance)` (bounded 0–1, higher = better). When `ENABLE_RERANKER=true`, `rerank_results` **overwrites** `score` with raw `CrossEncoder.predict` outputs — unbounded logits that are often negative for ms-marco models. Any threshold, sorting comparison across modes, or UI display now means different things depending on a config flag. Fix 5 (weak-retrieval routing) depends on a consistent score, so this must land first.

**Changes.**

1. In `backend/app/rag/reranker.py`, normalize cross-encoder logits with a sigmoid so both modes produce a 0–1 "higher is better" score:

```python
import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import SearchResult


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank_results(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    if not results:
        return results

    model = get_reranker()

    pairs = [(query, result.text) for result in results]
    scores = model.predict(pairs)

    scored = list(zip(results, scores))
    scored.sort(key=lambda item: float(item[1]), reverse=True)

    reranked = []
    for result, score in scored[:top_k]:
        # Normalize raw cross-encoder logits to (0, 1) so that scores are
        # comparable with the distance-derived similarity used when the
        # reranker is disabled. Sorting order is preserved (sigmoid is monotonic).
        result.score = round(_sigmoid(float(score)), 4)
        reranked.append(result)

    return reranked
```

2. Add a one-line comment in `search_chunks` (in `backend/app/rag/vectorstore.py`) above the score computation stating the invariant: `# Invariant: SearchResult.score is always in (0, 1], higher = more relevant, in both reranked and non-reranked modes.`

**Verify.** Start the backend with `ENABLE_RERANKER=true`, index a PDF, call `POST /search`, confirm all returned `score` values are in (0, 1). Toggle to `false`, repeat, confirm the same range.

**Commit message.** `fix(retrieval): normalize reranker scores to (0,1] to match non-reranked score scale`

---

## Fix 1 — Documentation and naming hygiene

**Why.** Contradictory docs and leftover project names are the cheapest bad impression to prevent. Reviewers read the README and .env.example before any code.

**Changes.**

1. `guideline.md` → Limitations section: **delete** the line "Retrieval uses simple top-k semantic search without reranking." **Replace** with: "Cross-encoder reranking (`ENABLE_RERANKER`) improves ranking precision but adds CPU latency per query; it is off by default for the fastest demo path."
2. `guideline.md` → add a short "Retrieval configuration" subsection documenting `ENABLE_RERANKER`, `RERANKER_MODEL`, `RERANK_TOP_N`, and the funnel: Chroma top-`RERANK_TOP_N` → cross-encoder → top-`TOP_K`.
3. `backend/app/main.py` → root endpoint: change `"Sounable Research Agent backend is running."` to `"Research Agent backend is running."` Also remove the stray leading space in `title=" Research Agent"` (occurs in `main.py` and in `frontend/app.py` — `gr.Blocks(title=" Research Agent")` and the `# Research Agent` markdown header).
4. Unify the old project name. Pick `research_agent` and apply consistently:
   - `.env.example`: `CHROMA_COLLECTION=paperops_documents` → `CHROMA_COLLECTION=research_documents` (this also fixes the mismatch with the `config.py` default, which is already `research_documents`).
   - `backend/app/config.py`: `SQLITE_PATH` default ends in `paperops.db` → change to `research_agent.db`.
   - `scripts/run_reference_benchmark.py` and `scripts/summarize_eval.py`: replace user-facing strings "PaperOps" with "Research Agent" (help texts, printed headers).
   - Do **not** rename the eval output files already committed under `evals/` — they are historical experiment artifacts; renaming would break the story your numbers tell.
5. **Migration note (do not skip):** changing `CHROMA_COLLECTION` or `SQLITE_PATH` orphans existing local data. After this commit, delete `data/chroma` and `data/sqlite` locally and re-index your test PDFs once. Mention this in the commit body.

**Verify.** `grep -ri "paperops\|sounable" --include="*.py" --include="*.md" --include="*.example" .` returns only hits inside `evals/` historical files.

**Commit message.** `docs: fix stale reranker limitation, unify naming, remove paperops remnants`

---

## Fix 2 — One judge rubric, independent judge, repeated runs

**Why.** Two rubrics exist (0–5 in `scripts/eval_llm_judge.py`, 0–3 in `run_reference_benchmark.py`), the judge defaults to the same local model that generated the answers, judge sampling is single-shot, and your saved judge outputs show faithfulness anchored at 2 on nearly every row — the judge is not discriminating. This fix makes your headline CV numbers defensible.

**Changes.**

1. Create `scripts/judge_common.py` — single source of truth for the rubric:

```python
"""Shared judge rubric and utilities for all evaluation scripts.

One rubric, one scale (0-3), used by every judge entrypoint so that
scores are comparable across experiments.
"""

import json
import re

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluation judge for a document-grounded QA system.

Score the model answer on a 0-3 scale for each dimension:

- answer_correctness: 0 = wrong or missing, 1 = partially correct,
  2 = mostly correct with minor gaps, 3 = fully correct vs the reference answer.
- faithfulness: 0 = contains claims contradicting or absent from the cited
  context, 1 = several unsupported claims, 2 = minor unsupported details,
  3 = every claim is supported by the cited context.
- citation_support: 0 = citations irrelevant or missing, 1 = weakly related,
  2 = mostly relevant, 3 = citations directly support the answer's claims.
- abstention_correctness: only for unanswerable questions.
  3 = correctly refused, 0 = answered when it should have refused.
  For answerable questions, output 0 and ignore this field downstream.

Be conservative: when unsure between two scores, choose the lower one.
Do not reward verbosity. Judge only against the provided reference and context.

Return ONLY valid JSON:
{"answer_correctness": 0-3, "faithfulness": 0-3, "citation_support": 0-3,
 "abstention_correctness": 0-3, "reason": "one short sentence"}
""".strip()

METRIC_KEYS = [
    "answer_correctness",
    "faithfulness",
    "citation_support",
    "abstention_correctness",
]


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {key: 0 for key in METRIC_KEYS} | {
            "reason": f"Could not parse judge output: {text[:200]}"
        }
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {key: 0 for key in METRIC_KEYS} | {
            "reason": f"Invalid JSON from judge: {text[:200]}"
        }
```

2. In `scripts/run_reference_benchmark.py`, upgrade the `judge` subcommand:
   - Import `JUDGE_SYSTEM_PROMPT`, `METRIC_KEYS`, `extract_json` from `judge_common` and delete the local duplicates.
   - Add arguments: `--judge-runs` (int, default `3`) and `--judge-temperature` (float, default `0.0`).
   - Add a **generator/judge identity warning**: the outputs file rows should already contain the generator model name (they come from `/ask` metrics — ADAPT to your row schema); at the start of `run_judge`, if `args.judge_model` equals the generator model AND `args.judge_base_url` equals the generator base URL, print `WARNING: judge model == generator model; scores are subject to self-preference bias.` Do not block — warn.
   - For each question, call the judge `judge_runs` times. Aggregate per metric:

```python
import statistics

def aggregate_judge_runs(run_dicts: list[dict]) -> dict:
    agg = {}
    for key in METRIC_KEYS:
        vals = [float(r.get(key, 0)) for r in run_dicts]
        agg[key] = statistics.mean(vals)
        agg[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    agg["reason"] = run_dicts[0].get("reason", "")
    agg["judge_runs"] = len(run_dicts)
    return agg
```

   - In the summary printout, report `mean ± std` per metric and add the count of questions where `*_std > 0` (i.e., where the judge disagreed with itself). That disagreement count is itself an interview talking point.
3. In `scripts/eval_llm_judge.py`: replace its 0–5 prompt and local `extract_json` with imports from `judge_common`, and update its summary lines to `/3`. Add a module docstring: `"""Legacy single-run judge. Prefer run_reference_benchmark.py judge (multi-run, aggregated)."""`
4. **Judge model choice.** Run the judge with a *different, stronger* model than the generator. Practical local option on your Mac: keep Qwen2.5-1.5B as generator on port 8002 and serve `Qwen2.5-7B-Instruct` GGUF (Q4_K_M) as judge on port 8003, then pass `--judge-base-url http://localhost:8003/v1`. If 7B is too heavy, 3B still breaks the self-judging pattern. Document the judge model in the output filename (you already do this convention with your `_reranker_top20_top5` suffixes — extend it, e.g. `_judge-qwen7b_k3.jsonl`).
5. **Re-run both headline experiments** (baseline and reranker configs) with the new judge and update `guideline.md` with the new table. If the faithfulness delta shrinks or disappears, keep the old numbers in the doc **alongside** the new ones with one sentence: "Re-evaluation with an independent multi-run judge revised these figures; the correctness and abstention improvements persisted, the faithfulness delta did not survive judge hardening." That sentence is worth more in an interview than the original delta ever was.

**Verify.** Run `python scripts/run_reference_benchmark.py judge --judge-runs 3 ...` on your existing outputs file. Confirm: warning fires when judge==generator; per-metric std appears in rows; summary shows `mean ± std`.

**Commit message.** `eval(judge): single 0-3 rubric, independent judge support, k-run aggregation with variance`

---

## Fix 3 — Citation verification: cite what was used, not what was retrieved

**Why.** `make_citations` / `citation_check_node` attach every retrieved chunk as a citation regardless of whether the answer used it. Citations must be evidence, not provenance dumps.

**Changes.**

1. **Prompt.** In `backend/app/rag/qa.py` `build_rag_prompt` AND in `backend/app/agents/prompts.py` `build_qa_prompt` (ADAPT: apply the same edit to `build_compare_prompt` and `build_lit_review_prompt`), add two rules to the Rules list:

```
- After each claim, cite the supporting source inline using its bracket tag, e.g. [Source 2].
- Only cite sources whose text actually supports the claim.
```

   The context builder already labels chunks `[Source 1]`, `[Source 2]`, … so no context change is needed.

2. **Parser.** Create `backend/app/rag/citation_filter.py`:

```python
"""Post-generation citation verification.

Parses inline [Source n] markers from the model answer and keeps only the
retrieved chunks that were actually cited. Falls back safely when the model
emits no valid markers.
"""

import re

from app.schemas import Citation, SearchResult

_MARKER_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def extract_cited_indices(answer: str, num_sources: int) -> list[int]:
    """Return sorted unique 1-based indices cited in the answer, bounded to valid range."""
    indices = set()
    for match in _MARKER_RE.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= num_sources:
            indices.add(idx)
    return sorted(indices)


def filter_citations(
    answer: str,
    results: list[SearchResult],
) -> tuple[list[Citation], list[str]]:
    """Build citations from only the chunks the answer actually cited.

    Returns (citations, warnings).
    Fallback: if the model produced an answer but no valid markers, all
    retrieved chunks are returned as citations with a warning, so the UI
    never silently loses provenance.
    """
    warnings: list[str] = []
    cited = extract_cited_indices(answer, len(results))

    if not results:
        return [], warnings

    if not cited:
        warnings.append(
            "Answer contained no valid inline [Source n] markers; "
            "falling back to all retrieved chunks as provenance."
        )
        selected = results
    else:
        selected = [results[i - 1] for i in cited]

    citations = [
        Citation(
            source=r.source,
            page=r.page,
            chunk_id=r.chunk_id,
            text=r.text[:500],
        )
        for r in selected
    ]
    return citations, warnings
```

3. **Wire into the agent.** In `backend/app/agents/graph.py` `citation_check_node`, replace the loop that converts all chunks into citations with:

```python
from app.rag.citation_filter import filter_citations
# ... inside citation_check_node:
citations, citation_warnings = filter_citations(draft_answer, chunks)
warnings = state.get("warnings", []) + citation_warnings
```

   Keep the existing no-chunks refusal behavior unchanged. Do NOT strip the `[Source n]` markers from the answer — they are now a feature; the frontend shows exactly which sentence rests on which source.

4. **Wire into the streaming path** the same way at the end of `stream_answer_question` in `backend/app/rag/qa.py`: after the token loop completes you have the accumulated answer only client-side today, so accumulate it server-side too (`full_answer += token` in the loop), then before the `done` event emit one extra event:

```python
final_citations, citation_warnings = filter_citations(full_answer, results)
yield json.dumps({
    "type": "citations_final",
    "citations": [c.model_dump() for c in final_citations],
    "warnings": citation_warnings,
}) + "\n"
```

   In `frontend/app.py`'s stream handler, add an `elif event_type == "citations_final": citations = event.get("citations", [])` branch so the UI replaces the provisional retrieved-chunk citations with the verified set.

5. **Benchmark metric.** In the benchmark `run` step (ADAPT to `run_outputs` row construction in `run_reference_benchmark.py`), record two new fields per row using `extract_cited_indices`: `inline_citation_count` and `inline_citation_valid` (bool: at least one valid marker on an answerable question). In `summarize_eval.py` / the benchmark summary, print `Inline citation rate: X%`. This gives you a *mechanically verified* citation metric that does not depend on the LLM judge at all — say exactly that sentence in interviews.

**Verify.** With the real model: ask a question, confirm the answer contains `[Source n]` markers, confirm the citations list length ≤ top_k and matches the cited indices. Ask an unanswerable question, confirm refusal still carries no citations. Break it on purpose: temporarily instruct the model not to cite, confirm the fallback warning appears.

**Commit message.** `feat(citations): inline [Source n] markers with post-generation verification and filtered citations`

---

## Fix 4 — Make memory read-write: inject project memory into generation

**Why.** `memory_update_node` writes telemetry strings to SQLite that nothing ever reads. Memory that changes no behavior is not memory.

**Changes.**

1. **State.** In `backend/app/agents/state.py`, add to `AgentState`: `memory_context: str`.

2. **Load node.** In `backend/app/agents/graph.py`, add:

```python
from app.db.memory import list_memories  # ADAPT: match the actual signature


def load_memory_node(state: AgentState) -> AgentState:
    """Loads recent project memory and formats it for the synthesis prompt."""
    try:
        memories = list_memories(
            user_id=state["user_id"],
            project_id=state["project_id"],
        )
    except Exception:
        return {"memory_context": ""}

    # ADAPT: keep the 5 most recent items; match your row/object shape.
    recent = memories[-5:] if memories else []
    if not recent:
        return {"memory_context": ""}

    lines = [f"- {m.memory_item}" for m in recent]  # ADAPT attribute/key name
    block = "Known project context from earlier sessions:\n" + "\n".join(lines)
    return {"memory_context": block[:800]}
```

3. **Graph wiring.** In `build_agent_graph()`:

```python
graph.add_node("load_memory", load_memory_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "load_memory")
graph.add_edge("load_memory", "retrieve")
# rest unchanged
```

4. **Use it.** In `synthesize_node`, prepend memory to the context passed to the LLM:

```python
context = build_context_from_chunks(chunks)
memory_context = state.get("memory_context", "")
if memory_context:
    context = f"{memory_context}\n\n{context}"
```

   Add one rule to the QA prompt (in `prompts.py`): `- The "Known project context" section is background from earlier sessions; prefer the [Source n] document context for factual claims and never cite the background section as a source.`

5. **Write something worth reading.** In `memory_update_node`, replace the three telemetry strings with a single useful record, written only for non-refusal answers:

```python
final_answer = state.get("final_answer", "")
user_query = state.get("user_query", "").strip()
refused = final_answer.startswith("I do not know")

memory_updates: list[str] = []
if user_query and final_answer and not refused:
    memory_updates.append(
        f"Q: {user_query} — A: {final_answer[:300]}"
    )
```

   Keep the `save_memory` loop as is. (If `create_memory_if_new` dedup exists in `app/db/memory.py`, route through it — ADAPT.)

**Verify.** Ask two related questions in one project. Check `GET /memory/{user}/{project}` shows the Q/A record after the first. Hit `/debug/agent` on the second question and confirm `memory_context` is populated in the debug state. Confirm memory never appears as a citation.

**Commit message.** `feat(memory): read path — inject recent project memory into synthesis; write Q/A summaries instead of telemetry`

---

## Fix 5 — Conditional edge: rewrite-and-retry on weak retrieval, then abstain

**Why.** The graph is a straight line; "agent" implies decisions. This adds one real decision with bounded retries — the smallest change that makes LangGraph's presence defensible, and a live demo of conditional routing.

**Changes.**

1. **Config.** In `backend/app/config.py` add:

```python
RETRIEVAL_MIN_SCORE: float = 0.25   # tune after Fix 0 normalization; see step 6
MAX_RETRIEVAL_ATTEMPTS: int = 2     # initial attempt + one rewrite
```

   Mirror both in `.env.example` with a comment that the score is on the unified (0,1] scale from Fix 0.

2. **State.** In `state.py` add: `active_query: str`, `retrieval_attempts: int`.

3. **Retrieve with the active query.** In `retrieve_node`, replace `query=state["user_query"]` with:

```python
query = state.get("active_query") or state["user_query"]
```

   and add to its return dict: `"retrieval_attempts": state.get("retrieval_attempts", 0) + 1`.

4. **Rewrite node.** Add to `graph.py`:

```python
def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrites the query once when retrieval confidence is low."""
    rewrite_prompt = (
        "Rewrite the following search query to maximize recall in a "
        "semantic search over academic PDF chunks. Expand abbreviations, "
        "add likely synonyms, keep it under 30 words. "
        "Return only the rewritten query.\n\n"
        f"Query: {state['user_query']}"
    )
    try:
        result = generate_answer(prompt=rewrite_prompt, context="")
        rewritten = (result.get("text") or "").strip().strip('"')
    except RuntimeError:
        rewritten = ""

    if not rewritten:
        rewritten = state["user_query"]

    warnings = state.get("warnings", []) + [
        f"Low retrieval confidence; retried with rewritten query: {rewritten[:120]}"
    ]
    return {"active_query": rewritten, "warnings": warnings}
```

5. **Router + wiring.** Add:

```python
def route_after_retrieve(state: AgentState) -> str:
    chunks = state.get("retrieved_chunks", [])
    attempts = state.get("retrieval_attempts", 1)
    top_score = max((c.score or 0.0) for c in chunks) if chunks else 0.0

    weak = (not chunks) or (top_score < settings.RETRIEVAL_MIN_SCORE)

    if weak and attempts < settings.MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite_query"
    return "synthesize"
```

   In `build_agent_graph()` replace `graph.add_edge("retrieve", "synthesize")` with:

```python
graph.add_node("rewrite_query", rewrite_query_node)
graph.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {"rewrite_query": "rewrite_query", "synthesize": "synthesize"},
)
graph.add_edge("rewrite_query", "retrieve")
```

   Existing abstention behavior handles the terminal weak case: if the retry still returns nothing, `synthesize_node`'s empty-chunks branch refuses, unchanged.

6. **Calibrate the threshold — do not guess it.** Run the retrieval eval over your benchmark and collect the top-1 score for every question, separated into `should_answer=true` vs `false`. Set `RETRIEVAL_MIN_SCORE` to the largest value that keeps ≥95% of answerable questions above it. Do this once per mode (`ENABLE_RERANKER` true/false); if the two calibrated values differ materially, add a second setting `RETRIEVAL_MIN_SCORE_RERANKED` and pick in the router based on `settings.ENABLE_RERANKER`. Save the little score-distribution table into `evals/` — that artifact *is* the interview answer to "how did you pick the threshold?"

7. **Re-run the full benchmark** and record deltas in `guideline.md`. Expect abstention behavior to be the metric most affected; watch it doesn't regress (a too-eager rewrite can turn a correct refusal into a hallucinated answer — if unsupported-abstention drops, raise the threshold).

**Verify.** `POST /debug/agent` with a deliberately vague query (e.g., "the thing about the second one") — debug state should show `retrieval_attempts: 2`, a rewrite warning, and `active_query` ≠ `user_query`. A normal query should show `retrieval_attempts: 1` and no rewrite.

**Commit message.** `feat(agent): conditional retrieval routing — score-gated query rewrite with bounded retries`

---

## Fix 6 (optional, do only if time remains) — Unify streaming with the agent

**Why.** `/ask/stream` bypasses planner, memory, and citation logic. Full LangGraph event streaming is the textbook fix, but the pragmatic unification below gets you 90% of the value in a day: same nodes, streaming only where streaming matters (generation).

**Changes.**

1. In `backend/app/agents/graph.py`, add a streaming runner that calls the existing node functions directly in sequence (they are plain functions on a dict-like state — no graph invocation needed for a linear-with-one-branch flow):

```python
from app.llm.client import stream_answer
from app.agents.prompts import build_qa_prompt, build_context_from_chunks
import json as _json


def run_agent_stream(user_id: str, project_id: str, user_query: str, top_k: int | None = None):
    """Streaming twin of run_agent. Reuses the same nodes; streams synthesis."""
    state: AgentState = {
        "user_id": user_id,
        "project_id": project_id,
        "user_query": user_query,
        "top_k": top_k or settings.TOP_K,
    }

    state.update(planner_node(state))
    state.update(load_memory_node(state))

    # retrieval with the same weak-retrieval retry policy as the graph
    state.update(retrieve_node(state))
    while route_after_retrieve(state) == "rewrite_query":
        state.update(rewrite_query_node(state))
        state.update(retrieve_node(state))

    chunks = state.get("retrieved_chunks", [])

    yield _json.dumps({
        "type": "metadata",
        "retrieval_latency_ms": state.get("retrieval_latency_ms", 0),
        "task_type": state.get("task_type", "qa"),
        "warnings": state.get("warnings", []),
        # provisional provenance; replaced by citations_final
        "citations": [
            {"source": c.source, "page": c.page, "chunk_id": c.chunk_id, "text": c.text[:500]}
            for c in chunks
        ],
    }) + "\n"

    if not chunks:
        yield _json.dumps({"type": "token", "text": "I do not know from the uploaded documents."}) + "\n"
        yield _json.dumps({"type": "done"}) + "\n"
        return

    context = build_context_from_chunks(chunks)
    memory_context = state.get("memory_context", "")
    if memory_context:
        context = f"{memory_context}\n\n{context}"

    prompt = build_qa_prompt(user_query)  # ADAPT: route by task_type like synthesize_node

    full_answer = ""
    for token in stream_answer(prompt=prompt, context=context):
        full_answer += token
        yield _json.dumps({"type": "token", "text": token}) + "\n"

    from app.rag.citation_filter import filter_citations
    final_citations, citation_warnings = filter_citations(full_answer, chunks)
    yield _json.dumps({
        "type": "citations_final",
        "citations": [c.model_dump() for c in final_citations],
        "warnings": citation_warnings,
    }) + "\n"

    state["final_answer"] = full_answer
    memory_update_node(state)

    yield _json.dumps({"type": "done"}) + "\n"
```

2. Point the `/ask/stream` endpoint in `main.py` at `run_agent_stream` instead of `stream_answer_question`. Keep `stream_answer_question` in `rag/qa.py` untouched for one release (delete it in a later cleanup commit).
3. Frontend already handles `metadata`/`token`/`done`; you added `citations_final` in Fix 3 step 4. No further frontend change.
4. In `guideline.md`, note honestly: "Streaming reuses the same agent nodes via a streaming runner; migrating to LangGraph `astream_events` for graph-native streaming is the planned next step." Knowing that API exists and why you deferred it is the interview answer.

**Verify.** Stream a question: metadata event includes `task_type` and any rewrite warning; final citations replace provisional ones; the Q/A memory record appears afterward. TTFT should be unchanged vs before (retrieval dominates the pre-stream time either way).

**Commit message.** `refactor(streaming): route /ask/stream through agent nodes (planner, memory, retry, citation verification)`

---

## Fix 7 — Engineering hygiene: real token counts, lifespan handler, smoke tests, pinned deps

**Changes.**

1. **Real token usage.** llama.cpp's OpenAI-compatible server returns `usage` on completions. In `backend/app/llm/client.py` `generate_answer`, replace the estimate with the server's number when available:

```python
usage = getattr(response, "usage", None)
completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

if completion_tokens:
    tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0.0
else:
    tokens_per_second = _estimate_tokens(text) / elapsed if elapsed > 0 else 0.0
```

   Rename the response field mentally from "estimated" to actual-when-available; keep the schema field name to avoid breaking the frontend, but update its docstring.

2. **Lifespan.** In `main.py`, replace the deprecated startup hook:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Research Agent", description=..., version="0.1.0", lifespan=lifespan)
```

   Delete the `@app.on_event("startup")` block.

3. **Smoke tests.** Create `backend/tests/test_smoke.py`:

```python
"""Smoke tests: app boots and core endpoints respond in mock mode."""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

settings.MOCK_LLM = True

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_unknown_project_refuses():
    response = client.post(
        "/ask",
        json={
            "user_id": "smoke-user",
            "project_id": "empty-project",
            "question": "What is the main contribution?",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "do not know" in body["answer"].lower()
    assert body["citations"] == []


def test_tools_listed():
    response = client.get("/tools")
    assert response.status_code == 200
    assert len(response.json()["tools"]) >= 4
```

   Add `pytest` and `httpx` to a `backend/requirements-dev.txt`. Run with `cd backend && python -m pytest tests/ -q`.

4. **Pin dependencies.** `pip freeze` from your working venv into `backend/requirements.txt` (or at minimum pin the ML-critical ones: `chromadb`, `sentence-transformers`, `langgraph`, `fastapi`, `openai`, `pydantic`). Unpinned `chromadb` in particular has broken APIs across minor versions.

5. **Optional CI (30 min).** `.github/workflows/ci.yml` running the smoke tests on push. An intern repo with green CI badges reads two levels above its weight class.

**Verify.** `pytest` green; `/health` still works; deprecation warning about `on_event` gone from startup logs.

**Commit message.** `chore: real token usage from server, lifespan handler, smoke tests, pinned requirements`

---

## Order recap and what each buys you in the interview

0. Score normalization → "I found and fixed a cross-mode score-scale bug" — a genuine bug story.
1. Docs hygiene → no cheap bad impressions.
2. Judge hardening → your CV numbers become defensible; possibly your best interview story ("I hardened my own eval and here's what survived").
3. Citation verification → "citations are verified evidence, not retrieval dumps" + a judge-free mechanical metric.
4. Memory read path → "memory changes behavior" is now literally true.
5. Conditional routing → "agent" is now defensible; calibrated threshold artifact answers the how-did-you-tune-it question.
6. Unified streaming → kills the divergent-paths critique.
7. Hygiene → tests, pinning, accurate metrics: the difference between a project and an engineered system.

After each fix that touches retrieval or generation (0, 3, 4, 5), re-run the 50-question benchmark and append one row to a results table in `guideline.md`. By the end you'll have a per-change ablation table — which is exactly the experiment-logging habit both JDs explicitly ask for.
