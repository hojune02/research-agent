"""
Shared judge rubric and utilities for all evaluation scripts.

One rubric, one 0-3 scale, used by every judge entrypoint so scores are
comparable across experiments and scripts.
"""

import json
import re
import statistics

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
- abstention_correctness: only meaningful for unanswerable questions.
  3 = correctly refused, 0 = answered when it should have refused.
  For answerable questions, output 0; it is ignored downstream.

Be conservative: when unsure between two scores, choose the lower one.
Do not reward verbosity. Judge only against the provided reference and context.

Return ONLY valid JSON, no other text:
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


def build_judge_user_prompt(row: dict) -> str:
    citations_text = "\n---\n".join(
        str(citation.get("text", "")) for citation in row.get("citations", [])
    ) or "(no citations)"

    return f"""
Question:
{row.get("question", "")}

Should the system answer this question (true = answerable from the corpus)?
{row.get("should_answer", "")}

Reference answer:
{row.get("reference_answer", "(none provided)")}

Model answer:
{row.get("answer", "")}

Cited context the model was given:
{citations_text}
""".strip()


def aggregate_judge_runs(run_dicts: list[dict]) -> dict:
    """Aggregate k judge samples for one question into mean and std per metric."""
    agg: dict = {}
    for key in METRIC_KEYS:
        vals = [float(run.get(key, 0) or 0) for run in run_dicts]
        agg[key] = round(statistics.mean(vals), 3)
        agg[f"{key}_std"] = round(
            statistics.pstdev(vals) if len(vals) > 1 else 0.0, 3
        )
    agg["reason"] = run_dicts[0].get("reason", "")
    agg["judge_runs"] = len(run_dicts)
    return agg