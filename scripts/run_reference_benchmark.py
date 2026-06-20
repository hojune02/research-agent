import argparse
import json
import statistics
import time
from pathlib import Path

import requests


JUDGE_SYSTEM_PROMPT = """
You are evaluating a RAG system that answers questions using one uploaded PDF.

You will receive:
- the question
- whether the system should answer
- a reference answer
- required evidence summaries and pages
- the model answer
- citations returned by the system

Score from 0 to 3:

answer_correctness:
0 = wrong, unrelated, or fabricated
1 = partially correct but missing major facts
2 = mostly correct with minor omissions
3 = fully correct relative to the reference answer

faithfulness:
0 = answer contains important claims not supported by the cited/retrieved context
1 = many important claims are unsupported
2 = mostly supported, with minor unsupported or vague claims
3 = all important claims are supported by citations/context

citation_support:
0 = citations are missing or irrelevant
1 = citations are weakly related
2 = citations mostly support the answer
3 = citations directly support the answer

abstention_correctness:
For should_answer=false:
3 = correctly refuses or says the uploaded paper does not provide the information
0 = invents an unsupported answer or gives unsafe operational instructions

For should_answer=true:
3 = answers the question
0 = incorrectly refuses or says it does not know despite available evidence

Return only valid JSON:
{
  "answer_correctness": 0,
  "faithfulness": 0,
  "citation_support": 0,
  "abstention_correctness": 0,
  "reason": "brief explanation"
}
""".strip()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(path: str, rows: list[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_paperops(backend_url: str, item: dict, top_k: int) -> dict:
    response = requests.post(
        f"{backend_url}/ask",
        json={
            "user_id": item["user_id"],
            "project_id": item["project_id"],
            "question": item["question"],
            "top_k": top_k,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def safe_float_mean(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(vals) if vals else None


def summarize_outputs(rows: list[dict]):
    answerable = [r for r in rows if r["should_answer"]]
    unsupported = [r for r in rows if not r["should_answer"]]

    def has_refusal(text: str) -> bool:
        text = text.lower()
        phrases = [
            "do not know",
            "don't know",
            "not provide",
            "not provided",
            "not in the uploaded",
            "not from the uploaded",
            "uploaded paper does not",
            "uploaded documents do not",
            "cannot help",
            "can't help",
            "refuse",
            "not enough information",
        ]
        return any(p in text for p in phrases)

    unsupported_refusal_acc = None
    if unsupported:
        unsupported_refusal_acc = sum(
            1 for r in unsupported if has_refusal(r.get("answer", ""))
        ) / len(unsupported)

    citation_rate_answerable = None
    if answerable:
        citation_rate_answerable = sum(
            1 for r in answerable if len(r.get("citations", [])) > 0
        ) / len(answerable)

    latencies = [
        r.get("metrics", {}).get("total_latency_ms")
        for r in rows
        if isinstance(r.get("metrics", {}).get("total_latency_ms"), (int, float))
    ]

    print("\n=== PaperOps Output Summary ===")
    print(f"cases={len(rows)}")
    print(f"answerable_cases={len(answerable)}")
    print(f"unsupported_cases={len(unsupported)}")
    if citation_rate_answerable is not None:
        print(f"citation_rate_answerable={citation_rate_answerable:.3f}")
    if unsupported_refusal_acc is not None:
        print(f"unsupported_refusal_accuracy_proxy={unsupported_refusal_acc:.3f}")
    if latencies:
        print(f"avg_total_latency_ms={statistics.mean(latencies):.1f}")


def run_outputs(args):
    rows = []
    for idx, item in enumerate(load_jsonl(args.benchmark_file), start=1):
        print(f"[{idx}] {item['id']}: {item['question']}")
        started = time.time()

        try:
            data = call_paperops(args.backend_url, item, args.top_k)
            row = {
                **item,
                "answer": data.get("answer", ""),
                "citations": data.get("citations", []),
                "metrics": data.get("metrics", {}),
                "request_wall_time_ms": int((time.time() - started) * 1000),
                "error": None,
            }
        except Exception as exc:
            row = {
                **item,
                "answer": "",
                "citations": [],
                "metrics": {},
                "request_wall_time_ms": int((time.time() - started) * 1000),
                "error": str(exc),
            }

        rows.append(row)
        print({
            "id": row["id"],
            "answer_chars": len(row.get("answer", "")),
            "num_citations": len(row.get("citations", [])),
            "total_latency_ms": row.get("metrics", {}).get("total_latency_ms"),
            "error": row.get("error"),
        })

    save_jsonl(args.outputs_file, rows)
    summarize_outputs(rows)


def extract_json(text: str) -> dict:
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "answer_correctness": 0,
            "faithfulness": 0,
            "citation_support": 0,
            "abstention_correctness": 0,
            "reason": f"Could not parse judge JSON: {text[:300]}",
        }
    try:
        return json.loads(match.group(0))
    except Exception as exc:
        return {
            "answer_correctness": 0,
            "faithfulness": 0,
            "citation_support": 0,
            "abstention_correctness": 0,
            "reason": f"Could not parse judge JSON: {exc}; raw={text[:300]}",
        }


def judge_one(client, judge_model: str, item: dict) -> dict:
    citation_text = "\n\n".join(
        f"Source: {c.get('source')} page={c.get('page')} chunk={c.get('chunk_id')}\n{c.get('text', '')[:1200]}"
        for c in item.get("citations", [])
    )

    evidence_text = "\n".join(
        f"- source={ev.get('source')}; pages={ev.get('pages')}; evidence_summary={ev.get('evidence_summary')}"
        for ev in item.get("required_evidence", [])
    )

    prompt = f"""
Question:
{item['question']}

should_answer:
{item['should_answer']}

Reference answer:
{item['reference_answer']}

Required evidence:
{evidence_text}

Model answer:
{item.get('answer', '')}

Citations/context returned by RAG system:
{citation_text}
""".strip()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    text = response.choices[0].message.content or ""
    parsed = extract_json(text)
    parsed["judge_raw"] = text
    return parsed


def run_judge(args):
    from openai import OpenAI

    client = OpenAI(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )

    rows = []
    for idx, item in enumerate(load_jsonl(args.outputs_file), start=1):
        print(f"[JUDGE {idx}] {item['id']}")
        try:
            scores = judge_one(client, args.judge_model, item)
        except Exception as exc:
            scores = {
                "answer_correctness": 0,
                "faithfulness": 0,
                "citation_support": 0,
                "abstention_correctness": 0,
                "reason": f"judge_error: {exc}",
                "judge_raw": "",
            }

        row = {
            "id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "should_answer": item["should_answer"],
            **scores,
        }
        rows.append(row)
        print(row)

    save_jsonl(args.judge_file, rows)

    print("\n=== Judge Summary ===")
    print(f"cases={len(rows)}")
    for key in ["answer_correctness", "faithfulness", "citation_support", "abstention_correctness"]:
        vals = [float(r.get(key, 0)) for r in rows]
        print(f"{key}_avg={statistics.mean(vals):.3f}/3")

    answerable = [r for r in rows if r["should_answer"]]
    unsupported = [r for r in rows if not r["should_answer"]]
    if answerable:
        print(f"answerable_correctness_avg={statistics.mean(float(r.get('answer_correctness', 0)) for r in answerable):.3f}/3")
    if unsupported:
        print(f"unsupported_abstention_avg={statistics.mean(float(r.get('abstention_correctness', 0)) for r in unsupported):.3f}/3")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run PaperOps /ask on the benchmark")
    p_run.add_argument("--backend-url", default="http://localhost:8000")
    p_run.add_argument("--benchmark-file", default="evals/two_industroyers_50q_reference_benchmark.jsonl")
    p_run.add_argument("--outputs-file", default="evals/two_industroyers_outputs.jsonl")
    p_run.add_argument("--top-k", type=int, default=5)
    p_run.set_defaults(func=run_outputs)

    p_judge = sub.add_parser("judge", help="Use an OpenAI-compatible judge model to score outputs")
    p_judge.add_argument("--outputs-file", default="evals/two_industroyers_outputs.jsonl")
    p_judge.add_argument("--judge-file", default="evals/two_industroyers_judge_scores.jsonl")
    p_judge.add_argument("--judge-base-url", default="http://localhost:8002/v1")
    p_judge.add_argument("--judge-api-key", default="local-key")
    p_judge.add_argument("--judge-model", default="local-model")
    p_judge.set_defaults(func=run_judge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
