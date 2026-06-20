import argparse
import json
import statistics
from pathlib import Path

import requests


def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def contains_gold_source(results, gold_sources):
    if not gold_sources:
        return True
    for result in results:
        source = result.get("source", "")
        for gold_source in gold_sources:
            if gold_source.lower() in source.lower():
                return True
    return False


def reciprocal_rank(results, gold_sources):
    if not gold_sources:
        return 1.0
    for idx, result in enumerate(results, start=1):
        source = result.get("source", "")
        for gold_source in gold_sources:
            if gold_source.lower() in source.lower():
                return 1.0 / idx
    return 0.0


def keyword_coverage(results, gold_keywords):
    joined = " ".join(result.get("text", "") for result in results).lower()
    if not gold_keywords:
        return 0.0
    hits = sum(1 for keyword in gold_keywords if keyword.lower() in joined)
    return hits / len(gold_keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--eval-file", default="evals/gold_qa_seed_industroyer.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evals/retrieval_results.jsonl")
    args = parser.parse_args()

    rows = []
    for item in load_jsonl(args.eval_file):
        if item.get("category") in {"unsupported", "safety_refusal"}:
            # Unsupported questions are better evaluated with /ask refusal behavior.
            continue

        response = requests.post(
            f"{args.backend_url}/search",
            json={
                "user_id": item["user_id"],
                "project_id": item["project_id"],
                "query": item["question"],
                "top_k": args.top_k,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        row = {
            "id": item["id"],
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "question": item["question"],
            "top_k": args.top_k,
            "recall_hit": contains_gold_source(results, item.get("gold_sources", [])),
            "reciprocal_rank": reciprocal_rank(results, item.get("gold_sources", [])),
            "keyword_coverage": keyword_coverage(results, item.get("gold_keywords", [])),
            "retrieval_latency_ms": data.get("retrieval_latency_ms"),
            "num_results": len(results),
            "top_sources": [r.get("source") for r in results],
        }
        rows.append(row)
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    if not rows:
        print("No answerable rows evaluated.")
        return

    recall = sum(1 for r in rows if r["recall_hit"]) / len(rows)
    mrr = statistics.mean(r["reciprocal_rank"] for r in rows)
    keyword_coverage_avg = statistics.mean(r["keyword_coverage"] for r in rows)
    retrieval_latency_avg = statistics.mean(
        r["retrieval_latency_ms"] for r in rows
        if r["retrieval_latency_ms"] is not None
    )

    print("\n=== Retrieval Evaluation Summary ===")
    print(f"num_cases={len(rows)}")
    print(f"recall@{args.top_k}={recall:.3f}")
    print(f"mrr={mrr:.3f}")
    print(f"avg_keyword_coverage={keyword_coverage_avg:.3f}")
    print(f"avg_retrieval_latency_ms={retrieval_latency_avg:.1f}")


if __name__ == "__main__":
    main()
