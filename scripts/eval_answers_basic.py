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


def keyword_coverage_text(answer: str, gold_keywords: list[str]) -> float:
    if not gold_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in gold_keywords if keyword.lower() in answer_lower)
    return hits / len(gold_keywords)


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    refusal_phrases = [
        "i do not know",
        "do not know",
        "not supported",
        "not found",
        "uploaded documents",
        "provided context",
        "insufficient context",
        "can't help",
        "cannot help",
        "not provide",
    ]
    return any(phrase in lower for phrase in refusal_phrases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--eval-file", default="evals/gold_qa_seed_industroyer.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evals/answer_results.jsonl")
    args = parser.parse_args()

    rows = []
    for item in load_jsonl(args.eval_file):
        response = requests.post(
            f"{args.backend_url}/ask",
            json={
                "user_id": item["user_id"],
                "project_id": item["project_id"],
                "question": item["question"],
                "top_k": args.top_k,
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        metrics = data.get("metrics", {})
        should_answer = item.get("should_answer", True)
        refused = is_refusal(answer)

        row = {
            "id": item["id"],
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "question": item["question"],
            "should_answer": should_answer,
            "answer": answer,
            "keyword_coverage": keyword_coverage_text(answer, item.get("gold_keywords", [])),
            "refused": refused,
            "refusal_correct": (not refused) if should_answer else refused,
            "citation_present": (len(citations) > 0) if should_answer else True,
            "num_citations": len(citations),
            "citations": citations,
            "total_latency_ms": metrics.get("total_latency_ms"),
            "generation_latency_ms": metrics.get("generation_latency_ms"),
            "retrieval_latency_ms": metrics.get("retrieval_latency_ms"),
            "tokens_per_second": metrics.get("estimated_tokens_per_second"),
            "cache_hit": metrics.get("cache_hit", False),
        }
        rows.append(row)
        print({
            "id": row["id"],
            "keyword_coverage": round(row["keyword_coverage"], 3),
            "refusal_correct": row["refusal_correct"],
            "citation_present": row["citation_present"],
            "total_latency_ms": row["total_latency_ms"],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    keyword_avg = statistics.mean(r["keyword_coverage"] for r in rows)
    refusal_acc = sum(1 for r in rows if r["refusal_correct"]) / len(rows)
    citation_rate = sum(1 for r in rows if r["citation_present"]) / len(rows)
    avg_latency = statistics.mean(
        r["total_latency_ms"] for r in rows
        if r["total_latency_ms"] is not None
    )

    answerable = [r for r in rows if r["should_answer"]]
    unsupported = [r for r in rows if not r["should_answer"]]
    answerable_kw = statistics.mean(r["keyword_coverage"] for r in answerable) if answerable else 0
    unsupported_refusal = (
        sum(1 for r in unsupported if r["refusal_correct"]) / len(unsupported)
        if unsupported else 0
    )

    print("\n=== Answer Evaluation Summary ===")
    print(f"num_cases={len(rows)}")
    print(f"avg_keyword_coverage_all={keyword_avg:.3f}")
    print(f"avg_keyword_coverage_answerable={answerable_kw:.3f}")
    print(f"refusal_accuracy_all={refusal_acc:.3f}")
    print(f"unsupported_refusal_accuracy={unsupported_refusal:.3f}")
    print(f"citation_presence_rate={citation_rate:.3f}")
    print(f"avg_total_latency_ms={avg_latency:.1f}")


if __name__ == "__main__":
    main()
