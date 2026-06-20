import argparse
import csv
import statistics
import time
from pathlib import Path

import requests


def run_once(
    backend_url: str,
    user_id: str,
    project_id: str,
    question: str,
    top_k: int,
) -> dict:
    start = time.time()

    response = requests.post(
        f"{backend_url}/ask",
        json={
            "user_id": user_id,
            "project_id": project_id,
            "question": question,
            "top_k": top_k,
        },
        timeout=300,
    )

    wall_time_ms = int((time.time() - start) * 1000)
    response.raise_for_status()

    data = response.json()
    metrics = data.get("metrics", {})

    answer = data.get("answer", "")
    citations = data.get("citations", [])

    return {
        "wall_time_ms": wall_time_ms,
        "retrieval_latency_ms": metrics.get("retrieval_latency_ms"),
        "generation_latency_ms": metrics.get("generation_latency_ms"),
        "total_latency_ms": metrics.get("total_latency_ms"),
        "estimated_tokens_per_second": metrics.get("estimated_tokens_per_second"),
        "llm_backend": metrics.get("llm_backend"),
        "llm_model": metrics.get("llm_model"),
        "mock": metrics.get("mock"),
        "answer_chars": len(answer),
        "answer_words": len(answer.split()),
        "num_citations": len(citations),
    }


def summarize(rows: list[dict]) -> dict:
    numeric_keys = [
        "wall_time_ms",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "estimated_tokens_per_second",
        "answer_chars",
        "answer_words",
        "num_citations",
    ]

    summary = {}

    for key in numeric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue

        summary[f"{key}_mean"] = round(statistics.mean(values), 2)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--user-id", default="hojune")
    parser.add_argument("--project-id", default="test-project")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--question",
        default="Explain in more than 500 words what Industroyer 1 is.",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Name of benchmark condition, e.g. qwen7b_q4_baseline",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/ask_latency.csv",
    )

    args = parser.parse_args()

    rows = []

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs}: {args.label}")
        row = run_once(
            backend_url=args.backend_url,
            user_id=args.user_id,
            project_id=args.project_id,
            question=args.question,
            top_k=args.top_k,
        )
        row["label"] = args.label
        row["run"] = i + 1
        rows.append(row)
        print(row)

    summary = summarize(rows)

    print("\nSummary:")
    print(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label",
        "run",
        "wall_time_ms",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "estimated_tokens_per_second",
        "llm_backend",
        "llm_model",
        "mock",
        "answer_chars",
        "answer_words",
        "num_citations",
    ]

    file_exists = out_path.exists()

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\nSaved rows to {out_path}")


if __name__ == "__main__":
    main()