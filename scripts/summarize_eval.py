import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(rows, key):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(vals) if vals else None


def pct(x):
    return f"{100*x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="evals/retrieval_results.jsonl")
    parser.add_argument("--answers", default="evals/answer_results.jsonl")
    args = parser.parse_args()

    retrieval = load_jsonl(args.retrieval)
    answers = load_jsonl(args.answers)

    print("# PaperOps Evaluation Summary\n")

    if retrieval:
        recall = sum(1 for r in retrieval if r.get("recall_hit")) / len(retrieval)
        mrr = mean(retrieval, "reciprocal_rank")
        kw = mean(retrieval, "keyword_coverage")
        lat = mean(retrieval, "retrieval_latency_ms")
        print("## Retrieval")
        print(f"- Cases: {len(retrieval)}")
        print(f"- Recall@k: {pct(recall)}")
        print(f"- MRR: {mrr:.3f}")
        print(f"- Avg retrieved keyword coverage: {kw:.3f}")
        print(f"- Avg retrieval latency: {lat:.1f} ms\n")

    if answers:
        answerable = [r for r in answers if r.get("should_answer")]
        unsupported = [r for r in answers if not r.get("should_answer")]
        kw_all = mean(answers, "keyword_coverage")
        kw_ans = mean(answerable, "keyword_coverage")
        refusal_all = sum(1 for r in answers if r.get("refusal_correct")) / len(answers)
        refusal_unsup = sum(1 for r in unsupported if r.get("refusal_correct")) / len(unsupported) if unsupported else 0
        citation_rate = sum(1 for r in answers if r.get("citation_present")) / len(answers)
        total_lat = mean(answers, "total_latency_ms")
        gen_lat = mean(answers, "generation_latency_ms")

        print("## Answering")
        print(f"- Cases: {len(answers)}")
        print(f"- Avg keyword coverage all: {kw_all:.3f}")
        print(f"- Avg keyword coverage answerable: {kw_ans:.3f}")
        print(f"- Refusal accuracy all: {pct(refusal_all)}")
        print(f"- Unsupported-question refusal accuracy: {pct(refusal_unsup)}")
        print(f"- Citation presence rate: {pct(citation_rate)}")
        print(f"- Avg total latency: {total_lat:.1f} ms")
        print(f"- Avg generation latency: {gen_lat:.1f} ms\n")


if __name__ == "__main__":
    main()
