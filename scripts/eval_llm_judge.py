import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


JUDGE_PROMPT = """
You are evaluating a RAG answer.

Score the answer from 0 to 5 for each criterion.

Criteria:
1. Correctness: Does the answer answer the question correctly?
2. Faithfulness: Are the claims supported by the retrieved/cited context?
3. Completeness: Does the answer include the important points?
4. Citation support: Do the citations appear relevant to the answer?

Return only valid JSON:
{
  "correctness": 0-5,
  "faithfulness": 0-5,
  "completeness": 0-5,
  "citation_support": 0-5,
  "reason": "short explanation"
}
"""


def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "correctness": 0,
            "faithfulness": 0,
            "completeness": 0,
            "citation_support": 0,
            "reason": f"Could not parse judge output: {text[:200]}",
        }

    return json.loads(match.group(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", default="evals/answer_results.jsonl")
    parser.add_argument("--base-url", default="http://localhost:8002/v1")
    parser.add_argument("--api-key", default="local-key")
    parser.add_argument("--judge-model", default="local-model")
    parser.add_argument("--out", default="evals/judge_results.jsonl")
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    rows = []

    for item in load_jsonl(args.answers_file):
        citations_text = "\n\n".join(
            citation.get("text", "")
            for citation in item.get("citations", [])
        )

        user_content = f"""
Question:
{item["question"]}

Generated answer:
{item["answer"]}

Retrieved/cited context:
{citations_text}
"""

        response = client.chat.completions.create(
            model=args.judge_model,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT.strip()},
                {"role": "user", "content": user_content.strip()},
            ],
            temperature=0,
            max_tokens=300,
        )

        judge_text = response.choices[0].message.content or ""
        judge = extract_json(judge_text)

        row = {
            "id": item["id"],
            "question": item["question"],
            **judge,
        }

        rows.append(row)
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    def avg(key):
        return sum(float(r[key]) for r in rows) / len(rows)

    print("\n=== Judge Evaluation Summary ===")
    print(f"correctness_avg={avg('correctness'):.2f}/5")
    print(f"faithfulness_avg={avg('faithfulness'):.2f}/5")
    print(f"completeness_avg={avg('completeness'):.2f}/5")
    print(f"citation_support_avg={avg('citation_support'):.2f}/5")


if __name__ == "__main__":
    main()