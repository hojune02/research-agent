import argparse
import time

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8002/v1")
    parser.add_argument("--api-key", default="local-key")
    parser.add_argument("--model", default="local-model")
    parser.add_argument(
        "--prompt",
        default="Explain in more than 500 words what Industroyer 1 is.",
    )
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    start = time.time()
    first_token_time = None
    chunks = []

    stream = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        temperature=0.2,
        max_tokens=args.max_tokens,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta.content

        if delta:
            if first_token_time is None:
                first_token_time = time.time()

            chunks.append(delta)
            print(delta, end="", flush=True)

    end = time.time()

    output = "".join(chunks)
    estimated_tokens = max(1, int(len(output.split()) * 1.3))

    print("\n\n--- Metrics ---")
    print(f"time_to_first_token_ms={int((first_token_time - start) * 1000)}")
    print(f"total_generation_ms={int((end - start) * 1000)}")
    print(f"estimated_output_tokens={estimated_tokens}")
    print(f"estimated_tokens_per_second={round(estimated_tokens / (end - start), 2)}")


if __name__ == "__main__":
    main()