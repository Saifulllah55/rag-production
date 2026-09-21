import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from data_loader import embed_texts
from vector_db import QdrantStorage

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RUN_GENERATION = False


def embed_query(text: str) -> list[float]:
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []


def search(query_vector: list[float], top_k: int) -> list[str]:
    storage = QdrantStorage()
    results = storage.search(query_vector, top_k)
    if isinstance(results, dict):
        return results.get("contexts", [])
    return results


def generate_answer(question: str, context_chunks: list[str]) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    context = "\n\n".join(context_chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {question}"

    resp = client.chat.completions.create(
        model="gemini-3.6-flash",
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided context.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Eval logic
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    id: int
    question: str
    difficulty: str
    expected_answer: str
    source_snippet: str
    retrieved_texts: list[str] = field(default_factory=list)
    hit: bool = False
    rank: int | None = None
    generated_answer: str | None = None
    error: str | None = None


def snippet_found(snippet: str, chunk_text: str) -> bool:
    """
    Loose containment check: normalizes whitespace/case since chunk boundaries
    and PDF extraction artifacts rarely match the source_snippet's formatting.
    """
    norm = lambda s: " ".join(s.lower().split())

    return (
        norm(snippet) in norm(chunk_text)
        or norm(chunk_text) in norm(snippet)
    )


def run_eval(eval_file: str, top_k: int) -> list[EvalResult]:
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[EvalResult] = []

    for item in data["eval_pairs"]:
        result = EvalResult(
            id=item["id"],
            question=item["question"],
            difficulty=item.get("difficulty", "unknown"),
            expected_answer=item["expected_answer"],
            source_snippet=item["source_snippet"],
        )

        try:
            query_vector = embed_query(item["question"])
            hits = search(query_vector, top_k)

            if isinstance(hits, dict):
                result.retrieved_texts = hits.get("contexts", [])
            elif hits and isinstance(hits[0], dict):
                result.retrieved_texts = [h.get("text", "") for h in hits]
            else:
                result.retrieved_texts = list(hits)

            for rank, chunk_text in enumerate(
                result.retrieved_texts,
                start=1,
            ):
                if snippet_found(item["source_snippet"], chunk_text):
                    result.hit = True
                    result.rank = rank
                    break

            if RUN_GENERATION:
                result.generated_answer = generate_answer(
                    item["question"],
                    result.retrieved_texts,
                )

        except NotImplementedError as e:
            result.error = str(e)

        except Exception as e:
            result.error = f"Unexpected error: {e}"

        results.append(result)

    return results


def print_report(results: list[EvalResult], top_k: int) -> None:
    if results and results[0].error:
        print(f"ERROR on first item: {results[0].error}")
        print(
            "\nFix the CONNECT HERE sections at the top of this script "
            "before running."
        )
        sys.exit(1)

    total = len(results)
    hits = sum(1 for r in results if r.hit)
    hit_rate = hits / total if total else 0.0

    reciprocal_ranks = [
        1 / r.rank
        for r in results
        if r.hit and r.rank
    ]

    mrr = sum(reciprocal_ranks) / total if total else 0.0

    by_difficulty: dict[str, list[EvalResult]] = {}

    for r in results:
        by_difficulty.setdefault(r.difficulty, []).append(r)

    print("=" * 70)
    print(f"RETRIEVAL EVAL REPORT  (top_k={top_k}, n={total})")
    print("=" * 70)
    print(f"Overall hit-rate: {hits}/{total} ({hit_rate:.1%})")
    print(f"MRR: {mrr:.3f}")
    print()

    print("By difficulty:")

    for diff, items in by_difficulty.items():
        d_hits = sum(1 for r in items if r.hit)

        print(
            f"  {diff:12s}: "
            f"{d_hits}/{len(items)} "
            f"({d_hits / len(items):.1%})"
        )

    print()

    print("Failures (retrieval missed the source chunk):")

    failures = [r for r in results if not r.hit]

    if not failures:
        print("  None -- all questions retrieved their source chunk.")

    for r in failures:
        print(f"  [{r.id}] ({r.difficulty}) {r.question}")
        print(
            f"      expected snippet: "
            f"{r.source_snippet[:80]}..."
        )

        if r.retrieved_texts:
            print(
                f"      top retrieved chunk (truncated): "
                f"{r.retrieved_texts[0][:80]}..."
            )
        else:
            print("      no chunks retrieved")

    print()

    if RUN_GENERATION:
        print("=" * 70)
        print("GENERATED ANSWERS (manual review)")
        print("=" * 70)

        for r in results:
            print(f"\n[{r.id}] {r.question}")
            print(f"  Expected : {r.expected_answer}")
            print(f"  Generated: {r.generated_answer}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality."
    )

    parser.add_argument(
        "--eval-file",
        default="graph-intro-eval-set.json",
        help="Path to the eval JSON file",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question",
    )

    args = parser.parse_args()

    results = run_eval(args.eval_file, args.top_k)
    print_report(results, args.top_k)


if __name__ == "__main__":
    main()