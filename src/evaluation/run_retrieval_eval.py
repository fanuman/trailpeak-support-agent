"""Score the search step on its own, against the golden set's chunk labels.

No LLM calls (one embedding per question, a fraction of a cent), so it is cheap
enough to run after every change. This is the check the CI gate will run.

    python -m src.evaluation.run_retrieval_eval                      # k = RETRIEVAL_K (5)
    python -m src.evaluation.run_retrieval_eval --k 3                # look at the top 3 only
    python -m src.evaluation.run_retrieval_eval --label vector-only  # name the run

Each run is saved to eval_results/retrieval_<date>_<time>.json so runs can be compared later.
"""
import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.core.config import EMBEDDING_MODEL, RETRIEVAL_K
from src.core.embeddings import get_embedding
from src.evaluation.golden_set import GOLDEN_SET
from src.evaluation.inspect_chunks import chunk_position
from src.evaluation.retrieval_metrics import average_scores, retrieval_scores
from src.rag.pipeline import RAGPipeline

RESULTS_DIR = "eval_results"


def chunk_label(hit):
    """Turn a search hit into the same "<file>#<position>" label the golden set uses."""
    return f"{hit['source']}#{chunk_position(hit['source'], hit['id'])}"


def run(k, label=""):
    pipeline = RAGPipeline(use_cache=False)
    results = []
    for case in GOLDEN_SET:
        hits = pipeline.search(get_embedding(case["query"]), k=k)
        retrieved = [chunk_label(hit) for hit in hits]
        scores = retrieval_scores(retrieved, case["expected_chunks"], k)
        missed = [chunk for chunk in case["expected_chunks"] if chunk not in retrieved]
        results.append({"query": case["query"], "expected": case["expected_chunks"],
                        "retrieved": retrieved, "missed": missed, "scores": scores})

        print(f"recall {scores['recall']:.2f} | mrr {scores['mrr']:.2f} | ndcg {scores['ndcg']:.2f} "
              f"| precision {scores['precision']:.2f} | {case['query']}")
        print(f"   got:    {', '.join(retrieved)}")
        if missed:
            print(f"   missed: {', '.join(missed)}")

    averages = average_scores([r["scores"] for r in results])
    print(f"\nAverages over {len(results)} questions (k={k}):")
    for name, value in averages.items():
        print(f"   {name:<10} {value:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"label": label, "k": k, "embedding_model": EMBEDDING_MODEL, "questions": len(results),
                   "averages": averages, "per_question": results}, f, indent=2)
    print(f"\nSaved {path}")
    return averages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the search step against the golden set.")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K, help="how many top results to score")
    parser.add_argument("--label", default="", help="a name for this run, e.g. vector-only")
    args = parser.parse_args()
    run(args.k, args.label)
