"""Score the search step on its own, against the golden set's section labels.

No LLM calls (one embedding per question, a fraction of a cent), so it is cheap
enough to run after every change. This is the check the CI gate will run.

    python -m src.evaluation.run_retrieval_eval                      # k = RETRIEVAL_K (5)
    python -m src.evaluation.run_retrieval_eval --k 3                # look at the top 3 only
    python -m src.evaluation.run_retrieval_eval --label vector-only  # name the run
    python -m src.evaluation.run_retrieval_eval --mode hybrid        # keywords + meaning

A hit counts when its chunk comes from an expected section ("<file> > <SECTION>"), so the labels
survive chunking changes. The run stops if a label matches no stored section, and warns when a
labelled section is split into several chunks (labels are then coarser than chunks).

Questions that should be refused (off-topic, prompt injection) have no relevant section,
so they are skipped here and scored as behaviour instead.
Each run is saved to eval_results/retrieval_<date>_<time>.json so runs can be compared later.
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import tiktoken
from dotenv import load_dotenv

load_dotenv()

from src.core.config import DEFAULT_MODEL, EMBEDDING_MODEL, RETRIEVAL_K, RETRIEVAL_MODE
from src.core.embeddings import get_embedding
from src.core.vectorstore import all_chunks
from src.evaluation.golden_set import GOLDEN_SET, VERSION
from src.evaluation.inspect_chunks import chunk_label, section_label
from src.evaluation.retrieval_metrics import average_scores, retrieval_scores
from src.rag.pipeline import RAGPipeline

RESULTS_DIR = "eval_results"


def print_table(per_category, overall, counts):
    print(f"\n{'category':<15} {'n':>3} {'recall':>7} {'hit':>6} {'mrr':>6} {'ndcg':>6} {'precision':>10}")
    rows = list(per_category.items()) + [("ALL", overall)]
    for name, s in rows:
        print(f"{name:<15} {counts[name]:>3} {s['recall']:>7.3f} {s['hit_rate']:>6.3f} {s['mrr']:>6.3f} "
              f"{s['ndcg']:>6.3f} {s['precision']:>10.3f}")


def heading(label):
    """'returns_policy.txt > REFUND TIMING' -> 'REFUND TIMING' (shorter for the printed report)."""
    return label.split(" > ", 1)[1]


def check_labels(cases):
    """Stop if a label names a section that is not in the collection (e.g. a renamed heading);
    warn when a labelled section is split into several chunks (labels are then coarser than chunks)."""
    stored = Counter(section_label(chunk) for chunk in all_chunks())
    labels = {label for case in cases for label in case["expected_sections"]}
    missing = sorted(labels - stored.keys())
    if missing:
        sys.exit("Golden-set labels not found in the collection (re-ingest, or fix the labels):\n  "
                 + "\n  ".join(missing))
    split = sorted(label for label in labels if stored[label] > 1)
    print(f"Labelled sections split into several chunks: {len(split)} of {len(labels)}"
          + (f" ({', '.join(split)})" if split else " (labels are exact)") + "\n")


def run(k, label="", mode=RETRIEVAL_MODE):
    pipeline = RAGPipeline(use_cache=False, retrieval_mode=mode)
    cases = [case for case in GOLDEN_SET if not case["should_refuse"]]
    skipped = len(GOLDEN_SET) - len(cases)
    check_labels(cases)
    encoder = tiktoken.encoding_for_model(DEFAULT_MODEL)
    results = []
    for case in cases:
        hits = pipeline.search(get_embedding(case["query"]), k=k, query_text=case["query"]).hits
        retrieved = [section_label(hit) for hit in hits]
        scores = retrieval_scores(retrieved, case["expected_sections"], k)
        missed = [section for section in case["expected_sections"] if section not in retrieved]
        context_tokens = sum(len(encoder.encode(hit["text"])) for hit in hits)
        results.append({"id": case["id"], "category": case["category"], "query": case["query"],
                        "expected": case["expected_sections"], "retrieved": retrieved,
                        "retrieved_chunks": [chunk_label(hit) for hit in hits],
                        "missed": missed, "context_tokens": context_tokens, "scores": scores})

        print(f"{case['id']:<9} recall {scores['recall']:.2f} | mrr {scores['mrr']:.2f} | "
              f"ndcg {scores['ndcg']:.2f} | {case['query']}")
        if scores["ndcg"] < 1:                      # show the order only when it was not perfect
            print(f"{'':<9} got:    {', '.join(heading(s) for s in retrieved)}")
        if missed:
            print(f"{'':<9} missed: {', '.join(heading(s) for s in missed)}")

    by_category = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result["scores"])
    per_category = {category: average_scores(scores) for category, scores in by_category.items()}
    overall = average_scores([result["scores"] for result in results])
    counts = {category: len(scores) for category, scores in by_category.items()} | {"ALL": len(results)}

    print_table(per_category, overall, counts)
    avg_tokens = sum(result["context_tokens"] for result in results) / len(results)
    print(f"\nmode={mode}, k={k}, golden set {VERSION}: {len(results)} questions scored, "
          f"{skipped} refuse questions skipped (scored as behaviour instead)")
    print(f"Context sent to the model at k={k}: {avg_tokens:.0f} tokens per question on average")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"label": label, "mode": mode, "golden_set": VERSION, "k": k, "embedding_model": EMBEDDING_MODEL,
                   "questions": len(results), "skipped_refuse": skipped,
                   "avg_context_tokens": round(avg_tokens), "averages": overall, "per_category": per_category, "per_question": results}, f, indent=2)
    print(f"Saved {path}")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the search step against the golden set.")
    parser.add_argument("--k", type=int, default=RETRIEVAL_K, help="how many top results to score")
    parser.add_argument("--label", default="", help="a name for this run, e.g. vector-only")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default=RETRIEVAL_MODE,
                        help="how to search (default: the RETRIEVAL_MODE setting)")
    args = parser.parse_args()
    run(args.k, args.label, args.mode)
