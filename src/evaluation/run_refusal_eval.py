"""Score the refusal gate: the distance check that sends the fallback message before the model is called.

The gate refuses a question when even the closest chunk is at least RELEVANCE_THRESHOLD away
(RAGPipeline._is_out_of_scope). It is scored as a yes/no classifier, positive = "should be refused":
  - off-topic questions should be refused (this is the gate's job);
  - real customer questions should be answered;
  - prompt-injection questions often mention the store and look close, so stopping them is the
    model's job: they are listed, but left out of the scores.
Then the same questions are scored at other thresholds (a sweep) to show the trade-off.
No LLM calls (one embedding per question).

    python -m src.evaluation.run_refusal_eval                 # the RETRIEVAL_MODE setting
    python -m src.evaluation.run_refusal_eval --mode hybrid   # the hybrid search path

Each run is saved to eval_results/refusal_<date>_<time>.json.
"""
import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.core.config import RETRIEVAL_K, RETRIEVAL_MODE
from src.core.embeddings import get_embedding
from src.evaluation.behaviour_metrics import classification_scores, confusion_counts
from src.evaluation.golden_set import GOLDEN_SET, VERSION
from src.rag.pipeline import RAGPipeline

RESULTS_DIR = "eval_results"
SWEEP = [round(0.50 + 0.02 * step, 2) for step in range(21)]   # 0.50, 0.52, ... 0.90


def measure(pipeline):
    """Search every question once; keep its closest distance and the gate's real decision."""
    rows = []
    for case in GOLDEN_SET:
        result = pipeline.search(get_embedding(case["query"]), k=RETRIEVAL_K, query_text=case["query"])
        rows.append({"id": case["id"], "category": case["category"], "query": case["query"],
                     "should_refuse": case["should_refuse"], "closest": result.closest,
                     "refused": pipeline._is_out_of_scope(result.closest)})
    return rows


def score(rows, decisions):
    """Confusion counts and scores for the gate's job: off-topic vs real questions (injections left out)."""
    judged = [(row, decided) for row, decided in zip(rows, decisions) if row["category"] != "injection"]
    counts = confusion_counts([decided for _, decided in judged], [row["should_refuse"] for row, _ in judged])
    return counts, classification_scores(counts)


def run(mode=RETRIEVAL_MODE):
    pipeline = RAGPipeline(use_cache=False, retrieval_mode=mode)
    threshold = pipeline.relevance_threshold
    rows = measure(pipeline)
    counts, scores = score(rows, [row["refused"] for row in rows])

    print(f"Refusal gate at RELEVANCE_THRESHOLD = {threshold}, mode={mode} (golden set {VERSION}, injections left out)")
    print(f"  TP {counts['tp']}  FP {counts['fp']}  FN {counts['fn']}  TN {counts['tn']}")
    print(f"  precision {scores['precision']:.3f} | recall {scores['recall']:.3f} | f1 {scores['f1']:.3f} | "
          f"false refusal rate {scores['false_positive_rate']:.3f} | accuracy {scores['accuracy']:.3f}")

    print("\nRefused, by category:")
    by_category = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    for category, items in by_category.items():
        refused = sum(row["refused"] for row in items)
        goal = "should all be refused" if items[0]["should_refuse"] else "should all be answered"
        print(f"  {category:<14} {refused:>2} / {len(items):<2} ({goal})")

    mistakes = sorted((row for row in rows if row["refused"] != row["should_refuse"]), key=lambda row: row["closest"])
    if mistakes:
        print("\nMistakes (closest distance, what happened):")
        for row in mistakes:
            if row["refused"]:
                what = "refused a real question"
            elif row["category"] == "injection":
                what = "let an injection through (the model's job, Part C)"
            else:
                what = "answered an off-topic question"
            print(f"  {row['id']:<9} {row['closest']:.3f}  {what}: {row['query']}")

    real = [row["closest"] for row in rows if not row["should_refuse"]]
    off_topic = [row["closest"] for row in rows if row["category"] == "off_topic"]
    print(f"\nClosest distance - real questions: {min(real):.3f} to {max(real):.3f}, "
          f"off-topic: {min(off_topic):.3f} to {max(off_topic):.3f}")

    print(f"\n{'threshold':>9}  TP FP FN TN  precision recall    f1  false refusals")
    sweep = []
    for value in SWEEP:
        sweep_counts, sweep_scores = score(rows, [row["closest"] >= value for row in rows])
        sweep.append({"threshold": value, "counts": sweep_counts, "scores": sweep_scores})
        marker = "  <- current" if abs(value - threshold) < 1e-9 else ""
        print(f"{value:>9.2f}  {sweep_counts['tp']:>2} {sweep_counts['fp']:>2} {sweep_counts['fn']:>2} "
              f"{sweep_counts['tn']:>2}  {sweep_scores['precision']:>9.3f} {sweep_scores['recall']:>6.3f} "
              f"{sweep_scores['f1']:>5.3f}  {sweep_scores['false_positive_rate']:>14.3f}{marker}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"refusal_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"golden_set": VERSION, "mode": mode, "threshold": threshold, "k": RETRIEVAL_K, "counts": counts,
                   "scores": scores, "sweep": sweep, "per_question": rows}, f, indent=2)
    print(f"\nSaved {path}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the refusal gate on the golden set.")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default=RETRIEVAL_MODE,
                        help="how to search (default: the RETRIEVAL_MODE setting)")
    run(parser.parse_args().mode)
