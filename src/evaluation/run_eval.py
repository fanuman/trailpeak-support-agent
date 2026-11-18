"""Score the bot's answers end to end with LLM judges, on golden set v1.

The real pipeline answers every question (cache off). Then:
  - the answerable questions get four answer scores: faithfulness, answer relevancy,
    context precision (source files) and context recall (LLM-judged: was the context enough?);
  - every question gets a "did it decline?" check, scored against should_refuse with precision,
    recall and the false refusal rate. A decline comes from the distance gate (the fallback
    message, no model call) or from the model itself (judged by refusal_check()).
LLM judges vary from run to run, so the same code gives slightly different scores: run it
several times to measure that noise before trusting a difference.

    python -m src.evaluation.run_eval                      # one run, saved to eval_results/
    python -m src.evaluation.run_eval --label noise-1      # name the run
    python -m src.evaluation.run_eval --compare-baseline   # also compare with the saved baseline

Costs real model calls (gpt-4o-mini), roughly ten per answerable question.
"""
import argparse
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.core.config import DEFAULT_MODEL
from src.evaluation.behaviour_metrics import classification_scores, confusion_counts
from src.evaluation.golden_set import GOLDEN_SET, VERSION
from src.evaluation.metrics import (answer_relevancy_score, context_precision, context_recall,
                                    faithfulness_score, refusal_check)
from src.evaluation.regression import check_regression
from src.rag.pipeline import RAGPipeline

RESULTS_DIR = "eval_results"
ANSWER_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def score_answer(case, result):
    """The four answer scores (each 0 to 1) for one answerable question, plus the recall judge's reasoning."""
    faithfulness, _ = faithfulness_score(result["answer"], result["full_context"])
    relevancy = answer_relevancy_score(case["query"], result["answer"])
    recall = context_recall(case["query"], result["full_context"])
    scores = {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy.score / 5,       # the judge answers 1 to 5
        "context_precision": context_precision(result["sources"], case["expected_sources"]),
        "context_recall": 1.0 if recall.sufficient else 0.0,
    }
    return scores, recall.reasoning


def who_declined(pipeline, case, result):
    """('gate' | 'model' | None, reason): the gate sends the fixed fallback message; otherwise the judge decides."""
    if result["answer"] == pipeline.FALLBACK_MESSAGE:
        return "gate", "the distance gate sent the fallback message"
    verdict = refusal_check(case["query"], result["answer"])
    return ("model" if verdict.declined else None), verdict.reasoning


def print_row(number, row):
    mismatch = (row["declined_by"] is not None) != row["should_refuse"]
    if row["scores"]:
        s = row["scores"]
        outcome = (f"faith {s['faithfulness']:.2f} | relev {s['answer_relevancy']:.2f} | "
                   f"prec {s['context_precision']:.2f} | recall {'yes' if s['context_recall'] else 'no '}")
    else:
        outcome = f"declined by {row['declined_by']}" if row["declined_by"] else "ANSWERED"
    if row["scores"] and row["declined_by"]:
        outcome += f" | declined by {row['declined_by']}"
    print(f"[{number:>2}/{len(GOLDEN_SET)}] {row['id']:<9} {outcome}{'   <-- refusal mistake' if mismatch else ''}")
    if mismatch:
        print(f"{'':<17}answer: {row['answer'][:140]}")
        print(f"{'':<17}judge:  {row['refusal_reasoning'][:140]}")
    elif row["scores"] and (row["scores"]["faithfulness"] < 1 or not row["scores"]["context_recall"]):
        print(f"{'':<17}answer: {row['answer'][:140]}")


def averages(rows):
    return {name: sum(row["scores"][name] for row in rows) / len(rows) for name in ANSWER_METRICS}


def run(label="", compare_baseline=False):
    pipeline = RAGPipeline(use_cache=False)
    started = time.time()
    rows = []
    for number, case in enumerate(GOLDEN_SET, start=1):
        result = pipeline.answer(case["query"])
        declined_by, reasoning = who_declined(pipeline, case, result)
        row = {"id": case["id"], "category": case["category"], "query": case["query"],
               "should_refuse": case["should_refuse"], "answer": result["answer"],
               "declined_by": declined_by, "refusal_reasoning": reasoning, "scores": None}
        if not case["should_refuse"]:
            row["scores"], row["recall_reasoning"] = score_answer(case, result)
        rows.append(row)
        print_row(number, row)

    answerable = [row for row in rows if row["scores"]]
    by_category = {}
    for row in answerable:
        by_category.setdefault(row["category"], []).append(row)
    per_category = {category: averages(items) for category, items in by_category.items()}
    overall = averages(answerable)

    print(f"\n{'category':<15} {'n':>3} {'faith':>6} {'relev':>6} {'prec':>6} {'recall':>7}")
    for name, s in list(per_category.items()) + [("ALL", overall)]:
        n = len(by_category[name]) if name in by_category else len(answerable)
        print(f"{name:<15} {n:>3} {s['faithfulness']:>6.3f} {s['answer_relevancy']:>6.3f} "
              f"{s['context_precision']:>6.3f} {s['context_recall']:>7.3f}")

    counts = confusion_counts([row["declined_by"] is not None for row in rows], [row["should_refuse"] for row in rows])
    refusal = classification_scores(counts)
    print(f"\nDeclining (gate + model, all {len(rows)} questions, injections included):")
    print(f"  TP {counts['tp']}  FP {counts['fp']}  FN {counts['fn']}  TN {counts['tn']}")
    print(f"  precision {refusal['precision']:.3f} | recall {refusal['recall']:.3f} | f1 {refusal['f1']:.3f} | "
          f"false refusal rate {refusal['false_positive_rate']:.3f}")
    print("\nDeclined, by category (gate / model):")
    categories = {}
    for row in rows:
        categories.setdefault(row["category"], []).append(row)
    for category, items in categories.items():
        gate = sum(row["declined_by"] == "gate" for row in items)
        model = sum(row["declined_by"] == "model" for row in items)
        goal = "should all be declined" if items[0]["should_refuse"] else "should all be answered"
        print(f"  {category:<14} {gate + model:>2} / {len(items):<2} (gate {gate}, model {model})  {goal}")

    minutes = (time.time() - started) / 60
    print(f"\nGolden set {VERSION}, model {DEFAULT_MODEL}, {minutes:.1f} minutes")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"llm_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump({"label": label, "golden_set": VERSION, "model": DEFAULT_MODEL, "minutes": round(minutes, 1),
                   "averages": overall, "per_category": per_category, "refusal_counts": counts,
                   "refusal_scores": refusal, "per_question": rows}, f, indent=2)
    print(f"Saved {path}")

    if compare_baseline:
        regressions = check_regression(overall)
        for message in regressions:
            print(message)
        if not regressions:
            print("No regressions against the baseline (threshold 0.04).")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the bot's answers with LLM judges on the golden set.")
    parser.add_argument("--label", default="", help="a name for this run, e.g. noise-1")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="compare with src/evaluation/baseline_scores.json (only meaningful once it is from golden set v1)")
    args = parser.parse_args()
    run(args.label, args.compare_baseline)
