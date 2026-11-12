"""Retrieval metrics: score one ranked search result against the answer key.

Every function takes:
    retrieved  chunk labels in the order the search returned them, best first,
               e.g. ["product_catalog.txt#4", "returns_policy.txt#0", ...]
    relevant   the answer key for the question (golden_set "expected_chunks")
    k          how many of the top results to look at

They are pure functions (no API calls), so they cost nothing to run and are
easy to test. Out-of-scope questions have no relevant chunks and are scored
separately, so an empty `relevant` raises an error instead of returning a
misleading number.
"""
import math


def _check(relevant, k):
    if not relevant:
        raise ValueError("relevant is empty: retrieval metrics need at least one expected chunk")
    if k < 1:
        raise ValueError("k must be at least 1")


def precision_at_k(retrieved, relevant, k):
    """Share of the top k results that are relevant."""
    _check(relevant, k)
    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k):
    """Share of the relevant chunks that appear in the top k."""
    _check(relevant, k)
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def hit_rate_at_k(retrieved, relevant, k):
    """1.0 if at least one relevant chunk is in the top k, else 0.0."""
    _check(relevant, k)
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank_at_k(retrieved, relevant, k):
    """1 / position of the first relevant chunk in the top k (0.0 if none).
    Averaged over questions this is MRR."""
    _check(relevant, k)
    for position, label in enumerate(retrieved[:k], start=1):
        if label in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    """Order quality of the top k compared with the best possible order (0 to 1)."""
    _check(relevant, k)
    relevant = set(relevant)
    dcg, counted = 0.0, set()
    for position, label in enumerate(retrieved[:k], start=1):
        if label in relevant and label not in counted:   # a repeated chunk earns credit only once
            counted.add(label)
            dcg += 1 / math.log2(position + 1)
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def retrieval_scores(retrieved, relevant, k):
    """All five metrics for one question."""
    return {
        "precision": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "hit_rate": hit_rate_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank_at_k(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def average_scores(per_question):
    """Average each metric over questions (score per question first, then average)."""
    if not per_question:
        raise ValueError("no questions to average")
    return {name: sum(scores[name] for scores in per_question) / len(per_question)
            for name in per_question[0]}
