"""Tests for src/evaluation/retrieval_metrics.py.
The expected numbers were worked out by hand (see the Day 1 notes), so these
tests pin the maths: a future change cannot silently break a metric."""
import pytest

from src.evaluation.retrieval_metrics import (
    average_scores,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    retrieval_scores,
)

RELEVANT = {"c2", "c4"}
SEARCH = ["c7", "c2", "c9", "c4", "c1"]      # relevant chunks at positions 2 and 4
REORDERED = ["c2", "c4", "c7", "c9", "c1"]   # the same five chunks, relevant ones first


def test_worked_example_from_the_notes():
    scores = retrieval_scores(SEARCH, RELEVANT, k=5)
    assert scores["precision"] == pytest.approx(0.4)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["hit_rate"] == 1.0
    assert scores["mrr"] == pytest.approx(0.5)
    assert scores["ndcg"] == pytest.approx(0.651, abs=0.001)


def test_reordering_changes_only_the_rank_aware_metrics():
    before = retrieval_scores(SEARCH, RELEVANT, k=5)
    after = retrieval_scores(REORDERED, RELEVANT, k=5)
    assert after["precision"] == before["precision"]
    assert after["recall"] == before["recall"]
    assert after["mrr"] == pytest.approx(1.0)
    assert after["ndcg"] == pytest.approx(1.0)


def test_smaller_k_looks_at_fewer_results():
    assert recall_at_k(SEARCH, RELEVANT, k=3) == pytest.approx(0.5)
    assert precision_at_k(SEARCH, RELEVANT, k=3) == pytest.approx(1 / 3)


def test_second_relevant_chunk_sinking_is_seen_by_ndcg_not_mrr():
    sunk = ["c2", "c7", "c9", "c1", "c4"]
    assert reciprocal_rank_at_k(sunk, RELEVANT, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(sunk, RELEVANT, k=5) == pytest.approx(0.850, abs=0.001)


def test_two_fact_question_with_one_fact_found():
    found_one = ["c2", "c7", "c9", "c1", "c3"]
    assert hit_rate_at_k(found_one, RELEVANT, k=5) == 1.0          # hit rate says "fine"
    assert recall_at_k(found_one, RELEVANT, k=5) == pytest.approx(0.5)  # recall shows the miss


def test_nothing_relevant_found_scores_zero():
    scores = retrieval_scores(["c7", "c9", "c1", "c3", "c5"], RELEVANT, k=5)
    assert scores == {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_duplicates_are_counted_once():
    assert recall_at_k(["c2", "c2", "c2"], RELEVANT, k=3) == pytest.approx(0.5)
    assert precision_at_k(["c2", "c2", "c2"], RELEVANT, k=3) == pytest.approx(1 / 3)


def test_more_relevant_chunks_than_k():
    relevant = {"a", "b", "c"}
    assert ndcg_at_k(["a", "b"], relevant, k=2) == pytest.approx(1.0)   # best possible with 2 slots
    assert recall_at_k(["a", "b"], relevant, k=2) == pytest.approx(2 / 3)


def test_empty_answer_key_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(SEARCH, set(), k=5)


def test_k_below_one_is_an_error():
    with pytest.raises(ValueError):
        precision_at_k(SEARCH, RELEVANT, k=0)


def test_average_scores_per_question_then_mean():
    averaged = average_scores([
        retrieval_scores(SEARCH, RELEVANT, k=5),      # mrr 0.5
        retrieval_scores(REORDERED, RELEVANT, k=5),   # mrr 1.0
    ])
    assert averaged["mrr"] == pytest.approx(0.75)
    assert averaged["recall"] == pytest.approx(1.0)


def test_average_of_no_questions_is_an_error():
    with pytest.raises(ValueError):
        average_scores([])


def test_a_repeated_chunk_cannot_push_ndcg_above_one():
    assert ndcg_at_k(["c2", "c2"], {"c2"}, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(["c2", "c2", "c4"], RELEVANT, k=3) <= 1.0
