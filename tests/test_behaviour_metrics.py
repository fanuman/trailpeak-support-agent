"""Tests for src/evaluation/behaviour_metrics.py, using the refusal demo from the Day 2 notes:
8 real questions and 4 off-topic ones, each with the distance to its closest chunk."""
import pytest

from src.evaluation.behaviour_metrics import classification_scores, confusion_counts

QUESTIONS = [
    (0.32, False), (0.41, False), (0.45, False), (0.52, False), (0.58, False),
    (0.61, False), (0.66, False), (0.71, False),            # real customer questions
    (0.64, True), (0.74, True), (0.79, True), (0.83, True),  # off-topic questions
]
TRUTHS = [should_refuse for _, should_refuse in QUESTIONS]


def refusals(threshold):
    return [distance >= threshold for distance, _ in QUESTIONS]


def test_counts_at_threshold_070():
    assert confusion_counts(refusals(0.70), TRUTHS) == {"tp": 3, "fp": 1, "fn": 1, "tn": 7}


def test_scores_at_threshold_070():
    scores = classification_scores(confusion_counts(refusals(0.70), TRUTHS))
    assert scores["precision"] == pytest.approx(0.75)
    assert scores["recall"] == pytest.approx(0.75)
    assert scores["f1"] == pytest.approx(0.75)
    assert scores["accuracy"] == pytest.approx(10 / 12)
    assert scores["false_positive_rate"] == pytest.approx(1 / 8)


def test_refusing_too_much_gives_perfect_recall_but_poor_precision():
    scores = classification_scores(confusion_counts(refusals(0.55), TRUTHS))
    assert scores["recall"] == pytest.approx(1.0)      # every off-topic question caught...
    assert scores["precision"] == pytest.approx(0.5)   # ...but half of the refusals were real customers


def test_never_refusing_looks_accurate_but_catches_nothing():
    scores = classification_scores(confusion_counts([False] * 12, TRUTHS))
    assert scores["accuracy"] == pytest.approx(8 / 12)
    assert scores["recall"] == 0.0
    assert scores["precision"] == 0.0                  # 0 / 0 is reported as 0.0
    assert scores["false_positive_rate"] == 0.0


def test_lists_of_different_length_are_an_error():
    with pytest.raises(ValueError):
        confusion_counts([True, False], [True])
