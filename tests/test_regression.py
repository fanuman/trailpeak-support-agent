"""Tests for src/evaluation/regression.py, with a small fake baseline (the real one is not read)."""
import pytest

from src.evaluation.regression import REGRESSION_THRESHOLD, check_regression

BASELINE = {"llm": {"mean": {"faithfulness": 0.977, "answer_relevancy": 0.978}}}


def test_a_drop_inside_the_noise_is_not_a_regression():
    assert check_regression({"faithfulness": 0.955, "answer_relevancy": 0.978}, BASELINE) == []


def test_a_drop_bigger_than_the_threshold_is_reported():
    messages = check_regression({"faithfulness": 0.930, "answer_relevancy": 0.978}, BASELINE)
    assert len(messages) == 1
    assert messages[0].startswith("faithfulness")


def test_every_metric_that_dropped_is_reported():
    messages = check_regression({"faithfulness": 0.90, "answer_relevancy": 0.90}, BASELINE)
    assert [m.split(":")[0] for m in messages] == ["faithfulness", "answer_relevancy"]


def test_a_better_score_is_never_a_regression():
    assert check_regression({"faithfulness": 1.0, "answer_relevancy": 1.0}, BASELINE) == []


def test_a_missing_metric_fails_loudly():
    with pytest.raises(KeyError):
        check_regression({"faithfulness": 0.977}, BASELINE)


def test_the_threshold_is_the_measured_one():
    assert REGRESSION_THRESHOLD == 0.04
