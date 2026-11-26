"""Tests for the golden set's section labels and the retrieval eval's label check (no OpenAI or Qdrant calls)."""
import glob
import os

import pytest

from src.evaluation import run_retrieval_eval
from src.evaluation.golden_set import GOLDEN_SET
from src.ingest import split_sections

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def sections_in_data():
    """Every '<file> > <SECTION>' label that ingest.py creates from data/."""
    labels = set()
    for path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        filename = os.path.basename(path)
        with open(path) as f:
            _, sections = split_sections(f.read(), default_title=filename)
        labels |= {f"{filename} > {name}" for name, _ in sections}
    return labels


def test_answerable_questions_have_sections_and_refusals_have_none():
    for case in GOLDEN_SET:
        assert bool(case["expected_sections"]) != case["should_refuse"], case["id"]


def test_every_label_matches_a_section_in_data():
    """A renamed heading in data/ fails here, in CI, before any evaluation runs."""
    labels = {label for case in GOLDEN_SET for label in case["expected_sections"]}
    assert labels - sections_in_data() == set()


def test_ids_are_unique():
    ids = [case["id"] for case in GOLDEN_SET]
    assert len(ids) == len(set(ids))


FAKE_STORED = [{"source": "a.txt", "section": "ONE"},
               {"source": "a.txt", "section": "TWO"}, {"source": "a.txt", "section": "TWO"}]


def test_check_labels_stops_when_a_label_matches_no_stored_section(monkeypatch):
    monkeypatch.setattr(run_retrieval_eval, "all_chunks", lambda: FAKE_STORED)
    with pytest.raises(SystemExit, match="a.txt > THREE"):
        run_retrieval_eval.check_labels([{"expected_sections": ["a.txt > ONE", "a.txt > THREE"]}])


def test_check_labels_reports_sections_split_into_several_chunks(monkeypatch, capsys):
    monkeypatch.setattr(run_retrieval_eval, "all_chunks", lambda: FAKE_STORED)
    run_retrieval_eval.check_labels([{"expected_sections": ["a.txt > ONE", "a.txt > TWO"]}])
    assert "split into several chunks: 1 of 2 (a.txt > TWO)" in capsys.readouterr().out
