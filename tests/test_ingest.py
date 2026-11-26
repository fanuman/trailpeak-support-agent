"""Tests for the structure-aware chunking in src/ingest.py (no OpenAI or Qdrant calls)."""
import pytest

from src.ingest import chunk_header, is_heading, load_chunks, split_sections

CATALOG = """TRAILPEAK OUTDOORS - PRODUCT CATALOG

===========================================
HYDROFLOW WATER FILTER
===========================================
Category: Water Treatment
SKU: TP-FILT-007

A hollow-fiber membrane water filter.

===========================================
NIGHTWATCH HEADLAMP
===========================================
Category: Lighting
SKU: TP-LAMP-009

A rechargeable LED headlamp.
"""

POLICY = """TRAILPEAK OUTDOORS - SHIPPING POLICY

DOMESTIC SHIPPING (United States)

Standard Shipping: 5-7 business days.

ADDRESS CHANGES

Address changes can only be made before an order ships.
"""


def write_data(tmp_path, monkeypatch, **files):
    """Make a temporary data/ folder with these files and run from its parent, as ingest expects."""
    (tmp_path / "data").mkdir()
    for name, text in files.items():
        (tmp_path / "data" / f"{name}.txt").write_text(text)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("line", ["REFUND TIMING", "DOMESTIC SHIPPING (United States)",
                                  "HYDROFLOW WATER FILTER", "  ALPINEPEAK 2-PERSON TENT  "])
def test_headings_are_recognised(line):
    assert is_heading(line)


@pytest.mark.parametrize("line", ["SKU: TP-FILT-007", "Category: Lighting", "- CLEARANCE ITEMS",
                                  "A rechargeable LED headlamp.", "===========", "", "OK"])
def test_other_lines_are_not_headings(line):
    assert not is_heading(line)


def test_catalog_splits_into_one_section_per_product():
    title, sections = split_sections(CATALOG, default_title="catalog.txt")
    assert title == "TRAILPEAK OUTDOORS - PRODUCT CATALOG"
    assert [name for name, _ in sections] == ["HYDROFLOW WATER FILTER", "NIGHTWATCH HEADLAMP"]
    assert sections[0][1].startswith("Category: Water Treatment")
    assert "headlamp" not in sections[0][1]          # nothing from the next product leaks in
    assert "=" not in sections[0][1]                 # the ===== rules are dropped


def test_text_under_the_title_is_its_own_section():
    _, sections = split_sections("TRAILPEAK OUTDOORS - ABOUT US\n\nWe sell outdoor gear.\n", "about.txt")
    assert sections == [("TRAILPEAK OUTDOORS - ABOUT US", "We sell outdoor gear.")]


def test_a_file_without_headings_is_one_section_named_after_the_file():
    title, sections = split_sections("just some notes\n", "notes.txt")
    assert title == "notes.txt"
    assert sections == [("notes.txt", "just some notes")]


def test_product_header_names_document_section_and_sku():
    title, sections = split_sections(CATALOG, "catalog.txt")
    assert chunk_header(title, *sections[0]) == \
        "TRAILPEAK OUTDOORS - PRODUCT CATALOG | HYDROFLOW WATER FILTER | SKU TP-FILT-007"


def test_policy_header_has_no_sku_and_names_the_document_once():
    title, sections = split_sections(POLICY, "policy.txt")
    assert chunk_header(title, *sections[1]) == "TRAILPEAK OUTDOORS - SHIPPING POLICY | ADDRESS CHANGES"
    about = "TRAILPEAK OUTDOORS - ABOUT US"
    assert chunk_header(about, about, "We sell gear.") == about


def test_every_chunk_starts_with_its_header_and_knows_its_section(tmp_path, monkeypatch):
    write_data(tmp_path, monkeypatch, catalog=CATALOG, policy=POLICY)
    chunks = load_chunks()
    assert [(c["source"], c["section"], c["position"]) for c in chunks] == [
        ("catalog.txt", "HYDROFLOW WATER FILTER", 0), ("catalog.txt", "NIGHTWATCH HEADLAMP", 1),
        ("policy.txt", "DOMESTIC SHIPPING (United States)", 0), ("policy.txt", "ADDRESS CHANGES", 1)]
    assert chunks[1]["text"].startswith(
        "TRAILPEAK OUTDOORS - PRODUCT CATALOG | NIGHTWATCH HEADLAMP | SKU TP-LAMP-009\n\n")
    assert len({c["id"] for c in chunks}) == 4


def test_a_long_section_is_split_inside_itself_with_the_header_on_every_piece(tmp_path, monkeypatch):
    long_body = "\n\n".join(f"Paragraph {i} about the filter. " * 8 for i in range(10))   # about 2,500 characters
    catalog = ("TRAILPEAK OUTDOORS - PRODUCT CATALOG\n\nHYDROFLOW WATER FILTER\nSKU: TP-FILT-007\n\n"
               + long_body + "\n\nNIGHTWATCH HEADLAMP\n\nA headlamp.\n")
    write_data(tmp_path, monkeypatch, catalog=catalog)
    filter_chunks = [c for c in load_chunks() if c["section"] == "HYDROFLOW WATER FILTER"]
    assert len(filter_chunks) >= 2
    for chunk in filter_chunks:
        assert chunk["text"].startswith("TRAILPEAK OUTDOORS - PRODUCT CATALOG | HYDROFLOW WATER FILTER | SKU TP-FILT-007\n\n")
        assert "headlamp" not in chunk["text"].lower()   # overlap never crosses into the next section
