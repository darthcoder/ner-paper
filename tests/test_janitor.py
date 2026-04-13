"""Test janitor output cleanliness.

Verifies that wikitext markup has been stripped from the clean corpus.
Run once per setup / after re-running janitor.py.
"""
import json
import re
from pathlib import Path

import pytest

CLEAN_FILE = Path("data/all_clean.jsonl")

WIKITEXT_PATTERNS = [
    (r"\{\{", "template opening {{"),
    (r"\[\[", "wikilink opening [["),
    (r"==.+==", "section heading ==...=="),
    (r"<ref", "reference tag <ref"),
    (r"\[\[File:", "file embed [[File:"),
    (r"\[\[Image:", "image embed [[Image:"),
]


@pytest.fixture(scope="module")
def clean_records():
    with CLEAN_FILE.open() as f:
        return [json.loads(line) for line in f]


def test_all_records_have_text(clean_records):
    """Every article must have a non-empty text field."""
    for record in clean_records:
        assert record.get("text"), f"[{record['title']}] has empty text"


def test_all_records_have_required_fields(clean_records):
    """Every record must have pageid, title, and text."""
    for record in clean_records:
        for field in ("pageid", "title", "text"):
            assert field in record, f"Missing field {field!r} in {record}"


@pytest.mark.parametrize("pattern,description", WIKITEXT_PATTERNS)
def test_no_wikitext_markup(clean_records, pattern, description):
    """Clean corpus must not contain wikitext markup."""
    violations = [
        record["title"]
        for record in clean_records
        if re.search(pattern, record["text"])
    ]
    assert not violations, (
        f"Found {description} in {len(violations)} article(s): "
        f"{violations[:5]}"
    )


def test_minimum_article_length(clean_records):
    """Every article should have at least 100 characters of text."""
    short = [r["title"] for r in clean_records if len(r["text"]) < 100]
    assert not short, f"Articles with < 100 chars: {short}"
