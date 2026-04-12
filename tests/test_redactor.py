"""Test redactor annotation correctness.

Verifies that every redaction annotation in the output files is internally
consistent: offsets point to [REDACTED] in the text, no overlaps, valid range.

Run often — fast, no model required.
"""
import json
from pathlib import Path

import pytest

MASK = "[REDACTED]"
REDACTED_FILES = [
    Path("data/train_redacted.jsonl"),
    Path("data/test_redacted.jsonl"),
]


def load_records(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("path", REDACTED_FILES)
def test_offsets_point_to_mask(path):
    """text[start:end] must equal '[REDACTED]' for every annotation."""
    for record in load_records(path):
        for r in record["redactions"]:
            actual = record["text"][r["start"]:r["end"]]
            assert actual == MASK, (
                f"[{record['title']}] offset [{r['start']}:{r['end']}] → "
                f"{actual!r}, expected {MASK!r}"
            )


@pytest.mark.parametrize("path", REDACTED_FILES)
def test_offsets_valid_range(path):
    """start < end, and end <= len(text)."""
    for record in load_records(path):
        n = len(record["text"])
        for r in record["redactions"]:
            assert r["start"] < r["end"], (
                f"[{record['title']}] start {r['start']} >= end {r['end']}"
            )
            assert r["end"] <= n, (
                f"[{record['title']}] end {r['end']} > text length {n}"
            )


@pytest.mark.parametrize("path", REDACTED_FILES)
def test_no_overlapping_redactions(path):
    """Redaction spans must not overlap."""
    for record in load_records(path):
        reds = sorted(record["redactions"], key=lambda r: r["start"])
        for i in range(len(reds) - 1):
            assert reds[i]["end"] <= reds[i + 1]["start"], (
                f"[{record['title']}] overlapping redactions: "
                f"{reds[i]} and {reds[i+1]}"
            )


@pytest.mark.parametrize("path", REDACTED_FILES)
def test_original_field_nonempty(path):
    """Every redaction must have a non-empty original field."""
    for record in load_records(path):
        for r in record["redactions"]:
            assert r.get("original"), (
                f"[{record['title']}] redaction missing original: {r}"
            )
