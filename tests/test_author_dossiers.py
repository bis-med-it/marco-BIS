"""Author-dossier renderer (scripts/tools/make_author_dossiers.py).

The dossier is the dictionary's defensibility test — it must show every
field an entry records, in the section its provenance belongs to, and
must not present non-paper target values as the paper's.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ENTRIES = sorted((ROOT / "skills" / "model-dictionary" / "entries").glob("*.yaml"))

_spec = importlib.util.spec_from_file_location(
    "make_author_dossiers", ROOT / "scripts" / "tools" / "make_author_dossiers.py")
dossiers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dossiers)


def _section(text: str, number: int) -> str:
    start = text.index(f"## {number}.")
    nxt = text.find(f"## {number + 1}.", start)
    return text[start:] if nxt == -1 else text[start:nxt]


def _nonempty(v) -> bool:
    return v not in (None, "", [], {})


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_every_entry_field_is_rendered(path):
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    text = dossiers.render(path)
    for key, value in entry.items():
        if key in ("id", "citation") or not _nonempty(value):
            continue
        if key == "implementation_notes":
            for sub, subval in value.items():
                if sub == "replication":
                    assert str(subval) in text, f"{path.stem}: replication path missing"
                elif _nonempty(subval):
                    assert f"{sub}:" in text, (
                        f"{path.stem}: implementation_notes.{sub} missing")
            continue
        assert f"{key}:" in text, f"{path.stem}: top-level field {key!r} missing"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_paper_attributed_fields_land_in_sections_1_and_2(path):
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    text = dossiers.render(path)
    attributed = _section(text, 1) + _section(text, 2)
    for block in ("model", "identification", "sample"):
        for key, value in (entry.get(block) or {}).items():
            if _nonempty(value):
                assert f"{key}:" in attributed, (
                    f"{path.stem}: {block}.{key} not in the attributed sections")


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_targets_are_not_labelled_as_paper_values(path):
    text = dossiers.render(path)
    assert "Paper value" not in text
    assert "benchmarked against the paper" not in text


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_deviations_section_carries_recorded_departures(path):
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    deviations = _section(dossiers.render(path), 4)
    proxies = dossiers._variable_proxies((entry.get("model") or {}).get("variables"))
    for name in proxies:
        assert name in deviations, (
            f"{path.stem}: proxy for {name!r} not listed under deviations")
    assert "data: null" not in deviations
    assert "Replication:" in deviations


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_internal_verification_tier_is_not_shown(path):
    """verified.status is internal metadata for the agent; the dossier shows
    the date and what the entry was checked against, not the status."""
    text = dossiers.render(path)
    assert "status:" not in _section(text, 7)
    assert "checked_against_paper" not in text


def test_citation_accepts_number_for_issue():
    cit = {"authors": "A", "year": 2000, "title": "T", "journal": "J",
           "volume": 1, "number": 2, "pages": "3-4"}
    assert "J, 1, 2, 3-4" in dossiers._fmt_citation(cit)
