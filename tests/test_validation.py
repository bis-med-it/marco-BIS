"""Tests for the paper-target validation framework."""

from __future__ import annotations

from pathlib import Path

import pytest

import lib


SAMPLE_TARGETS = [
    {
        "description": "Core CPI peak response",
        "location": "Figure 2, US column",
        "paper_value": 0.21,
        "paper_units": "percent (10% shock)",
        "our_key": "core_cpi_peak",
        "tolerance": 0.10,
        "tolerance_kind": "absolute",
    },
    {
        "description": "Energy FEV 2y horizon",
        "location": "Footnote 5",
        "paper_value": 0.19,
        "paper_units": "share (0-1)",
        "our_key": "energy_fev_2y",
        "tolerance": 0.30,
        "tolerance_kind": "relative",
    },
    {
        "description": "F-stat for Künzig instrument",
        "location": "Section 2.1",
        "paper_value": 23.1,
        "paper_units": "first-stage F",
        "our_key": "kaenzig_F",
        "tolerance": 30.0,
        "tolerance_kind": "absolute",
    },
]


class TestCompareToPaper:

    def test_all_pass(self):
        our = {"core_cpi_peak": 0.20, "energy_fev_2y": 0.22, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_targets == 3
        assert report.n_pass == 3
        assert report.n_fail == 0
        assert report.all_pass

    def test_one_fail(self):
        # core_cpi_peak deviates by 0.5 > tol 0.10
        our = {"core_cpi_peak": 0.71, "energy_fev_2y": 0.22, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_pass == 2
        assert report.n_fail == 1
        # The failing row is the core_cpi_peak one
        fail_rows = [r for r in report.rows if not r["ok"]]
        assert len(fail_rows) == 1
        assert fail_rows[0]["our_key"] == "core_cpi_peak"

    def test_missing_our_key(self):
        """A target whose our_key isn't in results should fail with a reason."""
        our = {"core_cpi_peak": 0.20}   # missing the other two
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_fail == 2
        for r in report.rows:
            if r["our_key"] != "core_cpi_peak":
                assert r["reason"] == "our_key not in results"
                assert r["our_value"] is None

    def test_relative_tolerance(self):
        """tolerance_kind='relative' uses |diff / paper| <= tol."""
        our = {
            "core_cpi_peak": 0.20,         # absolute tol, passes
            "energy_fev_2y": 0.246,         # |diff/paper| = 0.056/0.19 ≈ 0.295 < 0.30, passes
            "kaenzig_F": 25.0,
        }
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_pass == 3

        # Now nudge above the relative tolerance
        our["energy_fev_2y"] = 0.30        # |diff/paper| = 0.11/0.19 ≈ 0.579 > 0.30
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_pass == 2
        assert not [r for r in report.rows if r["our_key"] == "energy_fev_2y"][0]["ok"]

    def test_table_renders(self):
        our = {"core_cpi_peak": 0.20, "energy_fev_2y": 0.40, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        table = report.table()
        assert "Paper-target comparison" in table
        assert "Core CPI peak response" in table
        assert "Energy FEV 2y horizon" in table
        assert "Summary:" in table

    def test_table_includes_missing_reason(self):
        report = lib.compare_to_paper({}, SAMPLE_TARGETS)
        table = report.table()
        assert "our_key not in results" in table
        assert "(missing)" in table

    def test_to_dict_serializable(self):
        our = {"core_cpi_peak": 0.20, "energy_fev_2y": 0.22, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        d = report.to_dict()
        import json
        # round-trip through JSON to confirm serializability
        s = json.dumps(d, default=float)
        loaded = json.loads(s)
        assert loaded["n_pass"] == 3
        assert loaded["all_pass"] is True

    def test_load_from_yaml_path(self, tmp_path: Path):
        import yaml
        yaml_path = tmp_path / "paper_targets.yaml"
        yaml_path.write_text(yaml.safe_dump({"targets": SAMPLE_TARGETS}))
        our = {"core_cpi_peak": 0.20, "energy_fev_2y": 0.22, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, yaml_path)
        assert report.n_targets == 3
        assert report.all_pass

    def test_invalid_yaml_raises_clearly(self, tmp_path: Path):
        yaml_path = tmp_path / "broken.yaml"
        yaml_path.write_text("not a targets file: true")
        with pytest.raises(ValueError, match="expected top-level mapping"):
            lib.compare_to_paper({}, yaml_path)

    def test_missing_required_fields_raise(self):
        bad_targets = [
            {"description": "x", "location": "y"}   # missing paper_value etc.
        ]
        with pytest.raises(ValueError, match="missing required keys"):
            lib.compare_to_paper({}, bad_targets)

    def test_unknown_tolerance_kind_raises(self):
        our = {"core_cpi_peak": 0.20}
        bad_targets = [
            {**SAMPLE_TARGETS[0], "tolerance_kind": "fancy"},
        ]
        with pytest.raises(ValueError, match="tolerance_kind"):
            lib.compare_to_paper(our, bad_targets)

    def test_nan_our_value_treated_as_missing(self):
        """A NaN result is reported as 'replication did not produce this metric',
        not silently treated as a number that happens to fail the diff check.
        Surfaced during the Litterman 1986 retrofit when a typo in the
        result-extraction code passed NaN through."""
        import math
        our = {"core_cpi_peak": float("nan"), "energy_fev_2y": 0.22, "kaenzig_F": 25.0}
        report = lib.compare_to_paper(our, SAMPLE_TARGETS)
        assert report.n_fail == 1
        nan_row = [r for r in report.rows if r["our_key"] == "core_cpi_peak"][0]
        assert not nan_row["ok"]
        assert "NaN" in nan_row["reason"]
        # diff should be None (not NaN), so the table renders cleanly
        assert nan_row["diff"] is None

    def test_zero_paper_value_relative_tolerance_falls_back_to_absolute(self):
        """When paper_value == 0, relative tolerance applied as absolute."""
        targets = [
            {**SAMPLE_TARGETS[0], "paper_value": 0.0,
             "tolerance": 0.05, "tolerance_kind": "relative"}
        ]
        # diff = our - 0 = 0.04; absolute fallback: |0.04| <= 0.05, passes
        report = lib.compare_to_paper({"core_cpi_peak": 0.04}, targets)
        assert report.all_pass
        # diff = 0.06; fails
        report = lib.compare_to_paper({"core_cpi_peak": 0.06}, targets)
        assert not report.all_pass
