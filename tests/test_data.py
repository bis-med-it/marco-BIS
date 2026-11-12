"""Tests for ``lib.data`` — raw-data loader, per-column transforms, metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.data.transforms import (  # noqa: E402
    FRED_MD_CODE_TO_NAME,
    TRANSFORM_REGISTRY,
)
from lib.data.registry import get_dataset_path, load_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: synthetic project with dataset.csv + transform_codes.csv + registry
# ---------------------------------------------------------------------------

@pytest.fixture
def synth(tmp_path):
    project = tmp_path / "fake_project"
    data_dir = project / "data" / "test_ds"
    data_dir.mkdir(parents=True)

    dates = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    n = len(dates)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "GDP": 100 + np.cumsum(rng.normal(0.5, 1.0, n)),
            "CPI": 100 + np.cumsum(rng.normal(0.2, 0.5, n)),
            "FFR": 5 + rng.normal(0, 1, n),
        },
        index=dates,
    )
    df.index.name = "date"
    df.to_csv(data_dir / "dataset.csv")

    # FRED-MD-style transform_codes.csv: header + metadata row + per-series codes
    (data_dir / "transform_codes.csv").write_text(
        ",0\nsasdate,Transform:\nGDP,5.0\nCPI,5.0\nFFR,1.0\n"
    )

    registry = {
        "datasets": [
            {
                "id": "test_ds",
                "name": "Test Dataset",
                "description": "Synthetic test data",
                "path": "data/test_ds/",
                "frequency": "monthly",
                "vintage": "2099-99",
                "url": "https://example.org",
                "last_updated": "2099-99-99",
                "verified_by": "test fixture",
            }
        ]
    }
    registry_path = project / "data" / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry))

    return {"project": project, "registry_path": registry_path, "df": df}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_load_registry_merges_local_overlay(synth):
    """Cache-on-fetch entries live in registry.local.yaml, merged at load."""
    local = synth["registry_path"].parent / "registry.local.yaml"
    local.write_text(yaml.safe_dump({"datasets": [
        {"id": "local_cache_ds", "path": "data/local_cache_ds/"},
        {"id": "test_ds", "path": "data/SHOULD_NOT_OVERRIDE/"},  # dup id: main wins
    ]}))
    reg = load_registry(synth["registry_path"])
    ids = [e["id"] for e in reg["datasets"]]
    assert "local_cache_ds" in ids
    # the tracked registry's entry is not overridden by the overlay
    main_entry = [e for e in reg["datasets"] if e["id"] == "test_ds"]
    assert len(main_entry) == 1
    assert main_entry[0]["path"] != "data/SHOULD_NOT_OVERRIDE/"


def test_load_registry_returns_entries(synth):
    reg = load_registry(synth["registry_path"])
    assert any(e["id"] == "test_ds" for e in reg["datasets"])


def test_get_dataset_path_resolves_to_csv(synth):
    path = get_dataset_path("test_ds", registry_path=synth["registry_path"])
    assert path.exists()
    assert path.name == "dataset.csv"


def test_get_dataset_path_missing_source_raises(synth):
    with pytest.raises(KeyError, match="not_a_source"):
        get_dataset_path("not_a_source", registry_path=synth["registry_path"])


# ---------------------------------------------------------------------------
# Loader: returns raw data
# ---------------------------------------------------------------------------

def test_load_basic_dict_form(synth):
    df = lib.load_dataset(
        source="test_ds",
        series={"gdp": "GDP", "cpi": "CPI"},
        registry_path=synth["registry_path"],
    )
    assert list(df.columns) == ["gdp", "cpi"]
    assert len(df) == 132
    np.testing.assert_array_almost_equal(df["gdp"].values, synth["df"]["GDP"].values)


def test_load_list_form(synth):
    df = lib.load_dataset(
        source="test_ds",
        series=["GDP", "CPI", "FFR"],
        registry_path=synth["registry_path"],
    )
    assert list(df.columns) == ["GDP", "CPI", "FFR"]


def test_load_returns_raw_unchanged_values(synth):
    """Loader does NOT transform — values match the input CSV exactly."""
    df = lib.load_dataset(
        source="test_ds",
        series={"gdp": "GDP"},
        registry_path=synth["registry_path"],
    )
    np.testing.assert_array_almost_equal(df["gdp"].values, synth["df"]["GDP"].values)


def test_load_no_longer_accepts_transforms_param(synth):
    """The transforms parameter was removed; passing it should raise TypeError."""
    with pytest.raises(TypeError):
        lib.load_dataset(
            source="test_ds",
            series={"gdp": "GDP"},
            transforms={"gdp": "log"},  # type: ignore
            registry_path=synth["registry_path"],
        )


# ---------------------------------------------------------------------------
# Loader: sample filtering
# ---------------------------------------------------------------------------

def test_sample_start_end(synth):
    df = lib.load_dataset(
        source="test_ds",
        series=["GDP"],
        start="2005-01-01",
        end="2005-12-01",
        registry_path=synth["registry_path"],
    )
    assert df.index[0] == pd.Timestamp("2005-01-01")
    assert df.index[-1] == pd.Timestamp("2005-12-01")
    assert len(df) == 12


def test_sample_start_only(synth):
    df = lib.load_dataset(
        source="test_ds",
        series=["GDP"],
        start="2008-01-01",
        registry_path=synth["registry_path"],
    )
    assert df.index[0] == pd.Timestamp("2008-01-01")
    assert df.index[-1] == pd.Timestamp("2010-12-01")


# ---------------------------------------------------------------------------
# Loader: errors
# ---------------------------------------------------------------------------

def test_missing_series_raises(synth):
    with pytest.raises(KeyError, match="DOES_NOT_EXIST"):
        lib.load_dataset(
            source="test_ds",
            series=["DOES_NOT_EXIST"],
            registry_path=synth["registry_path"],
        )


def test_missing_source_raises(synth):
    with pytest.raises(KeyError, match="not_a_source"):
        lib.load_dataset(
            source="not_a_source",
            series=["GDP"],
            registry_path=synth["registry_path"],
        )


def test_missing_csv_file_raises(synth):
    registry = synth["registry_path"]
    reg_data = yaml.safe_load(registry.read_text())
    reg_data["datasets"].append(
        {"id": "no_csv", "path": "data/no_csv/", "frequency": "monthly"}
    )
    registry.write_text(yaml.safe_dump(reg_data))
    (synth["project"] / "data" / "no_csv").mkdir()
    with pytest.raises(FileNotFoundError, match="no_csv"):
        lib.load_dataset(source="no_csv", series=["GDP"], registry_path=registry)


# ---------------------------------------------------------------------------
# Loader: ignore_missing flag (BBE-style large information panels)
# ---------------------------------------------------------------------------

def test_ignore_missing_skips_with_warning(synth):
    """ignore_missing=True skips absent series with a UserWarning."""
    with pytest.warns(UserWarning, match="DOES_NOT_EXIST"):
        df = lib.load_dataset(
            source="test_ds",
            series=["GDP", "DOES_NOT_EXIST", "CPI"],
            ignore_missing=True,
            registry_path=synth["registry_path"],
        )
    assert list(df.columns) == ["GDP", "CPI"]


def test_ignore_missing_records_skipped_in_attrs(synth):
    """df.attrs['missing_series'] lists the codes that were skipped."""
    with pytest.warns(UserWarning):
        df = lib.load_dataset(
            source="test_ds",
            series=["GDP", "DOES_NOT_EXIST"],
            ignore_missing=True,
            registry_path=synth["registry_path"],
        )
    assert df.attrs["missing_series"] == ["DOES_NOT_EXIST"]


def test_ignore_missing_empty_attrs_when_nothing_missing(synth):
    """When all series load, missing_series is an empty list."""
    df = lib.load_dataset(
        source="test_ds", series=["GDP", "CPI"],
        registry_path=synth["registry_path"],
    )
    assert df.attrs["missing_series"] == []


def test_ignore_missing_raises_if_all_missing(synth):
    """If every requested series is missing, raise KeyError even with ignore_missing."""
    with pytest.raises(KeyError, match="None of the requested"):
        lib.load_dataset(
            source="test_ds",
            series=["NOT_HERE_1", "NOT_HERE_2"],
            ignore_missing=True,
            registry_path=synth["registry_path"],
        )


def test_ignore_missing_default_false_still_raises(synth):
    """Without ignore_missing, missing series still raise (regression check)."""
    with pytest.raises(KeyError, match="DOES_NOT_EXIST"):
        lib.load_dataset(
            source="test_ds",
            series=["GDP", "DOES_NOT_EXIST"],
            registry_path=synth["registry_path"],
        )


# ---------------------------------------------------------------------------
# Loader: metadata in df.attrs
# ---------------------------------------------------------------------------

def test_metadata_dataset_level_fields(synth):
    df = lib.load_dataset(
        source="test_ds", series={"gdp": "GDP"},
        registry_path=synth["registry_path"],
    )
    assert df.attrs["source"] == "test_ds"
    assert df.attrs["name"] == "Test Dataset"
    assert df.attrs["frequency"] == "monthly"
    assert df.attrs["vintage"] == "2099-99"
    assert df.attrs["url"] == "https://example.org"
    assert df.attrs["verified_by"] == "test fixture"


def test_metadata_sample_bounds_and_count(synth):
    df = lib.load_dataset(
        source="test_ds", series={"gdp": "GDP"},
        start="2005-01-01", end="2005-12-01",
        registry_path=synth["registry_path"],
    )
    assert df.attrs["sample_start_requested"] == "2005-01-01"
    assert df.attrs["sample_end_requested"] == "2005-12-01"
    assert df.attrs["sample_start_actual"] == "2005-01-01"
    assert df.attrs["sample_end_actual"] == "2005-12-01"
    assert df.attrs["n_obs"] == 12


def test_metadata_loaded_at_is_iso_timestamp(synth):
    df = lib.load_dataset(
        source="test_ds", series={"gdp": "GDP"},
        registry_path=synth["registry_path"],
    )
    assert "loaded_at" in df.attrs
    pd.Timestamp(df.attrs["loaded_at"])  # raises if not parseable


def test_metadata_per_series_includes_source_code(synth):
    df = lib.load_dataset(
        source="test_ds", series={"my_gdp": "GDP", "my_cpi": "CPI"},
        registry_path=synth["registry_path"],
    )
    assert df.attrs["series"]["my_gdp"]["source_code"] == "GDP"
    assert df.attrs["series"]["my_cpi"]["source_code"] == "CPI"


def test_metadata_per_series_includes_recommended_transform(synth):
    """When transform_codes.csv is present, per-series metadata carries the FRED-MD recommendation."""
    df = lib.load_dataset(
        source="test_ds", series={"gdp": "GDP", "ffr": "FFR"},
        registry_path=synth["registry_path"],
    )
    # GDP code 5 -> log_diff; FFR code 1 -> level
    assert df.attrs["series"]["gdp"]["recommended_transform"] == "log_diff"
    assert df.attrs["series"]["ffr"]["recommended_transform"] == "level"


def test_metadata_vintage_param_overrides_registry(synth):
    df = lib.load_dataset(
        source="test_ds", series=["GDP"],
        vintage="custom-vintage",
        registry_path=synth["registry_path"],
    )
    assert df.attrs["vintage"] == "custom-vintage"


# ---------------------------------------------------------------------------
# Transform function: applies per-column transforms
# ---------------------------------------------------------------------------

def _raw(synth, cols):
    return lib.load_dataset(
        source="test_ds", series=cols, registry_path=synth["registry_path"],
    )


def test_transform_log(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "log"})
    np.testing.assert_array_almost_equal(
        df["gdp"].values, np.log(synth["df"]["GDP"].values)
    )


def test_transform_diff(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "diff"})
    expected = synth["df"]["GDP"].diff().dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_log_diff(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "log_diff"})
    expected = np.log(synth["df"]["GDP"]).diff().dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_pct_change(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "pct_change"})
    expected = synth["df"]["GDP"].pct_change().dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_aliases_equivalent(synth):
    raw = _raw(synth, {"gdp": "GDP"})
    df1 = lib.transform(raw, {"gdp": "log_diff"})
    df2 = lib.transform(raw, {"gdp": "dlog"})
    np.testing.assert_array_equal(df1.values, df2.values)


def test_transform_string_applies_to_all(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP", "cpi": "CPI"}), "log")
    np.testing.assert_array_almost_equal(
        df["gdp"].values, np.log(synth["df"]["GDP"].values)
    )
    np.testing.assert_array_almost_equal(
        df["cpi"].values, np.log(synth["df"]["CPI"].values)
    )


def test_transform_unspecified_columns_default_to_level(synth):
    raw = _raw(synth, {"gdp": "GDP", "cpi": "CPI"})
    df = lib.transform(raw, {"gdp": "log"})  # cpi unspecified
    np.testing.assert_array_almost_equal(df["cpi"].values, synth["df"]["CPI"].values)


def test_transform_unknown_raises(synth):
    raw = _raw(synth, {"gdp": "GDP"})
    with pytest.raises(ValueError, match="bogus"):
        lib.transform(raw, {"gdp": "bogus"})


def test_transform_dropna_true(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "diff"}, dropna=True)
    assert not df["gdp"].isna().any()


def test_transform_dropna_false_keeps_leading_nan(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "diff"}, dropna=False)
    assert pd.isna(df["gdp"].iloc[0])


# Scaled variants
def test_transform_log_x_100(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "log_x_100"})
    expected = np.log(synth["df"]["GDP"]) * 100.0
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_dlog_x_100(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "dlog_x_100"})
    expected = (np.log(synth["df"]["GDP"]).diff() * 100.0).dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_dlog_x_400(synth):
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "dlog_x_400"})
    expected = (np.log(synth["df"]["GDP"]).diff() * 400.0).dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_scaled_aliases_equivalent(synth):
    raw = _raw(synth, {"gdp": "GDP"})
    df1 = lib.transform(raw, "log_diff_pct")
    df2 = lib.transform(raw, "dlog_x_100")
    df3 = lib.transform(raw, "dlog_pct")
    np.testing.assert_array_equal(df1.values, df2.values)
    np.testing.assert_array_equal(df1.values, df3.values)


def test_transform_abs_diff(synth):
    """abs_diff = |x.diff()| — used as volatility proxy (Diebold-Yilmaz)."""
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "abs_diff"})
    expected = synth["df"]["GDP"].diff().abs().dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


def test_transform_abs_log_diff(synth):
    """abs_log_diff = |log(x).diff()| — log-return volatility proxy."""
    df = lib.transform(_raw(synth, {"gdp": "GDP"}), {"gdp": "abs_log_diff"})
    expected = np.log(synth["df"]["GDP"]).diff().abs().dropna()
    np.testing.assert_array_almost_equal(df["gdp"].values, expected.values)


# ---------------------------------------------------------------------------
# Transform: metadata propagation
# ---------------------------------------------------------------------------

def test_transform_preserves_load_metadata(synth):
    raw = _raw(synth, {"gdp": "GDP"})
    df = lib.transform(raw, {"gdp": "log"})
    assert df.attrs["source"] == "test_ds"
    assert df.attrs["frequency"] == "monthly"
    assert df.attrs["series"]["gdp"]["source_code"] == "GDP"


def test_transform_records_applied_map(synth):
    raw = _raw(synth, {"gdp": "GDP", "cpi": "CPI"})
    df = lib.transform(raw, {"gdp": "log"})
    # Resolved map: gdp -> log, cpi -> level (default)
    assert df.attrs["transforms_applied"] == {"gdp": "log", "cpi": "level"}


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

def test_transform_registry_includes_fred_md_codes():
    assert "level" in TRANSFORM_REGISTRY        # 1
    assert "diff" in TRANSFORM_REGISTRY         # 2
    assert "log" in TRANSFORM_REGISTRY          # 4
    assert "log_diff" in TRANSFORM_REGISTRY     # 5
    assert "diff_log_diff" in TRANSFORM_REGISTRY  # 6
    assert "pct_change" in TRANSFORM_REGISTRY   # 7


def test_fred_md_code_to_name_mapping_valid():
    """Every FRED-MD code must map to a name that's in the registry."""
    for code, name in FRED_MD_CODE_TO_NAME.items():
        assert name in TRANSFORM_REGISTRY, f"code {code} -> {name!r} not in registry"


def test_load_rejects_duplicate_dates_in_cache(synth):
    """Read-side calendar guard: a cached dataset.csv with a duplicated
    date must be refused instead of silently duplicating rows in joins."""
    csv_path = synth["project"] / "data" / "test_ds" / "dataset.csv"
    df = synth["df"]
    dup = pd.concat([df, df.iloc[[5]]])
    dup.index.name = "date"
    dup.to_csv(csv_path)
    with pytest.raises(ValueError, match="DUPLICATE dates"):
        lib.load_dataset(
            source="test_ds", series={"gdp": "GDP"},
            registry_path=synth["registry_path"],
        )


def _set_frequency(synth, frequency):
    reg = yaml.safe_load(synth["registry_path"].read_text())
    reg["datasets"][0]["frequency"] = frequency
    synth["registry_path"].write_text(yaml.safe_dump(reg))


def test_load_rejects_absent_month(synth):
    """A month with no row at all is invisible to the NaN-based gap
    guard; without this check a two-month change becomes one 'growth'
    observation and lags treat Jan and Mar as adjacent."""
    csv_path = synth["project"] / "data" / "test_ds" / "dataset.csv"
    df = synth["df"].drop(pd.Timestamp("2005-02-01"))
    df.to_csv(csv_path)
    with pytest.raises(ValueError, match="absent from the monthly calendar"):
        lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                         registry_path=synth["registry_path"])
    # dropna=False is the caller-handles-gaps route: warn, do not raise.
    with pytest.warns(UserWarning, match="absent from the monthly calendar"):
        out = lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                               registry_path=synth["registry_path"],
                               dropna=False)
    assert len(out) == len(df)
    # A sample that excludes the gap loads cleanly.
    out = lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                           start="2006-01",
                           registry_path=synth["registry_path"])
    assert out.index[0] == pd.Timestamp("2006-01-01")


def test_load_quarterly_calendar_with_quarter_end_labels(synth):
    """FRED-QD labels quarters by the first day of the quarter-ending
    month (Mar/Jun/Sep/Dec). A complete grid must pass; an absent
    interior quarter must be rejected."""
    _set_frequency(synth, "quarterly")
    csv_path = synth["project"] / "data" / "test_ds" / "dataset.csv"
    dates = pd.date_range("2000-03-01", "2010-12-01", freq="3MS")
    q = pd.DataFrame({"GDP": np.arange(len(dates), dtype=float) + 100},
                     index=pd.Index(dates, name="date"))
    q.to_csv(csv_path)
    out = lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                           registry_path=synth["registry_path"])
    assert len(out) == len(q)
    q.drop(pd.Timestamp("2004-06-01")).to_csv(csv_path)
    with pytest.raises(ValueError, match="first: 2004Q2"):
        lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                         registry_path=synth["registry_path"])


def test_load_skips_calendar_check_for_mixed_frequency(synth):
    """Cache-on-fetch datasets are registered as 'mixed'; their calendar
    is not assumed regular, so an absent month is not an error there."""
    _set_frequency(synth, "mixed")
    csv_path = synth["project"] / "data" / "test_ds" / "dataset.csv"
    synth["df"].drop(pd.Timestamp("2005-02-01")).to_csv(csv_path)
    out = lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                           registry_path=synth["registry_path"])
    assert len(out) == len(synth["df"]) - 1


@pytest.mark.parametrize("start", [None, "2000-01"])
def test_load_rejects_unsplit_fred_csv_with_pointer(synth, start):
    """A raw FRED current.csv saved straight as dataset.csv keeps its
    'Transform:' row; the loader must say so instead of failing with a
    pandas TypeError/AttributeError."""
    csv_path = synth["project"] / "data" / "test_ds" / "dataset.csv"
    csv_path.write_text("sasdate,GDP\nTransform:,5\n1/1/2000,100\n2/1/2000,110\n")
    with pytest.raises(ValueError, match="UPDATE.md"):
        lib.load_dataset(source="test_ds", series={"gdp": "GDP"},
                         start=start, registry_path=synth["registry_path"])
