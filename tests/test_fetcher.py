"""Tests for ``lib.data.fetcher`` — cache-on-fetch infrastructure.

Mocks the HTTP layer of source-specific fetchers so tests are offline,
deterministic, and fast. Integration with real APIs is validated manually
when needed.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.data.fetcher import (  # noqa: E402
    BaseFetcher,
    fetch_data,
    get_fetcher,
    register_fetcher,
)


# ---------------------------------------------------------------------------
# Fixture: minimal project tree with an empty registry
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_project(tmp_path):
    project = tmp_path / "fake_project"
    (project / "data").mkdir(parents=True)
    registry_path = project / "data" / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"datasets": []}))
    return {"project": project, "registry_path": registry_path}


# ---------------------------------------------------------------------------
# A synthetic fetcher used to test the orchestration layer in isolation
# ---------------------------------------------------------------------------

class _SyntheticFetcher(BaseFetcher):
    """Returns canned monthly series for any requested alias."""

    name = "synthetic"
    description = "Test fixture fetcher"
    endpoint = "test://"

    SERIES: dict[str, pd.Series] = {}  # populated per-test

    def fetch(self, series, start=None, end=None):
        return {alias: self.SERIES[alias].copy() for alias in series if alias in self.SERIES}


@pytest.fixture
def synthetic_data():
    """Pre-built series the synthetic fetcher will return."""
    idx = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    n = len(idx)
    return {
        "a": pd.Series(np.arange(n, dtype=float), index=idx, name="a"),
        "b": pd.Series(np.arange(n, dtype=float) * 2, index=idx, name="b"),
    }


@pytest.fixture(autouse=True)
def _register_synthetic_fetcher():
    register_fetcher("synthetic", _SyntheticFetcher)
    yield
    # Clean up after each test
    from lib.data.fetcher import _FETCHER_REGISTRY
    _FETCHER_REGISTRY.pop("synthetic", None)


# ---------------------------------------------------------------------------
# Fetcher registry
# ---------------------------------------------------------------------------

def test_get_fetcher_returns_registered_instance():
    f = get_fetcher("synthetic")
    assert isinstance(f, _SyntheticFetcher)


def test_get_fetcher_unknown_raises():
    with pytest.raises(KeyError, match="not_a_real_api"):
        get_fetcher("not_a_real_api")


def test_builtin_fetchers_registered():
    """Built-in fetchers register themselves on import of lib.data.fetchers."""
    assert isinstance(get_fetcher("fred"), BaseFetcher)
    assert isinstance(get_fetcher("dbnomics"), BaseFetcher)
    assert isinstance(get_fetcher("ons"), BaseFetcher)
    assert isinstance(get_fetcher("boe_iadb"), BaseFetcher)
    assert isinstance(get_fetcher("csv_file"), BaseFetcher)
    assert isinstance(get_fetcher("external_archive"), BaseFetcher)


# ---------------------------------------------------------------------------
# fetch_data orchestration (cache write, metadata, registry)
# ---------------------------------------------------------------------------

def test_fetch_data_writes_cache_csv(empty_project, synthetic_data):
    _SyntheticFetcher.SERIES = synthetic_data
    df = fetch_data(
        api="synthetic",
        series={"a": "spec_a", "b": "spec_b"},
        dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )
    csv_path = empty_project["project"] / "data" / "testset" / "dataset.csv"
    assert csv_path.exists()
    cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    assert list(cached.columns) == ["a", "b"]
    assert len(cached) == len(synthetic_data["a"])
    # And the returned DataFrame has both columns
    assert set(df.columns) == {"a", "b"}


def test_fetch_data_writes_metadata_yaml(empty_project, synthetic_data):
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(
        api="synthetic",
        series={"a": "spec_a"},
        dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )
    meta_path = empty_project["project"] / "data" / "testset" / "metadata.yaml"
    assert meta_path.exists()
    meta = yaml.safe_load(meta_path.read_text())
    assert "variables" in meta
    entries = {v["series_id"]: v for v in meta["variables"]}
    assert "a" in entries
    assert entries["a"]["api"] == "synthetic"
    assert entries["a"]["source_spec"] == "spec_a"
    assert "fetched_at" in entries["a"]


def test_fetch_data_auto_registers_dataset(empty_project, synthetic_data):
    _SyntheticFetcher.SERIES = synthetic_data
    main_before = empty_project["registry_path"].read_text()
    fetch_data(
        api="synthetic",
        series={"a": "spec_a"},
        dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )
    # cache-on-fetch entries land in the gitignored overlay, NOT the
    # tracked registry (which must not churn on local fetches)
    local_path = empty_project["registry_path"].parent / "registry.local.yaml"
    assert local_path.exists()
    local = yaml.safe_load(local_path.read_text())
    assert "testset" in [e["id"] for e in local["datasets"]]
    assert empty_project["registry_path"].read_text() == main_before
    # and load_registry presents the merged view
    from lib.data.registry import load_registry
    merged = load_registry(empty_project["registry_path"])
    assert "testset" in [e["id"] for e in merged["datasets"]]


def test_fetch_data_cache_first_skips_api_on_second_call(empty_project, synthetic_data):
    """Second call to fetch_data shouldn't invoke the fetcher (cache hit)."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(
        api="synthetic",
        series={"a": "spec_a"},
        dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )

    # Patch the synthetic fetcher so any second .fetch() call would error
    with patch.object(_SyntheticFetcher, "fetch") as mocked:
        fetch_data(
            api="synthetic",
            series={"a": "spec_a"},
            dataset_id="testset",
            registry_path=empty_project["registry_path"],
        )
        mocked.assert_not_called()


def test_fetch_data_force_refresh_re_invokes_fetcher(empty_project, synthetic_data):
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(
        api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )
    with patch.object(_SyntheticFetcher, "fetch",
                       return_value=synthetic_data) as mocked:
        fetch_data(
            api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
            refresh="force",
            registry_path=empty_project["registry_path"],
        )
        mocked.assert_called_once()


def test_fetch_data_incremental_fetches_new_aliases_only(empty_project):
    """If a is cached but b is not, only b should be fetched, and b's full
    history survives the merge even where it lies outside a's calendar.

    The calendars differ on purpose: b starts two years before a and ends
    two years after it. Merging onto the cached index alone would silently
    truncate b to a's dates (the old cache-history loss), so the returned
    frame, the CSV on disk and a cache-only reload must all carry the
    union calendar, b's complete values and a's untouched values.
    """
    idx_a = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    idx_b = pd.date_range("1998-01-01", "2012-12-01", freq="MS")
    data = {
        "a": pd.Series(np.arange(len(idx_a), dtype=float), index=idx_a,
                       name="a"),
        "b": pd.Series(1000.0 + 2 * np.arange(len(idx_b), dtype=float),
                       index=idx_b, name="b"),
    }
    _SyntheticFetcher.SERIES = data
    # First: fetch a only
    fetch_data(
        api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
        registry_path=empty_project["registry_path"],
    )
    # Second: request a + b — only b should hit the fetcher
    captured = {}

    def capture(self, series, start=None, end=None):
        captured.update(series)
        return {alias: data[alias].copy() for alias in series}

    with patch.object(_SyntheticFetcher, "fetch", new=capture):
        df = fetch_data(
            api="synthetic", series={"a": "spec_a", "b": "spec_b"},
            dataset_id="testset", dropna=False,
            registry_path=empty_project["registry_path"],
        )
    assert set(captured) == {"b"}

    def assert_complete(frame):
        assert frame.index.equals(idx_b), (
            f"calendar {frame.index[0]}..{frame.index[-1]} ({len(frame)} rows); "
            f"expected {idx_b[0]}..{idx_b[-1]} ({len(idx_b)} rows)"
        )
        np.testing.assert_array_equal(frame["b"].to_numpy(),
                                      data["b"].to_numpy())
        a = frame["a"]
        np.testing.assert_array_equal(a.loc[idx_a].to_numpy(),
                                      data["a"].to_numpy())
        assert a.drop(idx_a).isna().all()

    # On return
    assert_complete(df)
    # On disk
    csv_path = empty_project["project"] / "data" / "testset" / "dataset.csv"
    assert_complete(pd.read_csv(csv_path, index_col=0, parse_dates=True))
    # After a cache-only reload
    with patch.object(_SyntheticFetcher, "fetch") as mocked:
        df_reload = fetch_data(
            api="synthetic", series={"a": "spec_a", "b": "spec_b"},
            dataset_id="testset", dropna=False,
            registry_path=empty_project["registry_path"],
        )
        b_only = fetch_data(
            api="synthetic", series={"b": "spec_b"}, dataset_id="testset",
            registry_path=empty_project["registry_path"],
        )
        mocked.assert_not_called()
    assert_complete(df_reload)
    assert b_only.index.equals(idx_b)
    np.testing.assert_array_equal(b_only["b"].to_numpy(), data["b"].to_numpy())


# ---------------------------------------------------------------------------
# csv_file fetcher
# ---------------------------------------------------------------------------

def test_csv_file_fetcher_loads_column(empty_project, tmp_path):
    csv = tmp_path / "stage.csv"
    idx = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    pd.DataFrame({"K222": np.arange(len(idx), dtype=float)},
                 index=idx).to_csv(csv, index_label="date")

    df = fetch_data(
        api="csv_file",
        series={"indpro": f"{csv}:K222"},
        dataset_id="csv_test",
        registry_path=empty_project["registry_path"],
    )
    assert "indpro" in df.columns
    assert len(df) == len(idx)


def test_csv_file_fetcher_bad_spec_raises(empty_project):
    with pytest.raises(ValueError, match="path.*column"):
        fetch_data(
            api="csv_file",
            series={"indpro": "no_colon"},
            dataset_id="csv_test",
            registry_path=empty_project["registry_path"],
        )


# ---------------------------------------------------------------------------
# ONS fetcher with mocked HTTP
# ---------------------------------------------------------------------------

_ONS_SAMPLE_JSON = {
    "months": [
        {"date": "2020 JAN", "value": "100.0"},
        {"date": "2020 FEB", "value": "101.0"},
        {"date": "2020 MAR", "value": "98.0"},
    ],
    "quarters": [],
    "years": [],
}


def _mock_urlopen_json(payload):
    """Build a context-manager that yields a urlopen-like object with .read()."""
    class _MockResponse:
        def __init__(self, data):
            self._data = data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self._data

    return _MockResponse(json.dumps(payload).encode("utf-8"))


def test_ons_fetcher_parses_monthly_json(empty_project):
    from lib.data.fetchers.ons import OnsFetcher
    fetcher = OnsFetcher()
    with patch("lib.data.fetchers.ons.request.urlopen",
                return_value=_mock_urlopen_json(_ONS_SAMPLE_JSON)):
        result = fetcher.fetch({"indpro": ("K222", "mm23")})
    assert "indpro" in result
    ser = result["indpro"]
    assert len(ser) == 3
    assert ser.iloc[0] == 100.0
    assert ser.index[0] == pd.Timestamp("2020-01-01")
    assert ser.index[-1] == pd.Timestamp("2020-03-01")


def test_ons_fetcher_handles_quarterly(empty_project):
    from lib.data.fetchers.ons import OnsFetcher
    payload = {
        "months": [],
        "quarters": [
            {"date": "2020 Q1", "value": "100"},
            {"date": "2020 Q2", "value": "95"},
        ],
        "years": [],
    }
    fetcher = OnsFetcher()
    with patch("lib.data.fetchers.ons.request.urlopen",
                return_value=_mock_urlopen_json(payload)):
        result = fetcher.fetch({"gdp": ("IHYR", "qna")})
    ser = result["gdp"]
    assert len(ser) == 2
    assert ser.index[0] == pd.Timestamp("2020-01-01")
    assert ser.index[1] == pd.Timestamp("2020-04-01")  # Q2 -> April


def test_ons_fetcher_rejects_bad_spec():
    from lib.data.fetchers.ons import OnsFetcher
    fetcher = OnsFetcher()
    with pytest.raises(ValueError, match="(code, dataset)"):
        fetcher.fetch({"x": "just_a_code"})


def test_ons_fetcher_http_error_warning_does_not_crash(empty_project):
    """On HTTPError the fetcher must warn cleanly and skip the failed
    alias instead of aborting. The warning formatter may only reference
    names in scope under the URL-path spec form — a stale `code` /
    `dataset` reference there would turn the HTTPError into a
    NameError."""
    from urllib.error import HTTPError
    from lib.data.fetchers.ons import OnsFetcher
    fetcher = OnsFetcher()

    def _raise_http(*args, **kwargs):
        raise HTTPError("https://x", 500, "Server Error", {}, None)

    with patch("lib.data.fetchers.ons.request.urlopen", side_effect=_raise_http):
        with pytest.warns(UserWarning, match="ONS fetch failed"):
            result = fetcher.fetch(
                {"indpro": "economy/economicoutputandproductivity/output/timeseries/k222/diop"},
            )
    # Failed alias is skipped; result is empty rather than raising NameError.
    assert result == {}


# ---------------------------------------------------------------------------
# BoE iadb fetcher with mocked HTTP
# ---------------------------------------------------------------------------

_BOE_SAMPLE_CSV = (
    "DATE,IUMABEDR\n"
    "01/Jan/2020,0.75\n"
    "01/Feb/2020,0.75\n"
    "01/Mar/2020,0.25\n"
)


def _mock_urlopen_text(text):
    class _MockResponse:
        def __init__(self, data):
            self._data = data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self._data
    return _MockResponse(text.encode("utf-8"))


def test_boe_iadb_fetcher_parses_csv():
    from lib.data.fetchers.boe_iadb import BoeIadbFetcher
    fetcher = BoeIadbFetcher()
    with patch("lib.data.fetchers.boe_iadb.request.urlopen",
                return_value=_mock_urlopen_text(_BOE_SAMPLE_CSV)):
        result = fetcher.fetch({"rate": "IUMABEDR"})
    ser = result["rate"]
    assert len(ser) == 3
    assert ser.iloc[0] == 0.75
    assert ser.iloc[-1] == 0.25
    assert ser.index[0] == pd.Timestamp("2020-01-01")
    assert ser.index[-1] == pd.Timestamp("2020-03-01")


def _boe_fetch_text(text):
    from lib.data.fetchers.boe_iadb import BoeIadbFetcher
    fetcher = BoeIadbFetcher()
    with patch.object(fetcher, "_http_get_text", return_value=text):
        return fetcher.fetch({"x": "IUMLPQ"})["x"]


def test_boe_iadb_two_quarterly_observations_not_relabelled_monthly():
    """Two quarter-end observations must not become two month starts."""
    with pytest.raises(ValueError, match="not monthly"):
        _boe_fetch_text("DATE,IUMLPQ\n31/Mar/2020,1\n30/Jun/2020,2\n")


def test_boe_iadb_single_observation_refused():
    with pytest.raises(ValueError, match="single observation"):
        _boe_fetch_text("DATE,IUMLPQ\n31/Mar/2020,1\n")


def test_boe_iadb_two_monthly_observations_normalised():
    ser = _boe_fetch_text("DATE,IUMLPQ\n31/Jan/2020,1\n29/Feb/2020,2\n")
    assert list(ser.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    assert list(ser.values) == [1.0, 2.0]


def test_boe_iadb_fetcher_rejects_non_string_spec():
    from lib.data.fetchers.boe_iadb import BoeIadbFetcher
    fetcher = BoeIadbFetcher()
    with pytest.raises(ValueError, match="bare string"):
        fetcher.fetch({"x": ("a", "b")})


# ---------------------------------------------------------------------------
# external_archive fetcher with mocked HTTP
# ---------------------------------------------------------------------------

# A small synthetic JK-style CSV: a date column and three shock columns.
_JK_ARCHIVE_CSV = (
    b"date,MP1,FF4,ED4\n"
    b"2010-01-01,0.05,0.02,0.01\n"
    b"2010-02-01,-0.03,-0.01,0.00\n"
    b"2010-03-01,0.10,0.07,0.05\n"
    b"2010-04-01,0.00,0.00,0.00\n"
)


def _mock_urlopen_bytes(payload: bytes):
    class _MockResponse:
        def __init__(self, data):
            self._data = data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self._data
    return _MockResponse(payload)


def test_external_archive_string_spec_loads_column():
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)):
        result = fetcher.fetch({"ff4": "https://example.com/jk.csv:FF4"})
    ser = result["ff4"]
    assert len(ser) == 4
    assert ser.iloc[0] == 0.02
    assert ser.iloc[2] == 0.07
    assert ser.index[0] == pd.Timestamp("2010-01-01")
    assert ser.name == "ff4"


def test_external_archive_dict_spec_loads_column():
    """Dict spec with explicit url + column."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    spec = {"url": "https://example.com/jk.csv", "column": "MP1"}
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)):
        result = fetcher.fetch({"mp1": spec})
    ser = result["mp1"]
    assert len(ser) == 4
    assert ser.iloc[0] == 0.05
    assert ser.iloc[1] == -0.03


def test_external_archive_dict_spec_with_skiprows():
    """skiprows passes through to pd.read_csv so banner rows are dropped."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv_with_banner = (
        b"# Author: Some Researcher\n"
        b"# Description: shock series\n"
        b"date,MP1\n"
        b"2010-01-01,0.05\n"
        b"2010-02-01,-0.03\n"
    )
    spec = {
        "url": "https://example.com/with_banner.csv",
        "column": "MP1",
        "skiprows": 2,
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_with_banner)):
        result = fetcher.fetch({"mp1": spec})
    assert len(result["mp1"]) == 2
    assert result["mp1"].iloc[0] == 0.05


def test_external_archive_dict_spec_with_custom_date_column():
    """date_column lets the user pick a non-first date column."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv_data = (
        b"id,observation_date,value\n"
        b"1,2010-01-01,100.0\n"
        b"2,2010-02-01,101.5\n"
        b"3,2010-03-01,99.7\n"
    )
    spec = {
        "url": "https://example.com/odd.csv",
        "column": "value",
        "date_column": "observation_date",
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_data)):
        result = fetcher.fetch({"v": spec})
    ser = result["v"]
    assert len(ser) == 3
    assert ser.index[0] == pd.Timestamp("2010-01-01")
    assert ser.iloc[1] == 101.5


def test_external_archive_dict_spec_with_date_columns():
    """date_columns (list) lets the user combine split year+month columns."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    # JK-style CSV: year and month as separate columns, no single date.
    csv_data = (
        b"year,month,MP_median,CBI_median\n"
        b"1990,2,0.00874,-0.00922\n"
        b"1990,3,-0.00297,-0.02378\n"
        b"1990,12,0.10000,0.20000\n"
        b"1991,1,0.30000,0.40000\n"
    )
    spec = {
        "url": "https://example.com/jk.csv",
        "column": "MP_median",
        "date_columns": ["year", "month"],
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_data)):
        result = fetcher.fetch({"mp": spec})
    ser = result["mp"]
    assert len(ser) == 4
    assert ser.index[0] == pd.Timestamp("1990-02-01")
    assert ser.index[2] == pd.Timestamp("1990-12-01")   # zero-pad worked
    assert ser.index[3] == pd.Timestamp("1991-01-01")   # year rollover
    assert ser.iloc[0] == 0.00874


def test_external_archive_date_columns_missing_raises():
    """A date_columns entry not in the CSV raises KeyError-as-warning."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv_data = b"year,value\n2020,1.0\n2021,2.0\n"
    spec = {
        "url": "https://example.com/yo.csv",
        "column": "value",
        "date_columns": ["year", "month"],  # 'month' is missing
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_data)):
        with pytest.warns(UserWarning, match="failed to parse archive"):
            result = fetcher.fetch({"v": spec})
    assert result == {}


def test_external_archive_dict_spec_with_date_format():
    """date_format lets the user override pandas auto-inference."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv_data = (
        b"date,value\n"
        b"01/2010,1.0\n"
        b"02/2010,2.0\n"
        b"03/2010,3.0\n"
    )
    spec = {
        "url": "https://example.com/mm.csv",
        "column": "value",
        "date_format": "%m/%Y",
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_data)):
        result = fetcher.fetch({"v": spec})
    assert result["v"].index[0] == pd.Timestamp("2010-01-01")
    assert result["v"].index[-1] == pd.Timestamp("2010-03-01")


def test_external_archive_caches_within_call():
    """Two aliases from the same URL trigger only one HTTP call."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)) as mocked:
        result = fetcher.fetch({
            "ff4": "https://example.com/jk.csv:FF4",
            "mp1": "https://example.com/jk.csv:MP1",
        })
    assert mocked.call_count == 1
    assert set(result) == {"ff4", "mp1"}


def test_external_archive_url_with_scheme_colon_parses_via_rpartition():
    """String spec splits on the last ':' so 'https://...:col' works."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    cfg = fetcher._parse_spec("https://example.com/path/data.csv:FF4")
    assert cfg["url"] == "https://example.com/path/data.csv"
    assert cfg["column"] == "FF4"


def test_external_archive_rejects_bad_string_spec():
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with pytest.raises(ValueError, match="<url>:<column>"):
        fetcher.fetch({"x": "no_colon_anywhere"})


def test_external_archive_rejects_dict_missing_url():
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with pytest.raises(ValueError, match="'url' and 'column'"):
        fetcher.fetch({"x": {"column": "FF4"}})


def test_external_archive_warns_and_skips_missing_column():
    """A wrong column name doesn't kill the whole fetch — it warns and skips."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)):
        with pytest.warns(UserWarning, match="column 'NOT_THERE' not in archive"):
            result = fetcher.fetch({"missing": "https://example.com/jk.csv:NOT_THERE"})
    assert result == {}


def test_external_archive_warns_on_http_error():
    """HTTP errors are warned, the alias is dropped, others continue."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    from urllib.error import HTTPError

    fetcher = ExternalArchiveFetcher()

    def fake_urlopen(req, timeout=60):
        raise HTTPError(req.full_url, 404, "Not Found", {}, None)

    with patch("lib.data.fetchers.external_archive.request.urlopen",
                side_effect=fake_urlopen):
        with pytest.warns(UserWarning, match="external_archive fetch failed"):
            result = fetcher.fetch({"ff4": "https://example.com/nothing.csv:FF4"})
    assert result == {}


def test_external_archive_filters_by_start_end():
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)):
        result = fetcher.fetch(
            {"ff4": "https://example.com/jk.csv:FF4"},
            start="2010-02-01",
            end="2010-03-01",
        )
    ser = result["ff4"]
    assert len(ser) == 2
    assert ser.index[0] == pd.Timestamp("2010-02-01")
    assert ser.index[-1] == pd.Timestamp("2010-03-01")


def test_external_archive_via_fetch_data_caches_to_disk(empty_project):
    """Full fetch_data path: archive parsed once, cached, served from disk on retry."""
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(_JK_ARCHIVE_CSV)) as mocked:
        df = fetch_data(
            api="external_archive",
            series={"ff4": "https://example.com/jk.csv:FF4"},
            dataset_id="jk_shocks",
            registry_path=empty_project["registry_path"],
        )
    assert "ff4" in df.columns
    assert mocked.call_count == 1
    csv_path = empty_project["project"] / "data" / "jk_shocks" / "dataset.csv"
    assert csv_path.exists()
    # The FF4 column of _JK_ARCHIVE_CSV, observation by observation.
    expected = pd.Series(
        [0.02, -0.01, 0.07, 0.00],
        index=pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01",
                              "2010-04-01"]),
    )

    def assert_ff4(series):
        assert pd.DatetimeIndex(series.index).equals(expected.index)
        np.testing.assert_allclose(series.to_numpy(dtype=float),
                                   expected.to_numpy(), rtol=0, atol=1e-12)

    assert_ff4(df["ff4"])
    assert_ff4(pd.read_csv(csv_path, index_col=0, parse_dates=True)["ff4"])

    # Second call: cache hit, no HTTP
    with patch("lib.data.fetchers.external_archive.request.urlopen") as mocked2:
        df2 = fetch_data(
            api="external_archive",
            series={"ff4": "https://example.com/jk.csv:FF4"},
            dataset_id="jk_shocks",
            registry_path=empty_project["registry_path"],
        )
        mocked2.assert_not_called()
    assert "ff4" in df2.columns
    assert_ff4(df2["ff4"])


def test_external_archive_handles_thousands_separator():
    """thousands kwarg passes through so '1,234.5'-style values parse as numeric."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv_data = (
        b"date,value\n"
        b"2010-01-01,\"1,234.5\"\n"
        b"2010-02-01,\"2,345.6\"\n"
    )
    spec = {
        "url": "https://example.com/thou.csv",
        "column": "value",
        "thousands": ",",
    }
    with patch("lib.data.fetchers.external_archive.request.urlopen",
                return_value=_mock_urlopen_bytes(csv_data)):
        result = fetcher.fetch({"v": spec})
    assert result["v"].iloc[0] == 1234.5
    assert result["v"].iloc[1] == 2345.6


# ---------------------------------------------------------------------------
# FRED fetcher with mocked HTTP
# ---------------------------------------------------------------------------

_FRED_SAMPLE_JSON = {
    "observations": [
        {"date": "2020-01-01", "value": "100.0"},
        {"date": "2020-02-01", "value": "."},      # FRED missing-value sentinel
        {"date": "2020-03-01", "value": "102.5"},
    ]
}


def test_fred_fetcher_parses_json(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test_key_123")
    from lib.data.fetchers.fred import FredFetcher
    with patch("lib.data.fetchers.fred.request.urlopen",
                return_value=_mock_urlopen_json(_FRED_SAMPLE_JSON)):
        result = FredFetcher().fetch({"rgdp": "GDPC1"})
    ser = result["rgdp"]
    assert len(ser) == 3
    assert ser.iloc[0] == 100.0
    assert np.isnan(ser.iloc[1])                       # "." -> NaN
    assert ser.iloc[2] == 102.5
    assert ser.index[0] == pd.Timestamp("2020-01-01")
    assert ser.index[-1] == pd.Timestamp("2020-03-01")
    assert ser.name == "rgdp"


def test_fred_fetcher_missing_key_raises(monkeypatch):
    """A missing FRED_API_KEY is a setup error -> clear RuntimeError, not a
    per-series warning."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from lib.data.fetchers.fred import FredFetcher
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredFetcher().fetch({"rgdp": "GDPC1"})


def test_fred_fetcher_rejects_bad_spec(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test_key_123")
    from lib.data.fetchers.fred import FredFetcher
    with pytest.raises(ValueError, match="bare series-ID"):
        FredFetcher().fetch({"x": ("a", "b")})


def test_fred_fetcher_http_error_warns_and_skips(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test_key_123")
    from urllib.error import HTTPError
    from lib.data.fetchers.fred import FredFetcher

    def _raise_400(*args, **kwargs):
        raise HTTPError("https://api.stlouisfed.org/fred", 400, "Bad Request", {}, None)

    with patch("lib.data.fetchers.fred.request.urlopen", side_effect=_raise_400):
        with pytest.warns(UserWarning, match="FRED fetch failed"):
            result = FredFetcher().fetch({"rgdp": "BADSERIES"})
    assert result == {}


def test_fred_fetcher_redacts_key_in_warning(monkeypatch):
    """The API key must never appear in a warning message."""
    monkeypatch.setenv("FRED_API_KEY", "secret_key_xyz")
    from urllib.error import URLError
    from lib.data.fetchers.fred import FredFetcher
    with patch("lib.data.fetchers.fred.request.urlopen", side_effect=URLError("boom")):
        with pytest.warns(UserWarning) as rec:
            FredFetcher().fetch({"rgdp": "GDPC1"})
    msg = str(rec[0].message)
    assert "secret_key_xyz" not in msg
    assert "<API_KEY>" in msg


# ---------------------------------------------------------------------------
# DBnomics fetcher with mocked HTTP
# ---------------------------------------------------------------------------

_DBNOMICS_SAMPLE = {
    "series": {
        "docs": [{
            "provider_code": "ECB", "dataset_code": "ICP",
            "series_code": "M.U2.N.000000.4.ANR",
            "period": ["2020-01", "2020-02", "2020-03"],
            "period_start_day": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "value": [1.4, 1.2, "NA"],          # "NA" = DBnomics missing sentinel
        }],
        "num_found": 1,
    },
    "errors": [],
}


def test_dbnomics_parses_json():
    from lib.data.fetchers.dbnomics import DbnomicsFetcher
    with patch("lib.data.fetchers.dbnomics.request.urlopen",
                return_value=_mock_urlopen_json(_DBNOMICS_SAMPLE)):
        result = DbnomicsFetcher().fetch({"ea_hicp": "ECB/ICP/M.U2.N.000000.4.ANR"})
    ser = result["ea_hicp"]
    assert len(ser) == 3
    assert ser.iloc[0] == 1.4
    assert np.isnan(ser.iloc[2])                       # "NA" -> NaN
    assert ser.index[0] == pd.Timestamp("2020-01-01")
    assert ser.name == "ea_hicp"


def test_dbnomics_quarterly_period_fallback():
    """No period_start_day -> fall back to parsing 'YYYY-Q#' period labels."""
    payload = {
        "series": {"docs": [{
            "provider_code": "IMF", "dataset_code": "IFS", "series_code": "Q.US.X",
            "period": ["1950-Q1", "1950-Q2", "1950-Q3"],
            "value": [100.0, 101.0, 102.0],
        }], "num_found": 1},
        "errors": [],
    }
    from lib.data.fetchers.dbnomics import DbnomicsFetcher
    with patch("lib.data.fetchers.dbnomics.request.urlopen",
                return_value=_mock_urlopen_json(payload)):
        result = DbnomicsFetcher().fetch({"x": "IMF/IFS/Q.US.X"})
    ser = result["x"]
    assert ser.index[0] == pd.Timestamp("1950-01-01")
    assert ser.index[1] == pd.Timestamp("1950-04-01")   # Q2 -> April
    assert ser.index[2] == pd.Timestamp("1950-07-01")   # Q3 -> July


def test_dbnomics_invalid_id_warns_and_skips():
    """Invalid id -> HTTP 200 with empty docs + an errors entry -> warn + skip."""
    payload = {
        "series": {"docs": [], "num_found": 0},
        "errors": [{"provider_code": "ECB", "dataset_code": "ICP",
                    "series_code": "NOPE", "message": "Could not load series"}],
    }
    from lib.data.fetchers.dbnomics import DbnomicsFetcher
    with patch("lib.data.fetchers.dbnomics.request.urlopen",
                return_value=_mock_urlopen_json(payload)):
        with pytest.warns(UserWarning, match="no data"):
            result = DbnomicsFetcher().fetch({"x": "ECB/ICP/NOPE"})
    assert result == {}


def test_dbnomics_rejects_bad_spec():
    from lib.data.fetchers.dbnomics import DbnomicsFetcher
    with pytest.raises(ValueError, match="PROVIDER/DATASET/SERIES"):
        DbnomicsFetcher().fetch({"x": "ECB/ICP"})       # only one '/'


def test_dbnomics_http_error_warns_and_skips():
    from urllib.error import HTTPError
    from lib.data.fetchers.dbnomics import DbnomicsFetcher

    def _raise_500(*args, **kwargs):
        raise HTTPError("https://api.db.nomics.world", 500, "Server Error", {}, None)

    with patch("lib.data.fetchers.dbnomics.request.urlopen", side_effect=_raise_500):
        with pytest.warns(UserWarning, match="DBnomics fetch failed"):
            result = DbnomicsFetcher().fetch({"x": "ECB/ICP/M.U2.N.000000.4.ANR"})
    assert result == {}


# ---------------------------------------------------------------------------
# Top-level lib export
# ---------------------------------------------------------------------------

def test_fetch_data_exposed_at_lib_top_level():
    assert lib.fetch_data is fetch_data


# ---------------------------------------------------------------------------
# Source-identity staleness (provider / spec changes) + calendar guards
# ---------------------------------------------------------------------------

class _SyntheticFetcherB(BaseFetcher):
    """Second provider returning different values for the same aliases."""

    name = "synthetic_b"
    description = "Test fixture fetcher (alternative provider)"
    endpoint = "test-b://"

    SERIES: dict[str, pd.Series] = {}

    def fetch(self, series, start=None, end=None):
        return {alias: self.SERIES[alias].copy()
                for alias in series if alias in self.SERIES}


@pytest.fixture
def _register_second_fetcher():
    register_fetcher("synthetic_b", _SyntheticFetcherB)
    yield
    from lib.data.fetcher import _FETCHER_REGISTRY
    _FETCHER_REGISTRY.pop("synthetic_b", None)


def test_provider_change_triggers_refetch(empty_project, synthetic_data,
                                          _register_second_fetcher):
    """The SAME alias + spec requested from a DIFFERENT provider is a
    different identity: it must be refetched, not served from the cache."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
               registry_path=empty_project["registry_path"])
    _SyntheticFetcherB.SERIES = {"a": synthetic_data["a"] * 10.0}
    with pytest.warns(UserWarning, match="refetching with the new identity"):
        df = fetch_data(api="synthetic_b", series={"a": "spec_a"},
                        dataset_id="testset",
                        registry_path=empty_project["registry_path"])
    assert np.allclose(df["a"].values, synthetic_data["a"].values * 10.0)


def test_spec_change_triggers_refetch(empty_project, synthetic_data):
    """Same provider, different source spec: also a new identity."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
               registry_path=empty_project["registry_path"])
    _SyntheticFetcher.SERIES = {"a": synthetic_data["a"] + 1.0}
    with pytest.warns(UserWarning, match="refetching with the new identity"):
        df = fetch_data(api="synthetic", series={"a": "spec_a_v2"},
                        dataset_id="testset",
                        registry_path=empty_project["registry_path"])
    assert np.allclose(df["a"].values, synthetic_data["a"].values + 1.0)


def test_failed_stale_refetch_raises_not_falls_back(empty_project,
                                                    synthetic_data):
    """A stale-identity alias whose refetch fails must raise — silently
    keeping the OLD source's values under the new label mislabels data."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
               registry_path=empty_project["registry_path"])
    with patch.object(_SyntheticFetcher, "fetch", return_value={}):
        with pytest.raises(RuntimeError, match="Refusing to fall back"):
            with pytest.warns(UserWarning,
                              match="refetching with the new identity"):
                fetch_data(api="synthetic", series={"a": "spec_a_v2"},
                           dataset_id="testset",
                           registry_path=empty_project["registry_path"])


@pytest.mark.parametrize("mode", ["metadata_absent", "alias_record_absent"])
def test_unrecorded_provenance_triggers_refetch(empty_project, synthetic_data,
                                                _register_second_fetcher, mode):
    """A cached alias whose provenance cannot be read (no metadata.yaml, or
    no record for the alias) cannot be matched to the requested identity:
    it must be refetched from the new provider, not served as its data."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
               registry_path=empty_project["registry_path"])
    meta = empty_project["project"] / "data" / "testset" / "metadata.yaml"
    if mode == "metadata_absent":
        meta.unlink()
    else:
        meta.write_text("variables: []\n", encoding="utf-8")
    _SyntheticFetcherB.SERIES = {"a": synthetic_data["a"] * 10.0}
    with patch.object(_SyntheticFetcherB, "fetch",
                      wraps=_SyntheticFetcherB().fetch) as spy:
        with pytest.warns(UserWarning, match="refetching with the new identity"):
            df = fetch_data(api="synthetic_b", series={"a": "spec_a"},
                            dataset_id="testset",
                            registry_path=empty_project["registry_path"])
    assert spy.call_count == 1
    assert np.allclose(df["a"].values, synthetic_data["a"].values * 10.0)


def test_unrecorded_provenance_failed_refetch_raises(empty_project,
                                                     synthetic_data):
    """Unverifiable cached values are not a fallback when the refetch fails."""
    _SyntheticFetcher.SERIES = synthetic_data
    fetch_data(api="synthetic", series={"a": "spec_a"}, dataset_id="testset",
               registry_path=empty_project["registry_path"])
    (empty_project["project"] / "data" / "testset" / "metadata.yaml").unlink()
    with patch.object(_SyntheticFetcher, "fetch", return_value={}):
        with pytest.raises(RuntimeError, match="Refusing to fall back"):
            with pytest.warns(UserWarning,
                              match="refetching with the new identity"):
                fetch_data(api="synthetic", series={"a": "spec_a"},
                           dataset_id="testset",
                           registry_path=empty_project["registry_path"])


def test_duplicate_dates_from_fetcher_rejected(empty_project):
    """Write-side calendar guard: a fetcher returning a nonunique calendar
    must be refused at the cache boundary."""
    idx = pd.to_datetime(["2000-01-01", "2000-01-01", "2000-02-01"])
    _SyntheticFetcher.SERIES = {
        "a": pd.Series([1.0, 2.0, 3.0], index=idx, name="a"),
    }
    with pytest.raises(ValueError, match="DUPLICATE dates"):
        fetch_data(api="synthetic", series={"a": "spec_a"},
                   dataset_id="dupset",
                   registry_path=empty_project["registry_path"])


def test_external_archive_numeric_date_column_rejected():
    """A bare-year date column without an explicit date_format must be
    refused — pandas would parse the integers as nanosecond offsets."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv = b"year,SHOCK\n1990,0.1\n1991,-0.2\n1992,0.05\n"
    with patch("lib.data.fetchers.external_archive.request.urlopen",
               return_value=_mock_urlopen_bytes(csv)):
        with pytest.warns(UserWarning, match="NANOSECOND"):
            result = fetcher.fetch({"s": {"url": "https://example.com/x.csv",
                                          "column": "SHOCK"}})
    # The alias is refused (not cached with a 1970s calendar); fetch_data
    # then raises its all-series-failed RuntimeError when nothing was cached.
    assert result == {}


def test_external_archive_numeric_date_with_format_parses():
    """The same file loads once the spec states the date format."""
    from lib.data.fetchers.external_archive import ExternalArchiveFetcher
    fetcher = ExternalArchiveFetcher()
    csv = b"year,SHOCK\n1990,0.1\n1991,-0.2\n1992,0.05\n"
    with patch("lib.data.fetchers.external_archive.request.urlopen",
               return_value=_mock_urlopen_bytes(csv)):
        result = fetcher.fetch({"s": {"url": "https://example.com/x.csv",
                                      "column": "SHOCK",
                                      "date_format": "%Y"}})
    assert result["s"].index[0] == pd.Timestamp("1990-01-01")
    assert len(result["s"]) == 3


_BOE_DAILY_CSV = (
    "DATE,IUMABEDR\n"
    "01/Jan/2020,0.75\n"
    "15/Jan/2020,0.70\n"
    "01/Feb/2020,0.65\n"
)

_BOE_QUARTERLY_CSV = (
    "DATE,IUMABEDR\n"
    "01/Jan/2020,0.75\n"
    "01/Apr/2020,0.50\n"
    "01/Jul/2020,0.25\n"
)


def test_boe_iadb_rejects_multiple_obs_per_month():
    """Daily/weekly data must not be silently collapsed to month starts."""
    from lib.data.fetchers.boe_iadb import BoeIadbFetcher
    fetcher = BoeIadbFetcher()
    with patch("lib.data.fetchers.boe_iadb.request.urlopen",
               return_value=_mock_urlopen_text(_BOE_DAILY_CSV)):
        with pytest.raises(ValueError, match="multiple observations"):
            fetcher.fetch({"rate": "IUMABEDR"})


def test_boe_iadb_rejects_non_monthly_spacing():
    """Quarterly data must not be relabelled as monthly."""
    from lib.data.fetchers.boe_iadb import BoeIadbFetcher
    fetcher = BoeIadbFetcher()
    with patch("lib.data.fetchers.boe_iadb.request.urlopen",
               return_value=_mock_urlopen_text(_BOE_QUARTERLY_CSV)):
        with pytest.raises(ValueError, match="not monthly"):
            fetcher.fetch({"rate": "IUMABEDR"})
