"""Tests for ``lib.sensitivity`` — hyperparameter-sweep driver.

Validation strategy:

1. **Single-axis sweep** — produces N rows for N grid points; metric
   values vary across the axis (using ``lambda_1``, which moves log_mdd
   under the symmetric NIW prior; ``lambda_2`` is a no-op under the
   symmetric prior by construction — see
   ``test_conventions.py::test_lambda_2_no_op_under_symmetric_mode``).
2. **Cartesian product** — multi-axis grid produces |axis1| × |axis2| rows.
3. **Default metric extractor** — pulls max_eig, AIC/BIC/HQIC, log_mdd.
4. **Error robustness** — a spec that fails to run produces a row with
   error column populated; other rows still complete.
5. **best() lookup** — finds the row minimising / maximising the chosen metric.
6. **Dotted-path parsing** — converts ``"a.b.c"`` to nested dict updates.
7. **keep_results=False** — table is built but results list is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.sensitivity import (  # noqa: E402
    SensitivityReport,
    sensitivity,
    _set_nested,
    _default_metric_fn,
)

# No module-level warning filter: an empirical run with `-W default` shows
# these tests emit no warnings, and pytest resets warning filters per test
# anyway, so a blanket import-time filter would only mask collection-time
# warnings while silently suppressing nothing useful.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FRED_QD_CSV = Path(__file__).resolve().parents[1] / "data" / "fred_qd" / "dataset.csv"


def _skip_without_data():
    if not _FRED_QD_CSV.exists():
        pytest.skip("fred_qd dataset.csv not downloaded - see data/fred_qd/UPDATE.md")


@pytest.fixture
def litterman_base():
    """Litterman 1986 spec without forecasts (keeps the sweep fast).

    Overrides the prior because these tests sweep ``lambda_1`` via
    ``log_mdd``, which is only defined under a *proper* natural-conjugate
    (NIW) prior:

    * ``asymmetric=False`` — the closed-form NIW MDD does not apply to
      Litterman's asymmetric, fixed-σ prior.
    * ``mu>0`` and ``delta>0`` — enable the sum-of-coefficients and
      dummy-initial-observation dummy blocks. Without them the core
      Minnesota dummies for this 7-variable, 6-lag system give exactly
      ``n_dummies == k`` (43 == 43), an *improper* prior whose MDD is
      ``-inf``. The paper-faithful Litterman replication sets
      ``asymmetric=True, mu=0, delta=0`` (both priors post-date 1986), so
      using MDD here is a deliberate test-only deviation from the paper.
    """
    _skip_without_data()
    return lib.Spec.from_replication("litterman_1986_bvar").update(
        forecast=None,
        irf={"horizon": 4, "n_draws": 100},   # small for test speed
        model={"prior": {"asymmetric": False, "mu": 1.0, "delta": 1.0}},
    )


@pytest.fixture
def small_var_base():
    """Tiny synthetic 3-variable VAR spec, for fast multi-axis tests."""
    _skip_without_data()
    return lib.Spec.from_dict({
        "data": {
            "source": "fred_qd",
            "sample_start": "1990-01-01",
            "sample_end": "2019-12-31",
            "series": [
                {"name": "rgdp", "fred_code": "GDPC1"},
                {"name": "cpi", "fred_code": "GDPCTPI"},
                {"name": "ffr", "fred_code": "FEDFUNDS"},
            ],
            "transforms": {"rgdp": "log", "cpi": "log", "ffr": "level"},
            "variable_order": ["rgdp", "cpi", "ffr"],
        },
        "model": {"class": "var", "lags": 2, "trend": "c"},
        "irf": {"method": "cholesky", "horizon": 4, "bands": "asymptotic_mc", "n_draws": 100},
    })


# ---------------------------------------------------------------------------
# Single-axis sweep
# ---------------------------------------------------------------------------


class TestSingleAxisSweep:
    def test_returns_sensitivity_report(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.05, 0.1, 0.2]},
        )
        assert isinstance(report, SensitivityReport)
        assert len(report.table) == 3
        assert len(report.results) == 3

    def test_grid_values_in_table(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.05, 0.1, 0.2]},
        )
        # Grid values must be the first column(s) of the table
        assert list(report.table["model.prior.lambda_1"]) == [0.05, 0.1, 0.2]

    def test_default_metrics_extracted(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.1, 0.5]},
        )
        for col in ("max_eig", "aic", "bic", "log_mdd"):
            assert col in report.table.columns, f"missing column {col}"
        # MDD should be finite for both
        assert report.table["log_mdd"].notna().all()

    def test_lambda_1_affects_mdd(self, litterman_base):
        """Sanity: varying lambda_1 actually moves the metric — verifies the
        sensitivity driver propagates the override end-to-end."""
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.05, 0.5]},
        )
        mdds = report.table["log_mdd"].values
        assert abs(mdds[0] - mdds[1]) > 1.0, (
            f"lambda_1 should move log MDD substantially; got {mdds}"
        )


# ---------------------------------------------------------------------------
# Multi-axis cartesian product
# ---------------------------------------------------------------------------


class TestCartesianProduct:
    def test_2x3_grid_produces_6_rows(self, small_var_base):
        report = sensitivity(
            small_var_base,
            grid={
                "model.lags": [1, 2, 3],
                "irf.horizon": [4, 8],
            },
        )
        assert len(report.table) == 6    # 3 * 2

    def test_all_combinations_present(self, small_var_base):
        report = sensitivity(
            small_var_base,
            grid={
                "model.lags": [1, 2],
                "irf.horizon": [4, 8],
            },
        )
        pairs = set(zip(report.table["model.lags"], report.table["irf.horizon"]))
        assert pairs == {(1, 4), (1, 8), (2, 4), (2, 8)}


# ---------------------------------------------------------------------------
# best() lookup
# ---------------------------------------------------------------------------


class TestBest:
    def test_best_by_metric_returns_spec_result(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.05, 0.1, 0.2, 0.5]},
        )
        best = report.best(by="log_mdd", ascending=False)
        assert best is not None
        # Verify the spec on best corresponds to the row with max log_mdd
        best_mdd = float(report.table["log_mdd"].max())
        best_lam1 = float(report.table.loc[report.table["log_mdd"].idxmax(),
                                            "model.prior.lambda_1"])
        assert best.spec.model["prior"]["lambda_1"] == best_lam1

    def test_best_ascending_uses_min(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.05, 0.1, 0.5]},
        )
        best = report.best(by="log_mdd", ascending=True)   # min mdd
        worst_mdd = float(report.table["log_mdd"].min())
        worst_lam1 = float(report.table.loc[report.table["log_mdd"].idxmin(),
                                             "model.prior.lambda_1"])
        assert best.spec.model["prior"]["lambda_1"] == worst_lam1

    def test_best_raises_on_missing_metric(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.1, 0.5]},
        )
        with pytest.raises(ValueError, match="not in table columns"):
            report.best(by="not_a_metric")

    def test_best_raises_when_results_dropped(self, litterman_base):
        report = sensitivity(
            litterman_base,
            grid={"model.prior.lambda_1": [0.1, 0.5]},
            keep_results=False,
        )
        with pytest.raises(ValueError, match="results list is empty"):
            report.best(by="log_mdd")


# ---------------------------------------------------------------------------
# Error robustness
# ---------------------------------------------------------------------------


class TestErrorRobustness:
    def test_failed_spec_records_error(self, small_var_base):
        """If a spec combination causes lib.run to fail, the sweep continues
        and the failed row's ``error`` column is populated."""
        # lags=0 is invalid → lib.fit_var raises
        report = sensitivity(
            small_var_base,
            grid={"model.lags": [0, 2]},   # 0 fails, 2 succeeds
        )
        assert len(report.table) == 2
        # The lags=0 row has a non-null error and NaN metrics
        error_row = report.table[report.table["model.lags"] == 0].iloc[0]
        assert error_row["error"] is not None
        assert pd.isna(error_row["max_eig"])
        # The lags=2 row succeeds
        ok_row = report.table[report.table["model.lags"] == 2].iloc[0]
        assert ok_row["error"] is None
        assert not pd.isna(ok_row["max_eig"])


# ---------------------------------------------------------------------------
# keep_results=False
# ---------------------------------------------------------------------------


class TestKeepResultsFalse:
    def test_keep_results_false_empties_list(self, small_var_base):
        report = sensitivity(
            small_var_base,
            grid={"model.lags": [1, 2]},
            keep_results=False,
        )
        assert report.results == []
        # Table is still populated
        assert len(report.table) == 2
        assert "max_eig" in report.table.columns


# ---------------------------------------------------------------------------
# Dotted-path helper
# ---------------------------------------------------------------------------


class TestSetNested:
    def test_single_key(self):
        d = {}
        _set_nested(d, "a", 1)
        assert d == {"a": 1}

    def test_two_levels(self):
        d = {}
        _set_nested(d, "a.b", 2)
        assert d == {"a": {"b": 2}}

    def test_three_levels(self):
        d = {}
        _set_nested(d, "a.b.c", 3)
        assert d == {"a": {"b": {"c": 3}}}

    def test_preserves_siblings(self):
        d = {"a": {"x": 1}}
        _set_nested(d, "a.b", 2)
        assert d == {"a": {"x": 1, "b": 2}}

    def test_overwrites_non_dict_intermediate(self):
        """If an intermediate value isn't a dict, it's replaced with one."""
        d = {"a": 5}
        _set_nested(d, "a.b", 2)
        assert d == {"a": {"b": 2}}


# ---------------------------------------------------------------------------
# Custom metric_fn
# ---------------------------------------------------------------------------


class TestCustomMetricFn:
    def test_custom_metric_columns_appear(self, small_var_base):
        def my_metrics(result) -> dict:
            return {
                "rgdp_h2_response": float(
                    result.irf.median[2, result.irf.variable_names.index("rgdp"),
                                          result.irf.variable_names.index("ffr")]
                ),
                "data_n_obs": int(len(result.data)),
            }

        report = sensitivity(
            small_var_base,
            grid={"model.lags": [1, 2]},
            metric_fn=my_metrics,
        )
        assert "rgdp_h2_response" in report.table.columns
        assert "data_n_obs" in report.table.columns
        # Both rows should have the same data_n_obs (data load is identical)
        assert len(set(report.table["data_n_obs"])) == 1


# ---------------------------------------------------------------------------
# Default metric_fn
# ---------------------------------------------------------------------------


class TestDefaultMetricFn:
    def test_extracts_max_eig_on_var(self, small_var_base):
        result = lib.run(small_var_base, seed=42)
        m = _default_metric_fn(result)
        assert "max_eig" in m
        assert isinstance(m["max_eig"], float)
        assert m["max_eig"] > 0

    def test_extracts_log_mdd_on_bvar(self, litterman_base):
        result = lib.run(litterman_base, seed=42)
        m = _default_metric_fn(result)
        assert "log_mdd" in m
        # MDD finite and finite-large for this fit
        assert np.isfinite(m["log_mdd"])
        # 7-variable, 6-lag system on ~237 obs → large-magnitude negative log MDD
        assert m["log_mdd"] < 0

    def test_no_log_mdd_on_var(self, small_var_base):
        result = lib.run(small_var_base, seed=42)
        m = _default_metric_fn(result)
        # log_mdd is only added for BVARFit
        assert "log_mdd" not in m


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_grid_rejected(self, small_var_base):
        with pytest.raises(ValueError, match="non-empty dict"):
            sensitivity(small_var_base, grid={})

    def test_non_dict_grid_rejected(self, small_var_base):
        with pytest.raises(ValueError, match="non-empty dict"):
            sensitivity(small_var_base, grid="not_a_dict")    # type: ignore[arg-type]

    def test_string_value_rejected(self, small_var_base):
        """A bare string can be iterated character by character — guard against
        the user passing a single string instead of a list of strings."""
        with pytest.raises(ValueError, match="list/tuple of values"):
            sensitivity(small_var_base, grid={"model.lags": "1234"})


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_lib_exports():
    assert lib.sensitivity is sensitivity
    assert lib.SensitivityReport is SensitivityReport


def test_lib_all_lists_sensitivity_surface():
    assert "sensitivity" in lib.__all__
    assert "SensitivityReport" in lib.__all__
