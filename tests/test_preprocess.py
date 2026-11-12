"""Tests for ``lib.preprocess`` — complex preprocessing utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.preprocess import aggregate, deflate, impute, nominalise  # noqa: E402


@pytest.fixture
def fred_like():
    """Synthetic monthly nominal series + a price deflator resembling FRED format."""
    dates = pd.date_range("2000-01-01", "2010-12-01", freq="MS")
    n = len(dates)
    rng = np.random.default_rng(42)
    nominal = pd.Series(
        100 + np.cumsum(rng.normal(0.5, 1.0, n)), index=dates, name="m1_nominal"
    )
    real = pd.Series(
        80 + np.cumsum(rng.normal(0.3, 0.8, n)), index=dates, name="m1_real"
    )
    # Deflator increasing from ~80 to ~120 with base 100 mid-sample
    deflator = pd.Series(
        np.linspace(80.0, 120.0, n), index=dates, name="gdpctpi"
    )
    return {"nominal": nominal, "real": real, "deflator": deflator, "dates": dates}


# ---------------------------------------------------------------------------
# deflate
# ---------------------------------------------------------------------------

def test_deflate_basic(fred_like):
    """real = nominal * 100 / deflator."""
    real = deflate(fred_like["nominal"], fred_like["deflator"])
    expected = fred_like["nominal"] * 100.0 / fred_like["deflator"]
    pd.testing.assert_series_equal(real, expected, check_names=False)


def test_deflate_preserves_name(fred_like):
    real = deflate(fred_like["nominal"], fred_like["deflator"])
    assert real.name == "m1_nominal"


def test_deflate_with_base_period(fred_like):
    """At the base period, real == nominal (deflator rebased to 100)."""
    base = "2005-01-01"
    real = deflate(fred_like["nominal"], fred_like["deflator"], base_period=base)
    assert abs(real.loc[base] - fred_like["nominal"].loc[base]) < 1e-10


def test_deflate_with_custom_scale(fred_like):
    """deflator_scale=1.0 treats deflator as ratio form (no base-100 division)."""
    deflator_ratio = fred_like["deflator"] / 100.0
    real = deflate(fred_like["nominal"], deflator_ratio, deflator_scale=1.0)
    expected = fred_like["nominal"] / deflator_ratio
    pd.testing.assert_series_equal(real, expected, check_names=False)


# ---------------------------------------------------------------------------
# nominalise
# ---------------------------------------------------------------------------

def test_nominalise_basic(fred_like):
    """nominal = real * deflator / 100."""
    nom = nominalise(fred_like["real"], fred_like["deflator"])
    expected = fred_like["real"] * fred_like["deflator"] / 100.0
    pd.testing.assert_series_equal(nom, expected, check_names=False)


def test_nominalise_preserves_name(fred_like):
    nom = nominalise(fred_like["real"], fred_like["deflator"])
    assert nom.name == "m1_real"


def test_nominalise_with_base_period(fred_like):
    """At the base period, nominal == real."""
    base = "2007-06-01"
    nom = nominalise(fred_like["real"], fred_like["deflator"], base_period=base)
    assert abs(nom.loc[base] - fred_like["real"].loc[base]) < 1e-10


# ---------------------------------------------------------------------------
# Inverse property
# ---------------------------------------------------------------------------

def test_deflate_nominalise_inverse(fred_like):
    """deflate(nominalise(x)) == x and vice versa."""
    nominal_original = fred_like["nominal"]
    deflator = fred_like["deflator"]

    roundtrip_nom = deflate(nominalise(nominal_original, deflator), deflator)
    pd.testing.assert_series_equal(roundtrip_nom, nominal_original, check_names=False)

    real_original = fred_like["real"]
    roundtrip_real = nominalise(deflate(real_original, deflator), deflator)
    pd.testing.assert_series_equal(roundtrip_real, real_original, check_names=False)


def test_inverse_with_base_period(fred_like):
    """Inverse holds even with a base_period rebase."""
    base = "2003-04-01"
    nominal = fred_like["nominal"]
    deflator = fred_like["deflator"]
    roundtrip = deflate(
        nominalise(nominal, deflator, base_period=base),
        deflator, base_period=base,
    )
    pd.testing.assert_series_equal(roundtrip, nominal, check_names=False)


# ---------------------------------------------------------------------------
# aggregate — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_panel():
    """Synthetic monthly panel with mixed flow / rate / stock columns.

    Spans 2000-Q1..2001-Q4 (24 months = 8 quarters). Deterministic values
    so each test can hand-check a sum or mean.
    """
    dates = pd.date_range("2000-01-01", periods=24, freq="MS")
    df = pd.DataFrame(
        {
            # Flow: 1, 2, ..., 24 (each quarter sums to 6, 15, 24, 33, 42, 51, 60, 69)
            "retail_sales": np.arange(1, 25, dtype=float),
            # Rate: 5.0 everywhere (each quarter mean = 5.0)
            "ffr": np.full(24, 5.0),
            # Stock: 100, 101, ..., 123 (each quarter end = 102, 105, 108, ...)
            "m1": np.arange(100, 124, dtype=float),
        },
        index=dates,
    )
    df.attrs = {
        "source": "synthetic",
        "frequency": "MS",
        "loaded_at": "2026-05-20T00:00:00Z",
    }
    return df


# ---------------------------------------------------------------------------
# aggregate — happy path
# ---------------------------------------------------------------------------

def test_aggregate_series_mean(monthly_panel):
    """A Series collapses to one value per target period via mean."""
    s = monthly_panel["ffr"]
    q = aggregate(s, "QS", "mean")
    assert len(q) == 8
    assert (q == 5.0).all()
    assert q.index[0] == pd.Timestamp("2000-01-01")
    assert q.index[-1] == pd.Timestamp("2001-10-01")


def test_aggregate_dataframe_scalar_method(monthly_panel):
    """A string method applies to all columns."""
    q = aggregate(monthly_panel, "QS", "mean")
    assert list(q.columns) == ["retail_sales", "ffr", "m1"]
    assert len(q) == 8
    # Each quarter's mean of 1..24 in chunks of 3: 2, 5, 8, 11, 14, 17, 20, 23
    np.testing.assert_array_equal(
        q["retail_sales"].to_numpy(), [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]
    )


def test_aggregate_dataframe_dict_method(monthly_panel):
    """Per-column methods produce the right per-quarter values."""
    q = aggregate(
        monthly_panel,
        "QS",
        {"retail_sales": "sum", "ffr": "mean", "m1": "end"},
    )
    # Sums of consecutive triples of 1..24
    np.testing.assert_array_equal(
        q["retail_sales"].to_numpy(), [6.0, 15.0, 24.0, 33.0, 42.0, 51.0, 60.0, 69.0]
    )
    assert (q["ffr"] == 5.0).all()
    # m1 quarter-end = last month of quarter: 102, 105, 108, ..., 123
    np.testing.assert_array_equal(
        q["m1"].to_numpy(), [102.0, 105.0, 108.0, 111.0, 114.0, 117.0, 120.0, 123.0]
    )


def test_aggregate_end_alias_matches_last(monthly_panel):
    """'end' is a synonym for 'last'."""
    a = aggregate(monthly_panel["m1"], "QS", "end")
    b = aggregate(monthly_panel["m1"], "QS", "last")
    pd.testing.assert_series_equal(a, b)


def test_aggregate_start_alias_matches_first(monthly_panel):
    """'start' is a synonym for 'first'."""
    a = aggregate(monthly_panel["m1"], "QS", "start")
    b = aggregate(monthly_panel["m1"], "QS", "first")
    pd.testing.assert_series_equal(a, b)


def test_aggregate_quarterly_to_annual():
    """Quarterly → annual aggregation collapses 4 quarters per year."""
    dates = pd.date_range("2000-01-01", periods=8, freq="QS")  # 2 years
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=dates)
    y = aggregate(s, "YS", "sum")
    assert len(y) == 2
    np.testing.assert_array_equal(y.to_numpy(), [10.0, 26.0])  # 1+2+3+4, 5+6+7+8


def test_aggregate_preserves_attrs_and_logs(monthly_panel):
    """df.attrs is copied; a 'preprocess_applied' entry is appended."""
    q = aggregate(monthly_panel, "QS", "mean")
    assert q.attrs["source"] == "synthetic"
    assert q.attrs["frequency"] == "MS"  # original (caller can update if they wish)
    assert "preprocess_applied" in q.attrs
    entry = q.attrs["preprocess_applied"][-1]
    assert entry["op"] == "aggregate"
    assert entry["target_freq"] == "QS"
    assert entry["method"] == "mean"


def test_aggregate_appends_to_existing_preprocess_log():
    """A pre-existing preprocess_applied list is extended, not overwritten."""
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    df = pd.DataFrame({"x": np.arange(12, dtype=float)}, index=dates)
    df.attrs = {"preprocess_applied": [{"op": "deflate", "base_period": "2000-01-01"}]}
    q = aggregate(df, "QS", "sum")
    assert len(q.attrs["preprocess_applied"]) == 2
    assert q.attrs["preprocess_applied"][0]["op"] == "deflate"
    assert q.attrs["preprocess_applied"][1]["op"] == "aggregate"


def test_aggregate_series_name_preserved(monthly_panel):
    """Series.name survives aggregation."""
    s = monthly_panel["m1"]
    q = aggregate(s, "QS", "end")
    assert q.name == "m1"


def test_aggregate_dataframe_column_order_preserved(monthly_panel):
    """Output columns match input order even when the dict iterates differently."""
    q = aggregate(
        monthly_panel,
        "QS",
        {"m1": "end", "retail_sales": "sum", "ffr": "mean"},  # different order
    )
    assert list(q.columns) == ["retail_sales", "ffr", "m1"]


def test_aggregate_to_month_end_anchor(monthly_panel):
    """QE (quarter-end) anchor: index values are end-of-quarter dates."""
    q = aggregate(monthly_panel, "QE", "mean")
    assert len(q) == 8
    assert q.index[0] == pd.Timestamp("2000-03-31")
    assert q.index[-1] == pd.Timestamp("2001-12-31")


# ---------------------------------------------------------------------------
# aggregate — error paths
# ---------------------------------------------------------------------------

def test_aggregate_raises_on_non_datetime_index():
    """A non-DatetimeIndex triggers TypeError before any resampling."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        aggregate(df, "QS", "mean")


def test_aggregate_raises_on_series_dict_method(monthly_panel):
    """A Series + dict method raises TypeError (dicts are DataFrame-only)."""
    with pytest.raises(TypeError, match="string method"):
        aggregate(monthly_panel["m1"], "QS", {"m1": "end"})


def test_aggregate_raises_on_dict_missing_column(monthly_panel):
    """Dict missing entries for some DataFrame columns is rejected."""
    with pytest.raises(ValueError, match="missing entries"):
        aggregate(monthly_panel, "QS", {"retail_sales": "sum"})  # ffr, m1 missing


def test_aggregate_raises_on_dict_extra_column(monthly_panel):
    """Dict with extra keys not in the DataFrame is rejected."""
    with pytest.raises(ValueError, match="unknown columns"):
        aggregate(
            monthly_panel,
            "QS",
            {
                "retail_sales": "sum",
                "ffr": "mean",
                "m1": "end",
                "phantom": "mean",  # not a column
            },
        )


def test_aggregate_raises_on_unknown_method(monthly_panel):
    """An unrecognised method name is rejected with the valid set listed."""
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregate(monthly_panel, "QS", "geomean")


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------

def test_lib_preprocess_exports():
    """Functions are accessible via lib.preprocess.<name>."""
    assert lib.preprocess.deflate is deflate
    assert lib.preprocess.nominalise is nominalise
    assert lib.preprocess.aggregate is aggregate
    assert lib.preprocess.impute is impute


def test_lib_preprocess_all_lists_public_surface():
    """__all__ advertises the public surface."""
    assert "deflate" in lib.preprocess.__all__
    assert "nominalise" in lib.preprocess.__all__
    assert "aggregate" in lib.preprocess.__all__
    assert "impute" in lib.preprocess.__all__


# ---------------------------------------------------------------------------
# impute
# ---------------------------------------------------------------------------


@pytest.fixture
def var1_data_with_nans():
    """Synthetic VAR(1) data on quarterly index with controlled NaN positions.

    The DGP is mildly persistent so all three imputation methods produce
    finite, comparable reconstructions. NaN positions: (a) two consecutive
    interior NaNs (rows 50, 51 of column 'a'), (b) a fully-NaN row (90),
    (c) a single NaN in column 'b' (row 95).
    """
    rng = np.random.default_rng(0)
    T = 100
    A = np.array([[0.7, 0.1], [0.0, 0.6]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(2) * 0.5
    df_true = pd.DataFrame(
        y, columns=["a", "b"],
        index=pd.date_range("2000-01-01", periods=T, freq="QS"),
    )
    df_nan = df_true.copy()
    df_nan.iloc[50, 0] = np.nan
    df_nan.iloc[51, 0] = np.nan
    df_nan.iloc[90, :] = np.nan
    df_nan.iloc[95, 1] = np.nan
    return {"true": df_true, "nan": df_nan}


def _imputed_errors(out, df_true, mask, label):
    """Errors at the cells that were missing and must now be filled.

    Selects the expected-to-be-filled positions first and requires every
    imputed value there to be finite, so an imputer that leaves cells
    unfilled cannot drop them silently from a pandas NaN-skipping sum.
    """
    mask = np.asarray(mask, dtype=bool)
    assert mask.any(), "fixture has no missing cells to score"
    est = np.asarray(out, dtype=float)[mask]
    truth = np.asarray(df_true, dtype=float)[mask]
    n_bad = int((~np.isfinite(est)).sum())
    assert n_bad == 0, (
        f"{label} left {n_bad} of {mask.sum()} missing cells unfilled"
    )
    return est - truth


class TestImpute:
    """Tests for ``lib.preprocess.impute``.

    Validation strategy:

    * **Identity on fully-observed input** — no NaNs in, no changes out
      (for every method).
    * **Observed values stay exact** — imputation only fills NaN
      positions; observed values pass through bit-identically.
    * **Method-specific limiting cases** — ffill matches the prior value;
      linear gives the midpoint on a single-gap; Kalman beats both on
      VAR(p) data (reconstruction RMSE ordering).
    * **All-NaN column rejected** with a clear error.
    * **df.attrs preserved** + ``preprocess_applied`` entry appended.
    """

    def test_identity_on_fully_observed(self, var1_data_with_nans):
        df = var1_data_with_nans["true"]
        for method in ("ffill", "linear", "kalman"):
            kwargs = {"lags": 1} if method == "kalman" else {}
            out = impute(df, method=method, **kwargs)
            # NaN-free input — every entry must be unchanged
            assert np.allclose(out.values, df.values, atol=1e-12), (
                f"{method} changed fully-observed data"
            )

    def test_observed_values_preserved(self, var1_data_with_nans):
        """Imputation only fills NaN positions; observed values must
        come through bit-identically."""
        df_nan = var1_data_with_nans["nan"]
        observed_mask = ~df_nan.isna()
        for method in ("ffill", "linear", "kalman"):
            kwargs = {"lags": 1} if method == "kalman" else {}
            out = impute(df_nan, method=method, **kwargs)
            # Where the input was observed, the output must equal the input
            obs = df_nan.where(observed_mask)
            out_obs = out.where(observed_mask)
            assert np.allclose(
                obs.fillna(0).values, out_obs.fillna(0).values, atol=1e-12,
            ), f"{method} altered an observed value"

    def test_ffill_matches_prior_value(self, var1_data_with_nans):
        df_nan = var1_data_with_nans["nan"]
        out = impute(df_nan, method="ffill")
        # Row 50 NaN in col 'a' → prior value at row 49
        assert out.iloc[50, 0] == df_nan.iloc[49, 0]
        # Row 51 NaN too → same prior (49) carried forward
        assert out.iloc[51, 0] == df_nan.iloc[49, 0]

    def test_linear_midpoint_on_single_gap(self):
        """One missing value between two observed → mean of neighbours."""
        df = pd.DataFrame(
            {"x": [1.0, np.nan, 3.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )
        out = impute(df, method="linear")
        assert out["x"].iloc[1] == pytest.approx(2.0)

    def test_kalman_beats_naive_on_var_data(self, var1_data_with_nans):
        """On VAR(p) DGP data, Kalman should reconstruct NaN positions
        with lower RMSE than ffill (which uses no model structure)."""
        df_true = var1_data_with_nans["true"]
        df_nan = var1_data_with_nans["nan"]
        nan_mask = df_nan.isna()
        out_kalman = impute(df_nan, method="kalman", lags=1)
        out_ffill = impute(df_nan, method="ffill")
        # Per-cell SE on NaN positions; every one must be filled
        se_k = float(np.sum(
            _imputed_errors(out_kalman, df_true, nan_mask, "kalman") ** 2))
        se_f = float(np.sum(
            _imputed_errors(out_ffill, df_true, nan_mask, "ffill") ** 2))
        assert se_k < se_f, (
            f"Kalman SE ({se_k:.4f}) should beat ffill SE ({se_f:.4f}) "
            f"on a VAR(p) DGP"
        )

    def test_all_nan_column_raises(self):
        df = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [np.nan, np.nan, np.nan]},
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )
        with pytest.raises(ValueError, match="fully-missing"):
            impute(df, method="linear")
        with pytest.raises(ValueError, match="fully-missing"):
            impute(df, method="kalman", lags=1)

    def test_unknown_method_raises(self, var1_data_with_nans):
        with pytest.raises(ValueError, match="Unknown method"):
            impute(var1_data_with_nans["nan"], method="cubic_spline")

    def test_attrs_preserved_and_logged(self, var1_data_with_nans):
        df_nan = var1_data_with_nans["nan"].copy()
        df_nan.attrs = {"source": "test", "preprocess_applied": [{"op": "prior"}]}
        out = impute(df_nan, method="linear")
        assert out.attrs["source"] == "test"
        applied = out.attrs["preprocess_applied"]
        assert applied[0] == {"op": "prior"}
        assert applied[-1]["op"] == "impute"
        assert applied[-1]["method"] == "linear"

    def test_series_input_returns_series(self):
        s = pd.Series(
            [1.0, np.nan, 3.0, np.nan, 5.0],
            index=pd.date_range("2020-01-01", periods=5, freq="MS"),
            name="x",
        )
        out = impute(s, method="linear")
        assert isinstance(out, pd.Series)
        assert out.name == "x"
        assert out.iloc[1] == pytest.approx(2.0)
        assert out.iloc[3] == pytest.approx(4.0)

    def test_kalman_fills_trailing_nan(self, var1_data_with_nans):
        """Trailing NaN (ragged edge) is the Kalman use case. ``linear``
        leaves it as NaN; ``kalman`` fills it with the model's forecast."""
        df_nan = var1_data_with_nans["nan"].copy()
        df_nan.iloc[-1, 0] = np.nan
        out_linear = impute(df_nan, method="linear")
        out_kalman = impute(df_nan, method="kalman", lags=1)
        # Linear leaves trailing NaN; Kalman fills it
        assert np.isnan(out_linear.iloc[-1, 0])
        assert not np.isnan(out_kalman.iloc[-1, 0])

    def test_fill_leading_option_for_ffill(self):
        df = pd.DataFrame(
            {"x": [np.nan, np.nan, 3.0, 4.0]},
            index=pd.date_range("2020-01-01", periods=4, freq="MS"),
        )
        # Without fill_leading, leading NaNs stay
        out_no = impute(df, method="ffill")
        assert np.isnan(out_no["x"].iloc[0])
        # With fill_leading, leading NaNs get the first observed value
        out_yes = impute(df, method="ffill", fill_leading=True)
        assert out_yes["x"].iloc[0] == 3.0
        assert out_yes["x"].iloc[1] == 3.0

    def test_requires_datetimeindex(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            impute(df, method="linear")


class TestImputeEM:
    """Tests for the EM-iteration extension on ``method="kalman"``.

    EM (Shumway-Stoffer 1982) alternates between Kalman smoothing
    (E-step) and re-estimating VAR(p) parameters from the smoother's
    sufficient statistics (M-step). Log-likelihood is monotone-increasing
    in expectation; iteration stops when relative gain falls below
    ``em_tol`` or after ``em_iter`` M-steps.

    Validation:

    * ``em_iter=0`` is byte-identical to the pre-EM behaviour
      (regression check — the default API path must not change).
    * Log-likelihood monotone-increasing across iterations.
    * On heavy-missingness VAR(p) data, EM beats single-pass Kalman in
      RMSE on imputed positions.
    * Internal parameter recovery: EM brings the VAR matrix closer to
      the true DGP than the linear-pre-fill initial fit.
    * ``em_converged`` flag and ``em_actual_iter`` counter are surfaced
      via ``df.attrs["preprocess_applied"]``.
    * ``em_iter < 0`` raises a clear error.
    """

    @staticmethod
    def _var1_heavy_missing(rng, T=600, missing_frac=0.5):
        # Cross-effects so the cross-variable correlation carries
        # information when one variable is missing (linear pre-fill
        # ignores this; EM exploits it through the VAR coupling).
        A = np.array([[0.5, 0.3], [0.3, 0.5]])
        Sigma = np.eye(2) * 0.5
        L = np.linalg.cholesky(Sigma)
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + L @ rng.standard_normal(2)
        df_true = pd.DataFrame(
            y, columns=["a", "b"],
            index=pd.date_range("1980-01-01", periods=T, freq="MS"),
        )
        mask = rng.random((T, 2)) < missing_frac
        # Anchor endpoints to keep pre-fill / linear interpolation well-defined
        mask[0] = False
        mask[-1] = False
        df_nan = df_true.mask(mask)
        return df_true, df_nan, mask, A

    def test_em_iter_zero_is_byte_identical(self, var1_data_with_nans):
        """``em_iter=0`` must reproduce the pre-EM single-pass behaviour
        exactly (regression check on the default API)."""
        df_nan = var1_data_with_nans["nan"]
        out_default = impute(df_nan, method="kalman", lags=1)
        out_em0 = impute(df_nan, method="kalman", lags=1, em_iter=0)
        assert np.array_equal(out_default.values, out_em0.values, equal_nan=True)

    def test_em_log_likelihood_monotone_increasing(self):
        """Each EM M-step should raise (or hold) the log-likelihood.

        Drives the internal helpers directly to read the log-lik
        trajectory; the public ``impute`` function only exposes the
        endpoint.
        """
        from lib.preprocess.imputation import (
            _build_companion_state_space, _em_m_step,
        )
        from lib import state_space as ss

        rng = np.random.default_rng(42)
        _, df_nan, _, _ = self._var1_heavy_missing(rng, T=400, missing_frac=0.5)
        n_vars = 2
        lags = 1
        Y_orig = df_nan.values.astype(float)

        pre = df_nan.interpolate("linear").ffill().bfill()
        fit = lib.fit_var(pre, lags=lags, trend="c")
        lag_mats = [np.asarray(A) for A in fit.lag_matrices()]
        Sigma_u = np.asarray(fit.Sigma_u)
        intercept = np.asarray(fit.B_hat)[-1, :]
        init_state = pre.iloc[0].values

        ll_history = []
        for _ in range(15):
            model = _build_companion_state_space(
                lag_mats, intercept, Sigma_u, init_state, n_vars, lags,
            )
            sm = ss.kalman_filter_smooth(model, Y_orig)
            ll_history.append(sm.log_likelihood)
            lag_mats, intercept, Sigma_u = _em_m_step(sm, n_vars, lags)

        # Allow tiny numerical regression (≤ 1e-6) but reject any
        # meaningful drop.
        for k in range(1, len(ll_history)):
            delta = ll_history[k] - ll_history[k - 1]
            assert delta > -1e-6, (
                f"Log-likelihood dropped at EM iter {k}: "
                f"{ll_history[k - 1]:.6f} -> {ll_history[k]:.6f}"
            )
        # And total gain over 15 iterations should be substantial on
        # heavy-missingness data (initial pre-fill is biased).
        assert ll_history[-1] - ll_history[0] > 1.0, (
            f"EM produced no meaningful log-lik improvement "
            f"({ll_history[-1] - ll_history[0]:.4f})"
        )

    def test_em_beats_single_pass_rmse_under_heavy_missingness(self):
        """On 50% MCAR VAR(1) data, EM should improve imputation RMSE
        over the single-pass linearly-pre-filled fit."""
        rng = np.random.default_rng(42)
        df_true, df_nan, mask, _ = self._var1_heavy_missing(
            rng, T=600, missing_frac=0.5,
        )
        out0 = impute(df_nan, method="kalman", lags=1, em_iter=0)
        out_em = impute(
            df_nan, method="kalman", lags=1, em_iter=30, em_tol=1e-7,
        )
        err0 = np.sqrt(np.mean(
            _imputed_errors(out0, df_true, mask, "single-pass") ** 2))
        err_em = np.sqrt(np.mean(
            _imputed_errors(out_em, df_true, mask, "EM") ** 2))
        assert err_em < err0, (
            f"EM RMSE ({err_em:.4f}) did not beat single-pass "
            f"({err0:.4f}) on heavy-missingness VAR(1) data"
        )

    def test_em_internal_parameters_closer_to_truth(self):
        """The whole point of EM is to remove the bias from fitting the
        VAR on a linearly-pre-filled copy. On a known DGP with 70% MCAR
        missingness, the EM's internal parameter estimate should be
        substantially closer to the true VAR matrix than the initial
        pre-fill-based fit."""
        from lib.preprocess.imputation import (
            _build_companion_state_space, _em_m_step,
        )
        from lib import state_space as ss

        rng = np.random.default_rng(0)
        _, df_nan, _, A_true = self._var1_heavy_missing(
            rng, T=800, missing_frac=0.7,
        )
        n_vars = 2
        lags = 1
        Y_orig = df_nan.values.astype(float)

        pre = df_nan.interpolate("linear").ffill().bfill()
        fit = lib.fit_var(pre, lags=lags, trend="c")
        lag_mats = [np.asarray(A) for A in fit.lag_matrices()]
        Sigma_u = np.asarray(fit.Sigma_u)
        intercept = np.asarray(fit.B_hat)[-1, :]
        init_state = pre.iloc[0].values

        err_initial = np.linalg.norm(lag_mats[0] - A_true)
        for _ in range(40):
            model = _build_companion_state_space(
                lag_mats, intercept, Sigma_u, init_state, n_vars, lags,
            )
            sm = ss.kalman_filter_smooth(model, Y_orig)
            lag_mats, intercept, Sigma_u = _em_m_step(sm, n_vars, lags)
        err_em = np.linalg.norm(lag_mats[0] - A_true)
        # On heavy missingness EM cuts parameter error by at least 2x
        assert err_em < err_initial / 2.0, (
            f"EM should halve parameter error vs initial pre-fill fit: "
            f"initial={err_initial:.4f}, EM={err_em:.4f}"
        )

    def test_em_converged_flag_set_on_early_stop(self):
        """With a very loose ``em_tol`` (1e10), the convergence check
        should trip on the FIRST comparison (after 1 M-step)."""
        rng = np.random.default_rng(1)
        _, df_nan, _, _ = self._var1_heavy_missing(
            rng, T=300, missing_frac=0.3,
        )
        # Need attrs to read em info back
        df_nan = df_nan.copy()
        df_nan.attrs = {"preprocess_applied": []}
        out = impute(
            df_nan, method="kalman", lags=1, em_iter=50, em_tol=1e10,
        )
        entry = out.attrs["preprocess_applied"][-1]
        assert entry["em_converged"] is True
        # First convergence check happens at k=1, after 1 M-step
        assert entry["em_actual_iter"] == 1

    def test_em_runs_full_em_iter_when_tol_unreachable(self):
        """With a microscopic ``em_tol``, all ``em_iter`` M-steps should
        run (no early stop)."""
        rng = np.random.default_rng(2)
        _, df_nan, _, _ = self._var1_heavy_missing(
            rng, T=300, missing_frac=0.4,
        )
        df_nan = df_nan.copy()
        df_nan.attrs = {"preprocess_applied": []}
        out = impute(
            df_nan, method="kalman", lags=1, em_iter=5, em_tol=1e-30,
        )
        entry = out.attrs["preprocess_applied"][-1]
        assert entry["em_actual_iter"] == 5
        assert entry["em_converged"] is False

    def test_em_attrs_recorded(self):
        """``preprocess_applied`` entry must record em_iter, em_actual_iter,
        em_converged when the Kalman path runs in EM mode."""
        rng = np.random.default_rng(3)
        _, df_nan, _, _ = self._var1_heavy_missing(
            rng, T=300, missing_frac=0.3,
        )
        df_nan = df_nan.copy()
        df_nan.attrs = {"source": "synth", "preprocess_applied": []}
        out = impute(
            df_nan, method="kalman", lags=2, em_iter=3, em_tol=1e-6,
        )
        entry = out.attrs["preprocess_applied"][-1]
        assert entry["op"] == "impute"
        assert entry["method"] == "kalman"
        assert entry["lags"] == 2
        assert entry["em_iter"] == 3
        assert "em_actual_iter" in entry
        assert "em_converged" in entry
        assert out.attrs["source"] == "synth"

    def test_em_attrs_record_em_iter_zero_path(self):
        """At ``em_iter=0`` the attrs entry records ``em_iter=0`` and
        ``em_actual_iter=0`` — uniform metadata across the Kalman path
        so users can inspect what was actually done."""
        rng = np.random.default_rng(4)
        _, df_nan, _, _ = self._var1_heavy_missing(
            rng, T=200, missing_frac=0.2,
        )
        df_nan = df_nan.copy()
        df_nan.attrs = {"preprocess_applied": []}
        out = impute(df_nan, method="kalman", lags=1, em_iter=0)
        entry = out.attrs["preprocess_applied"][-1]
        assert entry["em_iter"] == 0
        assert entry["em_actual_iter"] == 0
        assert entry["em_converged"] is False

    def test_em_negative_em_iter_raises(self, var1_data_with_nans):
        with pytest.raises(ValueError, match="em_iter"):
            impute(
                var1_data_with_nans["nan"],
                method="kalman", lags=1, em_iter=-1,
            )

    def test_em_observed_values_still_preserved(self):
        """Even in EM mode, observed values must pass through
        bit-identically (only NaN positions are filled)."""
        rng = np.random.default_rng(5)
        _, df_nan, mask, _ = self._var1_heavy_missing(
            rng, T=300, missing_frac=0.4,
        )
        out = impute(
            df_nan, method="kalman", lags=1, em_iter=10, em_tol=1e-7,
        )
        observed_mask = ~mask
        for col_idx in range(df_nan.shape[1]):
            col = df_nan.columns[col_idx]
            obs_idx = observed_mask[:, col_idx]
            assert np.allclose(
                out[col].values[obs_idx],
                df_nan[col].values[obs_idx],
                atol=1e-12,
            ), f"EM altered an observed value in column {col}"


class TestAggregationCompleteness:
    """Acceptance check: missing or partial bins must not look like
    complete economic observations."""

    def test_all_missing_and_partial_bins_are_nan(self):
        idx = pd.date_range("2000-01-01", periods=8, freq="MS")
        s = pd.Series([10, 10, 10, np.nan, np.nan, np.nan, 10, 10],
                      index=idx, name="flow")
        out = lib.preprocess.aggregate(s, "QS", "sum")
        assert out.iloc[0] == 30.0
        assert np.isnan(out.iloc[1])          # all-missing quarter, NOT 0
        assert np.isnan(out.iloc[2])          # two-month quarter, NOT a total

    def test_allow_partial_keeps_partial_but_not_empty(self):
        idx = pd.date_range("2000-01-01", periods=8, freq="MS")
        s = pd.Series([10, 10, 10, np.nan, np.nan, np.nan, 10, 10], index=idx)
        out = lib.preprocess.aggregate(s, "QS", "sum", allow_partial=True)
        assert out.iloc[0] == 30.0
        assert np.isnan(out.iloc[1])          # still no observations
        assert out.iloc[2] == 20.0            # explicit partial total

    def test_end_does_not_substitute_earlier_month(self):
        idx = pd.date_range("2000-01-01", periods=3, freq="MS")
        s = pd.Series([100.0, 110.0, np.nan], index=idx)
        out = lib.preprocess.aggregate(s, "QS", "end")
        assert np.isnan(out.iloc[0])          # scheduled endpoint missing
        out_partial = lib.preprocess.aggregate(s, "QS", "end",
                                               allow_partial=True)
        assert out_partial.iloc[0] == 110.0   # documented last-AVAILABLE

    def test_complete_bins_unchanged(self):
        idx = pd.date_range("2000-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.],
                           "b": [2, 4, 6, 8, 10, 12.]}, index=idx)
        out = lib.preprocess.aggregate(df, "QS", {"a": "mean", "b": "end"})
        assert out["a"].tolist() == [2.0, 5.0]
        assert out["b"].tolist() == [6.0, 12.0]

    def test_truncated_boundary_quarter_is_incomplete(self):
        idx = pd.date_range("2000-01-01", periods=8, freq="MS")  # Jan..Aug
        s = pd.Series(np.ones(8), index=idx)
        out = lib.preprocess.aggregate(s, "QS", "sum")
        assert out.iloc[-1:].isna().all()     # Q3 has only Jul-Aug rows


class TestNominaliseIndexPairing:
    def test_cpi_round_trips_a_cpi_deflated_series(self):
        """Reversing a CPI deflation requires CPI — a different price index
        leaves a relative-price component in the result (the M1REAL
        × GDP-deflator case)."""
        idx = pd.date_range("2000-01-01", periods=2, freq="QS")
        nominal = pd.Series([100.0, 100.0], index=idx)
        cpi = pd.Series([100.0, 110.0], index=idx)
        gdp_deflator = pd.Series([100.0, 105.0], index=idx)
        real = lib.preprocess.deflate(nominal, cpi)
        back_right = lib.preprocess.nominalise(real, cpi)
        back_wrong = lib.preprocess.nominalise(real, gdp_deflator)
        assert np.allclose(back_right.values, nominal.values)
        assert not np.allclose(back_wrong.values, nominal.values)
        assert abs(back_wrong.iloc[1] - 100.0 * 105.0 / 110.0) < 1e-9
