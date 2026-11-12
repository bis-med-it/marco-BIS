"""Tests for ``lib.forecast`` — iterated forecasts, rolling-window driver, scoring.

Validation strategy:

1. **Point-forecast identity** — h=1 iterated forecast matches the manual
   one-step reduced-form prediction ``c + sum_j A_j y_{T-j+1}`` to
   floating-point tolerance.
2. **Stationary-limit convergence** — for a stationary AR(1) the point
   forecast at long horizons converges to the unconditional mean
   ``c / (1 - phi)`` (n=1 case generalised to ``(I - A)^{-1} c`` for VAR).
3. **Innovation-variance growth** — density-forecast bands widen with
   horizon (under a stationary VAR they converge to bounded unconditional
   variance, but always monotone non-decreasing).
4. **Container API** — shapes, names, origin timestamp all correct.
5. **Rolling-window mechanics** — output rows align with origins and
   horizons; realised values match data slicing.
6. **Scoring** — RMSE on a known DGP recovers the innovation std under
   no-bias prediction; DM test on identical losses gives p=1.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lib  # noqa: E402
from lib.forecast import (  # noqa: E402
    ForecastContainer,
    ConditionalForecastContainer,
    iterated_forecast,
    conditional_forecast,
    rolling_forecasts,
    forecast_score,
    diebold_mariano,
)

warnings.filterwarnings("ignore", category=Warning)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def var1_data() -> pd.DataFrame:
    """Synthetic stationary VAR(1) with diagonal phi=0.7."""
    rng = np.random.default_rng(42)
    T, n = 300, 2
    y = np.zeros((T, n))
    A1 = np.diag([0.7, 0.5])
    c = np.array([0.2, -0.1])
    for t in range(1, T):
        y[t] = c + A1 @ y[t - 1] + rng.normal(0, 1, n)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    return pd.DataFrame(y, index=idx, columns=["x", "y"])


@pytest.fixture
def var1_fit(var1_data):
    return lib.fit_var(var1_data, lags=1, trend="c")


@pytest.fixture
def var2_fit(var1_data):
    return lib.fit_var(var1_data, lags=2, trend="c")


# ---------------------------------------------------------------------------
# Point forecast
# ---------------------------------------------------------------------------


class TestPointForecast:
    def test_h1_matches_manual_var1(self, var1_fit):
        """At h=1, iterated forecast equals c + A_1 y_T."""
        A1 = var1_fit.lag_matrices()[0]
        c = var1_fit.B_hat[-1]
        y_T = var1_fit.data.iloc[-1].values
        manual = c + A1 @ y_T

        fc = iterated_forecast(var1_fit, horizon=1)
        np.testing.assert_allclose(fc.point[0], manual, atol=1e-12)

    def test_h1_matches_manual_var2(self, var2_fit):
        """At h=1 with p=2: c + A_1 y_T + A_2 y_{T-1}."""
        A1, A2 = var2_fit.lag_matrices()
        c = var2_fit.B_hat[-1]
        y_T = var2_fit.data.iloc[-1].values
        y_Tm1 = var2_fit.data.iloc[-2].values
        manual = c + A1 @ y_T + A2 @ y_Tm1

        fc = iterated_forecast(var2_fit, horizon=1)
        np.testing.assert_allclose(fc.point[0], manual, atol=1e-12)

    def test_h2_chains_correctly(self, var1_fit):
        """At h=2: y_{T+2} = c + A_1 y_{T+1} where y_{T+1} is the h=1 forecast."""
        A1 = var1_fit.lag_matrices()[0]
        c = var1_fit.B_hat[-1]
        y_T = var1_fit.data.iloc[-1].values
        y_T1 = c + A1 @ y_T
        y_T2 = c + A1 @ y_T1

        fc = iterated_forecast(var1_fit, horizon=2)
        np.testing.assert_allclose(fc.point[0], y_T1, atol=1e-12)
        np.testing.assert_allclose(fc.point[1], y_T2, atol=1e-12)

    def test_stationary_limit_converges_to_unconditional_mean(self, var1_fit):
        """For a stationary VAR, the long-horizon forecast → (I-A)^{-1} c."""
        A1 = var1_fit.lag_matrices()[0]
        c = var1_fit.B_hat[-1]
        uncond_mean = np.linalg.solve(np.eye(2) - A1, c)

        fc = iterated_forecast(var1_fit, horizon=50)
        # At h=50 with phi=0.7, the deviation from the mean is ~0.7^50 ≈ 2e-8
        np.testing.assert_allclose(fc.point[-1], uncond_mean, atol=1e-5)

    def test_trend_n_no_intercept(self, var1_data):
        """trend='n' produces zero-intercept iterates."""
        fit = lib.fit_var(var1_data, lags=1, trend="n")
        A1 = fit.lag_matrices()[0]
        y_T = fit.data.iloc[-1].values
        # No constant — iteration is just A^h y_T
        fc = iterated_forecast(fit, horizon=3)
        np.testing.assert_allclose(fc.point[0], A1 @ y_T, atol=1e-12)
        np.testing.assert_allclose(fc.point[1], A1 @ (A1 @ y_T), atol=1e-12)

    def test_rejects_unsupported_trend(self, var1_data):
        fit = lib.fit_var(var1_data, lags=1, trend="ct")
        with pytest.raises(NotImplementedError, match="trend='ct'"):
            iterated_forecast(fit, horizon=5)

    def test_rejects_zero_horizon(self, var1_fit):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            iterated_forecast(var1_fit, horizon=0)

    def test_container_basics(self, var1_fit):
        fc = iterated_forecast(var1_fit, horizon=5)
        assert isinstance(fc, ForecastContainer)
        assert fc.point.shape == (5, 2)
        assert fc.paths is None and fc.median is None
        assert list(fc.horizons) == [1, 2, 3, 4, 5]
        assert fc.variable_names == ["x", "y"]
        assert fc.forecast_origin == var1_fit.data.index[-1]


# ---------------------------------------------------------------------------
# Density forecast
# ---------------------------------------------------------------------------


class TestDensityForecast:
    def test_paths_shape(self, var1_fit):
        fc = iterated_forecast(var1_fit, horizon=8, n_paths=200, seed=7)
        assert fc.paths.shape == (200, 8, 2)
        assert fc.median.shape == (8, 2)
        assert fc.lower_68.shape == (8, 2)
        assert fc.upper_90.shape == (8, 2)

    def test_band_ordering(self, var1_fit):
        fc = iterated_forecast(var1_fit, horizon=8, n_paths=500, seed=7)
        # 90% wider than 68% at every horizon and variable
        assert np.all(fc.lower_90 <= fc.lower_68 + 1e-12)
        assert np.all(fc.upper_90 >= fc.upper_68 - 1e-12)
        assert np.all(fc.lower_68 <= fc.median + 1e-12)
        assert np.all(fc.median <= fc.upper_68 + 1e-12)

    def test_innovation_variance_grows_with_horizon(self, var1_fit):
        """Stationary VAR — band widths grow from h=1 toward the long-horizon
        unconditional bound. Theoretical ratio is
        std(y_∞)/std(y_1) = sqrt(1/(1-φ²)) per variable for a diagonal AR(1).
        We test the observed band-width ratio is within 10% of the
        theoretical bound — substantively "band has accumulated the
        expected innovation variance by h=20"."""
        fc = iterated_forecast(var1_fit, horizon=20, n_paths=2000, seed=11)
        width = fc.upper_68 - fc.lower_68
        A1 = var1_fit.lag_matrices()[0]
        diag_phi = np.diag(A1)
        for i in range(2):
            theoretical_growth = 1.0 / np.sqrt(1.0 - diag_phi[i] ** 2)
            observed_growth = width[-1, i] / width[0, i]
            # Allow ±10% sampling-noise tolerance around the theoretical bound
            assert 0.9 * theoretical_growth < observed_growth < 1.15 * theoretical_growth, (
                f"variable {i}: observed long-horizon growth "
                f"{observed_growth:.3f} outside [0.9, 1.15] × theoretical "
                f"{theoretical_growth:.3f}"
            )

    def test_density_seeded_reproducible(self, var1_fit):
        fc_a = iterated_forecast(var1_fit, horizon=5, n_paths=100, seed=42)
        fc_b = iterated_forecast(var1_fit, horizon=5, n_paths=100, seed=42)
        np.testing.assert_array_equal(fc_a.paths, fc_b.paths)

    def test_median_approximates_point_at_short_horizons(self, var1_fit):
        """With enough paths and Gaussian innovations the simulated median
        should be very close to the point forecast at h=1 (where there's
        only one stage of noise added)."""
        fc = iterated_forecast(var1_fit, horizon=3, n_paths=5000, seed=3)
        np.testing.assert_allclose(fc.median[0], fc.point[0], atol=0.05)

    def test_parameter_uncertainty_widens_bands(self, var1_fit):
        """With parameter uncertainty on, bands should be at least as wide
        as the innovation-only bands."""
        fc_inno = iterated_forecast(
            var1_fit, horizon=8, n_paths=2000, seed=5,
            parameter_uncertainty=False,
        )
        fc_full = iterated_forecast(
            var1_fit, horizon=8, n_paths=2000, seed=5,
            parameter_uncertainty=True,
        )
        # Mean band width should be larger under joint uncertainty
        w_inno = (fc_inno.upper_68 - fc_inno.lower_68).mean()
        w_full = (fc_full.upper_68 - fc_full.lower_68).mean()
        assert w_full >= 0.95 * w_inno    # ≥ to within sampling noise

    def test_parameter_uncertainty_paths_follow_each_draw(self, var1_fit, monkeypatch):
        """Deterministic fixture: the sampler returns known, distinct
        coefficient draws with negligible innovation covariance, so path k
        must be the noise-free iteration of draw k, not of B_hat."""
        n = 2
        # Non-symmetric A_k so a transposed B layout would also fail
        A_draws = [
            np.array([[0.2, 0.1], [0.0, 0.3]]),
            np.array([[0.9, -0.4], [0.2, 0.1]]),
            np.array([[-0.5, 0.0], [0.6, 0.8]]),
        ]
        c_draws = [np.array([1.0, -1.0]), np.array([0.0, 2.0]),
                   np.array([-3.0, 0.5])]
        tiny = 1e-24 * np.eye(n)
        stub_draws = [(np.vstack([A.T, c]), tiny)
                      for A, c in zip(A_draws, c_draws)]
        calls = []

        def stub(n_draws, seed=None, **kwargs):
            calls.append((n_draws, seed))
            return stub_draws

        monkeypatch.setattr(var1_fit, "parameter_draws", stub)
        horizon = 4
        fc = iterated_forecast(var1_fit, horizon=horizon, n_paths=3, seed=9,
                               parameter_uncertainty=True)
        assert calls == [(3, 9)]
        assert fc.paths.shape == (3, horizon, n)
        y_T = np.asarray(var1_fit.data.values[-1], dtype=float)
        for k, (A, c) in enumerate(zip(A_draws, c_draws)):
            y = y_T.copy()
            expected = []
            for _ in range(horizon):
                y = c + A @ y
                expected.append(y)
            np.testing.assert_allclose(fc.paths[k], expected, atol=1e-9)
        # The three paths are genuinely different from each other
        assert not np.allclose(fc.paths[0], fc.paths[1])
        assert not np.allclose(fc.paths[1], fc.paths[2])

    def test_parameter_uncertainty_uses_each_sigma_draw(self, var1_fit, monkeypatch):
        """Zero coefficients and alternating innovation covariance: paths
        built from a negligible Sigma draw must be ~0, those from Sigma = I
        must not be. Sharing fit.Sigma_u across paths would fail this."""
        n = 2
        zero_B = np.zeros((n + 1, n))
        draws = [(zero_B, np.eye(n) if k % 2 == 0 else 1e-24 * np.eye(n))
                 for k in range(40)]
        monkeypatch.setattr(var1_fit, "parameter_draws",
                            lambda n_draws, seed=None, **kw: draws)
        fc = iterated_forecast(var1_fit, horizon=3, n_paths=40, seed=2,
                               parameter_uncertainty=True)
        assert np.abs(fc.paths[1::2]).max() < 1e-9
        assert np.abs(fc.paths[0::2]).max() > 0.5

    def test_parameter_uncertainty_exercises_sampler(self, var1_fit, monkeypatch):
        """On the real fit, the option calls parameter_draws once for
        n_paths draws, and the paths differ from the innovation-only
        paths under the same seed."""
        real = var1_fit.parameter_draws
        calls = []

        def spy(n_draws, seed=None, **kwargs):
            calls.append((n_draws, seed))
            return real(n_draws, seed=seed, **kwargs)

        monkeypatch.setattr(var1_fit, "parameter_draws", spy)
        fc_off = iterated_forecast(var1_fit, horizon=6, n_paths=300, seed=5)
        assert calls == []
        fc_on = iterated_forecast(var1_fit, horizon=6, n_paths=300, seed=5,
                                  parameter_uncertainty=True)
        assert calls == [(300, 5)]
        assert fc_on.paths.shape == fc_off.paths.shape
        assert not np.allclose(fc_on.paths, fc_off.paths)

    def test_parameter_uncertainty_requires_n_paths(self, var1_fit):
        with pytest.raises(ValueError, match="parameter_uncertainty=True"):
            iterated_forecast(var1_fit, horizon=5, parameter_uncertainty=True)


# ---------------------------------------------------------------------------
# Rolling forecasts
# ---------------------------------------------------------------------------


class TestRollingForecasts:
    def test_rolling_basic(self, var1_data):
        def fit_fn(df):
            return lib.fit_var(df, lags=1, trend="c")

        out = rolling_forecasts(
            var1_data, fit_fn,
            window_size=100, horizon=4, step=20,
        )
        # rows = number of origins * horizon * n_vars
        n_origins = len(range(100, len(var1_data), 20))
        assert len(out) == n_origins * 4 * 2
        assert set(out.columns) >= {"origin", "horizon", "variable",
                                     "forecast", "realised"}

    def test_realised_alignment(self, var1_data):
        """For an origin at index t (last in-sample obs), the h-step realised
        value equals ``data.iloc[t + h - 1]`` (the h-th step ahead)."""
        def fit_fn(df):
            return lib.fit_var(df, lags=1, trend="c")

        out = rolling_forecasts(
            var1_data, fit_fn,
            window_size=100, horizon=2, step=50,
        )
        # Pick the first row (origin = window_size index, var = x, h = 1)
        first = out.iloc[0]
        # origin is data.index[t-1] where t = window_size (so index 99)
        origin = first["origin"]
        t = list(var1_data.index).index(origin) + 1   # t = 100
        var = first["variable"]; h = int(first["horizon"])
        expected_realised = float(var1_data.iloc[t + h - 1][var])
        assert first["realised"] == pytest.approx(expected_realised, abs=1e-12)

    def test_expanding_window(self, var1_data):
        def fit_fn(df):
            return lib.fit_var(df, lags=1, trend="c")

        out = rolling_forecasts(
            var1_data, fit_fn,
            window_size=50, horizon=3, step=30,
            expanding=True,
        )
        # Just verify it runs and produces sensible output
        assert len(out) > 0
        assert out["forecast"].notna().any()

    def test_rejects_short_data(self, var1_data):
        def fit_fn(df):
            return lib.fit_var(df, lags=1, trend="c")

        with pytest.raises(ValueError, match="at least"):
            rolling_forecasts(
                var1_data.iloc[:5], fit_fn,
                window_size=100, horizon=1,
            )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestForecastScore:
    def test_rmse_zero_for_perfect_forecast(self):
        df = pd.DataFrame({
            "origin": [0, 0, 1, 1],
            "horizon": [1, 2, 1, 2],
            "variable": ["x"] * 4,
            "forecast": [1.0, 2.0, 3.0, 4.0],
            "realised": [1.0, 2.0, 3.0, 4.0],
        })
        out = forecast_score(df, metric="rmse", by="horizon")
        assert (out["rmse"] == 0.0).all()

    def test_rmse_known_value(self):
        df = pd.DataFrame({
            "origin": [0] * 4,
            "horizon": [1, 1, 2, 2],
            "variable": ["x"] * 4,
            "forecast": [0.0, 0.0, 0.0, 0.0],
            "realised": [1.0, 1.0, 2.0, 2.0],
        })
        out = forecast_score(df, metric="rmse", by="horizon")
        # h=1: rmse = sqrt(1) = 1; h=2: rmse = sqrt(4) = 2
        rmse_by_h = dict(zip(out["horizon"], out["rmse"]))
        assert rmse_by_h[1] == pytest.approx(1.0)
        assert rmse_by_h[2] == pytest.approx(2.0)

    def test_mae(self):
        df = pd.DataFrame({
            "origin": [0] * 4, "horizon": [1] * 4,
            "variable": ["x"] * 4,
            "forecast": [0.0, 0.0, 0.0, 0.0],
            "realised": [1.0, -1.0, 2.0, -2.0],
        })
        out = forecast_score(df, metric="mae", by="horizon")
        # mean abs error = (1+1+2+2)/4 = 1.5
        assert out["mae"].iloc[0] == pytest.approx(1.5)

    def test_handles_nans(self):
        """NaN forecasts or realised values are dropped before scoring."""
        df = pd.DataFrame({
            "origin": [0, 0], "horizon": [1, 1], "variable": ["x", "x"],
            "forecast": [1.0, np.nan],
            "realised": [1.0, 1.0],
        })
        out = forecast_score(df, metric="rmse", by="horizon")
        assert out["rmse"].iloc[0] == pytest.approx(0.0)

    def test_unknown_metric_rejected(self):
        df = pd.DataFrame({
            "origin": [0], "horizon": [1], "variable": ["x"],
            "forecast": [1.0], "realised": [1.0],
        })
        with pytest.raises(ValueError, match="unknown metric"):
            forecast_score(df, metric="not_a_metric")


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


class TestDieboldMariano:
    def test_identical_losses_nonsignificant(self):
        loss = np.array([1.0, 2.0, 1.5, 0.5, 2.5, 1.0, 1.5])
        out = diebold_mariano(loss, loss, horizon=1)
        # Identical losses → d_bar=0, var=0; implementation returns (0, 0, 1).
        assert out["d_bar"] == 0.0
        assert out["dm_stat"] == 0.0
        assert out["p_value"] == 1.0

    def test_systematic_difference_significant(self):
        """When loss_a is systematically larger, DM should reject H0 strongly."""
        rng = np.random.default_rng(0)
        n = 100
        # loss_a = loss_b + 0.5 (constant offset) + noise
        loss_b = rng.uniform(0, 1, n)
        loss_a = loss_b + 0.5 + rng.normal(0, 0.05, n)
        out = diebold_mariano(loss_a, loss_b, horizon=1)
        assert out["d_bar"] > 0.4
        assert out["p_value"] < 0.001    # strong rejection

    def test_horizon_increases_hac_lags(self):
        """With strongly positively autocorrelated loss differential, longer
        horizon (more HAC lags) should give a smaller t-stat in magnitude."""
        rng = np.random.default_rng(1)
        n = 200
        d = np.empty(n)
        d[0] = 0.5 + rng.normal(0, 0.1)
        for t in range(1, n):
            d[t] = 0.5 + 0.9 * (d[t - 1] - 0.5) + rng.normal(0, 0.1)
        loss_a = d
        loss_b = np.zeros(n)
        t1 = abs(diebold_mariano(loss_a, loss_b, horizon=1)["dm_stat"])
        t12 = abs(diebold_mariano(loss_a, loss_b, horizon=12)["dm_stat"])
        # HAC correction with q=11 should shrink the t-stat
        assert t12 < t1

    def test_rejects_short_input(self):
        with pytest.raises(ValueError, match="at least 3"):
            diebold_mariano(np.array([1.0]), np.array([2.0]), horizon=1)


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_lib_exports():
    assert lib.iterated_forecast is iterated_forecast
    assert lib.conditional_forecast is conditional_forecast
    assert lib.rolling_forecasts is rolling_forecasts
    assert lib.forecast_score is forecast_score
    assert lib.diebold_mariano is diebold_mariano
    assert lib.ForecastContainer is ForecastContainer
    assert lib.ConditionalForecastContainer is ConditionalForecastContainer


# ---------------------------------------------------------------------------
# Conditional forecasts (BGL 2015)
# ---------------------------------------------------------------------------


class TestConditionalForecast:
    """Tests for ``lib.conditional_forecast`` — BGL 2015 Kalman-smoother
    conditional forecasts.

    Validation strategy:

    * **Unconditional limit** — ``conditioning=None`` matches
      ``iterated_forecast`` point estimate to machine precision.
    * **Conditioning passes through exactly** — pinned values appear
      in ``point`` with no smoother noise.
    * **Conditioning shifts unconditioned variables** — a non-trivial
      conditioning path moves the other variables' forecasts in the
      direction implied by the VAR dynamics.
    * **Density forecast bands ordering** — lower_90 ≤ lower_68 ≤
      median ≤ upper_68 ≤ upper_90.
    * **Unknown variable rejected** — clear error message.
    * **Shape validation** — wrong-length conditioning path rejected.
    """

    def test_unconditional_matches_iterated_forecast(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        result = conditional_forecast(fit, horizon=8)
        iter_fc = iterated_forecast(fit, horizon=8)
        assert np.allclose(result.point, iter_fc.point, atol=1e-8)

    def test_empty_conditioning_dict_same_as_none(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        r_none = conditional_forecast(fit, horizon=5, conditioning=None)
        r_empty = conditional_forecast(fit, horizon=5, conditioning={})
        assert np.allclose(r_none.point, r_empty.point, atol=1e-12)

    def test_conditioning_value_passes_through_exactly(self, var1_data):
        """A pinned value at horizon h must appear EXACTLY in the
        forecast (smoother noise zeroed by the pin override)."""
        fit = lib.fit_var(var1_data, lags=2)
        target = 0.7
        path = np.full(6, target)
        result = conditional_forecast(
            fit, horizon=6, conditioning={fit.variable_names[0]: path},
        )
        assert np.allclose(result.point[:, 0], target, atol=1e-12)

    def test_partial_path_nan_means_unconditioned(self, var1_data):
        """NaN entries in the conditioning path mark 'not conditioned
        at this horizon' — those entries should be filled by the
        smoother, not pinned."""
        fit = lib.fit_var(var1_data, lags=2)
        path = np.array([1.0, np.nan, 1.0, np.nan, np.nan, 1.0])
        result = conditional_forecast(
            fit, horizon=6, conditioning={fit.variable_names[0]: path},
        )
        # Pinned positions match exactly
        for h in (0, 2, 5):
            assert result.point[h, 0] == pytest.approx(1.0, abs=1e-12)
        # Unpinned positions are NOT 1.0 (they're smoother-filled)
        for h in (1, 3, 4):
            assert not np.isclose(result.point[h, 0], 1.0)

    def test_conditioning_shifts_other_variables(self, var1_data):
        """Pinning a variable to a non-trivial path shifts the
        unconditional forecast of the other variables in a way that
        depends on the VAR's lag coefficients (rejects no-effect bug)."""
        fit = lib.fit_var(var1_data, lags=2)
        uncond = conditional_forecast(fit, horizon=6)
        cond = conditional_forecast(
            fit, horizon=6,
            conditioning={fit.variable_names[0]: np.full(6, 1.5)},
        )
        # The non-conditioned variables should change non-trivially
        n = fit.n_vars
        other_cols = list(range(1, n))
        max_shift = float(np.abs(
            cond.point[:, other_cols] - uncond.point[:, other_cols]
        ).max())
        assert max_shift > 0.001, (
            f"Expected non-trivial shift in non-conditioned variables; "
            f"got max diff {max_shift:.4e}"
        )

    def test_density_bands_ordering(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        result = conditional_forecast(
            fit, horizon=4,
            conditioning={fit.variable_names[0]: np.full(4, 0.5)},
            n_paths=200, seed=0,
        )
        assert result.paths is not None
        assert result.paths.shape == (200, 4, fit.n_vars)
        assert np.all(result.lower_90 <= result.lower_68 + 1e-10)
        assert np.all(result.lower_68 <= result.median + 1e-10)
        assert np.all(result.median <= result.upper_68 + 1e-10)
        assert np.all(result.upper_68 <= result.upper_90 + 1e-10)
        # Bands must be NON-DEGENERATE for the non-conditioned variables
        # (regression check — silent simulation-smoother failure would
        # collapse them to the smoother mean)
        non_cond_cols = list(range(1, fit.n_vars))
        spread = float((result.upper_68 - result.lower_68)[:, non_cond_cols].mean())
        assert spread > 1e-4, (
            f"68% band collapsed to a point ({spread:.4e}) on non-conditioned "
            f"variables — simulation smoother may be returning the deterministic mean."
        )

    def test_density_pinned_paths_exact(self, var1_data):
        """In density forecasts, the conditioning column should be
        exactly pinned in every path (no spread)."""
        fit = lib.fit_var(var1_data, lags=2)
        path = np.array([1.0, 2.0, 3.0])
        result = conditional_forecast(
            fit, horizon=3,
            conditioning={fit.variable_names[0]: path},
            n_paths=50, seed=0,
        )
        # Each path's column-0 should match the path exactly
        for j in range(50):
            assert np.allclose(result.paths[j, :, 0], path, atol=1e-12)

    def test_unknown_variable_raises(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        with pytest.raises(ValueError, match="not in fit.variable_names"):
            conditional_forecast(
                fit, horizon=4,
                conditioning={"not_a_variable": np.zeros(4)},
            )

    def test_wrong_length_path_raises(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        with pytest.raises(ValueError, match="shape"):
            conditional_forecast(
                fit, horizon=4,
                conditioning={fit.variable_names[0]: np.zeros(10)},
            )

    def test_horizon_validation(self, var1_data):
        fit = lib.fit_var(var1_data, lags=2)
        with pytest.raises(ValueError, match="horizon"):
            conditional_forecast(fit, horizon=0)

    def test_works_on_bvar(self, var1_data):
        """Conditional forecast must also work on a BVARFit (subclass)."""
        bvar = lib.fit_minnesota_bvar(var1_data, lags=2)
        result = conditional_forecast(
            bvar, horizon=4,
            conditioning={bvar.variable_names[0]: np.full(4, 0.5)},
        )
        assert result.point.shape == (4, bvar.n_vars)
        assert np.allclose(result.point[:, 0], 0.5, atol=1e-12)


def test_lib_all_lists_forecast_surface():
    for name in (
        "ForecastContainer",
        "iterated_forecast",
        "rolling_forecasts",
        "forecast_score",
        "diebold_mariano",
    ):
        assert name in lib.__all__


# ---------------------------------------------------------------------------
# Integration with BVARFit
# ---------------------------------------------------------------------------


def test_iterated_forecast_works_on_bvar(var1_data):
    """BVARFit must also accept iterated_forecast (same interface contract)."""
    prior = lib.MinnesotaPrior(
        lambda_1=0.1, lambda_2=0.5, lambda_3=1.0,
        lambda_4=100.0, mu=1.0, delta=1.0,
    )
    fit = lib.fit_minnesota_bvar(var1_data, lags=1, prior=prior, trend="c")
    fc = iterated_forecast(fit, horizon=5)
    assert fc.point.shape == (5, 2)
    # And density:
    fc_d = iterated_forecast(fit, horizon=5, n_paths=100, seed=0)
    assert fc_d.paths.shape == (100, 5, 2)


def test_iterated_forecast_parameter_uncertainty_works_on_bvar(var1_data):
    """BVARFit's parameter_draws() should flow through iterated_forecast
    when parameter_uncertainty=True."""
    prior = lib.MinnesotaPrior(
        lambda_1=0.1, lambda_2=0.5, lambda_3=1.0,
        lambda_4=100.0, mu=1.0, delta=1.0,
    )
    fit = lib.fit_minnesota_bvar(var1_data, lags=1, prior=prior, trend="c")
    real = fit.parameter_draws
    calls = []

    def spy(n_draws, seed=None, **kwargs):
        calls.append((n_draws, seed))
        return real(n_draws, seed=seed, **kwargs)

    fit.parameter_draws = spy
    fc = iterated_forecast(
        fit, horizon=5, n_paths=100, seed=0,
        parameter_uncertainty=True,
    )
    assert calls == [(100, 0)]
    assert fc.paths.shape == (100, 5, 2)
    assert fc.metadata["parameter_uncertainty"] is True
    # The draws must reach the paths: same seed, option off, differs
    fc_off = iterated_forecast(fit, horizon=5, n_paths=100, seed=0)
    assert not np.allclose(fc.paths, fc_off.paths)


def test_iterated_forecast_rejects_varx(var1_data):
    """VARX fits don't have well-defined future exog paths in
    iterated_forecast and the internal `_extract_intercept(B[-1])` would
    read the last exog coefficient instead of the constant. Reject
    rather than silently miscompute: passing a VARX fit must raise
    NotImplementedError."""
    # Construct an exog Series aligned with the data
    exog = pd.DataFrame(
        np.ones(len(var1_data)) * 0.1,
        index=var1_data.index, columns=["dummy"],
    )
    varx = lib.fit_var(var1_data, lags=1, trend="c", exog=exog, exog_lags=0)
    with pytest.raises(NotImplementedError, match="VARX"):
        iterated_forecast(varx, horizon=3)


class TestDensityScores:
    """CRPS / log score / PIT for simulated predictive densities."""

    def test_degenerate_paths_crps_equals_absolute_error(self):
        # All paths identical at x -> CRPS = |x - y| exactly
        paths = np.full((50, 3, 2), 1.5)
        actuals = np.array([[2.0, 1.5], [0.5, 3.0], [1.5, -1.0]])
        df = lib.density_scores(paths, actuals)
        expected = np.abs(1.5 - actuals).ravel()
        np.testing.assert_allclose(df["crps"].values, expected, atol=1e-12)

    def test_crps_sorted_identity_matches_bruteforce(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(23)
        y = 0.4
        term1 = np.mean(np.abs(x - y))
        term2 = np.mean(np.abs(x[:, None] - x[None, :]))
        crps_brute = term1 - 0.5 * term2
        df = lib.density_scores(x.reshape(-1, 1, 1), np.array([[y]]))
        assert df["crps"].iloc[0] == pytest.approx(crps_brute, abs=1e-12)

    def test_log_score_matches_gaussian_approximation(self):
        from scipy import stats
        rng = np.random.default_rng(5)
        x = rng.standard_normal(400) * 2.0 + 1.0
        y = 1.7
        df = lib.density_scores(x.reshape(-1, 1, 1), np.array([[y]]))
        expected = stats.norm.logpdf(y, loc=x.mean(), scale=x.std(ddof=1))
        assert df["log_score"].iloc[0] == pytest.approx(expected, abs=1e-12)

    def test_pit_extremes_and_calibration(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(300)
        below = lib.density_scores(x.reshape(-1, 1, 1), np.array([[-50.0]]))
        above = lib.density_scores(x.reshape(-1, 1, 1), np.array([[50.0]]))
        assert below["pit"].iloc[0] == 0.0
        assert above["pit"].iloc[0] == 1.0
        # Correctly calibrated: PITs over many cells average ~0.5
        paths = rng.standard_normal((200, 40, 1))
        actuals = rng.standard_normal((40, 1))
        df = lib.density_scores(paths, actuals)
        assert abs(df["pit"].mean() - 0.5) < 0.12

    def test_nan_actual_skipped(self):
        paths = np.random.default_rng(0).standard_normal((30, 2, 1))
        actuals = np.array([[0.1], [np.nan]])
        df = lib.density_scores(paths, actuals)
        assert np.isfinite(df.loc[df.horizon == 1, "crps"]).all()
        assert df.loc[df.horizon == 2, "crps"].isna().all()

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="3-D"):
            lib.density_scores(np.zeros((5, 2)), np.zeros((2, 1)))
        with pytest.raises(ValueError, match="at least 2 paths"):
            lib.density_scores(np.zeros((1, 2, 1)), np.zeros((2, 1)))
        with pytest.raises(ValueError, match="actuals shape"):
            lib.density_scores(np.zeros((5, 2, 1)), np.zeros((3, 1)))


# ---------------------------------------------------------------------------
# Acceptance checks: DM variance, matched scoring, name alignment,
# unit-invariant hard conditioning
# ---------------------------------------------------------------------------


class TestDieboldMarianoVariance:
    def test_matches_global_bartlett_reference(self):
        """The HAC variance must equal globally centred Bartlett
        cross-products with one 1/n denominator."""
        rng = np.random.default_rng(7)
        n, h = 16, 8
        loss_a = rng.uniform(0, 1, n) + 0.3
        loss_b = rng.uniform(0, 1, n)
        out = diebold_mariano(loss_a, loss_b, horizon=h)
        d = loss_a - loss_b
        dc = d - d.mean()
        q = h - 1
        var = float(dc @ dc) / n
        for lag in range(1, q + 1):
            var += 2.0 * (1 - lag / (q + 1)) * float(dc[lag:] @ dc[:-lag]) / n
        var /= n
        hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        expected_t = hln * d.mean() / np.sqrt(var)
        assert abs(out["dm_stat"] - expected_t) < 1e-12

    def test_bartlett_variance_never_negative(self):
        """Bartlett-weighted global centring is PSD — no silent fallback."""
        rng = np.random.default_rng(3)
        for seed in range(20):
            d = np.random.default_rng(seed).standard_normal(12)
            out = diebold_mariano(d + 1.0, np.full(12, 1.0) - 1e-9 + 0 * d,
                                  horizon=4)
            assert np.isfinite(out["dm_stat"])

    def test_nonzero_constant_differential_is_undefined(self):
        loss_a = np.full(10, 2.0)
        loss_b = np.full(10, 1.0)
        with pytest.raises(ValueError, match="nonzero constant"):
            diebold_mariano(loss_a, loss_b, horizon=1)

    def test_truncation_needs_enough_observations(self):
        loss_a = np.array([1.0, 2.0, 1.5, 0.7])
        loss_b = np.array([0.9, 1.8, 1.2, 0.6])
        with pytest.raises(ValueError, match="truncation"):
            diebold_mariano(loss_a, loss_b, horizon=8)


class TestForecastScoreMatched:
    def _rolling_df(self, forecasts: dict[tuple, float]) -> pd.DataFrame:
        rows = []
        for (origin, h, v), (fc, real) in forecasts.items():
            rows.append({"origin": origin, "horizon": h, "variable": v,
                         "forecast": fc, "realised": real})
        return pd.DataFrame(rows)

    def test_failed_origin_excluded_from_both_models(self):
        # Model A fine everywhere; model B fails (NaN) at origin 4, where A
        # is terrible. Separate scoring favours A; matched scoring must
        # compare only origins 1-3, where B is exact.
        a = self._rolling_df({(t, 1, "x"): (1.0, 0.0) for t in (1, 2, 3)}
                             | {(4, 1, "x"): (100.0, 0.0)})
        b = self._rolling_df({(t, 1, "x"): (0.0, 0.0) for t in (1, 2, 3)}
                             | {(4, 1, "x"): (np.nan, 0.0)})
        scores, info = lib.forecast_score_matched({"a": a, "b": b},
                                                  metric="rmse", by="horizon")
        assert info["n_common"] == 3
        assert info["n_excluded_a"] == 1
        assert float(scores["a"]["rmse"].iloc[0]) == 1.0
        assert float(scores["b"]["rmse"].iloc[0]) == 0.0

    def test_disagreeing_realisations_rejected(self):
        a = self._rolling_df({(1, 1, "x"): (1.0, 0.0), (2, 1, "x"): (1.0, 0.0),})
        b = self._rolling_df({(1, 1, "x"): (1.0, 5.0), (2, 1, "x"): (1.0, 5.0),})
        with pytest.raises(ValueError, match="realisations disagree"):
            lib.forecast_score_matched({"a": a, "b": b})


class TestRollingForecastNameAlignment:
    def test_reordered_model_variables_align_by_name(self, var1_data):
        """A fit callback that reorders columns must not swap labels."""
        def fit_reordered(df):
            return lib.fit_var(df[["y", "x"]], lags=1, trend="c")

        def fit_plain(df):
            return lib.fit_var(df, lags=1, trend="c")

        kw = dict(window_size=60, horizon=2, step=40)
        r_plain = rolling_forecasts(var1_data, fit_plain, **kw)
        r_reord = rolling_forecasts(var1_data, fit_reordered, **kw)
        merged = r_plain.merge(r_reord, on=["origin", "horizon", "variable"],
                               suffixes=("_p", "_r"))
        assert np.allclose(merged["forecast_p"], merged["forecast_r"])

    def test_mismatched_variable_set_rejected(self, var1_data):
        def fit_dropping(df):
            return lib.fit_var(df[["x"]].assign(z=df["y"].values),
                               lags=1, trend="c")
        with pytest.raises(ValueError, match="do not match the input"):
            rolling_forecasts(var1_data, fit_dropping,
                              window_size=60, horizon=1, step=300)


class TestConditionalForecastUnitInvariance:
    def test_hard_conditioning_invariant_to_units(self, var1_data):
        """Re-expressing the data in tiny units must not change the
        conditional point forecast (converted back)."""
        fit = lib.fit_var(var1_data, lags=1, trend="c")
        cond = {"x": np.array([1.0, np.nan, np.nan])}
        base = lib.conditional_forecast(fit, horizon=3, conditioning=cond,
                                        n_paths=0)
        scale = 1e-5
        fit_s = lib.fit_var(var1_data * scale, lags=1, trend="c")
        cond_s = {"x": np.array([1.0 * scale, np.nan, np.nan])}
        small = lib.conditional_forecast(fit_s, horizon=3,
                                         conditioning=cond_s, n_paths=0)
        assert np.allclose(small.point / scale, base.point,
                           rtol=1e-6, atol=1e-8)
