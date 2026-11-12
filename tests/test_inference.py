"""Tests for inference.py (bootstrap and asymptotic MC)."""

from __future__ import annotations

import numpy as np
import pytest

import lib


class TestAsymptoticMC:
    def test_returns_correct_shape(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        draws = lib.asymptotic_mc_draws(fit=fit, horizon=10, n_draws=100, seed=0)
        n = fit.n_vars
        assert draws.shape == (100, 11, n, n)

    def test_mean_recovers_point_irf(self, synthetic_var_data):
        """The average over many asymptotic-MC draws should approach the
        point-estimate Cholesky IRF."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        draws = lib.asymptotic_mc_draws(fit=fit, horizon=5, n_draws=3000, seed=0)
        # Point IRF
        chol = np.linalg.cholesky(fit.Sigma_u)
        Phi = fit.ma_rep(horizon=5)
        point = np.einsum("hij,jk->hik", Phi, chol)
        # The mean of draws should be close to the point IRF.
        # MC nonlinearity (B → IRF is nonlinear) means there can be small
        # bias; check within a generous tolerance.
        avg = draws.mean(axis=0)
        assert np.allclose(avg, point, atol=0.1)


class TestResidualBootstrap:
    def test_returns_correct_shape(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        draws = lib.residual_bootstrap_draws(
            fit=fit, horizon=5, n_draws=50, seed=0,
        )
        # Shape should be (≤ n_draws, H+1, n_vars, n_shocks)
        n = fit.n_vars
        assert draws.ndim == 4
        assert draws.shape[1] == 6
        assert draws.shape[2] == n
        assert draws.shape[3] == n


class TestDrawParameterSamples:
    def test_returns_requested_count(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        samples = lib.draw_parameter_samples(fit=fit, n_draws=100, seed=0)
        assert len(samples) == 100

    def test_mean_recovers_b_hat(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        samples = lib.draw_parameter_samples(fit=fit, n_draws=2000, seed=0)
        avg_B = np.mean([B for B, _ in samples], axis=0)
        assert np.allclose(avg_B, fit.B_hat, atol=0.02)


import pandas as pd


class TestBootstrapDtypeAndDivisor:
    """Acceptance check: integer input must not truncate simulated data,
    and each bootstrap replication must estimate the same covariance
    statistic as the point fit (VARX uses the ML divisor)."""

    @staticmethod
    def _sim(T=120, seed=0, integer=False):
        rng = np.random.default_rng(seed)
        y = np.zeros((T, 2))
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(2)
        y = np.round(y * 10)
        idx = pd.date_range("1990-01-01", periods=T, freq="QS")
        df = pd.DataFrame(y, columns=["a", "b"], index=idx)
        return df.astype(int) if integer else df.astype(float)

    def test_integer_and_float_input_give_identical_draws(self):
        df_i = self._sim(integer=True)
        df_f = self._sim(integer=False)
        fit_i = lib.fit_var(df_i.astype(float), lags=1, trend="c")
        # Re-fit on the integer frame so fit.data carries integer dtype
        import dataclasses
        fit_int = dataclasses.replace(fit_i, data=df_i)
        kw = dict(horizon=4, n_draws=30, seed=7, require_stable=False)
        d_f = lib.residual_bootstrap_draws(fit_i, **kw)
        d_i = lib.residual_bootstrap_draws(fit_int, **kw)
        assert np.allclose(d_f, d_i)

    def test_varx_bootstrap_divisor_matches_point_convention(self):
        rng = np.random.default_rng(3)
        T = 120
        exog = pd.Series(rng.standard_normal(T), name="x")
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.4 * y[t - 1] + 0.5 * exog.iloc[t] + rng.standard_normal(2)
        idx = pd.date_range("1990-01-01", periods=T, freq="QS")
        df = pd.DataFrame(y, columns=["a", "b"], index=idx)
        fit = lib.fit_var(df, lags=1, trend="c",
                          exog=pd.DataFrame({"x": exog.values}, index=idx))
        draws = lib.residual_bootstrap_draws(fit, horizon=2, n_draws=200,
                                             seed=1, require_stable=False)
        # The point fit's Sigma uses the ML divisor T_eff; the bootstrap
        # must estimate the SAME statistic, so the median h=0 own impact
        # should straddle the point impact rather than sit sqrt(T/(T-k))
        # above it.
        P_point = np.linalg.cholesky(np.asarray(fit.Sigma_u))
        med = np.median(draws[:, 0, 0, 0])
        ratio = med / P_point[0, 0]
        infl = np.sqrt(fit.t_eff / (fit.t_eff - fit.k))
        assert abs(ratio - 1.0) < abs(ratio - infl), (ratio, infl)
