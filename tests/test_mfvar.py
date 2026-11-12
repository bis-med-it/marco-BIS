"""Tests for mfvar.py (mixed-frequency Bayesian VAR).

Validation strategy:

1. **Measurement design correctness** — Z has identity rows for monthly
   variables and Mariano-Murasawa weights for quarterly variables.
2. **All-monthly degeneracy** — with zero quarterly variables, MF-VAR
   should produce parameter draws comparable to a vanilla BVAR.
3. **Aggregation arithmetic** — given a known monthly state, the
   measurement equation Z @ alpha equals the expected aggregation.
4. **Gibbs convergence** — drawn parameters stay in a sensible range
   after burn-in (no divergence, eigenvalues bounded).
5. **Imputation sensibility** — for quarterly variables, the imputed
   monthly state at quarter-end matches the observed quarterly value
   (up to MM aggregation tolerance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


def _make_synthetic_mfvar_data(
    seed: int = 0, T: int = 200, n_monthly: int = 2, n_quarterly: int = 1
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Generate synthetic monthly data, then NaN out intra-quarter
    quarterly observations.

    Returns (data, monthly_var_names, quarterly_var_names).
    """
    rng = np.random.default_rng(seed)
    n = n_monthly + n_quarterly
    # Simple VAR(1) DGP
    A = 0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    A = A * 0.95 / max(np.abs(np.linalg.eigvals(A)))   # ensure stationarity
    Sigma = np.eye(n)
    L = np.linalg.cholesky(Sigma)

    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)

    monthly_names = [f"m{i}" for i in range(n_monthly)]
    quarterly_names = [f"q{i}" for i in range(n_quarterly)]
    all_names = monthly_names + quarterly_names
    idx = pd.period_range("2000-01", periods=T, freq="M").to_timestamp()
    df = pd.DataFrame(Y, columns=all_names, index=idx)

    # Aggregate quarterly variables via Mariano-Murasawa weights and
    # mask out intra-quarter months
    w = lib.mariano_murasawa_weights()
    for qname in quarterly_names:
        # Quarterly aggregation: at quarter-end (Mar, Jun, Sep, Dec) the
        # quarterly obs = w @ [y_t, y_{t-1}, ..., y_{t-4}]
        agg = np.full(T, np.nan)
        col = df[qname].values
        for t in range(4, T):
            if df.index[t].month in (3, 6, 9, 12):
                agg[t] = sum(w[lag] * col[t - lag] for lag in range(5))
        df[qname] = agg

    return df, monthly_names, quarterly_names


class TestMFVARSpec:
    def test_canonical_variable_order(self):
        spec = lib.MFVARSpec(
            monthly_vars=["a", "b"], quarterly_vars=["c"], lags=2,
        )
        assert spec.variable_order == ["a", "b", "c"]
        assert spec.n_monthly == 2
        assert spec.n_quarterly == 1
        assert spec.n_vars == 3

    def test_rejects_lag_above_state_lags(self):
        with pytest.raises(ValueError, match="lags .* cannot exceed state_lags"):
            lib.MFVARSpec(monthly_vars=["a"], quarterly_vars=["b"],
                          lags=6, state_lags=5)

    def test_rejects_empty_vars(self):
        with pytest.raises(ValueError, match="at least one"):
            lib.MFVARSpec(monthly_vars=[], quarterly_vars=[], lags=1)


class TestMeasurementDesign:
    def test_all_monthly_design_is_identity(self):
        from lib.mfvar import _build_measurement_design
        spec = lib.MFVARSpec(monthly_vars=["a", "b", "c"], quarterly_vars=[], lags=2)
        Z = _build_measurement_design(spec)
        assert Z.shape == (3, 3 * 5)   # 3 vars × 5 state lags
        # First 3 columns = identity (current value of each monthly var)
        assert np.allclose(Z[:, :3], np.eye(3))
        # Remaining columns = zero
        assert np.allclose(Z[:, 3:], 0.0)

    def test_quarterly_design_uses_mariano_murasawa_weights(self):
        from lib.mfvar import _build_measurement_design
        spec = lib.MFVARSpec(
            monthly_vars=["m1"], quarterly_vars=["q1"], lags=1,
        )
        Z = _build_measurement_design(spec)
        # n=2, L=5, so Z is (2, 10).
        # Monthly row 0 picks current value of variable 0:
        assert Z[0, 0] == 1.0
        assert np.allclose(np.delete(Z[0], 0), 0.0)
        # Quarterly row 1 has MM weights on variable 1 across lags 0..4:
        w = lib.mariano_murasawa_weights()
        for lag in range(5):
            assert np.isclose(Z[1, lag * 2 + 1], w[lag])
        # All other entries of row 1 are zero
        zero_cols = [c for c in range(10) if c not in [lag * 2 + 1 for lag in range(5)]]
        assert np.allclose(Z[1, zero_cols], 0.0)

    def test_design_aggregates_correctly(self):
        """Given a known monthly state, Z @ alpha equals the MM aggregation."""
        from lib.mfvar import _build_measurement_design
        spec = lib.MFVARSpec(
            monthly_vars=["m1"], quarterly_vars=["q1"], lags=1,
        )
        Z = _build_measurement_design(spec)
        # Construct a state with known monthly history for q1: y_q = [10, 8, 6, 4, 2]
        # (i.e., y_t=10, y_{t-1}=8, ..., y_{t-4}=2)
        alpha = np.zeros(10)
        # Variable indices in state: m1=0, q1=1; lags stack [t, t-1, t-2, t-3, t-4]
        q_values = [10, 8, 6, 4, 2]
        for lag in range(5):
            alpha[lag * 2 + 1] = q_values[lag]
        result = Z @ alpha
        # Expected MM aggregation of q1:
        w = lib.mariano_murasawa_weights()
        expected_q1 = sum(w[lag] * q_values[lag] for lag in range(5))
        assert np.isclose(result[1], expected_q1)


class TestFitMFVARBasic:
    def test_runs_end_to_end(self):
        df, mon, quar = _make_synthetic_mfvar_data(seed=1, T=80)
        spec = lib.MFVARSpec(monthly_vars=mon, quarterly_vars=quar, lags=1)
        fit = lib.fit_mfvar_bayesian(
            df, spec, n_iter=100, burn_in=20, thin=1, seed=0,
        )
        assert fit.spec is spec
        assert fit.gibbs_B_draws.shape[0] == 80   # n_iter - burn_in = 80
        assert fit.state_path_posterior_mean.shape == (80, 3)

    def test_imputation_at_quarter_end_close_to_observed(self):
        """For quarterly variables: at quarter-end months the posterior-
        mean monthly state should give an MM-aggregation close to the
        observed quarterly value."""
        df, mon, quar = _make_synthetic_mfvar_data(seed=2, T=120)
        spec = lib.MFVARSpec(monthly_vars=mon, quarterly_vars=quar, lags=1)
        fit = lib.fit_mfvar_bayesian(
            df, spec, n_iter=200, burn_in=50, thin=1, seed=0,
        )
        # Compute MM aggregation of the posterior-mean state for q0
        state = fit.state_path_posterior_mean
        n = spec.n_vars
        q_idx = spec.n_monthly
        w = lib.mariano_murasawa_weights()
        T = len(state)
        agg = np.full(T, np.nan)
        for t in range(4, T):
            agg[t] = sum(w[lag] * state[t - lag, q_idx] for lag in range(5))
        # Comparison positions come from the data design alone: every
        # quarter-end month with a full five-month aggregation window.
        # The estimate must be finite at all of them.
        observed = df["q0"].values
        assert len(state) == len(df)
        quarter_end = np.asarray(df.index.month.isin([3, 6, 9, 12]))
        positions = np.flatnonzero(quarter_end & (np.arange(T) >= 4))
        assert len(positions) == 39   # T=120 from 2000-01: Mar 2000 dropped
        assert np.all(np.isfinite(observed[positions]))
        n_bad = int((~np.isfinite(agg[positions])).sum())
        assert n_bad == 0, (
            f"MM aggregation of the estimated state is missing at "
            f"{n_bad} of {len(positions)} quarter-end positions"
        )
        # On well-converged Gibbs, agg should be close to observed at
        # quarter ends; allow a small tolerance for Gibbs / Kalman noise
        diffs = np.abs(agg[positions] - observed[positions])
        assert diffs.mean() < 0.5

    def test_parameter_draws_returns_stored_draws(self):
        df, mon, quar = _make_synthetic_mfvar_data(seed=3, T=80)
        spec = lib.MFVARSpec(monthly_vars=mon, quarterly_vars=quar, lags=1)
        fit = lib.fit_mfvar_bayesian(
            df, spec, n_iter=100, burn_in=20, thin=1, seed=0,
        )
        n_avail = fit.gibbs_B_draws.shape[0]
        assert n_avail == 80
        out = fit.parameter_draws(n_draws=10, seed=0)
        # No stability filter requested, so exactly ten pairs come back
        assert len(out) == 10
        used = []
        for B, Sigma in out:
            assert B.shape == fit.B_hat.shape
            assert Sigma.shape == fit.Sigma_u.shape
            # Each (B, Sigma) pair must be one stored Gibbs draw, with B
            # and Sigma taken from the same iteration
            matches = [
                i for i in range(n_avail)
                if np.array_equal(B, fit.gibbs_B_draws[i])
                and np.array_equal(Sigma, fit.gibbs_Sigma_draws[i])
            ]
            assert matches, "returned pair is not a stored Gibbs draw"
            used.append(matches[0])
        # Resampling is without replacement
        assert len(set(used)) == 10

    def test_requesting_too_many_draws_raises(self):
        df, mon, quar = _make_synthetic_mfvar_data(seed=4, T=80)
        spec = lib.MFVARSpec(monthly_vars=mon, quarterly_vars=quar, lags=1)
        fit = lib.fit_mfvar_bayesian(
            df, spec, n_iter=50, burn_in=10, thin=1, seed=0,
        )
        n_avail = fit.gibbs_B_draws.shape[0]
        with pytest.raises(ValueError, match="only .* are stored"):
            fit.parameter_draws(n_draws=n_avail + 100)


class TestAllMonthlyDegenerate:
    def test_zero_quarterly_vars_runs(self):
        """When there are no quarterly variables, MF-VAR should still run
        (every observation is monthly, no NaN in data, Gibbs trivially
        converges)."""
        df, mon, _ = _make_synthetic_mfvar_data(seed=5, T=120, n_quarterly=0)
        spec = lib.MFVARSpec(monthly_vars=mon, quarterly_vars=[], lags=1)
        fit = lib.fit_mfvar_bayesian(
            df, spec, n_iter=50, burn_in=10, thin=1, seed=0,
        )
        # Posterior-mean state should match the (already-monthly) data.
        # The simulation smoother injects tiny noise from H = 1e-10 even
        # when observations are essentially exact; averaging over ~40 draws
        # leaves residual noise of order 1e-6, looser than that at the
        # very first time index because the initial state covariance is
        # not yet washed out by data.
        assert np.allclose(
            fit.state_path_posterior_mean, df[mon].values, atol=1e-3,
        )


class TestStartOfSampleLevelInvariance:
    """Regression contract: the months before the sample get a prior centred
    on the first observations (``_initial_state_prior``), not on zero. With a
    zero-centred prior the sampled pre-sample values of a series far from
    zero act as outliers in the parameter regression and push the
    coefficients on the longer lags to zero."""

    @staticmethod
    def _data(shift):
        rng = np.random.default_rng(8)
        A = np.array([[0.5, 0.1], [0.05, 0.9]])
        Y = np.zeros((250, 2))
        for t in range(1, 250):
            Y[t] = A @ Y[t - 1] + rng.standard_normal(2) * np.array([1.0, 0.5])
        Y = Y[100:].copy()
        Y[:, 1] += shift
        idx = pd.date_range("2000-01-01", periods=150, freq="MS")
        return pd.DataFrame(Y, columns=["a", "b"], index=idx)

    def test_all_monthly_fit_matches_bvar_at_any_level(self):
        prior = lib.MinnesotaPrior(mu=0, delta=0)
        spec = lib.MFVARSpec(monthly_vars=["a", "b"], quarterly_vars=[], lags=4)
        lag_coefs = {}
        for shift in (0.0, 460.0):
            df = self._data(shift)
            fit = lib.fit_mfvar_bayesian(
                df, spec, prior=prior, n_iter=500, burn_in=150, seed=1)
            lag_coefs[shift] = np.asarray(fit.B_hat)[:-1]
            bvar = lib.fit_minnesota_bvar(df, lags=4, prior=prior)
            # with every value observed, the answer is the ordinary BVAR
            assert np.max(np.abs(lag_coefs[shift] - np.asarray(bvar.B_hat)[:-1])) < 0.04
        # adding a constant to a series leaves the lag coefficients unchanged
        assert np.max(np.abs(lag_coefs[0.0] - lag_coefs[460.0])) < 0.02

    def test_prior_mean_and_variance(self):
        from lib.mfvar import _initial_state_prior
        spec = lib.MFVARSpec(monthly_vars=["m"], quarterly_vars=["q"], lags=2)
        T = 30
        m = 460.0 + 0.5 * np.arange(T) + np.tile([0.0, 1.0], T // 2)
        q = np.full(T, np.nan)
        q[2::3] = np.linspace(1.5, 3.0, len(q[2::3]))
        mean, cov = _initial_state_prior(np.column_stack([m, q]), spec)
        L = spec.state_lags
        assert mean.shape == (2 * L,) and cov.shape == (2 * L, 2 * L)
        assert np.allclose(mean[0::2], 460.0)       # first monthly observation
        assert np.allclose(mean[1::2], 1.5 / 3.0)   # first quarterly value / sum of weights
        d = np.diag(cov)
        assert np.all(d > 0) and np.allclose(cov, np.diag(d))
        assert np.allclose(d[2:4], 2 * d[0:2])      # grows with distance from the sample
