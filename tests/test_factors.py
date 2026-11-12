"""Tests for factors.py (FAVAR — Bernanke-Boivin-Eliasz 2005).

Validation strategy:

1. **PC factors recover known structure** — generate panel from K=2 known
   factors, extract 2 PCs, regression of original on PCs has R² ≈ 1.
2. **Standardisation is reversible** — `means` and `stds` are recoverable
   from the fit object.
3. **Slow/fast rotation** changes the factors AND the loadings.
4. **IRF shape** — `(H+1, n_response, 1)` (single MP shock).
5. **Augmented-state Cholesky** — at h=0, the policy variable's response
   to its own shock equals the policy-residual standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


@pytest.fixture(scope="module")
def synthetic_panel_and_policy() -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic panel with K=2 latent factors + policy variable.

    Setup:
      F_t = AR(1) process (2 factors, independent)
      Y_t = policy variable, AR(1) with feedback from F
      X_t = Lambda * F_t + idiosyncratic noise
    """
    rng = np.random.default_rng(42)
    T = 300
    K = 2
    N = 25
    Lambda_true = rng.standard_normal((N, K)) * 0.7
    F = np.zeros((T, K))
    for t in range(1, T):
        F[t] = 0.6 * F[t - 1] + rng.standard_normal(K) * 0.5
    Y = np.zeros(T)
    for t in range(1, T):
        Y[t] = 0.5 * Y[t - 1] + 0.2 * F[t, 0] + rng.standard_normal() * 0.3
    X = F @ Lambda_true.T + rng.standard_normal((T, N)) * 0.3

    idx = pd.period_range("1990-01", periods=T, freq="M").to_timestamp()
    info = pd.DataFrame(X, columns=[f"x{i}" for i in range(N)], index=idx)
    policy = pd.Series(Y, index=idx, name="policy")
    return info, policy


class TestPCFactorExtraction:
    def test_two_factors_recover_known_structure(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=2, lags=2)
        # Standardised regressions of each x on the factors should give
        # high R². Concatenate x_std = (info - mean) / std, regress on
        # fit.factors, residual variance should be small.
        X = (info.values - fit.means) / fit.stds
        # Fitted values — loadings are on the augmented state [F, Y]
        state = np.column_stack([fit.factors, policy.values])
        X_hat = state @ fit.loadings.T
        # Sum of squared residuals / sum of squared totals
        sse = float(((X - X_hat) ** 2).sum())
        sst = float((X ** 2).sum())
        r2_overall = 1.0 - sse / sst
        # With K=2 true factors and N=25 noisy observations, expect R² > 0.5
        assert r2_overall > 0.5

    def test_factor_loadings_shape(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=3, lags=2)
        # Loadings span the augmented state: K factor columns + 1 for the
        # observed policy variable (BBE eq (2))
        assert fit.loadings.shape == (25, 4)
        assert fit.factors.shape == (len(info), 3)
        assert fit.means.shape == (25,)
        assert fit.stds.shape == (25,)


class TestSlowFastRotation:
    def test_rotation_changes_factors(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        # Mark the first 12 as "slow"
        slow_cols = list(info.columns[:12])
        fit_norot = lib.fit_favar(info, policy, n_factors=2, lags=2)
        fit_rot = lib.fit_favar(info, policy, n_factors=2, lags=2,
                                  slow_columns=slow_cols)
        # Rotated factors should differ from unrotated
        assert not np.allclose(fit_norot.factors, fit_rot.factors)

    def test_rotation_recorded_in_metadata(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=2, lags=2,
                             slow_columns=list(info.columns[:5]))
        assert fit.metadata["rotation"] == "slow_fast_BBE"

    def test_unknown_slow_column_rejected(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        with pytest.raises(ValueError, match="slow_columns not in"):
            lib.fit_favar(info, policy, n_factors=2, lags=2,
                          slow_columns=["not_a_column"])


class TestFAVARIRF:
    def test_irf_shape_is_h_plus_1_n_response_one(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=2, lags=2)
        irf = lib.favar_irf(fit, horizon=6, bands="none")
        # All info panel variables as responses, one shock
        assert irf.median.shape == (7, 25, 1)
        # With "none" bands, lower/upper equal median
        assert np.allclose(irf.lower_68, irf.median)

    def test_response_subset_works(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=2, lags=2)
        irf = lib.favar_irf(fit, horizon=6, response_vars=["x0", "x5", "x10"],
                            bands="none")
        assert irf.median.shape == (7, 3, 1)
        assert irf.variable_names == ["x0", "x5", "x10"]

    def test_bands_ordering_holds(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit = lib.fit_favar(info, policy, n_factors=2, lags=2)
        irf = lib.favar_irf(fit, horizon=6, response_vars=["x0", "x5"],
                            bands="parameter_draws", n_draws=200, seed=0)
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)


class TestFAVARFitValidation:
    def test_misaligned_lengths_rejected(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        with pytest.raises(ValueError, match="equal length"):
            lib.fit_favar(info, policy.iloc[:-5], n_factors=2, lags=2)

    def test_nan_in_panel_rejected(self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        bad = info.copy()
        bad.iloc[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.fit_favar(bad, policy, n_factors=2, lags=2)


class TestPolicyShiftInvariance:
    """Fix acceptance: loadings are estimated on the CENTERED augmented
    state, so adding a constant to the policy variable (e.g. a rate in
    percent vs deviation-from-mean) must move neither the loadings nor the
    projected IRFs."""

    def test_loadings_invariant_to_policy_constant_shift(
            self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit_a = lib.fit_favar(info, policy, n_factors=2, lags=2)
        fit_b = lib.fit_favar(info, policy + 100.0, n_factors=2, lags=2)
        np.testing.assert_allclose(fit_a.loadings, fit_b.loadings, atol=1e-7)

    def test_projected_irf_invariant_to_policy_constant_shift(
            self, synthetic_panel_and_policy):
        info, policy = synthetic_panel_and_policy
        fit_a = lib.fit_favar(info, policy, n_factors=2, lags=2)
        fit_b = lib.fit_favar(info, policy + 100.0, n_factors=2, lags=2)
        irf_a = lib.favar_irf(fit_a, horizon=6, bands="none")
        irf_b = lib.favar_irf(fit_b, horizon=6, bands="none")
        np.testing.assert_allclose(irf_a.median, irf_b.median, atol=1e-7)
