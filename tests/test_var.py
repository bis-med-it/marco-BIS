"""Tests for the reduced-form VAR layer (var.py).

Validation strategy:

1. Pure-math primitives are checked against numpy reference computations.
2. ``fit_var`` is cross-checked against ``statsmodels.tsa.api.VAR`` — they
   should agree to floating-point precision because ``fit_var`` is a thin
   wrapper.
3. MA representation is checked against the explicit recursion plus a
   direct linearity check (Phi[0] = I).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR as _StatsmodelsVAR

import lib


class TestPrimitives:
    def test_build_design_matrices_shapes(self, synthetic_var_data):
        Y = synthetic_var_data.values
        Y_act, X_act = lib.build_design_matrices(Y, p=4, include_const=True)
        T, n = Y.shape
        assert Y_act.shape == (T - 4, n)
        assert X_act.shape == (T - 4, n * 4 + 1)
        # First row uses lags 1..4 of the rows before t=4
        expected_first = np.concatenate([Y[3], Y[2], Y[1], Y[0], [1.0]])
        assert np.allclose(X_act[0], expected_first)

    def test_extract_lag_matrices_recover_A_in_kronecker_form(self,
                                                                synthetic_var_dgp):
        """If we build B from known A matrices and extract back, we get A."""
        A1, A2 = synthetic_var_dgp["A1"], synthetic_var_dgp["A2"]
        n, p = synthetic_var_dgp["n"], synthetic_var_dgp["p"]
        # B has rows [A1_row1, A1_row2, ..., A2_row1, ..., const_row]^T per layout
        B = np.vstack([A1.T, A2.T, np.zeros((1, n))])   # const row of zeros
        lag_mats = lib.extract_lag_matrices(B, n=n, p=p)
        assert np.allclose(lag_mats[0], A1)
        assert np.allclose(lag_mats[1], A2)

    def test_companion_matrix_structure(self, synthetic_var_dgp):
        A1, A2 = synthetic_var_dgp["A1"], synthetic_var_dgp["A2"]
        n = synthetic_var_dgp["n"]
        C = lib.companion_matrix([A1, A2])
        assert C.shape == (2 * n, 2 * n)
        # Top block: [A1 | A2]
        assert np.allclose(C[:n, :n], A1)
        assert np.allclose(C[:n, n:], A2)
        # Bottom block: [I | 0]
        assert np.allclose(C[n:, :n], np.eye(n))
        assert np.allclose(C[n:, n:], np.zeros((n, n)))

    def test_max_eigenvalue_matches_numpy(self, synthetic_var_dgp):
        A1, A2 = synthetic_var_dgp["A1"], synthetic_var_dgp["A2"]
        ours = lib.max_eigenvalue_modulus([A1, A2])
        direct = float(np.max(np.abs(
            np.linalg.eigvals(lib.companion_matrix([A1, A2]))
        )))
        assert np.isclose(ours, direct)
        # DGP was chosen to be stationary
        assert ours < 1.0

    def test_ma_representation_initial_identity(self, synthetic_var_dgp):
        A1, A2 = synthetic_var_dgp["A1"], synthetic_var_dgp["A2"]
        Phi = lib.ma_representation([A1, A2], horizon=5)
        n = A1.shape[0]
        assert np.allclose(Phi[0], np.eye(n))
        assert np.allclose(Phi[1], A1)
        assert np.allclose(Phi[2], A1 @ A1 + A2)


class TestFitVAR:
    def test_coefficients_match_statsmodels(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sm = _StatsmodelsVAR(synthetic_var_data).fit(maxlags=2, ic=None, trend="c")
        # statsmodels exposes coefficients in different layout via .coefs
        # (p, n, n) where coefs[lag] = A_{lag+1}
        ours = fit.lag_matrices()
        for j in range(2):
            assert np.allclose(ours[j], sm.coefs[j], atol=1e-10)

    def test_aic_bic_match_statsmodels(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sm = _StatsmodelsVAR(synthetic_var_data).fit(maxlags=2, ic=None, trend="c")
        assert np.isclose(fit.fit_info["aic"], sm.aic)
        assert np.isclose(fit.fit_info["bic"], sm.bic)

    def test_sigma_u_matches_statsmodels(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        sm = _StatsmodelsVAR(synthetic_var_data).fit(maxlags=2, ic=None, trend="c")
        assert np.allclose(fit.Sigma_u, sm.sigma_u, atol=1e-10)

    def test_rejects_nan(self):
        import pandas as pd
        bad = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="NaN"):
            lib.fit_var(bad, lags=1)

    def test_rejects_lags_below_one(self, synthetic_var_data):
        with pytest.raises(ValueError, match="lags must be"):
            lib.fit_var(synthetic_var_data, lags=0)

    def test_trend_ct_adds_trend_row_to_b_hat(self, synthetic_var_data):
        """trend='ct' must extend B_hat with a trend coefficient row.

        Guards against a silent downgrade: a design matrix that only added
        a constant column would make fit_var(trend='ct') produce B_hat
        byte-identical to trend='c' while reporting a trend term.
        """
        fit_c = lib.fit_var(synthetic_var_data, lags=2, trend="c")
        fit_ct = lib.fit_var(synthetic_var_data, lags=2, trend="ct")
        # Extra row for the trend coefficient
        assert fit_ct.B_hat.shape[0] == fit_c.B_hat.shape[0] + 1
        assert fit_ct.k == fit_c.k + 1
        # X_act last column should be 1..T_eff (trend values), per the
        # statsmodels convention for post-lag samples
        T_eff = fit_ct.t_eff
        np.testing.assert_allclose(
            fit_ct.X_act[:, -1], np.arange(1, T_eff + 1, dtype=float)
        )
        # Lag matrices should DIFFER from trend='c' (the trend absorbs
        # systematic variation that would otherwise load on the lag block)
        assert not np.allclose(
            fit_c.lag_matrices()[0], fit_ct.lag_matrices()[0]
        )

    def test_trend_ct_matches_statsmodels_lag_matrices(self, synthetic_var_data):
        """Sanity check: with trend='ct', our lag matrices match statsmodels'."""
        fit = lib.fit_var(synthetic_var_data, lags=2, trend="ct")
        sm = _StatsmodelsVAR(synthetic_var_data).fit(
            maxlags=2, ic=None, trend="ct",
        )
        ours = fit.lag_matrices()
        for j in range(2):
            np.testing.assert_allclose(ours[j], sm.coefs[j], atol=1e-10)


class TestParameterDraws:
    def test_returns_requested_count(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        draws = fit.parameter_draws(n_draws=50, seed=0)
        assert len(draws) == 50
        for B, S in draws:
            assert B.shape == fit.B_hat.shape
            assert S.shape == fit.Sigma_u.shape

    def test_draws_centred_on_point_estimate(self, synthetic_var_data):
        """Average over many draws should approach the point estimate."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        draws = fit.parameter_draws(n_draws=2000, seed=0)
        avg_B = np.mean([B for B, _ in draws], axis=0)
        # Asymptotic distribution is centred on B_hat; 2000 draws should
        # bring the average within a few SEs of zero deviation per element.
        assert np.allclose(avg_B, fit.B_hat, atol=0.02)


class TestVARX:
    """VARX fits — exogenous block in fit_var."""

    @pytest.fixture
    def synthetic_varx_data(self):
        """Three endogenous, one exogenous, T=200."""
        rng = np.random.default_rng(7)
        T = 200
        idx = pd.date_range("1990-01-01", periods=T, freq="QS")
        x = rng.standard_normal(T).cumsum()
        Y = np.zeros((T, 3))
        for t in range(1, T):
            Y[t] = 0.5 * Y[t - 1] + 0.3 * x[t] + rng.standard_normal(3) * 0.5
        endo = pd.DataFrame(Y, index=idx, columns=["a", "b", "c"])
        ex = pd.DataFrame({"x": x}, index=idx)
        return endo, ex

    def test_no_exog_byte_identical_to_old_api(self, synthetic_varx_data):
        """fit_var(exog=None) reproduces the plain VAR exactly."""
        endo, _ = synthetic_varx_data
        fit_a = lib.fit_var(endo, lags=2, trend="c")
        fit_b = lib.fit_var(endo, lags=2, trend="c", exog=None)
        np.testing.assert_array_equal(fit_a.B_hat, fit_b.B_hat)
        np.testing.assert_array_equal(fit_a.Sigma_u, fit_b.Sigma_u)
        assert fit_a.has_exog is False
        assert fit_a.n_exog == 0

    def test_exog_fit_carries_metadata(self, synthetic_varx_data):
        endo, ex = synthetic_varx_data
        fit = lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=0)
        assert fit.has_exog is True
        assert fit.n_exog == 1
        assert fit.exog_lags == 0
        assert fit.exog_names == ["x"]
        # 2 lags x 3 vars + const + 1 contemp exog = 8
        assert fit.X_act.shape[1] == 8
        assert fit.fit_info["has_exog"] is True
        assert fit.fit_info["n_exog"] == 1

    def test_exog_coefs_recovers_dgp(self, synthetic_varx_data):
        """The exog coefficient should be near 0.3 (the DGP value)."""
        endo, ex = synthetic_varx_data
        fit = lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=0)
        exog_coefs = fit.exog_coefs()
        assert exog_coefs.shape == (1, 3)
        np.testing.assert_allclose(exog_coefs[0], [0.3, 0.3, 0.3], atol=0.07)

    def test_exog_lags_supported(self, synthetic_varx_data):
        """exog_lags=2 produces 3 contemp+lagged exog columns."""
        endo, ex = synthetic_varx_data
        fit = lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=2)
        # 2 endo lags x 3 + const + (1 exog x 3 positions) = 10
        assert fit.X_act.shape[1] == 10
        assert fit.exog_coefs().shape == (3, 3)

    def test_identification_works_on_varx(self, synthetic_varx_data):
        """Cholesky IRF on a VARX fit (no bootstrap)."""
        endo, ex = synthetic_varx_data
        fit = lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=0)
        irf = lib.cholesky_irf(fit, horizon=10, bands="none")
        assert irf.median.shape == (11, 3, 3)

    def test_residual_bootstrap_works_with_exog(self, synthetic_varx_data):
        """Cholesky IRF with residual bootstrap on VARX fit produces bands."""
        endo, ex = synthetic_varx_data
        fit = lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=0)
        irf = lib.cholesky_irf(
            fit, horizon=5, bands="residual_bootstrap", n_draws=50, seed=0,
        )
        assert irf.median.shape == (6, 3, 3)
        assert (irf.upper_68 - irf.lower_68).max() > 0

    def test_exog_index_mismatch_raises(self, synthetic_varx_data):
        endo, ex = synthetic_varx_data
        ex_misaligned = ex.iloc[5:].copy()
        with pytest.raises(ValueError, match="exog index must match"):
            lib.fit_var(endo, lags=2, trend="c", exog=ex_misaligned)

    def test_exog_lags_exceeds_lags_raises(self, synthetic_varx_data):
        endo, ex = synthetic_varx_data
        with pytest.raises(ValueError, match="exog_lags.*must be <= "):
            lib.fit_var(endo, lags=2, trend="c", exog=ex, exog_lags=3)


class TestSelectLagOrderAliases:
    """``select_lag_order`` accepts the documentation's Hannan-Quinn
    abbreviations: the skills and ``config.yaml`` write "hq"/"HQ", so the
    function must treat them as ``"hqic"`` rather than raising."""

    def test_hq_alias_matches_hqic(self, synthetic_var_data):
        p_alias = lib.select_lag_order(synthetic_var_data, maxlags=4,
                                       criterion="hq")
        p_canon = lib.select_lag_order(synthetic_var_data, maxlags=4,
                                       criterion="hqic")
        assert p_alias == p_canon

    def test_alias_is_case_insensitive(self, synthetic_var_data):
        p_upper = lib.select_lag_order(synthetic_var_data, maxlags=4,
                                       criterion="HQ")
        p_canon = lib.select_lag_order(synthetic_var_data, maxlags=4,
                                       criterion="hqic")
        assert p_upper == p_canon

    def test_unknown_criterion_still_raises(self, synthetic_var_data):
        with pytest.raises(ValueError, match="criterion must be one of"):
            lib.select_lag_order(synthetic_var_data, maxlags=4,
                                 criterion="schwarz")


class TestVecCovKroneckerOrdering:
    """``vec_cov`` pairs ``Sigma_u ⊗ (X'X)^{-1}`` with the EQUATION-MAJOR
    vec (Hamilton 1994, eq. 11.1.35). The transposed pairing would pass a
    mean-only check; the sharp pin is equality of the implied standard
    errors with statsmodels', equation by equation, after permuting the
    deterministic row (lib: [lags..., const]; statsmodels: [const, lags...])."""

    def test_vec_cov_diag_matches_statsmodels_stderr(self, synthetic_var_data):
        from statsmodels.tsa.api import VAR as SMVAR
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        sm = SMVAR(synthetic_var_data.to_numpy()).fit(maxlags=2, trend="c")
        k, n = fit.k, fit.n_vars
        diag = np.sqrt(np.diag(fit.vec_cov()))
        for j in range(n):
            lib_se = diag[j * k:(j + 1) * k]            # [lags..., const]
            sm_se = sm.bse[:, j]                        # [const, lags...]
            np.testing.assert_allclose(
                lib_se, np.r_[sm_se[1:], sm_se[:1]], rtol=1e-8
            )


class TestRecentredIWRecipe:
    """The joint-path Sigma draws come from an IW recentred on the point
    estimate (scale (T-k-n-1) Sigma_hat, df T-k): their MEAN must sit on
    Sigma_hat. At T=60, k=7, n=3 the textbook flat-prior posterior
    IW(U'U, T-k) has mean (T-k)/(T-k-n-1) = 53/49 ≈ 8.2% ABOVE Sigma_hat,
    so the 3% gate distinguishes the documented recipe from the flat-prior
    alternative (CONVENTIONS.md §4)."""

    def test_sigma_draws_mean_centred_on_point_estimate(self, synthetic_var_data):
        df = synthetic_var_data.iloc[:60]
        fit = lib.fit_var(data=df, lags=2, trend="c")
        draws = fit.parameter_draws(6000, seed=0)
        mean_sigma = np.mean([S for _B, S in draws], axis=0)
        ratio = np.trace(mean_sigma) / np.trace(np.asarray(fit.Sigma_u))
        assert abs(ratio - 1.0) < 0.03, f"trace ratio {ratio:.4f}"
        flat_prior_ratio = (fit.t_eff - fit.k) / (fit.t_eff - fit.k - fit.n_vars - 1)
        assert flat_prior_ratio - 1.0 > 0.05      # the gate separates the two


class TestVARXConventionPins:
    """VARX path conventions documented in CONVENTIONS.md §3 and the
    fit_var comments: ML Sigma divisor (T_eff), T-scaled information
    criteria equal to exactly T_eff x statsmodels' per-T VARX criteria,
    and an identical concentrated log-likelihood."""

    def _fit_pair(self, synthetic_var_data):
        from statsmodels.tsa.api import VAR as SMVAR
        df = synthetic_var_data
        ex = pd.DataFrame(
            {"x": np.sin(np.arange(len(df)) / 7.0)}, index=df.index
        )
        lib_fit = lib.fit_var(data=df, lags=2, trend="c", exog=ex, exog_lags=0)
        sm = SMVAR(df.to_numpy(), exog=ex.to_numpy()).fit(maxlags=2, trend="c")
        return lib_fit, sm

    def test_sigma_u_is_ml_divisor(self, synthetic_var_data):
        lib_fit, _ = self._fit_pair(synthetic_var_data)
        u = np.asarray(lib_fit.residuals)
        np.testing.assert_allclose(
            np.asarray(lib_fit.Sigma_u), (u.T @ u) / lib_fit.t_eff, atol=1e-13
        )

    def test_ic_is_t_eff_times_statsmodels_and_llf_identical(self, synthetic_var_data):
        lib_fit, sm = self._fit_pair(synthetic_var_data)
        T = lib_fit.t_eff
        np.testing.assert_allclose(lib_fit.fit_info["aic"], T * sm.aic, rtol=1e-9)
        np.testing.assert_allclose(lib_fit.fit_info["bic"], T * sm.bic, rtol=1e-9)
        np.testing.assert_allclose(lib_fit.fit_info["llf"], sm.llf, rtol=1e-10)


class TestTrendCtConstantConvention:
    """lib numbers the trend 1..T_eff; statsmodels numbers it p+1..T — a
    pure intercept reparameterisation. Slopes and lag matrices must be
    identical and the constants must relate as const_lib = const_sm +
    p * slope (the relation the docstring documents)."""

    def test_constant_relation_to_statsmodels(self, synthetic_var_data):
        from statsmodels.tsa.api import VAR as SMVAR
        p = 2
        fit = lib.fit_var(data=synthetic_var_data, lags=p, trend="ct")
        sm = SMVAR(synthetic_var_data.to_numpy()).fit(maxlags=p, trend="ct")
        np.testing.assert_allclose(
            np.asarray(fit.lag_matrices()), sm.coefs, atol=1e-10
        )
        const_lib, slope_lib = fit.B_hat[-2, :], fit.B_hat[-1, :]
        const_sm, slope_sm = sm.params[0, :], sm.params[1, :]
        np.testing.assert_allclose(slope_lib, slope_sm, atol=1e-12)
        np.testing.assert_allclose(const_lib, const_sm + p * slope_sm, atol=1e-10)


class TestRequireStableAttemptCeiling:
    """The fixed-Sigma draw loop must raise rather than spin forever when
    the stability filter rejects (nearly) everything."""

    def test_impossible_eig_tol_raises(self, synthetic_var_data):
        fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
        with pytest.raises(RuntimeError, match="stable"):
            fit.parameter_draws(
                5, seed=0, require_stable=True, eig_tol=0.01,
                include_sigma_uncertainty=False,
            )
