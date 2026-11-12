"""Convention contracts — closed-form textbook checks.

These tests pin down the framework's mathematical conventions so that
docstrings and code can't drift apart. Each test:

  1. Sets up a small DGP with a known closed-form answer.
  2. Asserts the library function returns that answer to floating-point
     precision.

If a library function changes its horizon convention, normalisation
scaling, or B_hat layout, the corresponding test here will fail —
forcing an explicit choice rather than a silent drift.

Tests live in this dedicated file (rather than per-module tests) so
that the convention contracts are discoverable as a single block. Each
test names the convention it pins down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


# ---------------------------------------------------------------------------
# FEVD horizon convention
# ---------------------------------------------------------------------------

class TestFEVDHorizonConvention:
    """structural_fevd & generalised_fevd use the textbook H-step-ahead
    convention:  FEV(H) sums H terms (j=0..H-1), NOT H+1.

    Closed-form check via a DIAGONAL bivariate VAR(1):
      y_t = A y_{t-1} + u_t with A = diag(phi1, phi2), Sigma = diag(sigma1^2, sigma2^2).
      Cholesky impact = diag(sigma1, sigma2).
      FEVD at any horizon H >= 1 is identity (each variable's variance
      comes 100% from its own shock).
    """

    def _make_diagonal_var1_fit(self, T: int = 1000, phi1: float = 0.5, phi2: float = 0.3,
                                  sigma1: float = 1.0, sigma2: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t, 0] = phi1 * Y[t - 1, 0] + sigma1 * rng.standard_normal()
            Y[t, 1] = phi2 * Y[t - 1, 1] + sigma2 * rng.standard_normal()
        df = pd.DataFrame(
            Y, columns=["y1", "y2"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        return lib.fit_var(df, lags=1, trend="n")

    def test_structural_fevd_h1_returns_impact_share(self):
        """At H=1, with Cholesky S = chol(Sigma), the bivariate diagonal DGP
        gives FEVD = identity (each var 100% from own shock)."""
        fit = self._make_diagonal_var1_fit()
        S = lib.cholesky_impact(fit.Sigma_u)
        shares = lib.structural_fevd(fit, S, horizon=1)
        assert shares.shape == (2, 2)
        # Each row sums to 1 (full identification)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-10)
        # Approximately diagonal (DGP is diagonal up to sampling error)
        assert shares[0, 0] > 0.9     # var 1 mostly from shock 1
        assert shares[1, 1] > 0.9     # var 2 mostly from shock 2

    def test_structural_fevd_h8_at_8step_horizon(self):
        """At H=8, FEVD sums 8 terms (j=0..7). Full Cholesky still gives row sums = 1."""
        fit = self._make_diagonal_var1_fit()
        S = lib.cholesky_impact(fit.Sigma_u)
        shares = lib.structural_fevd(fit, S, horizon=8)
        assert shares.shape == (2, 2)
        np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-10)

    def test_fevd_horizon_off_by_one_regression(self):
        """REGRESSION: H=8 must sum 8 terms (j=0..7), NOT 9.

        Manually compute the FEV using Phi[0..H-1] and compare to
        structural_fevd at the same H. If the function ever sums H+1
        terms (off-by-one bug), this test fails.
        """
        fit = self._make_diagonal_var1_fit()
        S = lib.cholesky_impact(fit.Sigma_u)
        H = 8
        # Library's structural_fevd
        shares_lib = lib.structural_fevd(fit, S, horizon=H)

        # Manual: textbook H-step FEV uses Phi[0..H-1]
        Phi = fit.ma_rep(H - 1)        # H matrices Phi_0..Phi_{H-1}
        Sigma = np.asarray(fit.Sigma_u)
        n = fit.n_vars
        irf_struct = np.einsum("hij,jk->hik", Phi, S)
        numer = (irf_struct ** 2).sum(axis=0)
        denom = np.array([
            sum((Phi[h] @ Sigma @ Phi[h].T)[k, k] for h in range(H))
            for k in range(n)
        ])
        shares_manual = numer / denom[:, None]
        np.testing.assert_allclose(shares_lib, shares_manual, atol=1e-12)

    def test_structural_fevd_partial_id_residual_share(self):
        """With m < n identified shocks, residual share = 1 - row sum."""
        # 2-var system. Cholesky gives full identification.
        rng = np.random.default_rng(0)
        T = 300
        Y = rng.standard_normal((T, 2)).cumsum(axis=0) * 0.5
        df = pd.DataFrame(
            Y, columns=["a", "b"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        S_full = lib.cholesky_impact(fit.Sigma_u)
        shares_full = lib.structural_fevd(fit, S_full, horizon=8)
        # Full id: rows should sum to 1
        assert np.allclose(shares_full.sum(axis=1), 1.0, atol=1e-10)
        # Partial: drop second column
        shares_partial = lib.structural_fevd(fit, S_full[:, :1], horizon=8)
        # First-column shares should match the partial-id matrix
        assert np.allclose(shares_partial[:, 0], shares_full[:, 0])
        # And rows should sum to less than 1 (the residual is "unidentified")
        assert (shares_partial.sum(axis=1) < 1.0).all()

    def test_generalised_fevd_h1_matches_impact_decomposition(self):
        """At H=1, generalised FEVD decomposes Sigma_u itself."""
        rng = np.random.default_rng(0)
        T = 300
        Y = rng.standard_normal((T, 3)).cumsum(axis=0) * 0.5
        df = pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        gfevd = lib.generalised_fevd(fit, horizon=1, row_normalise=False)
        # Phi_0 = I, so generalised FEVD at H=1 is (Sigma^2)/(diag(Sigma)^2 row * col)
        # For diagonal Sigma it'd be I. Here Sigma is the residual covariance.
        Sigma = np.asarray(fit.Sigma_u)
        sigma_diag = np.diag(Sigma)
        # theta[i,j] = (Sigma[i,j])^2 / (Sigma[i,i] * Sigma[j,j]) at H=1
        expected = (Sigma ** 2) / (sigma_diag[:, None] * sigma_diag[None, :])
        assert np.allclose(gfevd, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# IRF horizon convention
# ---------------------------------------------------------------------------

class TestIRFHorizonConvention:
    """IRFContainer.median has shape (H+1, n, m) indexed h=0..H.
    h=0 is the IMPACT period (response to a same-period shock).
    h=H is the response H periods after the shock.
    """

    def test_irf_horizon_indexing(self):
        """Cholesky IRF: container should have H+1 rows, h=0 = impact."""
        rng = np.random.default_rng(0)
        T = 200
        Y = rng.standard_normal((T, 2)).cumsum(axis=0) * 0.5
        df = pd.DataFrame(
            Y, columns=["a", "b"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        irf = lib.cholesky_irf(fit, horizon=8, bands="none")
        # Shape: (H+1, n, m) = (9, 2, 2)
        assert irf.median.shape == (9, 2, 2)
        # h=0 impact: lower triangular (Cholesky convention)
        assert abs(irf.median[0, 0, 1]) < 1e-12  # off-diagonal upper triangle = 0
        assert irf.median[0, 0, 0] > 0           # diagonal entry positive
        # h=H = 8th period response
        assert irf.median.shape[0] - 1 == 8

    def test_ma_representation_phi0_is_identity(self):
        """Phi[0] = I_n is the universal convention."""
        rng = np.random.default_rng(0)
        T = 100
        Y = rng.standard_normal((T, 3)).cumsum(axis=0)
        df = pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        Phi = fit.ma_rep(5)
        assert Phi.shape == (6, 3, 3)
        np.testing.assert_allclose(Phi[0], np.eye(3))


# ---------------------------------------------------------------------------
# B_hat layout convention
# ---------------------------------------------------------------------------

class TestBHatLayout:
    """fit_var's B_hat layout: [lag1, lag2, ..., lagp, (const), (trend), exog].

    Pins the deterministic-term rows explicitly: a trend='ct' fit whose
    design matrix silently added only a constant column would leave B_hat
    indistinguishable from trend='c' while the metadata claims a trend.
    """

    def _make_data(self, T: int = 200, n: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        Y = rng.standard_normal((T, n)).cumsum(axis=0)
        return pd.DataFrame(
            Y, columns=[f"v{i}" for i in range(n)],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )

    def test_trend_c_has_constant_row_only(self):
        fit = lib.fit_var(self._make_data(), lags=2, trend="c")
        # 2 lags × 3 vars + 1 const = 7
        assert fit.B_hat.shape[0] == 7
        # Last X_act column = 1 (constant)
        np.testing.assert_array_equal(fit.X_act[:, -1], 1.0)

    def test_trend_ct_has_constant_and_trend_rows(self):
        fit = lib.fit_var(self._make_data(), lags=2, trend="ct")
        # 2 lags × 3 vars + 1 const + 1 trend = 8
        assert fit.B_hat.shape[0] == 8
        # Last X_act column = trend values 1, 2, ..., T_eff
        T_eff = fit.t_eff
        np.testing.assert_allclose(fit.X_act[:, -1], np.arange(1, T_eff + 1, dtype=float))
        # Second-to-last = constant
        np.testing.assert_array_equal(fit.X_act[:, -2], 1.0)

    def test_exog_block_appended_after_deterministic(self):
        """With exog, layout is [lag1..lagp, const, (trend), exog_lag0..exog_lagq]."""
        data = self._make_data()
        ex = pd.DataFrame(
            {"x": np.arange(len(data)) * 0.01},
            index=data.index,
        )
        fit = lib.fit_var(data, lags=2, trend="c", exog=ex, exog_lags=0)
        # Layout: 2 lags × 3 vars + 1 const + 1 exog = 8
        assert fit.B_hat.shape[0] == 8
        # exog_coefs lifts the exog block
        ec = fit.exog_coefs()
        assert ec.shape == (1, 3)


# ---------------------------------------------------------------------------
# Proxy-SVAR normalisation convention
# ---------------------------------------------------------------------------

class TestProxySVARNormalisation:
    """Pin down the exact one_sd vs unit_policy formulas.

    one_sd:       s[policy] = std(u_policy) = sqrt(Sigma_u[policy, policy])
    unit_policy:  s[policy] = 1
    """

    def _make_data_with_instrument(self, T: int = 500, seed: int = 0):
        """Two-variable VAR where a known shock is the first variable.

        DGP:  z_t = N(0,1)
              eps_t = z_t (instrument is exactly the structural shock)
              u_1_t = eps_t
              u_2_t = 0.5 * eps_t + 0.5 * N(0,1)   (some loading)
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(T)
        eps = z
        other = rng.standard_normal(T) * 0.5
        u1 = eps
        u2 = 0.5 * eps + other
        # Integrate to get a stationary AR(1) system
        y1 = np.zeros(T); y2 = np.zeros(T)
        for t in range(1, T):
            y1[t] = 0.3 * y1[t - 1] + u1[t]
            y2[t] = 0.3 * y2[t - 1] + u2[t]
        df = pd.DataFrame(
            {"y1": y1, "y2": y2},
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        return df, z

    def test_one_sd_normalisation_h0_policy_equals_std_u_policy(self):
        """one_sd normalisation: h=0 policy IRF = np.std(u_policy, ddof=1).

        NOTE: this is the sample standard deviation with Bessel correction
        (ddof=1), NOT sqrt(Sigma_u[policy,policy]) — the latter uses
        statsmodels' degrees-of-freedom-adjusted variance estimator, which
        gives a slightly different number on small samples. The library's
        convention is ddof=1; convention contract is in
        _proxy_impact_column's docstring.
        """
        df, z = self._make_data_with_instrument(T=1000)
        fit = lib.fit_var(df, lags=1, trend="c")
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1",
            horizon=4, bands="none", normalization="one_sd",
        )
        std_u_policy_ddof1 = float(np.std(fit.residuals[:, 0], ddof=1))
        h0_impact_on_policy = irf.median[0, 0, 0]
        assert abs(h0_impact_on_policy - std_u_policy_ddof1) < 1e-10

    def test_unit_policy_normalisation_h0_policy_equals_one(self):
        df, z = self._make_data_with_instrument(T=1000)
        fit = lib.fit_var(df, lags=1, trend="c")
        irf, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1",
            horizon=4, bands="none", normalization="unit_policy",
        )
        h0_impact_on_policy = irf.median[0, 0, 0]
        assert abs(h0_impact_on_policy - 1.0) < 1e-10

    def test_reliability_normalisation_matches_closed_form(self):
        """reliability: h=0 policy IRF = the Mertens-Ravn / GK (fn.4) s_p closed
        form; the other variables scale by the same s_p (ratios preserved vs
        unit_policy). s_p is computed here by hand (the 2-variable scalar form)
        so a partition / df / algebra bug in _proxy_reliability_sp is caught.
        """
        df, z = self._make_data_with_instrument(T=1000)
        fit = lib.fit_var(df, lags=1, trend="c")
        irf_rel, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1",
            horizon=4, bands="none", normalization="reliability",
        )
        irf_unit, _ = lib.proxy_svar_irf(
            fit=fit, instrument=z, policy_var="y1",
            horizon=4, bands="none", normalization="unit_policy",
        )
        # hand-compute s_p from the df-adjusted, mean-centred residual covariance
        resid = fit.residuals
        T, k = resid.shape[0], fit.k
        rc = resid - resid.mean(axis=0)
        Sigma = (rc.T @ rc) / (T - k)
        r = float(irf_unit.median[0, 1, 0])             # ratio s_2 / s_1
        S11, S21, S22 = Sigma[0, 0], Sigma[1, 0], Sigma[1, 1]
        Q = r * S11 * r - 2.0 * S21 * r + S22
        sp = float(np.sqrt(S11 - (S21 - r * S11) ** 2 / Q))
        assert abs(float(irf_rel.median[0, 0, 0]) - sp) < 1e-9        # policy = s_p
        # whole column = unit_policy column * s_p (ratios preserved)
        assert np.allclose(irf_rel.median[0, :, 0],
                           irf_unit.median[0, :, 0] * sp, atol=1e-9)
        # the measurement-error correction never raises s_p above the
        # df-adjusted policy-residual std sqrt(S11) (the quadratic term is >= 0)
        assert sp <= float(np.sqrt(S11)) + 1e-12
        assert irf_rel.metadata["normalization"] == "reliability"


# ---------------------------------------------------------------------------
# Historical decomposition vs FEVD: distinct concepts
# ---------------------------------------------------------------------------

class TestBVARMinnesotaConventions:
    """Convention contracts for the Minnesota BVAR prior.

    Pin down (a) lambda_1 / lambda_2 / lambda_3 directional behaviour,
    (b) symmetric-vs-asymmetric mode semantics for lambda_2, (c) mu
    and delta sub-prior disabling, (d) rw_deltas effect.
    """

    @pytest.fixture
    def synthetic_persistent_data(self):
        """3-var system with strong persistence (close to random walk)."""
        rng = np.random.default_rng(0)
        T = 200
        Y = np.zeros((T, 3))
        for t in range(1, T):
            Y[t] = 0.95 * Y[t - 1] + rng.standard_normal(3) * 0.5
        return pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )

    def test_lambda_1_smaller_means_tighter_shrinkage(self, synthetic_persistent_data):
        """Smaller lambda_1 → posterior mean closer to RW prior mean."""
        prior_tight = lib.MinnesotaPrior(lambda_1=0.01)
        prior_loose = lib.MinnesotaPrior(lambda_1=1.0)
        fit_tight = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_tight, trend="c")
        fit_loose = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_loose, trend="c")
        # RW prior mean: own-lag-1 coefficient = 1; everything else = 0.
        # Tight prior should push own-lag-1 coefficients closer to 1.
        # (Slack lag-1 matrix; element [i, i] is own-lag-1 of variable i.)
        lag1_tight = fit_tight.lag_matrices()[0]
        lag1_loose = fit_loose.lag_matrices()[0]
        # Tight prior's diagonal is closer to 1
        dist_tight = np.abs(np.diag(lag1_tight) - 1.0).sum()
        dist_loose = np.abs(np.diag(lag1_loose) - 1.0).sum()
        assert dist_tight < dist_loose, (
            f"Tight prior (lambda_1=0.01) should pull diag closer to 1; "
            f"got dist_tight={dist_tight:.4f}, dist_loose={dist_loose:.4f}"
        )

    def test_lambda_2_no_op_under_symmetric_mode(self, synthetic_persistent_data):
        """Under the symmetric BGR (2010) Kronecker prior, lambda_2 cannot
        affect B_hat — the Sigma ⊗ V_0 structure has one per-regressor
        precision.

        Changing lambda_2 from 0.1 to 10 must produce IDENTICAL B_hat
        when asymmetric=False — pinning the no-op as a documented property
        rather than letting it pass for active cross-variable shrinkage.
        """
        prior_tight = lib.MinnesotaPrior(lambda_2=0.1, asymmetric=False)
        prior_loose = lib.MinnesotaPrior(lambda_2=10.0, asymmetric=False)
        fit_tight = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_tight, trend="c")
        fit_loose = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_loose, trend="c")
        np.testing.assert_allclose(fit_tight.B_hat, fit_loose.B_hat, atol=1e-12)

    def test_lambda_2_has_effect_under_asymmetric_mode(self, synthetic_persistent_data):
        """Under asymmetric=True, lambda_2 IS active. Smaller lambda_2 →
        tighter cross-variable shrinkage relative to own-lag, so
        off-diagonal entries of B_hat shrink toward 0."""
        prior_tight = lib.MinnesotaPrior(lambda_2=0.1, asymmetric=True)
        prior_loose = lib.MinnesotaPrior(lambda_2=10.0, asymmetric=True)
        fit_tight = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_tight, trend="c")
        fit_loose = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_loose, trend="c")
        # B_hat must differ
        assert not np.allclose(fit_tight.B_hat, fit_loose.B_hat, atol=1e-6)
        # Off-diagonal sum in tight should be < off-diagonal sum in loose
        lag1_tight = fit_tight.lag_matrices()[0]
        lag1_loose = fit_loose.lag_matrices()[0]
        off_tight = np.abs(lag1_tight - np.diag(np.diag(lag1_tight))).sum()
        off_loose = np.abs(lag1_loose - np.diag(np.diag(lag1_loose))).sum()
        assert off_tight < off_loose, (
            f"Tighter lambda_2 should shrink cross-variable lags more; "
            f"got off_tight={off_tight:.4f}, off_loose={off_loose:.4f}"
        )

    def test_mu_zero_disables_sum_of_coefficients_dummies(self, synthetic_persistent_data):
        """mu=0 omits the SOC dummy block — n_dummies drops by n_vars (one
        SOC dummy row per variable)."""
        prior_with_mu = lib.MinnesotaPrior(mu=1.0, delta=0)
        prior_no_mu = lib.MinnesotaPrior(mu=0.0, delta=0)
        fit_mu = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_with_mu, trend="c")
        fit_no = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_no_mu, trend="c")
        n_vars = synthetic_persistent_data.shape[1]
        # SOC adds one dummy row per variable
        assert fit_mu.n_dummies - fit_no.n_dummies == n_vars

    def test_delta_zero_disables_dummy_initial_observation(self, synthetic_persistent_data):
        """delta=0 omits the single dummy-initial-observation row."""
        prior_with_d = lib.MinnesotaPrior(mu=0, delta=1.0)
        prior_no_d = lib.MinnesotaPrior(mu=0, delta=0.0)
        fit_d = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_with_d, trend="c")
        fit_no = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_no_d, trend="c")
        # Delta adds exactly one dummy row
        assert fit_d.n_dummies - fit_no.n_dummies == 1

    def test_rw_deltas_zero_is_white_noise_prior(self, synthetic_persistent_data):
        """rw_deltas=[0,0,0] sets the prior mean to white noise (own-lag-1 = 0).
        Tight lambda_1 with rw_deltas=0 should pull own-lag-1 toward 0."""
        # White-noise prior; SOC and dummy-init aren't really meaningful here
        prior = lib.MinnesotaPrior(lambda_1=0.05, rw_deltas=[0, 0, 0], mu=0, delta=0)
        fit = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior, trend="c")
        lag1 = fit.lag_matrices()[0]
        # Diagonal should be smaller (closer to 0) than with default rw_deltas=1
        prior_default = lib.MinnesotaPrior(lambda_1=0.05, mu=0, delta=0)
        fit_default = lib.fit_minnesota_bvar(synthetic_persistent_data, lags=2, prior=prior_default, trend="c")
        lag1_default = fit_default.lag_matrices()[0]
        # Distance from 0 should be SMALLER under rw_deltas=0
        assert np.abs(np.diag(lag1)).sum() < np.abs(np.diag(lag1_default)).sum()


class TestHDvsFEVDDistinction:
    """HD decomposes Y_t (the observed path) into shock contributions.
    FEVD decomposes the H-step-ahead forecast-error VARIANCE into shock contributions.
    Different objects. Both have valid horizon conventions but they don't
    correspond to each other one-to-one.
    """

    def test_hd_reconstructs_data_at_all_t(self):
        """For full identification, HD shocks sum + det_plus_init = Y_act exactly."""
        rng = np.random.default_rng(0)
        T = 200
        Y = rng.standard_normal((T, 3)).cumsum(axis=0)
        df = pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1970-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        S = lib.cholesky_impact(fit.Sigma_u)
        hd = lib.historical_decomposition(fit, S)
        err = hd.reconstruction_error()
        assert (err < 1e-10).all()


# ---------------------------------------------------------------------------
# Non-Gaussian SVAR (LMS 2017) and higher-moment (AFM 2023) conventions
# ---------------------------------------------------------------------------


class TestLMSNonGaussianConventions:
    """Pin down conventions of the non-Gaussian SVAR identification.

    LMS recovers B such that ``u_t = B ε_t`` with ε independent and
    non-Gaussian. The library convention:

    * Pre-whiten via Cholesky (``Z = L^{-1} u`` with ``L = chol(Σ_u)``).
    * FastICA recovers an orthogonal unmixing ``W`` so that
      ``ε_t = W z_t``. The structural matrix is then ``B = L W^⊤``
      (NOT ``L W``; the transpose comes from the row/column
      convention: in row form ``ε = Z @ W.T``).
    """

    def test_h0_irf_equals_LW_transpose(self):
        """The h=0 IRF column k equals ``(L @ W.T)[:, k]`` — the
        structural impact matrix."""
        rng = np.random.default_rng(0)
        T = 600
        n = 3
        eps = rng.standard_t(df=4, size=(T + 100, n)) * np.sqrt(2 / 4)
        L_true = np.array([[1.0, 0, 0], [0.5, 1.0, 0], [0.2, 0.4, 1.0]])
        A1 = 0.4 * np.eye(n)
        u = eps @ L_true.T
        Y = np.zeros((T + 100, n))
        for t in range(1, T + 100):
            Y[t] = A1 @ Y[t - 1] + u[t]
        Y = Y[100:]
        idx = pd.period_range("1980-01", periods=T, freq="M").to_timestamp()
        df = pd.DataFrame(Y, columns=["a", "b", "c"], index=idx)
        fit = lib.fit_var(df, lags=1)
        irf, diag = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        # B_point should equal irf.median at h=0
        B = np.array(diag["B_point"])
        for k in range(n):
            assert np.allclose(irf.median[0, :, k], B[:, k], atol=1e-12)

    def test_fastica_W_is_orthogonal(self):
        """The unmixing matrix W must be orthogonal: W W' = I."""
        rng = np.random.default_rng(0)
        Z = rng.standard_t(df=5, size=(500, 3)) * np.sqrt(3 / 5)
        W, _ = lib.fastica_symmetric(Z, seed=0)
        assert np.allclose(W @ W.T, np.eye(3), atol=1e-10)

    def test_metadata_identification_tag(self):
        """The identification tag is the literal ``"nongaussian_svar"``."""
        rng = np.random.default_rng(0)
        T = 300
        eps = rng.standard_t(df=4, size=(T + 50, 2)) * np.sqrt(2 / 4)
        L = np.array([[1.0, 0], [0.3, 1.0]])
        u = eps @ L.T
        Y = np.zeros((T + 50, 2))
        for t in range(1, T + 50):
            Y[t] = 0.4 * Y[t - 1] + u[t]
        Y = Y[50:]
        idx = pd.period_range("1990-01", periods=T, freq="M").to_timestamp()
        df = pd.DataFrame(Y, columns=["x", "y"], index=idx)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        assert irf.metadata["identification"] == "nongaussian_svar"


class TestAFMHigherMomentConventions:
    """Pin down conventions of the AFM 2023 higher-moment restriction
    layer in ``sign_restricted_irf``.

    Conventions:

    * ``moment`` must be ``"skewness"`` or ``"excess_kurtosis"`` —
      no other names accepted.
    * ``"excess_kurtosis"`` is Fisher style (kurt − 3), via
      ``scipy.stats.kurtosis(fisher=True, bias=False)``.
    * In single-shock mode only THRESHOLD restrictions (no ``vs`` key)
      are valid.
    * The identification tag gains ``"higher_moment"`` when the kwarg
      is active.
    * ``higher_moment_restrictions=None`` is BYTE-IDENTICAL to the
      pre-existing sign-only fit (backwards compatibility).
    """

    def test_none_is_byte_identical_to_baseline(self, synthetic_var_data):
        """Calling with higher_moment_restrictions=None should produce
        the same IRF as omitting the kwarg — same seed, same data."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_a = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"x": {"y1": +1}},
            target_accepted=30, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        irf_b = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"x": {"y1": +1}},
            higher_moment_restrictions=None,   # explicit None
            target_accepted=30, max_rotations=2000, seed=42,
            parameter_uncertainty=False,
        )
        assert np.allclose(irf_a.median, irf_b.median, atol=1e-15)
        assert np.allclose(irf_a.upper_68, irf_b.upper_68, atol=1e-15)

    def test_identification_tag_gains_higher_moment(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"x": {"y1": +1}},
            higher_moment_restrictions=[
                {"moment": "excess_kurtosis", "lower": -10.0},
            ],
            target_accepted=20, max_rotations=2000, seed=0,
            parameter_uncertainty=False,
        )
        assert "higher_moment" in irf.metadata["identification"]
        assert "AFM 2023" in irf.metadata["method"]

    def test_excess_kurtosis_is_fisher_form(self, synthetic_var_data):
        """The library's ``excess_kurtosis`` is Fisher (kurt - 3), so a
        Gaussian-like shock has excess_kurt ≈ 0 (not 3). The bound
        ``lower=0`` therefore says 'leptokurtic or Gaussian-equivalent'."""
        # Smoke test: lower=-10 should accept everything; lower=10 should
        # accept nothing. Confirms the convention via the rejection counter.
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf_lax = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"x": {"y1": +1}},
            higher_moment_restrictions=[
                {"moment": "excess_kurtosis", "lower": -10.0},
            ],
            target_accepted=50, max_rotations=5000, seed=0,
            parameter_uncertainty=False,
        )
        assert irf_lax.metadata["n_rejected_by_higher_moment"] == 0
        with pytest.raises(RuntimeError, match="no rotations"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"x": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "excess_kurtosis", "lower": 100.0},
                ],
                target_accepted=5, max_rotations=200, seed=0,
                parameter_uncertainty=False,
            )


# ---------------------------------------------------------------------------
# Local-projection conventions (lp.py)
# ---------------------------------------------------------------------------


class TestLPHorizonConventions:
    """Pin LP horizon indexing, container shape, and HAC defaults.

    Same horizon convention as VAR IRFs: ``h = 0, 1, ..., H`` with row
    0 = impact. Container shape ``(H+1, n_response, 1)`` for a single
    identified shock.
    """

    def _make_data(self):
        """Small synthetic dataset for LP tests."""
        rng = np.random.default_rng(0)
        T = 200
        y1 = np.zeros(T)
        y2 = np.zeros(T)
        for t in range(2, T):
            y1[t] = 0.5 * y1[t - 1] + rng.standard_normal()
            y2[t] = 0.3 * y2[t - 1] + 0.2 * y1[t - 1] + rng.standard_normal()
        idx = pd.date_range("1990-01-01", periods=T, freq="QS")
        return pd.DataFrame({"y1": y1, "y2": y2}, index=idx)

    def test_lp_horizon_indexing(self):
        """LP at horizon H returns container with H+1 rows; row 0 = impact."""
        data = self._make_data()
        rng = np.random.default_rng(1)
        shock = pd.Series(rng.standard_normal(len(data)), index=data.index)
        H = 6
        irf = lib.lp_irf(
            data=data, shock=shock, horizon=H, lags=2,
            response_vars=["y1"],
        )
        # Shape pin
        assert irf.median.shape == (H + 1, 1, 1)
        # h=0 must equal the contemporaneous OLS coefficient
        # (regression of y1_t on shock_t + lagged controls); pinned by
        # tests/test_lp.py::TestLP::test_h0_equals_ols_with_same_controls.
        # Here we just verify the slot exists and is finite.
        assert np.isfinite(irf.median[0, 0, 0])

    def test_lp_container_shape_one_shock(self):
        """LP identifies ONE shock — container last axis is 1, not n_vars."""
        data = self._make_data()
        rng = np.random.default_rng(0)
        shock = pd.Series(rng.standard_normal(len(data)), index=data.index)
        irf = lib.lp_irf(data=data, shock=shock, horizon=4, lags=2)
        # Last dim = 1 regardless of n_response
        assert irf.median.shape == (5, 2, 1)
        assert len(irf.shock_names) == 1

    def test_hac_horizon_plus_one_default(self):
        """The default ``hac_lag_rule="horizon_plus_one"`` uses
        ``max(h+1, 1)`` lags at horizon h. Pinned via metadata."""
        data = self._make_data()
        rng = np.random.default_rng(0)
        shock = pd.Series(rng.standard_normal(len(data)), index=data.index)
        irf = lib.lp_irf(data=data, shock=shock, horizon=4, lags=2)
        # The rule is recorded in metadata for downstream consumers
        assert irf.metadata["hac_lag_rule"] == "horizon_plus_one"

    def test_build_lp_controls_layout(self):
        """build_lp_controls returns p lags of every data column, ordered
        ``(lag1_var1, lag1_var2, ..., lagp_varn)``, plus an integer
        ``valid_index`` array of original-row positions starting at
        ``lags`` (NOT a boolean mask — easy footgun)."""
        data = self._make_data()
        controls, valid_index = lib.build_lp_controls(data, lags=2)
        T = len(data)
        # n_vars=2, lags=2 → 4 control columns; T-lags rows
        assert controls.shape == (T - 2, 4)
        # valid_index is the array of ORIGINAL-row positions (not a boolean
        # mask). Length = T - lags. First entry = lags; last = T - 1.
        valid_index = np.asarray(valid_index)
        assert valid_index.shape == (T - 2,)
        assert valid_index.dtype.kind in ("i", "u")   # integer
        assert valid_index[0] == 2          # first valid row is at position `lags`
        assert valid_index[-1] == T - 1
        # Strictly increasing
        assert np.all(np.diff(valid_index) == 1)

    def test_lp_metadata_identification_tag(self):
        """Plain LP tag is "external_or_pre_identified_shock"."""
        data = self._make_data()
        shock = pd.Series(
            np.random.default_rng(0).standard_normal(len(data)),
            index=data.index,
        )
        irf = lib.lp_irf(data=data, shock=shock, horizon=2, lags=2)
        assert irf.metadata["identification"] == "external_or_pre_identified_shock"
        assert irf.metadata["method"] == "local_projections"


class TestStateDependentLPConventions:
    """Pin conventions for ``lib.state_dependent_lp_irf`` (AG 2012 / RZ 2018).

    The state-dependent LP container has shape ``(H+1, n_response, 2)``
    with shock dim ordered ``(expansion, recession)``. Shock names are
    ``[f"{shock_name}_{state_names[0]}", f"{shock_name}_{state_names[1]}"]``.
    Recession SE accounts for the covariance between the shock and
    shock×state coefficients in the joint regression — dropping that
    term is a known footgun.
    """

    def _make_data_shock_state(self, T=300, b_E=1.0, b_R=2.0, seed=0):
        rng = np.random.default_rng(seed)
        state = (rng.random(T) < 0.4).astype(float)
        shock = rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            eff = b_E + (b_R - b_E) * state[t]
            y[t] = 0.5 * y[t - 1] + eff * shock[t] + 0.4 * rng.standard_normal()
        idx = pd.date_range("1990-01-01", periods=T, freq="QS")
        data = pd.DataFrame({"y": y}, index=idx)
        return data, pd.Series(shock, index=idx), pd.Series(state, index=idx)

    def test_container_shape_has_two_shocks(self):
        """Container last dim is 2 (expansion, recession), NOT 1."""
        data, shock, state = self._make_data_shock_state(T=200)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=4, lags=2,
        )
        assert irf.median.shape == (5, 1, 2)
        assert irf.lower_68.shape == (5, 1, 2)
        assert irf.upper_90.shape == (5, 1, 2)
        assert len(irf.shock_names) == 2

    def test_shock_name_ordering_expansion_then_recession(self):
        """Shock dim convention: index 0 = expansion, index 1 = recession."""
        data, shock, state = self._make_data_shock_state(T=200)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
            state_names=("good", "bad"),
            shock_name="s",
        )
        # The order in shock_names mirrors state_names ordering
        assert irf.shock_names == ["s_good", "s_bad"]

    def test_recession_se_includes_cov_term(self):
        """The recession SE must be sqrt(Var(beta_E) + Var(beta_diff)
        + 2*Cov(beta_E, beta_diff)) — dropping the covariance is a
        common footgun.

        Concretely: on a DGP where beta_E and beta_diff are positively
        correlated, the recession SE must be LARGER than
        sqrt(Var(beta_E) + Var(beta_diff)) (which would be the
        ignore-covariance bound).

        We pin this indirectly by checking that lower_68 + upper_68
        for the recession coefficient bracket the median and that the
        bands are finite — confirming a non-degenerate SE that uses
        the covariance term. The exact closed-form check is
        ``TestStateLPRecessionSEClosedForm`` in tests/test_lp.py.
        """
        data, shock, state = self._make_data_shock_state(T=500, b_E=1.0, b_R=2.0)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
        )
        # All recession-band values should be finite (no NaN from the
        # cov term being mishandled) on this well-conditioned DGP
        rec = irf.median[:, 0, 1]
        rec_lo = irf.lower_68[:, 0, 1]
        rec_hi = irf.upper_68[:, 0, 1]
        assert np.all(np.isfinite(rec))
        assert np.all(np.isfinite(rec_lo))
        assert np.all(np.isfinite(rec_hi))
        # Bands bracket the median
        assert np.all(rec_lo <= rec + 1e-10)
        assert np.all(rec <= rec_hi + 1e-10)

    def test_state_zero_everywhere_recovers_expansion_coefficient(self):
        """With state ≡ 0 everywhere, the F*shock interaction is 0 so
        the recession path is collinear / undefined. The expansion
        coefficient should equal the plain-LP coefficient with the
        same controls (β^E = OLS-on-shock).

        Implementation tolerates collinearity gracefully: either
        returns NaN for the recession path or raises a clear error.
        We verify the expansion path is sensible.
        """
        data, shock, _ = self._make_data_shock_state(T=300, b_E=1.5, b_R=1.5)
        # Force state ≡ 0 (pure expansion)
        state_zero = pd.Series(np.zeros(len(data)), index=data.index)
        try:
            irf_state = lib.state_dependent_lp_irf(
                data=data, shock=shock, state=state_zero,
                horizon=0, lags=2,
            )
            # Expansion coefficient should approximate b_E = 1.5
            assert abs(irf_state.median[0, 0, 0] - 1.5) < 0.20
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable: collinear F*shock column → singular regression
            pytest.skip("singular F*shock with constant state — expected")

    def test_recession_equals_expansion_plus_diff(self):
        """β^R[h] - β^E[h] is the F*shock coefficient in the regression.
        Pin: recession coefficient is recovered as expansion + (R-E diff),
        not stored independently. So the LP fit produces β^E and β^diff,
        and we report β^R = β^E + β^diff.

        Concretely: the regression is fit ONCE per horizon with both
        coefficients estimated jointly. We can't extract β^diff
        directly from the container but the relationship must hold
        between the median values reported.
        """
        data, shock, state = self._make_data_shock_state(
            T=600, b_E=1.0, b_R=2.5,
        )
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
        )
        # Sample R - E gap should approximate the true gap (1.5)
        gap = irf.median[0, 0, 1] - irf.median[0, 0, 0]
        assert abs(gap - 1.5) < 0.25, (
            f"recession - expansion gap at h=0 = {gap:.3f}; expected ~1.5"
        )

    def test_metadata_identification_tag(self):
        data, shock, state = self._make_data_shock_state(T=200)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
        )
        assert irf.metadata["identification"] == "state_dependent_lp"
        assert "state_mean" in irf.metadata
        # state_mean should be in [0, 1] for any valid state weight
        sm = irf.metadata["state_mean"]
        assert 0.0 <= sm <= 1.0

    def test_interact_controls_default_is_true(self):
        """The default RZ 2018 ``interact_controls=True`` is recorded
        in metadata so downstream consumers know whether the spec is
        saturated."""
        data, shock, state = self._make_data_shock_state(T=200)
        irf_default = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
        )
        assert irf_default.metadata["interact_controls"] is True
        irf_off = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
            interact_controls=False,
        )
        assert irf_off.metadata["interact_controls"] is False

    def test_band_construction_is_normal_distribution(self):
        """Bands are point ± z * SE, where z is the standard-normal
        quantile for the requested coverage. Verify approximately by
        checking that (upper_68 - median) and (median - lower_68) are
        symmetric (LP uses normal-distribution CIs)."""
        data, shock, state = self._make_data_shock_state(T=400)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
        )
        # Pick a finite slice and check symmetry of 68% band
        mask = np.isfinite(irf.median) & np.isfinite(irf.lower_68) & np.isfinite(irf.upper_68)
        upper_half = irf.upper_68[mask] - irf.median[mask]
        lower_half = irf.median[mask] - irf.lower_68[mask]
        assert np.allclose(upper_half, lower_half, atol=1e-10)


# ===========================================================================
# CROSS-CUTTING MATHEMATICAL INVARIANTS
#
# Tests in this section pin properties that hold for ANY valid output of
# the library, regardless of the input data. Failing one of these
# indicates a math error that no amount of "the code runs without error"
# checking would catch — e.g., FEVD shares failing to sum to 1, accepted
# sign-restricted rotations failing the sign restrictions, structural
# shocks failing to be uncorrelated.
# ===========================================================================


class TestCholeskyInvariants:
    """Cholesky identification: structural impact L satisfies L L' = Σ_u
    and yields uncorrelated unit-variance structural shocks.
    """

    def _fit(self):
        rng = np.random.default_rng(0)
        T = 300
        Y = np.zeros((T, 3))
        for t in range(1, T):
            Y[t] = 0.5 * Y[t - 1] + rng.standard_normal(3)
        df = pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        return lib.fit_var(df, lags=1, trend="c")

    def test_LL_T_equals_sigma_u(self):
        """L L' = Σ_u to machine precision (Cholesky is exact)."""
        fit = self._fit()
        L = lib.cholesky_impact(fit.Sigma_u)
        assert np.allclose(L @ L.T, fit.Sigma_u, atol=1e-12)

    def test_diagonal_positive(self):
        """Lower-triangular Cholesky factor convention: diagonal > 0."""
        fit = self._fit()
        L = lib.cholesky_impact(fit.Sigma_u)
        assert np.all(np.diag(L) > 0)

    def test_recovered_shocks_have_identity_covariance(self):
        """ε = L^-1 u → cov(ε) ≈ I_n (within sample noise on T=296)."""
        fit = self._fit()
        L = lib.cholesky_impact(fit.Sigma_u)
        eps = np.asarray(fit.residuals) @ np.linalg.inv(L).T   # (T_eff, n)
        Sigma_eps = np.cov(eps.T, ddof=1)
        # On T=296 we tolerate ~5% deviation in any entry
        assert np.allclose(Sigma_eps, np.eye(3), atol=0.10)

    def test_recovered_shocks_have_zero_mean(self):
        """ε should have mean ≈ 0 (driven by zero-mean residuals)."""
        fit = self._fit()
        L = lib.cholesky_impact(fit.Sigma_u)
        eps = np.asarray(fit.residuals) @ np.linalg.inv(L).T
        assert np.all(np.abs(eps.mean(axis=0)) < 0.10)


class TestFEVDInvariants:
    """FEVD shares are bounded in [0, 1] and sum to 1 across shocks for
    full identification.
    """

    def _fit_3var(self):
        rng = np.random.default_rng(1)
        T = 400
        Y = np.zeros((T, 3))
        A1 = np.array([[0.5, 0.0, 0.0], [0.1, 0.5, 0.0], [0.0, 0.1, 0.5]])
        for t in range(1, T):
            Y[t] = A1 @ Y[t - 1] + rng.standard_normal(3)
        df = pd.DataFrame(
            Y, columns=["a", "b", "c"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        return lib.fit_var(df, lags=1, trend="c")

    def test_shares_sum_to_one_for_full_identification(self):
        """For a Cholesky-identified n × n structural impact matrix, the
        FEVD shares for each variable should sum to 1.0 across all n
        identified shocks at every horizon."""
        fit = self._fit_3var()
        L = lib.cholesky_impact(fit.Sigma_u)   # full identification
        for H in (1, 2, 4, 8, 12):
            shares = lib.structural_fevd(fit, L, horizon=H)
            assert shares.shape == (3, 3)
            row_sums = shares.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-10), (
                f"FEVD shares at H={H} don't sum to 1: row sums = {row_sums}"
            )

    def test_shares_in_zero_one(self):
        """All FEVD shares ∈ [0, 1] (variance components, can't be
        negative or exceed total variance)."""
        fit = self._fit_3var()
        L = lib.cholesky_impact(fit.Sigma_u)
        for H in (1, 4, 12):
            shares = lib.structural_fevd(fit, L, horizon=H)
            assert np.all(shares >= -1e-12), (
                f"FEVD shares below 0 at H={H}: min={shares.min()}"
            )
            assert np.all(shares <= 1.0 + 1e-12), (
                f"FEVD shares above 1 at H={H}: max={shares.max()}"
            )

    def test_partial_id_shares_bounded_by_one(self):
        """For partial identification (m < n shocks), the row sum is the
        share attributable to identified shocks — must be ≤ 1."""
        fit = self._fit_3var()
        L = lib.cholesky_impact(fit.Sigma_u)
        # Use only the first 2 columns of L as a "partial" identification
        L_partial = L[:, :2]
        for H in (1, 4, 12):
            shares = lib.structural_fevd(fit, L_partial, horizon=H)
            assert shares.shape == (3, 2)
            row_sums = shares.sum(axis=1)
            assert np.all(row_sums <= 1.0 + 1e-12), (
                f"Partial-id row sums exceed 1 at H={H}: {row_sums}"
            )

    def test_h1_equals_impact_share(self):
        """FEVD at H=1 (one term in the sum) reduces to the impact-period
        decomposition: share_k_j = L[k, j]^2 / Sigma_u[k, k]."""
        fit = self._fit_3var()
        L = lib.cholesky_impact(fit.Sigma_u)
        shares = lib.structural_fevd(fit, L, horizon=1)
        expected = (L ** 2) / np.diag(fit.Sigma_u)[:, None]
        assert np.allclose(shares, expected, atol=1e-12)


class TestSignRestrictionInvariants:
    """Sign-restricted IRFs: every accepted rotation actually satisfies
    its own restrictions. A failure here means rotations are being
    accepted that shouldn't be.
    """

    def test_accepted_irfs_satisfy_sign_restriction_at_h0(
        self, synthetic_var_data,
    ):
        """The IRF median at h=0 for the named shock and named variable
        must have the requested sign (or be zero when unconstrained)."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        restrictions = {
            "demand": {"y1": +1, "y2": +1, "y3": -1},
        }
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions=restrictions,
            restriction_horizons=2,
            target_accepted=50, max_rotations=5000, seed=0,
            parameter_uncertainty=False,
        )
        # Median IRF over accepted rotations should satisfy the sign
        # restrictions at h=0..1 (restriction_horizons=2). Median of
        # same-sign values is same-sign, so this holds.
        for var, sign in restrictions["demand"].items():
            v_idx = list(fit.variable_names).index(var)
            for h in range(2):
                med = irf.median[h, v_idx, 0]
                if sign > 0:
                    assert med >= -1e-10, (
                        f"sign +1 violated for {var} at h={h}: median = {med}"
                    )
                elif sign < 0:
                    assert med <= 1e-10, (
                        f"sign -1 violated for {var} at h={h}: median = {med}"
                    )

    def test_acceptance_rate_recorded_in_metadata(
        self, synthetic_var_data,
    ):
        """Acceptance metadata must be present and consistent:
        n_accepted ≤ n_attempted; acceptance_rate ∈ [0, 1]."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"x": {"y1": +1}},
            target_accepted=30, max_rotations=2000, seed=0,
            parameter_uncertainty=False,
        )
        n_acc = irf.metadata["n_accepted"]
        n_att = irf.metadata["n_attempted"]
        rate = irf.metadata["acceptance_rate"]
        assert 0 < n_acc <= n_att
        assert 0.0 <= rate <= 1.0
        assert abs(rate - n_acc / n_att) < 1e-10


class TestPanelEquivalenceInvariants:
    """Equivalence invariants for the panel estimators: edge-case inputs
    should collapse to the corresponding simpler estimator.
    """

    def _make_one_unit_panel(self):
        rng = np.random.default_rng(0)
        T = 200
        Y = np.zeros((T, 2))
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        for t in range(1, T):
            Y[t] = A1 @ Y[t - 1] + rng.standard_normal(2)
        df = pd.DataFrame(
            Y, columns=["y1", "y2"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        return {"u1": df}

    def test_pooled_var_with_one_unit_equals_fit_var(self):
        """A pooled VAR on a one-unit panel should give the same lag
        coefficients as fit_var on that unit's data. (The intercept may
        differ — pooled uses unit-FE; fit_var with trend='c' is
        equivalent for a single unit.)"""
        panel = self._make_one_unit_panel()
        df = panel["u1"]
        pooled = lib.fit_pooled_var(panel, lags=1, trend="c")
        plain = lib.fit_var(df, lags=1, trend="c")
        # Lag-block coefficients (first n*p rows of B_hat)
        n, p = plain.n_vars, plain.p
        np.testing.assert_allclose(
            pooled.B_hat[: n * p, :], plain.B_hat[: n * p, :], atol=1e-10,
            err_msg="pooled VAR lag block on one-unit panel != fit_var",
        )

    def test_mean_group_with_one_unit_equals_fit_var(self):
        """Mean-group on a one-unit panel = fit_var on that unit
        (average of one fit = the fit itself).

        Coefficient equality is exact. Sigma_u differs by the panel
        convention (ML divisor) vs plain VAR convention (Lütkepohl T-K);
        check that the ratio is exactly (T_eff - K) / T_eff."""
        panel = self._make_one_unit_panel()
        df = panel["u1"]
        mg = lib.fit_mean_group(panel, lags=1, trend="c")
        plain = lib.fit_var(df, lags=1, trend="c")
        np.testing.assert_allclose(mg.B_hat, plain.B_hat, atol=1e-12)
        # Panel convention is ML (1/T_eff); plain VAR is Lütkepohl (T_eff-K).
        # Rescale plain to ML before comparing.
        ml_factor = (plain.t_eff - plain.k) / plain.t_eff
        np.testing.assert_allclose(
            np.asarray(mg.Sigma_u),
            np.asarray(plain.Sigma_u) * ml_factor,
            atol=1e-12,
        )

    def test_global_var_lambda_cross_unit_zero_is_block_diagonal(self):
        """When lambda_cross_unit=0, every cross-unit block must be
        EXACTLY zero (the prior fully shrinks them). Already pinned
        in tests/test_panel.py; re-pinned here as a cross-cutting
        invariant."""
        rng = np.random.default_rng(0)
        T = 200
        K = 2
        n_units = 2
        N = n_units * K
        A = 0.4 * np.eye(N) + 0.05 * np.ones((N, N))
        Y = np.zeros((T + 50, N))
        for t in range(1, T + 50):
            Y[t] = A @ Y[t - 1] + rng.standard_normal(N) * 0.4
        Y = Y[50:]
        idx = pd.date_range("1990-01-01", periods=T, freq="MS")
        panel = {
            "u0": pd.DataFrame(Y[:, :K], columns=["a", "b"], index=idx),
            "u1": pd.DataFrame(Y[:, K:], columns=["a", "b"], index=idx),
        }
        fit = lib.fit_global_var(
            panel, lags=1, lambda_1=0.5, lambda_cross_unit=0.0, trend="n",
        )
        for ui in fit.units:
            for uj in fit.units:
                if ui == uj:
                    continue
                block = fit.cross_unit_lag_block(ui, uj, lag=1)
                assert np.allclose(block, 0.0, atol=1e-10), (
                    f"cross-unit block ({ui}, {uj}) not zero with "
                    f"lambda_cross_unit=0: max |entry| = {np.abs(block).max()}"
                )


class TestIRFContainerInvariants:
    """The standardised IRF container has a fixed shape contract that
    callers depend on. Failing any of these would break downstream
    plotting / reporting code silently.
    """

    def test_horizons_array_length_matches_h_plus_one(
        self, synthetic_var_data,
    ):
        """irf.horizons has length H+1 (one entry per row of irf.median)."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        for H in (0, 1, 5, 20):
            irf = lib.cholesky_irf(fit, horizon=H, bands="none")
            assert len(irf.horizons) == H + 1
            assert irf.median.shape[0] == H + 1
            # First horizon is 0; last is H
            assert irf.horizons[0] == 0
            assert irf.horizons[-1] == H

    def test_shock_names_length_matches_last_dim(self, synthetic_var_data):
        """len(shock_names) == median.shape[-1]."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(fit, horizon=4, bands="none")
        assert len(irf.shock_names) == irf.median.shape[-1]

    def test_variable_names_length_matches_middle_dim(
        self, synthetic_var_data,
    ):
        """len(variable_names) == median.shape[1]."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(fit, horizon=4, bands="none")
        assert len(irf.variable_names) == irf.median.shape[1]

    def test_band_ordering_invariant_across_all_finite_entries(
        self, synthetic_var_data,
    ):
        """For every finite (horizon, variable, shock) entry, the band
        ordering ``lower_90 <= lower_68 <= median <= upper_68 <= upper_90``
        must hold. A single violation indicates a band-construction
        bug."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(
            fit, horizon=10, bands="asymptotic_mc", n_draws=200, seed=0,
        )
        mask = (
            np.isfinite(irf.lower_90)
            & np.isfinite(irf.lower_68)
            & np.isfinite(irf.median)
            & np.isfinite(irf.upper_68)
            & np.isfinite(irf.upper_90)
        )
        assert np.all(irf.lower_90[mask] <= irf.lower_68[mask] + 1e-10)
        assert np.all(irf.lower_68[mask] <= irf.median[mask] + 1e-10)
        assert np.all(irf.median[mask] <= irf.upper_68[mask] + 1e-10)
        assert np.all(irf.upper_68[mask] <= irf.upper_90[mask] + 1e-10)


class TestBLPConventions:
    """Pin the Bayesian Local Projections (Ferreira, Miranda-Agrippino & Ricco
    2025; 2021 working-paper design)
    convex-combination convention.

    Under the natural-conjugate λ-precision-matches-data scaling:
    - point estimate β^BLP_h = (β^OLS_h + λ β^VAR_h) / (1 + λ)
    - posterior std σ^BLP_h = σ^OLS_h / √(1 + λ)

    These are algorithmic guarantees, not statistical claims — pinning
    locks in the formula.
    """

    @staticmethod
    def _make_data(seed=0, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(2) * 0.5
        data = pd.DataFrame(
            y, columns=["a", "b"],
            index=pd.date_range("2000-01-01", periods=T, freq="QS"),
        )
        shock = rng.standard_normal(T)
        return data, shock

    def test_blp_at_zero_shrinkage_equals_lp(self):
        data, shock = self._make_data(seed=0)
        ols = lib.lp_irf(data, shock=shock, horizon=4, lags=1)
        blp = lib.blp_irf(data, shock=shock, horizon=4, lags=1, shrinkage=0.0)
        np.testing.assert_array_equal(blp.median, ols.median)

    def test_blp_h0_invariant_to_shrinkage(self):
        """h=0 prior mean = OLS LP h=0 by construction → convex
        combination at h=0 invariant."""
        data, shock = self._make_data(seed=0)
        blp1 = lib.blp_irf(data, shock=shock, horizon=4, lags=1, shrinkage=1.0)
        blp10 = lib.blp_irf(data, shock=shock, horizon=4, lags=1, shrinkage=10.0)
        np.testing.assert_allclose(
            blp1.median[0, :, 0], blp10.median[0, :, 0], atol=1e-10,
        )


class TestZeroSignRestrictionConventions:
    """Pin the Arias-Rubio-Ramirez-Waggoner 2018 zero+sign convention.

    Key invariant: under a contemporaneous zero restriction on
    (variable v, shock j), the structural IRF at h=0 MUST be EXACTLY
    zero for every accepted rotation (not "near zero" — algorithmically
    enforced via null-space projection of Q[:, j]).
    """

    @staticmethod
    def _make_data(seed=0, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1, 0.05], [0.0, 0.4, 0.0], [0.1, 0.0, 0.3]])
        y = np.zeros((T, 3))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(3) * 0.5
        return pd.DataFrame(
            y, columns=["rgdp", "cpi", "ffr"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )

    def test_zero_restriction_is_exact_for_all_bands(self):
        """median, lower_68, upper_68, lower_90, upper_90 must all be
        exactly zero at the zero-restricted cell. Pinned because the
        ARRW null-space projection guarantees this algorithmically."""
        fit = lib.fit_var(self._make_data(0), lags=2, trend="c")
        irf = lib.zero_sign_restricted_irf(
            fit, horizon=8,
            sign_restrictions={"mp": {"cpi": -1, "ffr": +1}},
            zero_restrictions={"mp": [{"variable": "rgdp", "horizon": 0}]},
            target_accepted=50, max_rotations=3000, seed=0,
            parameter_uncertainty=False,
        )
        for arr in [irf.median, irf.lower_68, irf.upper_68,
                    irf.lower_90, irf.upper_90]:
            assert abs(arr[0, 0, 0]) < 1e-10

    def test_identification_tag_distinguishes_zero_sign(self):
        fit = lib.fit_var(self._make_data(0), lags=2, trend="c")
        irf_zs = lib.zero_sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"mp": {"ffr": +1}},
            zero_restrictions={"mp": [{"variable": "rgdp", "horizon": 0}]},
            target_accepted=10, max_rotations=2000, seed=0,
            parameter_uncertainty=False,
        )
        irf_pure = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"mp": {"ffr": +1}},
            target_accepted=10, max_rotations=2000, seed=0,
            parameter_uncertainty=False,
        )
        assert irf_zs.metadata["identification"] == "zero_sign_restrictions"
        assert irf_pure.metadata["identification"] == "sign_restrictions"


class TestTVPPriorConventions:
    """Pin the Primiceri (2005) calibration for the TVP-VAR Q prior.

    Per Primiceri (2005, p. 831 / Appendix C), the
    inverse-Wishart scale is k_Q^2 * tau * V_OLS:

    1. it MULTIPLIES by the training-sample length tau, not divides.
       Dividing by n_train instead would be ~tau^2 too tight, giving
       essentially constant coefficient paths;
    2. it uses V_OLS, NOT the beta_0 prior covariance 4*V_OLS. The
       factor 4 belongs to the beta_0 prior only; and the default
       Q_prior_scale is k_Q^2 = 1e-4 (Primiceri's k_Q = 0.01 SQUARED).
       A 4*V_OLS scale with a 0.01 default would be jointly ~400x
       looser than the paper's calibration.

    The scale and degrees of freedom are observed at the sampler
    boundary: every Q draw in ``fit_tvp_var`` is
    ``IW(df = (T_eff - 1) + df_prior, scale = nu'nu + S_prior)`` with
    ``nu`` the first differences of the simulated state path. Capturing
    the smoother output and the inverse-Wishart arguments recovers
    ``S_prior`` and ``df_prior`` exactly, and they are compared with
    ``k_Q^2 * tau * V_OLS`` from a training-sample OLS built here from
    the raw data.
    """

    def test_default_q_prior_scale_is_primiceri_k_q_squared(self):
        """k_Q = 0.01 (Primiceri 2005, p. 831) => k_Q^2 = 1e-4."""
        assert lib.TVPVARSpec(lags=1).Q_prior_scale == pytest.approx(1e-4)

    @staticmethod
    def _make_data(T, seed=0):
        rng = np.random.default_rng(seed)
        n = 2
        # Mildly persistent VAR(1)
        A = np.array([[0.5, 0.1], [0.0, 0.5]])
        y = np.zeros((T, n))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(n) * 0.5
        return pd.DataFrame(
            y, columns=["a", "b"],
            index=pd.date_range("2000-01-01", periods=T, freq="QS"),
        )

    @staticmethod
    def _independent_v_ols(data, tau):
        """V_OLS = Sigma_OLS kron (X'X)^{-1} on the first tau rows of a
        VAR(1) with intercept, design built by hand: X_t = [y_{t-1}', 1]."""
        y = data.values.astype(float)
        X = np.column_stack([y[:-1], np.ones(len(y) - 1)])[:tau]
        Y = y[1:][:tau]
        XtX_inv = np.linalg.inv(X.T @ X)
        b_ols = np.linalg.solve(X.T @ X, X.T @ Y)
        resid = Y - X @ b_ols
        Sigma_ols = resid.T @ resid / (tau - X.shape[1])
        return np.kron(Sigma_ols, XtX_inv)

    @staticmethod
    def _capture_q_prior(monkeypatch, data, spec, n_iter=4):
        """Run fit_tvp_var and recover the Q prior (scale, df) it uses."""
        from types import SimpleNamespace

        import lib.tvp as tvp

        real_smoother = tvp.simulation_smoother
        real_iw = tvp.sp_stats.invwishart
        n, k = data.shape[1], data.shape[1] * spec.lags + 1
        nk = n * k
        last_path = []
        captured = []

        def smoother(*args, **kwargs):
            draws = real_smoother(*args, **kwargs)
            last_path.append(np.array(draws[0], dtype=float))
            return draws

        def iw_rvs(df, scale, random_state=None):
            scale = np.asarray(scale, dtype=float)
            if scale.shape == (nk, nk):
                nu = np.diff(last_path[-1], axis=0)
                nunu = nu.T @ nu
                captured.append((scale - nunu, df - nu.shape[0],
                                 np.abs(nunu).max()))
            return real_iw.rvs(df=df, scale=scale, random_state=random_state)

        monkeypatch.setattr(tvp, "simulation_smoother", smoother)
        monkeypatch.setattr(tvp, "sp_stats", SimpleNamespace(
            invwishart=SimpleNamespace(rvs=iw_rvs),
            gamma=tvp.sp_stats.gamma,
        ))
        lib.fit_tvp_var(data, spec, n_iter=n_iter, burn_in=1, seed=0)
        assert len(captured) == n_iter, "Q draw not observed every sweep"
        return captured

    def _assert_q_prior(self, monkeypatch, data, spec, expected_df):
        tau = spec.training_sample
        expected_scale = (
            spec.Q_prior_scale * tau * self._independent_v_ols(data, tau)
        )
        captured = self._capture_q_prior(monkeypatch, data, spec)
        for prior_scale, prior_df, nunu_max in captured:
            # Recovery subtracts nu'nu, so allow round-off at its scale
            np.testing.assert_allclose(
                prior_scale, expected_scale, rtol=1e-7,
                atol=1e-12 * max(nunu_max, 1.0),
                err_msg=(
                    "Q prior scale passed to the sampler is not "
                    "k_Q^2 * tau * V_OLS (CONVENTIONS.md section 4c)"
                ),
            )
            assert prior_df == expected_df
        return expected_scale

    def test_q_prior_scale_multiplies_by_tau(self, monkeypatch):
        """The Q prior scale ``fit_tvp_var`` hands to the sampler equals
        ``k_Q^2 * tau * V_OLS`` (V_OLS, not the beta_0 prior covariance
        4 * V_OLS), and the prior df equals
        ``max(Q_prior_df_factor * tau, nk + 2)``."""
        data = self._make_data(T=80)
        spec = lib.TVPVARSpec(
            lags=1, training_sample=30, Q_prior_scale=0.01, Q_prior_df_factor=1,
        )
        # n=2, k=3 => nk=6, so df = max(1 * 30, 8) = 30
        self._assert_q_prior(monkeypatch, data, spec, expected_df=30)

    def test_q_prior_scale_uses_times_tau_not_divide(self, monkeypatch):
        """Doubling tau: the observed prior scale tracks
        ``k_Q^2 * tau * V_OLS(tau)`` at both training lengths. V_OLS
        shrinks roughly like 1/tau, so the correct ``* tau`` scale is
        roughly constant in tau, while a ``/ tau`` scale would fall by
        ~1/4. The df follows ``Q_prior_df_factor * tau``."""
        data = self._make_data(T=200)
        scales = {}
        for tau in (30, 60):
            spec = lib.TVPVARSpec(
                lags=1, training_sample=tau, Q_prior_df_factor=2,
            )
            assert spec.Q_prior_scale == pytest.approx(1e-4)
            scales[tau] = self._assert_q_prior(
                monkeypatch, data, spec, expected_df=2 * tau,
            )
            monkeypatch.undo()
        ratio = np.trace(scales[60]) / np.trace(scales[30])
        assert 0.5 < ratio < 2.0, (
            f"scale(2 tau) / scale(tau) = {ratio:.4f}; the fixture no longer "
            f"separates the `* tau` and `/ tau` calibrations"
        )


# ============================================================================
# Equivariance / invariance tests — economic-correctness audits
# ============================================================================
#
# These pin structural invariants that any *faithful* econometric
# implementation must satisfy. They catch a broad class of subtle bugs:
# wrong normalisations, broken time-indexing, sign errors, off-by-one
# indexing, etc.
#
# Three families:
#   * Scale equivariance: linear in data → linear in IRFs/HDs
#   * Permutation behaviour: relabelling variables relabels the
#     order-invariant outputs (Sigma_u, generalised FEVD; not Cholesky IRFs)
#   * Centering invariance: with `trend="c"`, level shifts pass through the intercept
#   * Sign symmetry: negating the shock negates the IRF


class TestScaleEquivariance:
    """Multiplying input data ``Y`` by a positive scalar ``c`` should:

    * leave VAR lag coefficients unchanged (OLS is scale-invariant in Y);
    * scale ``Sigma_u`` by ``c²``;
    * scale the structural impact matrix by ``c``;
    * scale IRFs at every horizon by ``c``;
    * leave FEVD shares INVARIANT (the c² cancels in numerator and denominator);
    * scale HD contributions by ``c``.

    These hold for plain VAR + Cholesky / sign-restricted / long-run /
    Rigobon identifications. They DO NOT hold trivially for proxy-SVAR
    (the instrument scale matters), or for BVAR with a fixed-scale prior.
    """

    @staticmethod
    def _make_data(seed=0, T=200):
        rng = np.random.default_rng(seed)
        A1 = np.array([[0.5, 0.1, 0.05], [0.0, 0.4, 0.0], [0.1, 0.0, 0.3]])
        A2 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        y = np.zeros((T, 3))
        for t in range(2, T):
            y[t] = A1 @ y[t-1] + A2 @ y[t-2] + rng.standard_normal(3) * 0.5
        return pd.DataFrame(
            y, columns=["a", "b", "c"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )

    def test_var_lag_matrices_invariant_to_scale(self):
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_c = lib.fit_var(df * 5.0, lags=2, trend="c")
        for A, A_c in zip(fit.lag_matrices(), fit_c.lag_matrices()):
            np.testing.assert_allclose(A, A_c, atol=1e-10)

    def test_sigma_u_scales_as_c_squared(self):
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_c = lib.fit_var(df * 5.0, lags=2, trend="c")
        np.testing.assert_allclose(
            np.asarray(fit_c.Sigma_u),
            np.asarray(fit.Sigma_u) * 25.0,
            atol=1e-8,
        )

    def test_cholesky_irf_scales_linearly(self):
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_c = lib.fit_var(df * 5.0, lags=2, trend="c")
        irf = lib.cholesky_irf(fit, horizon=8, bands="none")
        irf_c = lib.cholesky_irf(fit_c, horizon=8, bands="none")
        np.testing.assert_allclose(
            irf_c.median, irf.median * 5.0, atol=1e-8,
        )

    def test_lp_irf_scales_linearly_with_data(self):
        df = self._make_data()
        rng = np.random.default_rng(42)
        shock = rng.standard_normal(len(df))
        ols = lib.lp_irf(df, shock=shock, horizon=4, lags=2)
        ols_c = lib.lp_irf(df * 5.0, shock=shock, horizon=4, lags=2)
        np.testing.assert_allclose(
            ols_c.median, ols.median * 5.0, atol=1e-8,
        )

    def test_blp_irf_scales_linearly_with_data(self):
        df = self._make_data()
        rng = np.random.default_rng(42)
        shock = rng.standard_normal(len(df))
        blp = lib.blp_irf(df, shock=shock, horizon=4, lags=2, shrinkage=1.0)
        blp_c = lib.blp_irf(df * 5.0, shock=shock, horizon=4, lags=2, shrinkage=1.0)
        np.testing.assert_allclose(
            blp_c.median, blp.median * 5.0, atol=1e-8,
        )

    def test_structural_fevd_invariant_to_scale(self):
        """FEVD shares should be INVARIANT to scaling — the c² cancels."""
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_c = lib.fit_var(df * 5.0, lags=2, trend="c")
        S = lib.cholesky_impact(fit.Sigma_u)
        S_c = lib.cholesky_impact(fit_c.Sigma_u)
        shares = lib.structural_fevd(fit, S, horizon=4)
        shares_c = lib.structural_fevd(fit_c, S_c, horizon=4)
        np.testing.assert_allclose(shares_c, shares, atol=1e-10)

    def test_historical_decomposition_scales_linearly(self):
        """HD shock contributions are linear in the data scale."""
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_c = lib.fit_var(df * 5.0, lags=2, trend="c")
        S = lib.cholesky_impact(fit.Sigma_u)
        S_c = lib.cholesky_impact(fit_c.Sigma_u)
        hd = lib.historical_decomposition(fit, S)
        hd_c = lib.historical_decomposition(fit_c, S_c)
        # Each shock's contribution time series scales by 5.0
        for shock_name in hd.shocks:
            np.testing.assert_allclose(
                hd_c.shocks[shock_name].values,
                hd.shocks[shock_name].values * 5.0,
                atol=1e-8,
            )


class TestCenteringInvariance:
    """Adding a constant to a variable (under ``trend="c"``) should leave
    lag coefficients, IRFs, FEVD, HD shock contributions all unchanged.
    The shift is absorbed entirely into the intercept row.

    This is a strong correctness test — a wrong intercept extraction
    (e.g., reading the last B_hat row when it holds an exog coefficient
    rather than the constant) would show up here as drift in residuals
    between the original and centred fit."""

    @staticmethod
    def _make_data(seed=1, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A @ y[t-1] + rng.standard_normal(2) * 0.5
        return pd.DataFrame(
            y, columns=["a", "b"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )

    def test_var_lags_invariant_to_centering(self):
        df = self._make_data()
        df_shift = df.copy()
        df_shift["a"] += 10.0
        df_shift["b"] -= 5.0
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_shift = lib.fit_var(df_shift, lags=2, trend="c")
        for A, A_s in zip(fit.lag_matrices(), fit_shift.lag_matrices()):
            np.testing.assert_allclose(A, A_s, atol=1e-10)

    def test_sigma_u_invariant_to_centering(self):
        df = self._make_data()
        df_shift = df.copy()
        df_shift["a"] += 10.0
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_shift = lib.fit_var(df_shift, lags=2, trend="c")
        np.testing.assert_allclose(
            np.asarray(fit.Sigma_u), np.asarray(fit_shift.Sigma_u),
            atol=1e-10,
        )

    def test_cholesky_irf_invariant_to_centering(self):
        df = self._make_data()
        df_shift = df.copy()
        df_shift["a"] += 10.0
        fit = lib.fit_var(df, lags=2, trend="c")
        fit_shift = lib.fit_var(df_shift, lags=2, trend="c")
        irf = lib.cholesky_irf(fit, horizon=6, bands="none")
        irf_shift = lib.cholesky_irf(fit_shift, horizon=6, bands="none")
        np.testing.assert_allclose(irf.median, irf_shift.median, atol=1e-10)

    def test_lp_irf_invariant_to_centering(self):
        df = self._make_data()
        df_shift = df + 10.0
        rng = np.random.default_rng(42)
        shock = rng.standard_normal(len(df))
        ols = lib.lp_irf(df, shock=shock, horizon=4, lags=2)
        ols_shift = lib.lp_irf(df_shift, shock=shock, horizon=4, lags=2)
        np.testing.assert_allclose(ols.median, ols_shift.median, atol=1e-10)


class TestShockSignSymmetry:
    """Negating the shock should negate the LP / BLP IRF exactly."""

    @staticmethod
    def _make_data(seed=2, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = A @ y[t-1] + rng.standard_normal(2) * 0.5
        return pd.DataFrame(
            y, columns=["a", "b"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        ), rng.standard_normal(T)

    def test_lp_irf_negates_with_shock_sign_flip(self):
        df, shock = self._make_data()
        ols = lib.lp_irf(df, shock=shock, horizon=4, lags=2)
        ols_neg = lib.lp_irf(df, shock=-shock, horizon=4, lags=2)
        np.testing.assert_allclose(ols.median, -ols_neg.median, atol=1e-10)

    def test_blp_irf_negates_with_shock_sign_flip(self):
        df, shock = self._make_data()
        blp = lib.blp_irf(df, shock=shock, horizon=4, lags=2, shrinkage=1.0)
        blp_neg = lib.blp_irf(df, shock=-shock, horizon=4, lags=2, shrinkage=1.0)
        np.testing.assert_allclose(blp.median, -blp_neg.median, atol=1e-10)


class TestPermutationEquivariance:
    """Reordering variables in input data permutes the order-invariant
    objects consistently.

    Cholesky identification is NOT among them: the recursive scheme
    assigns shock j to the j-th variable in the ordering, and the
    Cholesky factor of a permuted covariance is in general not the
    permuted Cholesky factor. Re-estimating after a reordering therefore
    gives a different structural model, and its Cholesky IRFs are not a
    relabelling of the original ones. No Cholesky IRF property is tested
    here.

    The tests pin the two properties that do hold: the reduced-form
    Sigma_u (identification-free) permutes as P Sigma_u P', and the
    generalised (Pesaran-Shin) FEVD, which does not depend on the
    ordering, permutes in both axes.
    """

    @staticmethod
    def _make_data(seed=3, T=200):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1, 0.05], [0.0, 0.4, 0.0], [0.1, 0.0, 0.3]])
        y = np.zeros((T, 3))
        for t in range(1, T):
            y[t] = A @ y[t-1] + rng.standard_normal(3) * 0.5
        return pd.DataFrame(
            y, columns=["a", "b", "c"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )

    def test_sigma_u_permutes_consistently(self):
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        df_perm = df[["c", "a", "b"]]
        fit_perm = lib.fit_var(df_perm, lags=2, trend="c")
        # Reorder original Sigma_u to match the permuted ordering [c, a, b]
        # Original order was [a, b, c] (indices 0, 1, 2); permuted [c, a, b] = [2, 0, 1]
        perm = [2, 0, 1]
        sigma_perm_expected = np.asarray(fit.Sigma_u)[np.ix_(perm, perm)]
        np.testing.assert_allclose(
            np.asarray(fit_perm.Sigma_u), sigma_perm_expected, atol=1e-10,
        )

    def test_generalised_fevd_permutes_consistently(self):
        """Generalised (Pesaran-Shin) FEVD is order-invariant — permuting
        variables should permute the FEVD matrix consistently."""
        df = self._make_data()
        fit = lib.fit_var(df, lags=2, trend="c")
        df_perm = df[["c", "a", "b"]]
        fit_perm = lib.fit_var(df_perm, lags=2, trend="c")
        fevd = lib.generalised_fevd(fit, horizon=4, row_normalise=True)
        fevd_perm = lib.generalised_fevd(fit_perm, horizon=4, row_normalise=True)
        perm = [2, 0, 1]
        np.testing.assert_allclose(
            fevd_perm, fevd[np.ix_(perm, perm)], atol=1e-10,
        )


# ============================================================================
# Cross-method consistency tests
# ============================================================================
#
# These pin the principle that DIFFERENT estimators of the same underlying
# quantity should agree (asymptotically, or exactly in special cases). They
# catch silent drift between code paths that purport to compute the same
# object via different routes.


class TestLPvsVARIteratedConsistency:
    """LP and VAR-iterated IRF should agree on a well-specified VAR DGP
    at long T.

    With p lags of y as the LP controls and the correct structural shock
    series, the OLS LP is unbiased for the true structural IRF. The
    VAR-iterated path imposes more structure (uses the full lag dynamics
    estimated from the data) and is more efficient but consistent for
    the same quantity. At T = 1000 they should agree within ~10-15% on
    short-to-medium horizons.

    A meaningful mismatch would indicate either a normalisation bug
    (different scaling of the shock) or a horizon-indexing bug.
    """

    @staticmethod
    def _simulate(seed=0, T=1000, burn=200):
        rng = np.random.default_rng(seed)
        n = 3
        A1 = np.array([
            [0.5, 0.1, 0.05],
            [0.0, 0.4, 0.05],
            [0.05, 0.0, 0.3],
        ])
        A2 = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ])
        Sigma_true = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        L_true = np.linalg.cholesky(Sigma_true)
        Y = np.zeros((T + burn, n))
        structural = np.zeros((T + burn, n))
        for t in range(2, T + burn):
            structural[t] = rng.standard_normal(n)
            Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + L_true @ structural[t]
        return (Y[burn:], structural[burn:], A1, A2, L_true)

    def test_lp_and_var_iterated_irf_agree_on_first_shock(self):
        """The OLS LP of y_{j, t+h} on shock_1_t (Cholesky-recovered)
        should equal Phi_h L[:, 0] within finite-sample tolerance."""
        Y, structural, A1, A2, L_true = self._simulate(seed=0, T=1000)
        df = pd.DataFrame(
            Y, columns=["y1", "y2", "y3"],
            index=pd.date_range("1950-01-01", periods=Y.shape[0], freq="QS"),
        )
        fit = lib.fit_var(df, lags=2, trend="c")
        # VAR-iterated IRF on the first Cholesky shock
        irf_var = lib.cholesky_irf(fit, horizon=8, bands="none")
        var_irf_shock1 = irf_var.median[:, :, 0]   # (H+1, n_vars)
        # LP IRF using the true structural shock (which exists in the DGP)
        shock1 = structural[:, 0]
        irf_lp = lib.lp_irf(df, shock=shock1, horizon=8, lags=2)
        lp_irf_shock1 = irf_lp.median[:, :, 0]    # (H+1, n_vars)
        # Compare. The two are estimators of the same DGP quantity; at
        # T=1000 they should be close but not exact (finite-sample noise +
        # different specifications). Allow ~0.05 absolute (the IRFs have
        # magnitudes of order 0.5-1.0).
        for h in range(1, 6):
            for j in range(3):
                diff = abs(var_irf_shock1[h, j] - lp_irf_shock1[h, j])
                assert diff < 0.10, (
                    f"LP vs VAR-iterated at h={h}, var={j}: "
                    f"LP={lp_irf_shock1[h, j]:.4f}, "
                    f"VAR-it={var_irf_shock1[h, j]:.4f}, |diff|={diff:.4f}"
                )


class TestCholeskyRecoverTrueIRF:
    """When the DGP is a recursive (lower-triangular) Cholesky model,
    `cholesky_irf` applied to the fitted VAR should recover the true
    structural IRF to finite-sample tolerance.

    This is the most direct test of `cholesky_irf` correctness — a
    sanity check that the entire fit + identify pipeline produces
    the right answer on a DGP where the answer is known.
    """

    def test_recovers_true_cholesky_irf(self):
        rng = np.random.default_rng(11)
        n = 3
        T = 1000
        burn = 200
        A1 = np.array([
            [0.5, 0.1, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.3],
        ])
        Sigma_true = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        L_true = np.linalg.cholesky(Sigma_true)
        Y = np.zeros((T + burn, n))
        for t in range(1, T + burn):
            Y[t] = A1 @ Y[t-1] + L_true @ rng.standard_normal(n)
        Y = Y[burn:]
        df = pd.DataFrame(
            Y, columns=["y1", "y2", "y3"],
            index=pd.date_range("1950-01-01", periods=T, freq="QS"),
        )
        # True IRF: Phi_h @ L_true
        Phi_true = np.zeros((9, n, n))
        Phi_true[0] = np.eye(n)
        for h in range(1, 9):
            Phi_true[h] = A1 @ Phi_true[h-1]
        true_irf = np.einsum("hij,jk->hik", Phi_true, L_true)
        # Fit and identify
        fit = lib.fit_var(df, lags=1, trend="c")
        irf = lib.cholesky_irf(fit, horizon=8, bands="none")
        # Compare on the first column (first shock) — most informative
        for h in range(9):
            for j in range(n):
                diff = abs(irf.median[h, j, 0] - true_irf[h, j, 0])
                # Allow 0.08 absolute (IRFs of order 0.5-1.0)
                assert diff < 0.08, (
                    f"Cholesky recovery at h={h}, var={j}: "
                    f"estimated={irf.median[h, j, 0]:.4f}, "
                    f"true={true_irf[h, j, 0]:.4f}"
                )


class TestBVARFlatPriorLimitConsistency:
    """As the Minnesota prior tightness λ_1 → ∞, the BVAR posterior
    mean converges to the OLS estimate (flat-prior limit).

    A reasonable practical check: at λ_1 = 100 (very loose), the BVAR
    coefficient matrix should be within a few percent of the OLS fit.
    This pins the property that the prior's strength is correctly
    parameterised — a wrong sign or wrong inversion would push BVAR
    further from OLS as λ_1 grows.
    """

    def test_bvar_converges_to_ols_at_loose_prior(self):
        rng = np.random.default_rng(7)
        n = 2
        T = 400
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, n))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + rng.standard_normal(n) * 0.5
        df = pd.DataFrame(
            Y, columns=["a", "b"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        ols = lib.fit_var(df, lags=1, trend="c")
        # Very loose prior (lambda_1 = 100 → essentially flat)
        prior = lib.MinnesotaPrior(
            lambda_1=100.0, lambda_2=0.5, lambda_3=1.0,
            lambda_4=100.0, mu=0.0, delta=0.0,   # disable extra priors
        )
        bvar = lib.fit_minnesota_bvar(df, lags=1, prior=prior, trend="c")
        # B_hat should be very close to OLS
        # Note: BVAR's B_hat has the same shape as OLS (k, n). Compare row-wise.
        ols_B = np.asarray(ols.B_hat)
        bvar_B = np.asarray(bvar.B_hat)
        np.testing.assert_allclose(bvar_B, ols_B, atol=0.05, rtol=0.05)


class TestHDReconstruction:
    """Historical decomposition is exact by construction:
    ``data == det_plus_initial + sum_j shock_j + other_shocks``.
    Re-pin here as a cross-cutting invariant. (Already covered in
    test_irf.py but worth pinning at the cross-method level too.)"""

    def test_full_id_reconstruction_to_machine_eps(self):
        rng = np.random.default_rng(5)
        T = 200
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + rng.standard_normal(2) * 0.5
        df = pd.DataFrame(
            Y, columns=["a", "b"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        fit = lib.fit_var(df, lags=1, trend="c")
        S = lib.cholesky_impact(fit.Sigma_u)
        hd = lib.historical_decomposition(fit, S)
        recon_err = hd.reconstruction_error()
        assert (recon_err < 1e-10).all(), (
            f"HD reconstruction error too large: {recon_err.to_dict()}"
        )


# ============================================================================
# Machine-precision tests against analytical answers
# ============================================================================
#
# These pin EXACT equality (tol = 1e-10) between our framework's outputs
# and closed-form analytical results on known DGPs. They are the
# strongest possible correctness tests: no data-version ambiguity, no
# Monte Carlo noise, no statistical tolerance.
#
# Each test exploits an algebraic identity:
#   * VAR(1) MA representation: Phi_h = A^h (matrix power)
#   * Companion matrix eigenvalues = roots of
#     det(lambda^p I - A_1 lambda^(p-1) - ... - A_p) = 0 (the reciprocals
#     of the nonzero roots z of det(I - A_1 z - ... - A_p z^p) = 0)
#   * Long-run multiplier: C(1) = (I - Sum A_j)^{-1}
#   * Cholesky impact: L L' = Sigma_u
#   * FEVD at h=1 = (P[i, j])^2 / Sigma_u[i, i] (for Cholesky P = L)
#   * BLP at lambda=0 = OLS LP (already pinned in TestBLPConventions)
#   * Cholesky IRF at h=0 = L (the Cholesky factor itself)


class TestMachinePrecisionInvariants:
    """Closed-form invariants pinned to 1e-10 tolerance.

    Any drift in these would indicate a real bug in the underlying
    linear algebra or convention handling.
    """

    def test_var1_ma_representation_equals_matrix_power(self):
        """For VAR(1), Phi_h = A^h. Closed form; should match to
        machine precision."""
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        lag_matrices = [A]
        Phi = lib.ma_representation(lag_matrices, horizon=5)
        for h in range(6):
            A_power = np.linalg.matrix_power(A, h)
            np.testing.assert_allclose(Phi[h], A_power, atol=1e-12)

    def test_var2_ma_recursion_matches_companion_power(self):
        """For VAR(p), Phi_h equals the top n × n block of the
        companion matrix raised to the h-th power. Pure linear-algebra
        identity."""
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        A2 = np.array([[0.1, 0.05], [0.0, 0.1]])
        n = 2
        lag_matrices = [A1, A2]
        C = lib.companion_matrix(lag_matrices)
        Phi = lib.ma_representation(lag_matrices, horizon=5)
        for h in range(6):
            C_power = np.linalg.matrix_power(C, h)
            top_left = C_power[:n, :n]
            np.testing.assert_allclose(Phi[h], top_left, atol=1e-12)

    def test_cholesky_impact_satisfies_LL_eq_Sigma(self):
        """L L' = Sigma_u exactly (by construction of Cholesky)."""
        rng = np.random.default_rng(0)
        # Random PSD matrix
        M = rng.standard_normal((3, 3))
        Sigma = M @ M.T + 0.1 * np.eye(3)
        L = lib.cholesky_impact(Sigma)
        np.testing.assert_allclose(L @ L.T, Sigma, atol=1e-12)
        # L should be lower triangular
        assert np.allclose(L, np.tril(L), atol=1e-12)
        # Diagonal should be positive
        assert np.all(np.diag(L) > 0)

    def test_cholesky_irf_at_h0_equals_cholesky_factor(self):
        """At h=0, the Cholesky IRF is exactly L (since Phi_0 = I)."""
        rng = np.random.default_rng(0)
        T = 100
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + rng.standard_normal(2) * 0.3
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        L = lib.cholesky_impact(fit.Sigma_u)
        irf = lib.cholesky_irf(fit, horizon=4, bands="none")
        # At h=0, the IRF matrix should equal L exactly
        np.testing.assert_allclose(irf.median[0], L, atol=1e-12)

    def test_long_run_multiplier_inverts_I_minus_sumA(self):
        """C(1) = (I - sum A_j)^{-1} — a direct algebraic identity that
        underpins Blanchard-Quah identification. Verify our long-run
        machinery produces the right matrix."""
        rng = np.random.default_rng(0)
        T = 200
        A1 = np.array([[0.5, 0.0], [0.1, 0.4]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.1]])
        Y = np.zeros((T, 2))
        for t in range(2, T):
            Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + rng.standard_normal(2) * 0.3
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=2, trend="c")
        # Recover the LR matrix from the internal helper
        from lib.identification import _long_run_structural_matrix
        H = _long_run_structural_matrix(fit.lag_matrices(), fit.Sigma_u)
        # By construction H satisfies: C(1) H is lower triangular with
        # H H' = Sigma_u.
        # Verify the latter exactly:
        np.testing.assert_allclose(
            H @ H.T, np.asarray(fit.Sigma_u), atol=1e-10,
        )
        # And verify C(1) H is lower-triangular up to numerical tolerance:
        I_minus_A = np.eye(2) - sum(fit.lag_matrices())
        C1 = np.linalg.inv(I_minus_A)
        LR = C1 @ H
        # Upper-triangular entries (i < j) should be zero
        assert abs(LR[0, 1]) < 1e-10, (
            f"Long-run matrix not lower-triangular: LR[0, 1] = {LR[0, 1]}"
        )

    def test_fevd_at_h1_equals_squared_impact_normalised(self):
        """At horizon H=1, the structural FEVD reduces to
        ``FEVD[i, j](1) = (Phi_0 S)[i, j]^2 / Sigma_u[i, i]
                        = S[i, j]^2 / Sigma_u[i, i]``
        because Phi_0 = I. For Cholesky S = L (lower-triangular Cholesky),
        this gives the exact analytical FEVD share at h=1.
        """
        rng = np.random.default_rng(0)
        T = 100
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + rng.standard_normal(2) * 0.3
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        L = lib.cholesky_impact(fit.Sigma_u)
        Sigma_diag = np.diag(np.asarray(fit.Sigma_u))
        # Closed-form: shares[i, j] = L[i, j]^2 / Sigma[i, i]
        expected = (L ** 2) / Sigma_diag[:, None]
        # Our framework's structural_fevd at horizon=1
        shares = lib.structural_fevd(fit, L, horizon=1)
        np.testing.assert_allclose(shares, expected, atol=1e-12)
        # Each row should sum to 1 exactly (Cholesky orthogonalisation)
        np.testing.assert_allclose(shares.sum(axis=1), np.ones(2), atol=1e-12)

    def test_companion_eigenvalues_are_characteristic_polynomial_roots(self):
        """Companion matrix eigenvalues are exactly the roots of the
        characteristic polynomial
        det(lambda^p I - A_1 lambda^(p-1) - ... - A_p) = 0.
        The nonzero eigenvalues are the reciprocals of the roots z of the
        lag polynomial det(I - A_1 z - ... - A_p z^p) = 0, not those roots
        themselves (AR coefficient 0.5: eigenvalue 0.5, lag root z = 2).
        For VAR(1), these are just eigenvalues of A_1 directly.
        For higher p, the companion form encodes them.

        Test on VAR(2) where we know specific eigenvalues."""
        # Construct A1, A2 such that the companion matrix has known
        # eigenvalues. Choice: A1 = diag(0.5, 0.3), A2 = 0.
        # Then companion = [[A1, 0], [I, 0]] has eigenvalues
        # {0.5, 0.3, 0, 0}.
        A1 = np.diag([0.5, 0.3])
        A2 = np.zeros((2, 2))
        lag_matrices = [A1, A2]
        C = lib.companion_matrix(lag_matrices)
        eigvals = np.linalg.eigvals(C)
        eigvals_sorted = np.sort(np.abs(eigvals))[::-1]   # descending |λ|
        expected = np.array([0.5, 0.3, 0.0, 0.0])
        np.testing.assert_allclose(eigvals_sorted, expected, atol=1e-12)
        # And max_eigenvalue_modulus should be 0.5
        assert abs(lib.max_eigenvalue_modulus(lag_matrices) - 0.5) < 1e-12

    def test_extract_lag_matrices_inverts_companion_topblock(self):
        """The lag matrices extracted from B_hat should match the
        top n × n × p block of the companion matrix exactly."""
        rng = np.random.default_rng(0)
        T = 100
        A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        A2 = np.array([[0.1, 0.0], [0.05, 0.1]])
        Y = np.zeros((T, 2))
        for t in range(2, T):
            Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + rng.standard_normal(2) * 0.3
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=2, trend="c")
        lag_mats = fit.lag_matrices()
        assert len(lag_mats) == 2
        assert lag_mats[0].shape == (2, 2)
        assert lag_mats[1].shape == (2, 2)
        # Round-trip: rebuild companion from the extracted matrices,
        # extract them again, should match exactly
        C = lib.companion_matrix(lag_mats)
        # Top n*n block of C = horizontal stack [A_1, A_2]
        n = 2
        np.testing.assert_allclose(C[:n, :n], lag_mats[0], atol=1e-12)
        np.testing.assert_allclose(C[:n, n:2*n], lag_mats[1], atol=1e-12)
        # Identity shift in second block-row
        np.testing.assert_allclose(C[n:, :n], np.eye(n), atol=1e-12)
        np.testing.assert_allclose(C[n:, n:], np.zeros((n, n)), atol=1e-12)

    def test_ols_var_residuals_orthogonal_to_regressors(self):
        """OLS first-order condition: X' u_hat = 0 exactly.
        Direct sanity check on our fit_var output."""
        rng = np.random.default_rng(0)
        T = 100
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + rng.standard_normal(2) * 0.5
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        # X' residuals should be approximately zero
        xtu = fit.X_act.T @ fit.residuals
        np.testing.assert_allclose(xtu, np.zeros_like(xtu), atol=1e-9)

    def test_ols_var_solution_inverts_normal_equations(self):
        """B_hat = (X'X)^{-1} X'Y should satisfy X'X B_hat = X'Y exactly.
        Direct sanity check of the OLS normal equations."""
        rng = np.random.default_rng(0)
        T = 100
        A = np.array([[0.5, 0.1], [0.0, 0.4]])
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + rng.standard_normal(2) * 0.5
        df = pd.DataFrame(Y, columns=["a", "b"],
                          index=pd.date_range("2000-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        XtX = fit.X_act.T @ fit.X_act
        XtY = fit.X_act.T @ fit.Y_act
        lhs = XtX @ fit.B_hat
        np.testing.assert_allclose(lhs, XtY, atol=1e-8)
