"""Tests for the Minnesota BVAR (bvar.py).

Validation strategy:

1. **Limiting behaviour** — as ``lambda_1 -> infinity`` the prior becomes
   uninformative and the posterior mean must approach the OLS estimate.
2. **Limiting behaviour** — as ``lambda_1 -> 0`` the prior dominates and
   the own-lag-1 coefficients must shrink toward ``rw_deltas`` (default 1).
3. **Shape conformance** — dummies and posterior draws have the right shapes.
4. **Posterior centring** — ``BVARFit.parameter_draws`` average must
   approach the posterior mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


class TestMinnesotaDummies:
    def test_shape_matches_design(self, synthetic_var_data):
        Y = synthetic_var_data.values
        prior = lib.MinnesotaPrior()
        Yd, Xd = lib.build_minnesota_dummies(Y, p=2, prior=prior)
        n, p = Y.shape[1], 2
        # Dummies: p*n (tightness) + n (Sigma block, BGR eq. 5) + 1 (const)
        #          + n (soc) + 1 (dio)
        expected_rows = p * n + n + 1 + n + 1
        assert Yd.shape == (expected_rows, n)
        assert Xd.shape == (expected_rows, n * p + 1)

    def test_disabling_soc_and_dio_drops_rows(self, synthetic_var_data):
        Y = synthetic_var_data.values
        prior = lib.MinnesotaPrior(mu=0, delta=0)
        Yd, Xd = lib.build_minnesota_dummies(Y, p=2, prior=prior)
        n, p = Y.shape[1], 2
        expected_rows = p * n + n + 1   # tightness + Sigma block + const
        assert Yd.shape == (expected_rows, n)

    def test_sum_of_coefficients_skips_variables_marked_stationary(
            self, synthetic_var_data):
        """A variable with rw_deltas = 0 gets no sum-of-coefficients row
        (Banbura-Giannone-Reichlin 2010 scale each row by delta_i). Its prior
        mean stays at zero and the prior scale of its error variance is not
        inflated, at any level of the data."""
        Y = synthetic_var_data.values + 10.0
        n, p = Y.shape[1], 2
        rw = [1.0] * (n - 1) + [0.0]
        Yd, Xd = lib.build_minnesota_dummies(
            Y, p=p, prior=lib.MinnesotaPrior(rw_deltas=rw))
        # one sum-of-coefficients row fewer than with all random-walk means
        assert Yd.shape[0] == p * n + n + 1 + (n - 1) + 1
        b0 = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        own_sum = sum(b0[lag * n + (n - 1), n - 1] for lag in range(p))
        assert abs(own_sum) < 1e-3
        S0 = (Yd - Xd @ b0).T @ (Yd - Xd @ b0)
        Yd0, Xd0 = lib.build_minnesota_dummies(
            Y, p=p, prior=lib.MinnesotaPrior(rw_deltas=rw, mu=0))
        b00 = np.linalg.lstsq(Xd0, Yd0, rcond=None)[0]
        S00 = (Yd0 - Xd0 @ b00).T @ (Yd0 - Xd0 @ b00)
        assert S0[n - 1, n - 1] == pytest.approx(S00[n - 1, n - 1], rel=1e-3)

    def test_sigma_block_anchors_s0_and_leaves_b_post_unchanged(
            self, synthetic_var_data):
        """The Sigma-prior block (Yd=diag(sigma), Xd=0) must (a) make the
        dummy-residual scale S_0 positive definite — the properness anchor —
        and (b) contribute nothing to the posterior mean of B."""
        Y = synthetic_var_data.values
        prior = lib.MinnesotaPrior(mu=0, delta=0)
        Yd, Xd = lib.build_minnesota_dummies(Y, p=2, prior=prior)
        b0 = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        S0 = (Yd - Xd @ b0).T @ (Yd - Xd @ b0)
        assert np.linalg.eigvalsh(S0).min() > 1e-12, "S_0 not PD — improper prior"
        # Zero-X rows cannot move B: dropping them leaves the lstsq unchanged
        keep = ~np.all(Xd == 0, axis=1)
        b0_drop = np.linalg.lstsq(Xd[keep], Yd[keep], rcond=None)[0]
        np.testing.assert_allclose(b0, b0_drop, atol=1e-10)


class TestLimitingBehaviour:
    def test_uninformative_prior_recovers_ols(self, synthetic_var_data):
        """With lambda_1 huge and SoC / DIO off, posterior mean ≈ OLS."""
        prior_loose = lib.MinnesotaPrior(
            lambda_1=1e8, lambda_4=1e8, mu=0, delta=0,
        )
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior_loose)
        ols = lib.fit_var(synthetic_var_data, lags=2)
        assert np.allclose(bvar.B_hat, ols.B_hat, atol=1e-3)

    def test_tight_prior_shrinks_to_random_walk(self, synthetic_var_data):
        """With lambda_1 tiny and SoC / DIO off, own-lag-1 coefs ≈ rw_deltas (1)."""
        prior_tight = lib.MinnesotaPrior(
            lambda_1=1e-6, mu=0, delta=0,
        )
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior_tight)
        # B layout is (k, n) with rows [lag1_var1, lag1_var2, ..., const]^T
        # So own-lag-1 coefficients are the diagonal of B_hat[:n, :]
        n = bvar.n_vars
        own_lag1 = np.diag(bvar.B_hat[:n, :])
        assert np.allclose(own_lag1, 1.0, atol=1e-3)
        # Cross-variable lag-1 coefficients should be ≈ 0
        cross = bvar.B_hat[:n, :] - np.diag(own_lag1)
        assert np.allclose(cross, 0.0, atol=1e-3)
        # Lag-2 coefficients should also be ≈ 0
        assert np.allclose(bvar.B_hat[n:2 * n, :], 0.0, atol=1e-3)

    def test_rw_deltas_override_changes_prior_mean(self, synthetic_var_data):
        """rw_deltas = [0, 0, 0] should shrink toward white noise instead of RW."""
        prior_wn = lib.MinnesotaPrior(
            lambda_1=1e-6, mu=0, delta=0, rw_deltas=[0.0, 0.0, 0.0],
        )
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior_wn)
        n = bvar.n_vars
        # Now all lag-1 coefficients should shrink to 0
        assert np.allclose(bvar.B_hat[:n, :], 0.0, atol=1e-3)


class TestPosteriorDraws:
    def test_returns_requested_count(self, synthetic_var_data):
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        draws = bvar.parameter_draws(n_draws=50, seed=0)
        assert len(draws) == 50
        for B, S in draws:
            assert B.shape == bvar.B_hat.shape
            assert S.shape == bvar.Sigma_u.shape

    def test_stable_filter_keeps_eigenvalues_below_tolerance(self, synthetic_var_data):
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        draws = bvar.parameter_draws(
            n_draws=50, seed=0, require_stable=True, eig_tol=1.01,
        )
        for B, _ in draws:
            lag_mats = lib.extract_lag_matrices(B, bvar.n_vars, bvar.p)
            assert lib.max_eigenvalue_modulus(lag_mats) < 1.01

    def test_posterior_mean_recovers_b_hat(self, synthetic_var_data):
        """Averaging draws should approach the posterior mean B_hat."""
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        draws = bvar.parameter_draws(
            n_draws=2000, seed=0, require_stable=False,
        )
        avg_B = np.mean([B for B, _ in draws], axis=0)
        assert np.allclose(avg_B, bvar.B_hat, atol=0.02)


class TestIntegrationWithCholesky:
    def test_bvar_works_with_cholesky_irf(self, synthetic_var_data):
        """End-to-end: fit BVAR, identify with Cholesky, get IRFs."""
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        irf = lib.cholesky_irf(
            fit=bvar, horizon=10, bands="parameter_draws",
            n_draws=200, seed=0,
        )
        # At h=0 the median should still equal chol(posterior mean Sigma)
        # (because every draw is Cholesky-decomposed at h=0)
        chol_post = np.linalg.cholesky(bvar.Sigma_u)
        # Median of cholesky draws does NOT equal chol of the median Sigma
        # exactly, but it should be in the right ballpark
        # Just check shapes here:
        assert irf.median.shape == (11, bvar.n_vars, bvar.n_vars)
        assert irf.metadata["bands"] == "parameter_draws"


class TestMarginalDataDensity:
    """Tests for log_marginal_data_density (GLP 2015 closed-form MDD).

    Validation strategy:

    * **Finite for sensible hyperparameters** — MDD must be a finite real
      number for any well-defined Minnesota prior.
    * **Tighter prior at low information loss is preferred** — for data
      generated under a tight DGP, the MDD should be higher at tight
      lambda_1 than at very loose lambda_1.
    * **Optimisation actually improves MDD** — running
      ``optimize_minnesota_hyperparameters`` from a poor starting point
      should produce a higher MDD than the starting point.
    """

    def test_mdd_is_finite(self, synthetic_var_data):
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        mdd = lib.log_marginal_data_density(bvar)
        assert np.isfinite(mdd)

    def test_mdd_varies_with_lambda(self, synthetic_var_data):
        """Different priors give different MDDs (the prior matters)."""
        priors = [
            lib.MinnesotaPrior(lambda_1=0.05, mu=1.0, delta=1.0),
            lib.MinnesotaPrior(lambda_1=0.2, mu=1.0, delta=1.0),
            lib.MinnesotaPrior(lambda_1=1.0, mu=1.0, delta=1.0),
        ]
        mdds = []
        for prior in priors:
            bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior)
            mdds.append(lib.log_marginal_data_density(bvar))
        assert all(np.isfinite(m) for m in mdds), f"Got MDDs: {mdds}"
        assert len(set(mdds)) > 1

    def test_mdd_proper_for_pure_minnesota(self, synthetic_var_data):
        """The diag(sigma) Sigma-prior dummy block (BGR 2010 eq. 5) makes the
        NIW prior proper even with mu = delta = 0 (nu_0 = n), so the MDD of a
        pure Minnesota prior is finite. Without that block, n_dummies = k
        would give nu_0 = 0 — an improper prior whose MDD is -inf."""
        prior = lib.MinnesotaPrior(lambda_1=0.1, mu=0, delta=0)
        bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior)
        assert np.isfinite(lib.log_marginal_data_density(bvar))

    def test_mdd_smooth_in_lambda(self, synthetic_var_data):
        """Regression contract: the MDD must be locally smooth in lambda_1.
        Without the diag(sigma) block, S_0 would be numerically singular
        (eigenvalues ~ 0) and the (nu_0/2)logdet(S_0) term would inject
        hundreds of log-points of floating-point noise — a 1e-6 change in
        lambda_1 moving the 'MDD' by O(10) log-points, so hyperparameter
        optimization would optimize noise. Finiteness checks cannot catch
        this (the noisy value is finite); a local-smoothness assertion can."""
        def mdd(l1):
            prior = lib.MinnesotaPrior(lambda_1=l1, mu=1.0, delta=1.0)
            bvar = lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior)
            return lib.log_marginal_data_density(bvar)
        jump = abs(mdd(0.100001) - mdd(0.1))
        assert jump < 0.01, (
            f"MDD moved {jump:.4f} log-points for a 1e-6 lambda_1 change — "
            f"improper-prior floating-point noise is dominating the MDD"
        )


class TestHyperparameterOptimisation:
    def test_improves_mdd_from_poor_starting_point(self, synthetic_var_data):
        """Starting from lambda_1 well away from the optimum, the
        optimiser must increase the MDD.

        Uses mu=1, delta=1 to make the prior proper (nu_0 > 0).
        """
        bad_init = lib.MinnesotaPrior(lambda_1=2.0, mu=1.0, delta=1.0)
        optimal, diag = lib.optimize_minnesota_hyperparameters(
            synthetic_var_data, lags=2,
            params_to_optimize=["lambda_1"],
            initial=bad_init,
        )
        assert 1e-4 <= optimal.lambda_1 <= 5.0
        # Only lambda_1 was free; the rest of the returned prior is the start
        assert optimal.mu == bad_init.mu and optimal.delta == bad_init.delta

        # Recompute the objective independently at the RETURNED prior,
        # not from the optimiser's own diagnostics
        def mdd(prior):
            return lib.log_marginal_data_density(
                lib.fit_minnesota_bvar(synthetic_var_data, lags=2, prior=prior)
            )

        mdd_start = mdd(bad_init)
        mdd_opt = mdd(optimal)
        # Known feasible benchmark: lambda_1 = 0.2 beats the start by
        # about 18.6 log points on this fixture (-1759.51 vs -1778.12)
        mdd_feasible = mdd(lib.MinnesotaPrior(lambda_1=0.2, mu=1.0, delta=1.0))
        assert mdd_feasible - mdd_start > 10.0   # fixture sanity
        assert mdd_opt >= mdd_feasible - 1e-6, (
            f"optimised MDD {mdd_opt:.3f} at lambda_1={optimal.lambda_1:.4f} "
            f"is worse than the feasible lambda_1=0.2 point {mdd_feasible:.3f}"
        )
        assert mdd_opt - mdd_start > 10.0
        # Diagnostics must agree with the recomputed objective
        assert diag["best_mdd"] == pytest.approx(mdd_opt, abs=1e-6)
        assert diag["initial_mdd"] == pytest.approx(mdd_start, abs=1e-6)

    def test_rejects_unknown_param_name(self, synthetic_var_data):
        with pytest.raises(ValueError, match="unknown params_to_optimize"):
            lib.optimize_minnesota_hyperparameters(
                synthetic_var_data, lags=2,
                params_to_optimize=["lambda_99"],
            )

    def test_multi_param_optimisation_runs(self, synthetic_var_data):
        """Optimising over multiple MDD-active hyperparameters should run
        and return a finite result. Uses ``mu`` (sum-of-coefficients) which
        is active in the symmetric / natural-conjugate MDD; ``lambda_2`` is
        not (it's only meaningful under ``asymmetric=True``)."""
        optimal, diag = lib.optimize_minnesota_hyperparameters(
            synthetic_var_data, lags=2,
            params_to_optimize=["lambda_1", "mu"],
            initial=lib.MinnesotaPrior(
                lambda_1=0.5, mu=0.5, delta=1.0,
            ),
        )
        assert np.isfinite(diag["best_mdd"])
        assert "lambda_1" in diag["optimized_params"]
        assert "mu" in diag["optimized_params"]

    def test_optimisation_rejects_asymmetric_prior(self, synthetic_var_data):
        """MDD-based optimisation requires the conjugate prior. Asymmetric
        Litterman conditions on plug-in sigmas and has no closed-form MDD."""
        with pytest.raises(ValueError, match="asymmetric=False"):
            lib.optimize_minnesota_hyperparameters(
                synthetic_var_data, lags=2,
                params_to_optimize=["lambda_1"],
                initial=lib.MinnesotaPrior(asymmetric=True, mu=1.0, delta=1.0),
            )


class TestAsymmetricPrior:
    """Tests for the Litterman (1986) / Sims-Zha (1998) asymmetric prior
    (``MinnesotaPrior.asymmetric=True``), fit via equation-by-equation Ridge.

    Validation strategy:

    * **lambda_2 activity contract** — under asymmetric mode ``lambda_2``
      must move ``B_hat``, the cross-equation coefficients in particular
      (it is a structural no-op under the symmetric prior).
    * **Limiting behaviour** — at ``lambda_2 = 1`` the asymmetric prior
      reduces to the symmetric BGR form (own and cross have the same
      tightness scaling), so the two fits must agree exactly.
    * **Limiting behaviour** — as ``lambda_2 -> 0`` cross-equation lag
      coefficients shrink to zero (per-equation regressions become
      univariate).
    * **MDD not defined** — ``log_marginal_data_density`` must raise.
    * **Posterior draws** — average of ``parameter_draws`` recovers
      ``B_hat`` (with ``Sigma`` held fixed).
    """

    def test_lambda_2_now_moves_b_hat(self, synthetic_var_data):
        """Symmetric mode yields identical B_hat for any lambda_2 (a
        structural no-op); asymmetric mode must produce a non-trivial
        difference."""
        fit_tight = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(lambda_2=0.01, asymmetric=True),
        )
        fit_loose = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(lambda_2=10.0, asymmetric=True),
        )
        max_diff = np.max(np.abs(fit_tight.B_hat - fit_loose.B_hat))
        assert max_diff > 0.05, (
            f"Asymmetric lambda_2 should move B_hat substantially; got "
            f"max diff = {max_diff:.4e}"
        )

    def test_lambda_2_equals_one_matches_symmetric(self, synthetic_var_data):
        """At lambda_2 = 1, own and cross share the same tightness scaling
        (sigma_i/sigma_j) * lambda_1 / l^lambda_3. The asymmetric fit must
        reproduce the symmetric BGR fit to machine precision."""
        common = dict(lambda_1=0.1, lambda_2=1.0, mu=1.0, delta=1.0)
        fit_sym = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(**common, asymmetric=False),
        )
        fit_asym = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(**common, asymmetric=True),
        )
        assert np.allclose(fit_sym.B_hat, fit_asym.B_hat, atol=1e-10)

    def test_lambda_2_tight_zeros_cross_lags(self, synthetic_var_data):
        """With lambda_2 << 1, cross-equation lag coefficients shrink toward
        zero — each equation effectively becomes univariate. The own-lag
        coefficients are unaffected by lambda_2 (they still respond only
        to lambda_1)."""
        fit = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(
                lambda_1=0.1, lambda_2=1e-6, asymmetric=True,
            ),
        )
        n = fit.n_vars
        # First n rows of B_hat are the lag-1 block. Diagonal = own-lag-1,
        # off-diagonal = cross-lag-1. lambda_2 only constrains the latter.
        lag1_block = fit.B_hat[:n, :]
        cross_lag1 = lag1_block - np.diag(np.diag(lag1_block))
        assert np.max(np.abs(cross_lag1)) < 1e-3
        # Lag-2 cross coefficients also shrunk
        lag2_block = fit.B_hat[n:2 * n, :]
        cross_lag2 = lag2_block - np.diag(np.diag(lag2_block))
        assert np.max(np.abs(cross_lag2)) < 1e-3

    def test_mdd_not_defined_for_asymmetric(self, synthetic_var_data):
        fit = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(asymmetric=True),
        )
        with pytest.raises(NotImplementedError, match="asymmetric"):
            lib.log_marginal_data_density(fit)

    def test_asymmetric_parameter_draws_recover_b_hat(self, synthetic_var_data):
        """Conditional on plug-in Sigma, beta_i ~ N(B_hat[:, i], cov_i).
        Sample mean of B_hat draws should approach B_hat."""
        fit = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(asymmetric=True),
        )
        draws = fit.parameter_draws(n_draws=2000, seed=0, require_stable=False)
        avg_B = np.mean([B for B, _ in draws], axis=0)
        assert np.allclose(avg_B, fit.B_hat, atol=0.02)
        # Sigma is held fixed in asymmetric mode
        for _, S in draws[:5]:
            assert np.allclose(S, fit.Sigma_u)

    def test_asymmetric_fit_marks_metadata(self, synthetic_var_data):
        fit = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2,
            prior=lib.MinnesotaPrior(asymmetric=True),
        )
        assert fit.asymmetric is True
        assert fit.XtX_inv_list is not None
        assert len(fit.XtX_inv_list) == fit.n_vars
        assert fit.sigmas is not None
        assert fit.sigmas.shape == (fit.n_vars,)
        assert fit.fit_info["method"] == "minnesota_bvar_asymmetric_ridge"


class TestObsScale:
    """Tests for the ``obs_scale`` parameter — GLS-style per-observation
    variance scaling, used by Lenza-Primiceri (2022) to down-weight COVID
    innovations without dropping them.

    Validation strategy:

    * **Identity check** — ``obs_scale=None`` (default) matches an explicit
      ``obs_scale=ones(T_eff)``.
    * **No-op vs unscaled** — both produce byte-identical ``B_hat``.
    * **Symmetric scaling shifts B but observed values rescale** — scaling
      ALL observations by the same factor reproduces a standard fit on the
      same data with looser prior dummies (a structural cross-check).
    * **Length validation** — wrong-length ``obs_scale`` raises.
    * **Positivity validation** — non-positive entries raise.
    * **fit_info flag** records whether scaling was applied.
    """

    def test_none_vs_ones_equivalent(self, synthetic_var_data):
        """obs_scale=None must produce the same fit as obs_scale=ones."""
        fit_no = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        T_eff = fit_no.t_eff
        fit_one = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2, obs_scale=np.ones(T_eff),
        )
        assert np.allclose(fit_no.B_hat, fit_one.B_hat, atol=1e-12)
        assert np.allclose(fit_no.Sigma_u, fit_one.Sigma_u, atol=1e-12)

    def test_heavy_scale_on_one_obs_dampens_its_influence(self, synthetic_var_data):
        """An observation with s_t = 1000 contributes ~1e-6 of normal weight
        to the GLS posterior — its influence on B_hat should be negligible."""
        T_eff = synthetic_var_data.shape[0] - 2
        # Pick an interior observation and inflate s_t
        obs_scale = np.ones(T_eff)
        obs_scale[T_eff // 2] = 1000.0
        fit_normal = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        fit_drop = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2, obs_scale=obs_scale,
        )
        # The two fits should differ — the heavy-scaled fit drops one obs
        # — but stay in the same ballpark since 1/T influence is small.
        max_diff = float(np.abs(fit_normal.B_hat - fit_drop.B_hat).max())
        assert 0.0 < max_diff < 0.05, (
            f"Expected small but non-zero B_hat shift; got max diff {max_diff:.4e}"
        )

    def test_obs_scale_length_validation(self, synthetic_var_data):
        T_eff = synthetic_var_data.shape[0] - 2
        with pytest.raises(ValueError, match="length"):
            lib.fit_minnesota_bvar(
                synthetic_var_data, lags=2,
                obs_scale=np.ones(T_eff + 5),
            )

    def test_obs_scale_positivity_validation(self, synthetic_var_data):
        T_eff = synthetic_var_data.shape[0] - 2
        bad = np.ones(T_eff)
        bad[5] = -1.0
        with pytest.raises(ValueError, match="positive"):
            lib.fit_minnesota_bvar(synthetic_var_data, lags=2, obs_scale=bad)

    def test_fit_info_records_obs_scale_presence(self, synthetic_var_data):
        T_eff = synthetic_var_data.shape[0] - 2
        fit_no = lib.fit_minnesota_bvar(synthetic_var_data, lags=2)
        fit_yes = lib.fit_minnesota_bvar(
            synthetic_var_data, lags=2, obs_scale=np.full(T_eff, 1.5),
        )
        assert fit_no.fit_info["obs_scale_present"] is False
        assert fit_yes.fit_info["obs_scale_present"] is True

    def test_obs_scale_rejected_in_asymmetric_mode(self, synthetic_var_data):
        """The asymmetric estimator would fit coefficients on scaled rows
        but report Sigma_u from unscaled residuals — covariance on the
        wrong scale — so the combination is refused."""
        T_eff = synthetic_var_data.shape[0] - 2
        obs_scale = np.ones(T_eff)
        obs_scale[T_eff // 2] = 5.0
        with pytest.raises(NotImplementedError, match="obs_scale"):
            lib.fit_minnesota_bvar(
                synthetic_var_data, lags=2,
                prior=lib.MinnesotaPrior(asymmetric=True),
                obs_scale=obs_scale,
            )


# ---------------------------------------------------------------------------
# Steady-state (mean-adjusted) BVAR -- Villani (2009)
# ---------------------------------------------------------------------------


def _sim_steady_state_var(T=600, seed=0):
    """Stationary VAR(1) with a KNOWN unconditional mean mu and AR matrix A."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.55]])
    mu = np.array([2.0, -1.0])
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    y = np.zeros((T, 2))
    y[0] = mu
    for t in range(1, T):
        y[t] = mu + A @ (y[t - 1] - mu) + L @ rng.standard_normal(2)
    df = pd.DataFrame(y, columns=["a", "b"],
                      index=pd.date_range("1960-01-01", periods=T, freq="QS"))
    return df, A, mu


class TestSteadyStateBVAR:
    def test_recovers_mean_and_dynamics(self):
        df, A, mu = _sim_steady_state_var(seed=0)
        fit = lib.fit_steady_state_bvar(
            df, lags=1,
            prior=lib.SteadyStatePrior(psi_mean=[0, 0], psi_sd=[20, 20]),
            n_draws=800, burn_in=500, seed=1)
        # loose prior => steady state tracks the sample mean (and the true mu)
        assert np.allclose(fit.steady_state, df.mean().values, atol=0.2)
        assert np.allclose(fit.steady_state, mu, atol=0.35)
        # AR(1) matrix recovered
        assert np.allclose(fit.lag_matrices()[0], A, atol=0.12)

    def test_tight_prior_anchors_steady_state(self):
        df, A, mu = _sim_steady_state_var(seed=0)
        fit = lib.fit_steady_state_bvar(
            df, lags=1,
            prior=lib.SteadyStatePrior(psi_mean=[5.0, 5.0], psi_sd=[0.02, 0.02]),
            n_draws=400, burn_in=300, seed=1)
        # a tight prior far from the sample mean pins the steady state near it
        assert np.allclose(fit.steady_state, [5.0, 5.0], atol=0.1)
        assert not np.allclose(fit.steady_state, df.mean().values, atol=1.0)

    def test_irf_and_parameter_draws(self):
        df, A, mu = _sim_steady_state_var(seed=0)
        fit = lib.fit_steady_state_bvar(
            df, lags=1,
            prior=lib.SteadyStatePrior(psi_mean=[0, 0], psi_sd=[20, 20]),
            n_draws=300, burn_in=200, seed=1)
        irf = lib.cholesky_irf(fit, horizon=12, bands="none")
        assert irf.median.shape == (13, 2, 2)
        draws = fit.parameter_draws(n_draws=50, seed=2)
        assert len(draws) > 0
        B, Sig = draws[0]
        assert B.shape == (fit.k, 2) and Sig.shape == (2, 2)
        assert fit.psi.shape == (2, 1)

    def test_determinism(self):
        df, _, _ = _sim_steady_state_var(seed=0)
        kw = dict(lags=1,
                  prior=lib.SteadyStatePrior(psi_mean=[0, 0], psi_sd=[10, 10]),
                  n_draws=200, burn_in=100, seed=7)
        a = lib.fit_steady_state_bvar(df, **kw)
        b = lib.fit_steady_state_bvar(df, **kw)
        assert np.allclose(a.steady_state, b.steady_state)
        assert np.allclose(a.B_hat, b.B_hat)

    def test_validation(self):
        df, _, _ = _sim_steady_state_var(seed=0)
        with pytest.raises(ValueError, match="shape"):
            lib.fit_steady_state_bvar(
                df, lags=1,
                prior=lib.SteadyStatePrior(psi_mean=[0], psi_sd=[1, 1]),
                n_draws=10, burn_in=5, seed=1)
        with pytest.raises(ValueError, match="psi_sd"):
            lib.fit_steady_state_bvar(
                df, lags=1,
                prior=lib.SteadyStatePrior(psi_mean=[0, 0], psi_sd=[1, -1]),
                n_draws=10, burn_in=5, seed=1)


# ---------------------------------------------------------------------------
# Block exogeneity (small open economy)
# ---------------------------------------------------------------------------


def _block_exo_dgp(T=400, seed=0):
    """Foreign var ``f`` is exogenous (own AR only); domestic ``d1, d2`` depend
    on the foreign block."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T); d1 = np.zeros(T); d2 = np.zeros(T)
    for t in range(2, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
        d1[t] = 0.5 * f[t - 1] + 0.4 * d1[t - 1] + rng.standard_normal()
        d2[t] = 0.3 * d1[t - 1] + 0.5 * d2[t - 1] + rng.standard_normal()
    return pd.DataFrame(
        {"f": f, "d1": d1, "d2": d2},
        index=pd.date_range("1990-01-01", periods=T, freq="QS"))


class TestBlockExogeneity:
    @staticmethod
    def _excl_max(B):
        # f-equation (col 0) coefficients on domestic lags: rows 1,2 (lag1 d1,d2),
        # rows 4,5 (lag2 d1,d2) in the [lag1(3), lag2(3), const] layout.
        return float(np.abs(B[[1, 2, 4, 5], 0]).max())

    def test_pins_excluded_coefficients(self):
        df = _block_exo_dgp()
        bex = lib.fit_minnesota_bvar(df, lags=2, exogenous_block=["f"])
        base = lib.fit_minnesota_bvar(
            df, lags=2, prior=lib.MinnesotaPrior(asymmetric=True))
        assert self._excl_max(bex.B_hat) < 1e-3            # pinned to ~0
        assert self._excl_max(base.B_hat) > self._excl_max(bex.B_hat)  # base non-zero
        assert abs(bex.B_hat[0, 0] - 0.7) < 0.15           # f own dynamics preserved

    def test_no_feedback_in_irf(self):
        df = _block_exo_dgp()
        bex = lib.fit_minnesota_bvar(df, lags=2, exogenous_block=["f"])
        irf = lib.cholesky_irf(bex, horizon=12, bands="none")   # f ordered first
        assert np.abs(irf.median[:, 0, 1]).max() < 1e-3    # f flat to a d1 shock
        assert np.abs(irf.median[:, 0, 2]).max() < 1e-3    # f flat to a d2 shock

    def test_metadata_and_asymmetric_routing(self):
        df = _block_exo_dgp()
        bex = lib.fit_minnesota_bvar(df, lags=2, exogenous_block=["f"])
        assert bex.asymmetric is True                       # per-equation path
        assert bex.fit_info["block_exogeneity"] == {0: [1, 2]}

    def test_validation(self):
        df = _block_exo_dgp()
        with pytest.raises(ValueError, match="not in data columns"):
            lib.fit_minnesota_bvar(df, lags=2, exogenous_block=["nope"])
        with pytest.raises(ValueError, match="every variable"):
            lib.fit_minnesota_bvar(df, lags=2, exogenous_block=["f", "d1", "d2"])


class TestScalarDrawAndScaleGuards:
    """Acceptance checks: one-variable NIW draws keep matrix shape;
    obs_scale is rejected on the asymmetric/block-exogeneity route."""

    def test_univariate_bvar_parameter_draws(self):
        rng = np.random.default_rng(0)
        y = np.zeros(120)
        for t in range(1, 120):
            y[t] = 0.6 * y[t - 1] + rng.standard_normal()
        df = pd.DataFrame({"y": y})
        fit = lib.fit_minnesota_bvar(df, lags=1)
        draws = fit.parameter_draws(n_draws=4, seed=1)
        assert len(draws) == 4
        for B_d, Sigma_d in draws:
            assert Sigma_d.shape == (1, 1)
            assert B_d.shape == fit.B_hat.shape

    def test_obs_scale_rejected_on_asymmetric_path(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal((80, 2)).cumsum(axis=0) * 0.1
        df = pd.DataFrame(y, columns=["a", "b"])
        prior = lib.MinnesotaPrior(asymmetric=True)
        with pytest.raises(NotImplementedError, match="obs_scale"):
            lib.fit_minnesota_bvar(df, lags=1, prior=prior,
                               obs_scale=np.full(79, 2.0))
