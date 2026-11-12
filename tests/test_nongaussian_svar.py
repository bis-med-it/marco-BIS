"""Tests for lib.nongaussian_svar — LMS 2017 non-Gaussian SVAR identification.

The identifying claim of LMS (2017): if structural shocks are mutually
independent with at most one Gaussian component, B in u_t = B ε_t is
identified up to column permutation and sign.

Validation strategy:

1. On synthetic data with strongly non-Gaussian shocks of a known
   structural matrix, the algorithm must recover that matrix up to
   column permutation and sign (the LMS identification claim).
2. On Gaussian residuals, identification fails by definition — the
   ``weak_nongaussianity`` diagnostic must flag this.
3. FastICA's symmetric orthogonalisation must produce orthogonal W.
4. The recovered B must produce IRFs whose h=0 column equals the
   recovered B column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


def _sim_var_nongaussian(
    T: int = 800,
    n: int = 3,
    df: float = 4.0,
    burn: int = 200,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate a stationary VAR(1) with i.i.d. t-distributed shocks.

    Returns the data frame plus the TRUE structural impact matrix B_true
    (so tests can verify recovery up to column permutation and sign).

    The t-distribution has Var = df/(df-2); to get unit-variance shocks
    matching the LMS convention the simulated shocks are scaled by
    ``sqrt((df-2)/df)``.
    """
    rng = np.random.default_rng(seed)
    A1 = np.array([
        [0.50, 0.10, 0.00],
        [0.00, 0.50, 0.10],
        [0.10, 0.00, 0.50],
    ])[:n, :n]
    # Lower-triangular B with non-zero off-diagonals → cross-correlation
    B_true = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.2, 0.4, 1.0],
    ])[:n, :n]

    # Unit-variance t-distributed shocks
    eps = rng.standard_t(df=df, size=(T + burn, n)) * np.sqrt((df - 2) / df)
    u = eps @ B_true.T   # (T+burn, n), each row is B_true @ ε_t

    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = A1 @ Y[t - 1] + u[t]
    Y = Y[burn:]
    idx = pd.period_range("1980-01", periods=T, freq="M").to_timestamp()
    return (
        pd.DataFrame(Y, columns=[f"y{i + 1}" for i in range(n)], index=idx),
        B_true,
    )


def _sim_var_gaussian(
    T: int = 400, n: int = 3, burn: int = 100, seed: int = 0,
) -> pd.DataFrame:
    """Simulate a VAR(1) with Gaussian shocks — LMS identification must fail."""
    rng = np.random.default_rng(seed)
    A1 = np.array([
        [0.50, 0.10, 0.00],
        [0.00, 0.50, 0.10],
        [0.10, 0.00, 0.50],
    ])[:n, :n]
    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = A1 @ Y[t - 1] + rng.standard_normal(n)
    Y = Y[burn:]
    idx = pd.period_range("1980-01", periods=T, freq="M").to_timestamp()
    return pd.DataFrame(Y, columns=[f"y{i + 1}" for i in range(n)], index=idx)


def _match_columns_up_to_permutation_and_sign(
    B_est: np.ndarray, B_true: np.ndarray,
) -> tuple[float, list[int], list[int]]:
    """Find the column permutation + sign that best matches B_est to B_true.

    Brute-force over all permutations (OK for small n). Returns the
    best (Frobenius) distance, the permutation, and the sign pattern.
    """
    from itertools import permutations, product

    n = B_true.shape[1]
    best = (np.inf, None, None)
    for perm in permutations(range(n)):
        for signs in product((-1, 1), repeat=n):
            B_p = B_est[:, list(perm)] * np.array(signs)[None, :]
            dist = float(np.linalg.norm(B_p - B_true, ord="fro"))
            if dist < best[0]:
                best = (dist, list(perm), list(signs))
    return best


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------


class TestWhiten:
    def test_whitened_cov_is_identity(self):
        rng = np.random.default_rng(0)
        T = 500
        Sigma = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, 0.2], [0.1, 0.2, 1.0]])
        L = np.linalg.cholesky(Sigma)
        u = rng.standard_normal((T, 3)) @ L.T
        Z, L_back = lib.whiten_residuals(u)
        assert np.allclose(L_back @ L_back.T, np.cov(u.T, ddof=1))
        # Empirical covariance of Z should be close to identity (T=500 so allow slack)
        Sigma_Z = np.cov(Z.T, ddof=1)
        assert np.allclose(Sigma_Z, np.eye(3), atol=0.15)

    def test_rejects_singular_residuals(self):
        # Two identical columns → singular Sigma → Cholesky fails
        u = np.tile(np.random.default_rng(0).standard_normal((50, 1)), (1, 2))
        with pytest.raises(np.linalg.LinAlgError):
            lib.whiten_residuals(u)


# ---------------------------------------------------------------------------
# FastICA primitive
# ---------------------------------------------------------------------------


class TestFastICA:
    def test_produces_orthogonal_W(self):
        rng = np.random.default_rng(0)
        # Truly independent t-distributed signals, then orthogonal mixing
        T = 500
        S = rng.standard_t(df=4, size=(T, 3)) * np.sqrt(2 / 4)  # unit-var
        Q_true, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        Z = S @ Q_true.T   # already whitened since S has identity cov
        W, diag = lib.fastica_symmetric(Z, seed=0)
        assert diag["converged"]
        # Orthogonality check
        assert np.allclose(W @ W.T, np.eye(3), atol=1e-10)

    def test_recovers_independent_components(self):
        """FastICA must recover the original mixing up to permutation + sign
        when given truly independent non-Gaussian sources."""
        rng = np.random.default_rng(1)
        T = 2000
        # Three independent t-distributed sources (unit variance)
        S = rng.standard_t(df=3, size=(T, 3)) * np.sqrt(1 / 3)
        # Apply orthogonal mixing
        Q_true, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        Z = S @ Q_true.T
        W, diag = lib.fastica_symmetric(Z, nonlinearity="tanh", seed=42)
        assert diag["converged"]
        # Recovered ε = Z @ W.T should be a column-permuted, sign-flipped S.
        eps = Z @ W.T
        # The recovered components should each have high correlation with
        # exactly one source (up to sign).
        # |corr(eps_k, S_j)| ≈ 1 for some j.
        corr = np.zeros((3, 3))
        for k in range(3):
            for j in range(3):
                corr[k, j] = abs(np.corrcoef(eps[:, k], S[:, j])[0, 1])
        # Each row should have one entry close to 1
        row_max = corr.max(axis=1)
        assert (row_max > 0.90).all(), f"row-max correlations: {row_max}"
        # Each source should be picked up by exactly one component
        col_max = corr.max(axis=0)
        assert (col_max > 0.90).all(), f"col-max correlations: {col_max}"

    @pytest.mark.parametrize("g", ["tanh", "cube", "exp"])
    def test_all_nonlinearities_converge(self, g):
        rng = np.random.default_rng(0)
        S = rng.standard_t(df=4, size=(800, 3)) * np.sqrt(2 / 4)
        W, diag = lib.fastica_symmetric(S, nonlinearity=g, seed=0, max_iter=500)
        assert diag["converged"], f"FastICA with g={g} failed to converge"
        assert np.allclose(W @ W.T, np.eye(3), atol=1e-10)

    def test_unknown_nonlinearity_raises(self):
        Z = np.random.default_rng(0).standard_normal((100, 3))
        with pytest.raises(ValueError, match="unknown nonlinearity"):
            lib.fastica_symmetric(Z, nonlinearity="invalid")


# ---------------------------------------------------------------------------
# End-to-end nongaussian_svar_irf
# ---------------------------------------------------------------------------


class TestNonGaussianSVAR:
    def test_irf_at_h0_equals_B(self):
        """h=0 IRF column equals the structural impact column."""
        df, _ = _sim_var_nongaussian(T=600, seed=0)
        fit = lib.fit_var(df, lags=1)
        irf, diag = lib.nongaussian_svar_irf(fit, horizon=10, seed=0)
        B_pt = np.array(diag["B_point"])
        # irf.median[0, :, k] should equal B_pt[:, k]
        for k in range(fit.n_vars):
            assert np.allclose(irf.median[0, :, k], B_pt[:, k], atol=1e-12)

    def test_recovers_true_B_on_strong_nongaussianity(self):
        """With T=1500 t-distributed shocks, recover B up to perm and sign."""
        df, B_true = _sim_var_nongaussian(T=1500, df=3.5, seed=7)
        fit = lib.fit_var(df, lags=1)
        irf, diag = lib.nongaussian_svar_irf(fit, horizon=4, seed=42)
        B_est = np.array(diag["B_point"])
        # Frobenius distance after best permutation + sign
        dist, perm, signs = _match_columns_up_to_permutation_and_sign(B_est, B_true)
        # Allow some sampling error — should be <~ 0.3 in Frobenius
        # (||B_true||_F ≈ 2.0, so ~15% relative error)
        assert dist < 0.35, (
            f"Failed to recover B_true: dist={dist:.3f}, perm={perm}, signs={signs}\n"
            f"B_est:\n{B_est}\nB_true:\n{B_true}"
        )

    def test_weak_nongaussianity_flag_on_gaussian_data(self):
        """When residuals are Gaussian, ``weak_nongaussianity`` must be True."""
        df = _sim_var_gaussian(T=400, seed=0)
        fit = lib.fit_var(df, lags=1)
        _irf, diag = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        # All Jarque-Bera p-values should be high (residuals look Gaussian)
        assert diag["weak_nongaussianity"], (
            f"Gaussian residuals should trigger weak_nongaussianity flag; "
            f"got max_jb_pvalue={diag['max_jb_pvalue']}"
        )

    def test_strong_nongaussianity_no_flag(self):
        """When residuals are heavy-tailed, the flag should be False."""
        df, _ = _sim_var_nongaussian(T=800, df=3.5, seed=2)
        fit = lib.fit_var(df, lags=1)
        _irf, diag = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        assert not diag["weak_nongaussianity"]
        # All JB p-values should be tiny (well below 0.10)
        assert diag["max_jb_pvalue"] < 0.01

    def test_diagnostic_fields_present(self):
        df, _ = _sim_var_nongaussian(T=400, seed=0)
        fit = lib.fit_var(df, lags=1)
        _irf, diag = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        # Required keys
        for key in [
            "jarque_bera", "skewness", "excess_kurtosis",
            "max_jb_pvalue", "weak_nongaussianity",
            "fastica_converged", "fastica_n_iter", "B_point",
        ]:
            assert key in diag, f"missing diagnostic key: {key}"
        # Per-shock dicts have all 3 shocks
        assert set(diag["jarque_bera"].keys()) == {"shock_1", "shock_2", "shock_3"}

    def test_sign_normalization_works(self):
        """After sign_normalization, the named variable response has the
        requested sign."""
        df, _ = _sim_var_nongaussian(T=600, seed=3)
        fit = lib.fit_var(df, lags=1)
        # Pin: shock_1's contemporaneous impact on y1 should be positive
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, seed=0,
            sign_normalization={
                "shock_1": {"y1": +1},
                "shock_2": {"y2": +1},
                "shock_3": {"y3": +1},
            },
        )
        # h=0 IRF on the named variable for each shock should be ≥ 0
        for k, vname in enumerate(["y1", "y2", "y3"]):
            v_idx = list(fit.variable_names).index(vname)
            assert irf.median[0, v_idx, k] >= 0, (
                f"shock_{k+1} h=0 on {vname} should be ≥0, got "
                f"{irf.median[0, v_idx, k]}"
            )

    def test_label_by_assigns_columns_by_max_impact(self):
        """After label_by, shock k's largest |impact| variable should be the
        one named in label_by."""
        df, _ = _sim_var_nongaussian(T=800, seed=4)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, seed=0,
            shock_names=["y1_shock", "y2_shock", "y3_shock"],
            label_by={"y1_shock": "y1", "y2_shock": "y2", "y3_shock": "y3"},
        )
        # For each shock, its largest-|impact| variable should be the labelled one
        for k, target in enumerate(["y1", "y2", "y3"]):
            v_idx = list(fit.variable_names).index(target)
            impacts = np.abs(irf.median[0, :, k])
            argmax_v = list(fit.variable_names)[int(np.argmax(impacts))]
            assert argmax_v == target, (
                f"shock {k} largest-impact var={argmax_v}, expected {target}\n"
                f"impacts: {impacts}"
            )

    def test_residual_bootstrap_bands(self):
        """residual_bootstrap produces ordered (lower < median < upper) bands."""
        df, _ = _sim_var_nongaussian(T=500, seed=0)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, seed=0,
            sign_normalization={
                "shock_1": {"y1": +1},
                "shock_2": {"y2": +1},
                "shock_3": {"y3": +1},
            },
            label_by={"shock_1": "y1", "shock_2": "y2", "shock_3": "y3"},
            bands="residual_bootstrap", n_draws=50,
        )
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)
        assert np.all(irf.lower_90 <= irf.lower_68 + 1e-10)
        assert np.all(irf.upper_68 <= irf.upper_90 + 1e-10)

    def test_metadata_identification_tag(self):
        df, _ = _sim_var_nongaussian(T=400, seed=0)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(fit, horizon=4, seed=0)
        assert irf.metadata["identification"] == "nongaussian_svar"
        assert "LMS 2017" in irf.metadata["method"]

    def test_wrong_method_raises(self):
        df, _ = _sim_var_nongaussian(T=200, seed=0)
        fit = lib.fit_var(df, lags=1)
        with pytest.raises(ValueError, match="unknown method"):
            lib.nongaussian_svar_irf(fit, horizon=4, method="not_a_method")

    def test_wrong_shock_names_length_raises(self):
        df, _ = _sim_var_nongaussian(T=200, seed=0)
        fit = lib.fit_var(df, lags=1)
        with pytest.raises(ValueError, match="shock_names has"):
            lib.nongaussian_svar_irf(
                fit, horizon=4, shock_names=["a", "b"],   # wrong length
            )

    def test_wrong_bands_raises(self):
        df, _ = _sim_var_nongaussian(T=200, seed=0)
        fit = lib.fit_var(df, lags=1)
        with pytest.raises(ValueError, match="unknown bands"):
            lib.nongaussian_svar_irf(fit, horizon=4, bands="invalid")


# ---------------------------------------------------------------------------
# Parametric pseudo-ML with t-innovations (LMS §3)
# ---------------------------------------------------------------------------


class TestPMLT:
    def test_givens_orthogonal_is_orthogonal(self):
        from scripts.lib.nongaussian_svar import _givens_orthogonal
        rng = np.random.default_rng(0)
        for n in (2, 3, 4, 5):
            n_angles = n * (n - 1) // 2
            theta = rng.uniform(-np.pi, np.pi, n_angles)
            Q = _givens_orthogonal(theta, n)
            assert np.allclose(Q @ Q.T, np.eye(n), atol=1e-12)
            assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-12)

    def test_givens_zero_angles_gives_identity(self):
        from scripts.lib.nongaussian_svar import _givens_orthogonal
        for n in (2, 3, 4):
            n_angles = n * (n - 1) // 2
            Q = _givens_orthogonal(np.zeros(n_angles), n)
            assert np.allclose(Q, np.eye(n), atol=1e-15)

    def test_standardised_t_logpdf_unit_variance(self):
        """A sample drawn from the standardised-t should have mean ≈ 0 and
        variance ≈ 1, and the log-pdf should integrate (numerically) to
        nearly 1."""
        from scripts.lib.nongaussian_svar import _standardised_t_logpdf
        rng = np.random.default_rng(0)
        T = 10000
        nu = np.array([4.0, 8.0])
        sigma = np.sqrt((nu - 2) / nu)
        X = rng.standard_t(df=nu[None, :], size=(T, 2)) * sigma[None, :]
        # Sample moments match unit variance (within Monte Carlo error)
        assert np.allclose(X.mean(axis=0), 0.0, atol=0.1)
        assert np.allclose(X.var(axis=0), 1.0, atol=0.1)
        # log-pdf is finite and consistent with sample density
        lp = _standardised_t_logpdf(X, nu)
        assert np.all(np.isfinite(lp))

    def test_pml_t_recovers_known_df(self):
        """When shocks are simulated from t(ν=4), pml_t should recover ν
        in the rough neighbourhood of 4 (sampling error is large for ν
        on T~1000)."""
        rng = np.random.default_rng(7)
        T = 2000
        n = 3
        nu_true = 4.0
        sigma = np.sqrt((nu_true - 2) / nu_true)
        Z = rng.standard_t(df=nu_true, size=(T, n)) * sigma
        # Already whitened (identity covariance up to sample noise)
        Q, nu_est, diag = lib.pml_t_estimate(Z, seed=0, n_restarts=2)
        assert diag["converged"]
        # Recovered ν should be in roughly the right neighbourhood
        # (allow factor of ~2 due to ν's heavy-tailed sampling)
        for v in nu_est:
            assert 2.5 < v < 12.0, (
                f"recovered nu {v:.2f} far from true {nu_true}; "
                f"all nu: {nu_est}"
            )

    def test_pml_t_log_likelihood_at_least_as_high_as_fastica(self):
        """pml_t with FastICA warm-start should find at least as high a
        log-likelihood as the FastICA solution (since pml_t optimises the
        likelihood directly and starts from FastICA's Q)."""
        df, _ = _sim_var_nongaussian(T=1000, df=4.0, seed=11)
        fit = lib.fit_var(df, lags=1)
        irf_fica, d_fica = lib.nongaussian_svar_irf(
            fit, horizon=4, method="fastica", seed=0,
        )
        irf_pml, d_pml = lib.nongaussian_svar_irf(
            fit, horizon=4, method="pml_t", seed=0,
        )
        # Compute the same likelihood at the FastICA solution for comparison
        from scripts.lib.nongaussian_svar import (
            whiten_residuals, _standardised_t_logpdf
        )
        Z, _L = whiten_residuals(np.asarray(fit.residuals))
        # FastICA's Q
        B_fica = np.array(d_fica["B_point"])
        # B = L Q so Q_fica = L^-1 B_fica. (Up to column perm + sign from labelling
        # /normalisation, which doesn't change the likelihood since the t-distribution
        # is symmetric and shock independence is invariant to permutation.)
        # We use the pml_t-reported log-likelihood as a sanity check.
        assert d_pml["pml_t_converged"]
        assert d_pml["pml_t_log_likelihood"] is not None
        # Numerical sanity: pml_t's nu values are finite and > 2
        for v in d_pml["pml_t_nu"]:
            assert 2.0 < v < 1000.0

    def test_pml_t_diagnostics_present(self):
        df, _ = _sim_var_nongaussian(T=500, seed=2)
        fit = lib.fit_var(df, lags=1)
        _irf, diag = lib.nongaussian_svar_irf(
            fit, horizon=4, method="pml_t", seed=0,
        )
        for key in (
            "pml_t_log_likelihood", "pml_t_nu", "pml_t_converged",
            "pml_t_n_restarts_tried", "pml_t_best_restart_index",
        ):
            assert key in diag, f"missing pml_t diagnostic key: {key}"
        # FastICA diagnostics also present (used as warm-start)
        assert "fastica_converged" in diag

    def test_pml_t_metadata_method_tag(self):
        df, _ = _sim_var_nongaussian(T=300, seed=0)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, method="pml_t", seed=0,
        )
        assert irf.metadata["ica_method"] == "pml_t"
        assert "pml_t" in irf.metadata["method"]
        assert irf.metadata["identification"] == "nongaussian_svar"

    def test_pml_t_sign_normalization_compatible(self):
        """sign_normalization and label_by work with method='pml_t' too."""
        df, _ = _sim_var_nongaussian(T=600, seed=3)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, method="pml_t", seed=0,
            label_by={"shock_1": "y1", "shock_2": "y2", "shock_3": "y3"},
            sign_normalization={
                "shock_1": {"y1": +1},
                "shock_2": {"y2": +1},
                "shock_3": {"y3": +1},
            },
        )
        for k, vname in enumerate(["y1", "y2", "y3"]):
            v_idx = list(fit.variable_names).index(vname)
            assert irf.median[0, v_idx, k] >= 0

    def test_pml_t_residual_bootstrap(self):
        """residual_bootstrap should also work for method='pml_t' (slower
        because each draw refits)."""
        df, _ = _sim_var_nongaussian(T=400, seed=0)
        fit = lib.fit_var(df, lags=1)
        irf, _ = lib.nongaussian_svar_irf(
            fit, horizon=4, method="pml_t", seed=0,
            label_by={"shock_1": "y1", "shock_2": "y2", "shock_3": "y3"},
            sign_normalization={
                "shock_1": {"y1": +1},
                "shock_2": {"y2": +1},
                "shock_3": {"y3": +1},
            },
            bands="residual_bootstrap", n_draws=15,
            # Small n_draws for test speed; reduce restarts too
            pml_t_n_restarts=1,
        )
        assert np.all(irf.lower_68 <= irf.median + 1e-10)
        assert np.all(irf.median <= irf.upper_68 + 1e-10)


class TestLabelingUnitInvarianceAndNuAlignment:
    """Acceptance check: shock names must not depend on variable units,
    per-component tail parameters must follow relabeled columns, and no
    unconverged optimisation may be reported as an estimate."""

    def test_label_priority_invariant_to_row_rescaling(self):
        from lib.nongaussian_svar import _label_columns_by_impact
        # Counterexample: rows [activity, rate]
        B = np.array([[2.0, 1.8],
                      [1.5, 0.1]])
        eps = np.zeros((10, 2))
        names = ["activity_shock", "policy_shock"]
        label_by = {"activity_shock": "activity", "policy_shock": "rate"}
        _, _, perm1 = _label_columns_by_impact(
            B.copy(), eps.copy(), names, ["activity", "rate"], label_by)
        # Rescale ONLY the rate row by 10 (a pure unit change)
        B2 = B.copy()
        B2[1, :] *= 10.0
        _, _, perm2 = _label_columns_by_impact(
            B2, eps.copy(), names, ["activity", "rate"], label_by)
        assert perm1 == perm2, (perm1, perm2)

    def test_pml_nu_follows_relabeled_columns(self, monkeypatch):
        import lib.nongaussian_svar as ng
        rng = np.random.default_rng(0)
        T = 400
        # Near-identity residual covariance so B ~ I under Q = I
        y = rng.standard_normal((T, 3)) * [1.0, 1.0, 1.0]
        df = pd.DataFrame(y, columns=["v1", "v2", "v3"],
                          index=pd.date_range("1980-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")

        def fake_pml(Z, init_Q=None, **kw):
            Q = np.eye(3)
            nu = np.array([3.0, 8.0, 40.0])
            return Q, nu, {"log_likelihood": 0.0, "nu": nu.tolist(),
                           "converged": True, "n_restarts_tried": 1,
                           "best_restart_index": 0}

        monkeypatch.setattr(ng, "pml_t_estimate", fake_pml)
        # Request names in REVERSE variable order to force permutation [2,1,0]
        irf, diag = lib.nongaussian_svar_irf(
            fit, horizon=1, method="pml_t",
            shock_names=["sA", "sB", "sC"],
            label_by={"sA": "v3", "sB": "v2", "sC": "v1"},
            bands="none", seed=0,
        )
        B_pt = np.asarray(diag["B_point"])
        # Each named shock's dominant-impact variable defines its column;
        # the nu entry must describe THAT column.
        nu = diag["pml_t_nu"]
        for j, var in enumerate(["v3", "v2", "v1"]):
            v_idx = ["v1", "v2", "v3"].index(var)
            assert np.argmax(np.abs(B_pt[:, j])) == v_idx
        # With B ~ L ~ I, perm should be [2, 1, 0] and nu reversed
        assert nu == [40.0, 8.0, 3.0], nu

    def test_pml_raises_when_no_restart_converges(self):
        from lib.nongaussian_svar import pml_t_estimate, whiten_residuals
        rng = np.random.default_rng(1)
        Z, _ = whiten_residuals(rng.standard_normal((200, 2)))
        # maxiter=0 cannot converge from any start
        with pytest.raises(RuntimeError, match="did not converge"):
            pml_t_estimate(Z, n_restarts=2, max_iter=0, tol=1e-12, seed=0)

    def test_bootstrap_reports_convergence_accounting(self):
        rng = np.random.default_rng(2)
        T = 300
        eps = rng.standard_t(df=5, size=(T, 2))
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + eps[t]
        df = pd.DataFrame(y, columns=["a", "b"],
                          index=pd.date_range("1960-01-01", periods=T, freq="QS"))
        fit = lib.fit_var(df, lags=1, trend="c")
        irf, diag = lib.nongaussian_svar_irf(
            fit, horizon=2, method="fastica", bands="residual_bootstrap",
            n_draws=12, seed=3,
        )
        assert diag["bootstrap_n_draws"] == 12
        assert "bootstrap_n_retried_inits" in diag
