"""Tests for dfm.py (Bayesian DFM — Bańbura-Modugno 2014).

Validation strategy:

1. **Spec validation** — invalid n_factors rejected.
2. **End-to-end Gibbs runs** — output shapes correct.
3. **Recovery of synthetic factors** — generate a 2-factor DGP; Gibbs
   should recover the factor R² > 0.5 (factors are identified up to
   rotation/sign, so we compare the SPACE spanned, not exact values).
4. **Missing data handled** — Gibbs continues to converge with NaN
   blocks in the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


@pytest.fixture(scope="module")
def two_factor_dgp() -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a 2-factor DGP for testing factor recovery."""
    rng = np.random.default_rng(42)
    T = 200
    r = 2
    n = 20
    Lambda_true = rng.standard_normal((n, r)) * 0.8
    Phi_true = np.array([[0.6, 0.1], [0.0, 0.5]])
    Q_true = np.eye(r) * 0.5
    L_Q = np.linalg.cholesky(Q_true)
    R_true = rng.uniform(0.1, 0.5, n)

    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = Phi_true @ F[t - 1] + L_Q @ rng.standard_normal(r)
    Y = F @ Lambda_true.T + rng.standard_normal((T, n)) * np.sqrt(R_true)

    idx = pd.period_range("2000-01", periods=T, freq="M").to_timestamp()
    df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(n)], index=idx)
    return df, F


class TestDFMSpec:
    def test_valid(self):
        spec = lib.DFMSpec(n_factors=3)
        assert spec.n_factors == 3
        assert spec.standardize is True

    def test_rejects_zero_factors(self):
        with pytest.raises(ValueError, match="n_factors"):
            lib.DFMSpec(n_factors=0)


class TestFitDFMBasic:
    def test_runs_end_to_end(self, two_factor_dgp):
        data, _ = two_factor_dgp
        spec = lib.DFMSpec(n_factors=2)
        fit = lib.fit_dfm_bayesian(
            data, spec, n_iter=30, burn_in=10, seed=0,
        )
        T = len(data)
        n = data.shape[1]
        assert fit.factor_path_posterior_mean.shape == (T, 2)
        assert fit.loadings_posterior_mean.shape == (n, 2)
        assert fit.factor_AR_posterior_mean.shape == (2, 2)
        assert fit.Q_posterior_mean.shape == (2, 2)
        assert fit.R_posterior_mean.shape == (n,)

    def test_rejects_too_short_data(self):
        rng = np.random.default_rng(0)
        short = pd.DataFrame(rng.standard_normal((10, 5)),
                              columns=[f"y{i}" for i in range(5)])
        spec = lib.DFMSpec(n_factors=2)
        with pytest.raises(ValueError, match="too short"):
            lib.fit_dfm_bayesian(short, spec, n_iter=10, burn_in=5)

    def test_rejects_non_dataframe(self):
        spec = lib.DFMSpec(n_factors=2)
        rng = np.random.default_rng(0)
        with pytest.raises(TypeError, match="DataFrame"):
            lib.fit_dfm_bayesian(rng.standard_normal((100, 5)), spec)


class TestFactorRecovery:
    def test_factors_capture_DGP_variance(self, two_factor_dgp):
        """With 200 obs and r=2 true factors, the Gibbs-mean factors should
        capture most of the variance of the data — same as PC factors do."""
        data, true_F = two_factor_dgp
        spec = lib.DFMSpec(n_factors=2)
        fit = lib.fit_dfm_bayesian(
            data, spec, n_iter=80, burn_in=20, seed=0,
        )
        F_est = fit.factor_path_posterior_mean   # (T, r)

        # Factors are identified up to rotation/sign; compute R² of the
        # observed data on the estimated factor space.
        Y_std = (data.values - fit.means) / fit.stds
        X = np.column_stack([np.ones(len(Y_std)), F_est])
        # Regress each y on factors; compute average R²
        r2_list = []
        for j in range(Y_std.shape[1]):
            mask = ~np.isnan(Y_std[:, j])
            beta, *_ = np.linalg.lstsq(X[mask], Y_std[mask, j], rcond=None)
            resid = Y_std[mask, j] - X[mask] @ beta
            sse = float(resid @ resid)
            sst = float(np.var(Y_std[mask, j], ddof=1) * (mask.sum() - 1))
            if sst > 0:
                r2_list.append(1.0 - sse / sst)
        avg_r2 = np.mean(r2_list)
        # Should explain > 50% of variance on average with 2 factors
        assert avg_r2 > 0.5, f"average R² = {avg_r2:.3f}"


class TestMissingData:
    def test_runs_with_nan_block(self, two_factor_dgp):
        """Inject a block of NaN values for one variable and ensure
        the Gibbs sampler still produces finite output."""
        data, _ = two_factor_dgp
        data_with_nan = data.copy()
        data_with_nan.iloc[50:80, 3] = np.nan   # 30-month gap in y3
        spec = lib.DFMSpec(n_factors=2)
        fit = lib.fit_dfm_bayesian(
            data_with_nan, spec, n_iter=30, burn_in=10, seed=0,
        )
        # Posterior outputs should be finite
        assert np.all(np.isfinite(fit.factor_path_posterior_mean))
        assert np.all(np.isfinite(fit.loadings_posterior_mean))
        assert np.all(np.isfinite(fit.R_posterior_mean))


class TestPhiOrientation:
    """Regression contract: the factor-AR matrix is stored as Phi, not Phi'.

    The Gibbs draw of the factor VAR(1) matrix samples the REGRESSION
    matrix B = Phi' (regressors x equations); storing it untransposed
    would make iterations >= 1 use Phi' where the initializer (_ols_var1,
    which returns solve(...).T) and all consumers (transition T=Phi,
    residual F_lhs - F_rhs Phi') expect Phi. Eigenvalue or fit checks
    cannot catch a transpose; orientation consistency can: the posterior
    mean of Phi must match OLS on the posterior-mean factor path, not
    its transpose.
    """

    def test_posterior_phi_orientation_consistent(self):
        from lib.dfm import _ols_var1
        rng = np.random.default_rng(5)
        T, N, r = 300, 12, 2
        Phi_true = np.array([[0.5, 0.5], [0.0, 0.4]])   # strongly asymmetric
        F = np.zeros((T, r))
        for t in range(1, T):
            F[t] = Phi_true @ F[t - 1] + rng.standard_normal(r)
        Lam = np.zeros((N, r))
        Lam[:6, 0] = rng.uniform(0.8, 1.2, 6)
        Lam[6:, 1] = rng.uniform(0.8, 1.2, 6)
        X = F @ Lam.T + 0.3 * rng.standard_normal((T, N))
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N)])

        fit = lib.fit_dfm_bayesian(df, lib.DFMSpec(n_factors=2),
                                   n_iter=300, burn_in=100, seed=2)
        ph = np.asarray(fit.factor_AR_posterior_mean)
        B = _ols_var1(np.asarray(fit.factor_path_posterior_mean))
        d_same = np.linalg.norm(ph - B)
        d_trans = np.linalg.norm(ph - B.T)
        # The estimated factor basis makes Phi-tilde genuinely asymmetric,
        # so the transpose is far away; under a transpose error the two
        # distances swap.
        assert d_same * 2 < d_trans, (
            f"Phi posterior mean matches the TRANSPOSE of OLS-on-factors "
            f"(d_same={d_same:.4f}, d_transpose={d_trans:.4f})"
        )


class TestCommonComponentAndOneFactor:
    """Acceptance checks: draw-wise common component for imputation;
    one-factor estimation survives SciPy's scalar inverse-Wishart draws."""

    def test_common_component_shape_and_fit(self, two_factor_dgp):
        data, F_true = two_factor_dgp
        spec = lib.DFMSpec(n_factors=2)
        fit = lib.fit_dfm_bayesian(data, spec, n_iter=120, burn_in=60,
                                   thin=2, seed=4)
        cc = fit.common_component_posterior_mean
        assert cc.shape == (len(data), data.shape[1])
        assert np.all(np.isfinite(cc))
        # The draw-wise reconstruction must track the (standardised) data's
        # common part: average per-variable correlation clearly positive.
        X_std = (data.values - fit.means) / fit.stds
        cors = [np.corrcoef(cc[:, j], X_std[:, j])[0, 1]
                for j in range(data.shape[1])]
        assert np.nanmean(cors) > 0.3

    def test_one_factor_dfm_completes(self, two_factor_dgp):
        """SciPy squeezes 1-D inverse-Wishart draws to scalars; the sampler
        must keep matrix shape and run more than one iteration."""
        data, _ = two_factor_dgp
        spec = lib.DFMSpec(n_factors=1)
        fit = lib.fit_dfm_bayesian(data, spec, n_iter=40, burn_in=20,
                                   thin=1, seed=5)
        assert fit.Q_posterior_mean.shape == (1, 1)
        assert fit.metadata["n_draws_stored"] >= 15
