"""Standing differential harness: lib vs statsmodels on random DGPs.

Every quantity both implementations compute is compared at (near) machine
precision across several system sizes and lag orders. The comparisons
differ in how independent they are:

- Sigma_u is NOT independent: plain ``lib.fit_var`` fits statsmodels'
  VAR internally and passes its ``sigma_u`` through, so that test checks
  the wrapper and the df convention it exposes, not the covariance
  algebra.
- Residuals and lag matrices come from lib's own design matrix and
  normal-equation solve, but it is the same textbook OLS formula on the
  same data; agreement pins lib's design-matrix layout, lag ordering and
  constant handling against statsmodels' conventions rather than
  validating OLS itself.
- MA coefficients, orthogonalised IRFs and the FEVD are computed by
  lib's own recursions (``ma_representation``, ``cholesky_impact``,
  ``structural_fevd``), so these are genuine cross-implementation checks
  of the MA recursion, Cholesky orientation and FEVD normalisation, given
  the shared Sigma_u.

This generalises the one-time development cross-check
(DEVELOPMENT.md §"Independent cross-checks") into a permanent, randomised
test.

Convention notes encoded here (drift in either library breaks the test):
- both report the df-adjusted Sigma_u (Lütkepohl T - K divisor);
- both orthogonalise with the lower Cholesky factor of that Sigma_u;
- statsmodels' FEVD ``decomp[:, t, :]`` is the (t+1)-step-ahead share,
  matching ``lib.structural_fevd(horizon=t+1)`` (H-term sums).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib

from statsmodels.tsa.api import VAR as SMVAR

CONFIGS = [
    {"n": 2, "p": 1, "T": 300, "seed": 11},
    {"n": 3, "p": 2, "T": 400, "seed": 22},
    {"n": 4, "p": 3, "T": 600, "seed": 33},
]


def _simulate(n, p, T, seed, burn=100):
    """Stable VAR(p) with non-trivial cross-correlation."""
    rng = np.random.default_rng(seed)
    A = [rng.standard_normal((n, n)) * 0.9 / (n * p) for _ in range(p)]
    # Rescale to a comfortable spectral radius via the companion matrix
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = np.hstack(A)
    if p > 1:
        comp[n:, :-n] = np.eye(n * (p - 1))
    rad = np.max(np.abs(np.linalg.eigvals(comp)))
    A = [a * (0.7 / max(rad, 0.7)) for a in A]

    L = np.linalg.cholesky(np.eye(n) + 0.3 * np.ones((n, n)))
    Y = np.zeros((T + burn, n))
    for t in range(p, T + burn):
        Y[t] = sum(A[j] @ Y[t - 1 - j] for j in range(p)) + L @ rng.standard_normal(n)
    Y = Y[burn:]
    idx = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()
    return pd.DataFrame(Y, columns=[f"v{i}" for i in range(n)], index=idx)


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: f"n{c['n']}p{c['p']}")
def fit_pair(request):
    c = request.param
    df = _simulate(**c)
    lib_fit = lib.fit_var(data=df, lags=c["p"], trend="c")
    sm_res = SMVAR(df.to_numpy()).fit(maxlags=c["p"], trend="c")
    return lib_fit, sm_res


class TestDifferentialStatsmodels:

    def test_residuals(self, fit_pair):
        lib_fit, sm_res = fit_pair
        np.testing.assert_allclose(
            np.asarray(lib_fit.residuals), sm_res.resid, atol=1e-9
        )

    def test_sigma_u(self, fit_pair):
        lib_fit, sm_res = fit_pair
        np.testing.assert_allclose(
            np.asarray(lib_fit.Sigma_u), sm_res.sigma_u, atol=1e-11
        )

    def test_lag_matrices(self, fit_pair):
        lib_fit, sm_res = fit_pair
        np.testing.assert_allclose(
            np.asarray(lib_fit.lag_matrices()), sm_res.coefs, atol=1e-11
        )

    def test_ma_rep(self, fit_pair):
        lib_fit, sm_res = fit_pair
        np.testing.assert_allclose(
            lib_fit.ma_rep(12), sm_res.ma_rep(12), atol=1e-11
        )

    def test_orthogonalised_irf(self, fit_pair):
        lib_fit, sm_res = fit_pair
        irf = lib.cholesky_irf(fit=lib_fit, horizon=12, bands="none")
        np.testing.assert_allclose(
            np.asarray(irf.median), sm_res.irf(12).orth_irfs, atol=1e-9
        )

    @pytest.mark.parametrize("horizon", [1, 4, 12])
    def test_fevd(self, fit_pair, horizon):
        lib_fit, sm_res = fit_pair
        S = lib.cholesky_impact(lib_fit.Sigma_u)
        ours = lib.structural_fevd(lib_fit, S, horizon=horizon)
        theirs = sm_res.fevd(12).decomp[:, horizon - 1, :]
        np.testing.assert_allclose(ours, theirs, atol=1e-9)
