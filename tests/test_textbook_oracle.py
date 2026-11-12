"""Textbook-exact oracle: Lütkepohl (2005) on the E1 dataset.

The West German investment/income/consumption example is worked in the
book to published precision, and its ingredients ship locally with
statsmodels: ``e1.dat`` (the frozen dataset — no vintage confound) and
``E1_Results`` (the book's p. 120 IRF standard errors as validated by the
statsmodels maintainers against the text). lib is run on the IDENTICAL
sample — diff(log), estimation window ending 1978Q4, VAR(2) with constant
— and checked two ways:

1. **Machine-precision differential vs statsmodels' VAR** (itself
   validated upstream against the book, JMulTi and R). Plain
   ``lib.fit_var`` runs statsmodels internally and passes its
   ``sigma_u`` through, so the Sigma_u comparison is a wrapper check;
   the residual and lag-matrix comparisons pin lib's own design-matrix
   and OLS conventions against statsmodels on the same formula. The MA
   coefficients, orthogonalised IRFs and FEVD come from lib's own
   recursions and are the genuinely separate calculations here.
2. **Textbook-analytic oracle for the inference machinery**: the standard
   deviation of lib's asymptotic ``parameter_draws`` MA coefficients must
   reproduce Lütkepohl's delta-method IRF standard errors (p. 120). A
   wrong (X'X)^{-1} scaling, a missing degrees-of-freedom adjustment, or
   a mis-specified coefficient covariance would miss these values by far
   more than the Monte Carlo tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib

sm_test_var = pytest.importorskip("statsmodels.tsa.vector_ar.tests.test_var")
from statsmodels.tsa.api import VAR as SMVAR  # noqa: E402


def _e1_adjusted():
    """The book's estimation data: diff(log(e1)), last 16 obs held out."""
    sdata, dates = sm_test_var.get_lutkepohl_data("e1")
    data = sm_test_var.data_util.struct_to_ndarray(sdata)
    adj = np.diff(np.log(data), axis=0)[:-16]
    idx = pd.period_range("1960Q2", periods=len(adj), freq="Q").to_timestamp()
    return pd.DataFrame(adj, columns=["inv", "inc", "cons"], index=idx)


@pytest.fixture(scope="module")
def e1_fits():
    df = _e1_adjusted()
    lib_fit = lib.fit_var(data=df, lags=2, trend="c")
    sm_res = SMVAR(df.to_numpy()).fit(maxlags=2, trend="c")
    return lib_fit, sm_res


class TestLutkepohlE1Differential:
    """lib vs statsmodels on the book's exact sample, machine precision."""

    def test_residuals_identical(self, e1_fits):
        lib_fit, sm_res = e1_fits
        np.testing.assert_allclose(
            np.asarray(lib_fit.residuals), sm_res.resid, atol=1e-10
        )

    def test_sigma_u_identical(self, e1_fits):
        # Both df-adjusted (Lütkepohl T - K divisor)
        lib_fit, sm_res = e1_fits
        np.testing.assert_allclose(
            np.asarray(lib_fit.Sigma_u), sm_res.sigma_u, atol=1e-12
        )

    def test_lag_matrices_identical(self, e1_fits):
        lib_fit, sm_res = e1_fits
        np.testing.assert_allclose(
            np.asarray(lib_fit.lag_matrices()), sm_res.coefs, atol=1e-12
        )

    def test_ma_rep_identical(self, e1_fits):
        lib_fit, sm_res = e1_fits
        np.testing.assert_allclose(
            lib_fit.ma_rep(10), sm_res.ma_rep(10), atol=1e-12
        )

    def test_orthogonalised_irf_identical(self, e1_fits):
        lib_fit, sm_res = e1_fits
        irf = lib.cholesky_irf(fit=lib_fit, horizon=10, bands="none")
        np.testing.assert_allclose(
            np.asarray(irf.median), sm_res.irf(10).orth_irfs, atol=1e-10
        )

    @pytest.mark.parametrize("horizon", [1, 5, 10])
    def test_fevd_identical(self, e1_fits, horizon):
        lib_fit, sm_res = e1_fits
        S = lib.cholesky_impact(lib_fit.Sigma_u)
        ours = lib.structural_fevd(lib_fit, S, horizon=horizon)
        theirs = sm_res.fevd(10).decomp[:, horizon - 1, :]
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestLutkepohlP120StderrOracle:
    """lib's asymptotic parameter draws vs the book's published IRF
    standard errors (Lütkepohl 2005, p. 120, horizons 1-3, non-orth).

    The draws sample vec(B) from N(vec(B_hat), Sigma ⊗ (X'X)^{-1}); the
    std of the implied MA coefficients across draws must match the
    delta-method values printed in the book up to Monte Carlo error and
    the first-order nature of the delta method.
    """

    @staticmethod
    def _phi_from_lags(A_list, horizon):
        """Independent MA recursion (Lütkepohl eq. 2.1.22) — NOT lib's."""
        n = A_list[0].shape[0]
        p = len(A_list)
        phis = [np.eye(n)]
        for h in range(1, horizon + 1):
            acc = np.zeros((n, n))
            for j in range(1, min(h, p) + 1):
                acc += phis[h - j] @ A_list[j - 1]
            phis.append(acc)
        return np.array(phis)

    def test_recursion_helper_agrees_with_statsmodels(self, e1_fits):
        lib_fit, sm_res = e1_fits
        mine = self._phi_from_lags(list(sm_res.coefs), 10)
        np.testing.assert_allclose(mine, sm_res.ma_rep(10), atol=1e-12)

    def test_ma_draw_std_matches_book(self, e1_fits):
        lib_fit, _ = e1_fits
        n, p = lib_fit.n_vars, lib_fit.p
        draws = lib_fit.parameter_draws(4000, seed=0)
        phi_draws = np.empty((len(draws), 4, n, n))
        for d, (B, _Sigma) in enumerate(draws):
            A_list = [B[n * j:n * (j + 1), :].T for j in range(p)]
            phi_draws[d] = self._phi_from_lags(A_list, 3)
        sim_std = phi_draws.std(axis=0, ddof=1)[1:]      # h = 1..3

        book = sm_test_var.E1_Results().irf_stderr        # h = 1..3
        signed = (sim_std - book) / book

        # h=1: Phi_1 = A_1 is LINEAR in the coefficients, so the simulated
        # std must equal the book's analytic stderr up to MC noise — the
        # sharp oracle for the draw covariance. (A missing df adjustment
        # or wrong (X'X)^{-1} scaling shifts this row by ~10%; observed
        # agreement is within ±2%.)
        assert np.abs(signed[0]).max() < 0.05, (
            f"h=1 dev {np.abs(signed[0]).max():.3f}\n{signed[0]}"
        )
        # h=2..3: Phi_h is a degree-h polynomial in the coefficients; the
        # book's first-order delta method UNDERSTATES the dispersion at
        # T=75, so the simulation may sit above it (observed +1% to +17%,
        # one-signed, growing with h) but must never sit materially BELOW
        # it — sitting below is the scaling-bug signature, not a
        # delta-method artefact.
        assert signed[1:].min() > -0.05, f"below-book dev\n{signed[1:]}"
        assert signed[1:].max() < 0.25, f"above-book dev\n{signed[1:]}"
