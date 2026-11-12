"""Monte Carlo coverage studies — the gold standard for "are the bands honest?"

For each band-producing function, we:

1. Define a known data-generating process (DGP) — a stationary VAR(p).
2. Compute the TRUE impulse response from the DGP's parameters.
3. Run N replications: simulate data → fit model → compute bands.
4. Compute the EMPIRICAL coverage rate: fraction of replications where
   the true IRF value falls within the band's [lower, upper] envelope.
5. Assert empirical coverage is close to nominal (68% / 90%).

This catches issues no other test does: bands that LOOK right (correct
shapes, monotone-ordered, reasonable widths) but UNDER-cover or
OVER-cover the truth. A frequentist coverage rate of, say, 50% under
a nominal 68% band would indicate a wrong SE formula or miscalibrated
bootstrap.

These tests are marked ``@pytest.mark.slow`` because they require
hundreds of model fits. Run with::

    pytest tests/test_coverage.py --runslow

Tolerance budget: with N_REPS = 200 replications, the standard error
of the empirical coverage rate is sqrt(p (1-p) / N), about 3.3 pp for
p = 0.68 and 2.1 pp for p = 0.90. Every check allows ±8 percentage
points (TOLERANCE_PP), about 2.4σ for the 68% band and 3.8σ for the
90% band, to keep false-positive flakiness low; the 90% check is
therefore the looser of the two. The budget holds only if all N_REPS
replications complete, so an estimation error fails the study instead
of silently removing that replication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import lib  # noqa: E402
from conftest import VAR_DGP_A1, VAR_DGP_A2, VAR_DGP_SIGMA  # noqa: E402

# Test budget knobs
N_REPS = 200          # Monte Carlo replications per test
N_DRAWS_FAST = 500    # number of band draws for fast (asymptotic-MC) paths
N_DRAWS_BOOTSTRAP = 200  # bootstrap draws (slower)
TOLERANCE_PP = 8.0    # acceptable empirical-coverage deviation from nominal, in pp
                      # (2.5σ at N=200 for p=0.68 ⇒ ~8 pp ⇒ ~1% nominal false-positive rate)


# ---------------------------------------------------------------------------
# DGP definitions — shared across coverage tests
# ---------------------------------------------------------------------------


def _make_var_dgp(n=3):
    """A stationary VAR(2) DGP with known parameters.

    Same parameter matrices as the conftest synthetic-VAR fixtures
    (single source of truth: conftest.VAR_DGP_*).
    """
    return {"A1": VAR_DGP_A1, "A2": VAR_DGP_A2, "Sigma": VAR_DGP_SIGMA,
            "n": n, "p": 2}


def _simulate_var(dgp: dict, T: int, burn: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one realisation of the VAR DGP."""
    n = dgp["n"]
    A1 = dgp["A1"]
    A2 = dgp["A2"]
    L = np.linalg.cholesky(dgp["Sigma"])
    Y = np.zeros((T + burn, n))
    for t in range(2, T + burn):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + L @ rng.standard_normal(n)
    return Y[burn:]


def _true_cholesky_irf(dgp: dict, horizon: int) -> np.ndarray:
    """True Cholesky-identified IRF tensor from the DGP."""
    A1 = dgp["A1"]
    A2 = dgp["A2"]
    n = dgp["n"]
    L = np.linalg.cholesky(dgp["Sigma"])
    # Wold MA representation
    Phi = np.zeros((horizon + 1, n, n))
    Phi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        if h >= 1:
            Phi[h] = Phi[h] + A1 @ Phi[h - 1]
        if h >= 2:
            Phi[h] = Phi[h] + A2 @ Phi[h - 2]
    # Structural IRF: Phi_h @ L
    return np.einsum("hij,jk->hik", Phi, L)


def _coverage_rate(
    true_value: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of replications where true_value falls in [lower, upper]."""
    covered = (lower <= true_value) & (true_value <= upper)
    return float(covered.mean())


def _assert_all_replications(records_68: dict, records_90: dict) -> None:
    """Every replication must contribute a band.

    The tolerance budget assumes N_REPS records. A coverage rate computed
    from a reduced survivor sample would carry a larger, unstated
    sampling error and could be selected by the failures themselves.
    """
    for records in (records_68, records_90):
        for h, rec in records.items():
            assert len(rec["lower"]) == N_REPS, (
                f"h={h}: {len(rec['lower'])}/{N_REPS} replications recorded"
            )
            assert len(rec["upper"]) == N_REPS
            assert np.all(np.isfinite(rec["lower"]))
            assert np.all(np.isfinite(rec["upper"]))


# ---------------------------------------------------------------------------
# Cholesky IRF — asymptotic-MC bands
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCholeskyAsymptoticMCCoverage:
    """Coverage study for `cholesky_irf(bands="asymptotic_mc")`.

    Asymptotic bands draw from the asymptotic-Normal distribution of
    vec(B_hat) and re-apply the identification per draw. Should achieve
    nominal coverage at large T.
    """

    def test_coverage_at_h2_and_h4(self):
        dgp = _make_var_dgp()
        true_irf = _true_cholesky_irf(dgp, horizon=6)
        # Target horizons
        target_h = [2, 4]
        # Track band endpoints per replication for shock 0, response 1 (off-diagonal)
        # at each target horizon.
        rng_master = np.random.default_rng(0)
        T = 250
        burn = 100

        records_68 = {h: {"lower": [], "upper": []} for h in target_h}
        records_90 = {h: {"lower": [], "upper": []} for h in target_h}

        for rep in range(N_REPS):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            Y = _simulate_var(dgp, T=T, burn=burn, rng=rng)
            df = pd.DataFrame(Y, columns=["y1", "y2", "y3"],
                              index=pd.date_range("1980-01-01", periods=T, freq="QS"))
            # No exception handling: the DGP is fixed and well behaved, so
            # any estimation error is a defect, not a discarded replication.
            fit = lib.fit_var(df, lags=dgp["p"], trend="c")
            irf = lib.cholesky_irf(
                fit, horizon=6, bands="asymptotic_mc",
                n_draws=N_DRAWS_FAST,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for h in target_h:
                # Shock 0 → response 1 at horizon h (off-diagonal entry)
                records_68[h]["lower"].append(irf.lower_68[h, 1, 0])
                records_68[h]["upper"].append(irf.upper_68[h, 1, 0])
                records_90[h]["lower"].append(irf.lower_90[h, 1, 0])
                records_90[h]["upper"].append(irf.upper_90[h, 1, 0])

        _assert_all_replications(records_68, records_90)

        for h in target_h:
            true_val = true_irf[h, 1, 0]
            cov_68 = _coverage_rate(
                true_val,
                np.asarray(records_68[h]["lower"]),
                np.asarray(records_68[h]["upper"]),
            )
            cov_90 = _coverage_rate(
                true_val,
                np.asarray(records_90[h]["lower"]),
                np.asarray(records_90[h]["upper"]),
            )
            # Coverage should be within tolerance of nominal
            assert abs(cov_68 - 0.68) * 100 <= TOLERANCE_PP, (
                f"Cholesky asymp-MC: at h={h}, empirical 68% coverage = "
                f"{cov_68:.3f} (target 0.68, tolerance ±{TOLERANCE_PP / 100:.2f})"
            )
            assert abs(cov_90 - 0.90) * 100 <= TOLERANCE_PP, (
                f"Cholesky asymp-MC: at h={h}, empirical 90% coverage = "
                f"{cov_90:.3f} (target 0.90, tolerance ±{TOLERANCE_PP / 100:.2f})"
            )


# ---------------------------------------------------------------------------
# Cholesky IRF — residual-bootstrap bands
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCholeskyBootstrapCoverage:
    """Coverage study for `cholesky_irf(bands="residual_bootstrap")`.

    Residual bootstrap re-samples residuals, refits the VAR, recomputes
    IRFs per bootstrap iteration. Should achieve nominal coverage at
    large T (asymptotically).
    """

    def test_coverage_at_h2_and_h4(self):
        dgp = _make_var_dgp()
        true_irf = _true_cholesky_irf(dgp, horizon=6)
        target_h = [2, 4]
        rng_master = np.random.default_rng(1)
        T = 250
        burn = 100

        records_68 = {h: {"lower": [], "upper": []} for h in target_h}
        records_90 = {h: {"lower": [], "upper": []} for h in target_h}

        for rep in range(N_REPS):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            Y = _simulate_var(dgp, T=T, burn=burn, rng=rng)
            df = pd.DataFrame(Y, columns=["y1", "y2", "y3"],
                              index=pd.date_range("1980-01-01", periods=T, freq="QS"))
            fit = lib.fit_var(df, lags=dgp["p"], trend="c")
            irf = lib.cholesky_irf(
                fit, horizon=6, bands="residual_bootstrap",
                n_draws=N_DRAWS_BOOTSTRAP,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for h in target_h:
                records_68[h]["lower"].append(irf.lower_68[h, 1, 0])
                records_68[h]["upper"].append(irf.upper_68[h, 1, 0])
                records_90[h]["lower"].append(irf.lower_90[h, 1, 0])
                records_90[h]["upper"].append(irf.upper_90[h, 1, 0])

        _assert_all_replications(records_68, records_90)

        for h in target_h:
            true_val = true_irf[h, 1, 0]
            cov_68 = _coverage_rate(
                true_val,
                np.asarray(records_68[h]["lower"]),
                np.asarray(records_68[h]["upper"]),
            )
            cov_90 = _coverage_rate(
                true_val,
                np.asarray(records_90[h]["lower"]),
                np.asarray(records_90[h]["upper"]),
            )
            assert abs(cov_68 - 0.68) * 100 <= TOLERANCE_PP, (
                f"Cholesky bootstrap: at h={h}, empirical 68% coverage = "
                f"{cov_68:.3f} (target 0.68, tolerance ±{TOLERANCE_PP / 100:.2f})"
            )
            assert abs(cov_90 - 0.90) * 100 <= TOLERANCE_PP, (
                f"Cholesky bootstrap: at h={h}, empirical 90% coverage = "
                f"{cov_90:.3f} (target 0.90, tolerance ±{TOLERANCE_PP / 100:.2f})"
            )


# ---------------------------------------------------------------------------
# LP — HAC-robust bands
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLPHACCoverage:
    """Coverage study for `lp_irf` HAC bands.

    LP is a per-horizon regression with HAC-robust SEs (Newey-West,
    truncation = h+1). Bands are constructed as
    ``point ± z * HAC_SE``.

    On a stationary VAR(2) DGP with a structural shock injected as the
    first-equation innovation, the true LP-IRF at horizon h for response
    j is the (j, 1) entry of `Phi_h @ L` (Cholesky impact). With well-
    specified controls (lagged y), the LP should be unbiased and HAC
    bands should achieve nominal coverage at large T.
    """

    def test_coverage_at_h2_and_h4(self):
        dgp = _make_var_dgp()
        true_irf = _true_cholesky_irf(dgp, horizon=6)
        target_h = [2, 4]
        rng_master = np.random.default_rng(2)
        T = 250
        burn = 100

        records_68 = {h: {"lower": [], "upper": []} for h in target_h}
        records_90 = {h: {"lower": [], "upper": []} for h in target_h}

        L_true = np.linalg.cholesky(dgp["Sigma"])

        for rep in range(N_REPS):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            # Generate shocks first so we can inject them
            n = dgp["n"]
            T_total = T + burn
            shocks = rng.standard_normal((T_total, n))
            innov = shocks @ L_true.T  # (T_total, n) reduced-form innovations
            Y = np.zeros((T_total, n))
            for t in range(2, T_total):
                Y[t] = dgp["A1"] @ Y[t - 1] + dgp["A2"] @ Y[t - 2] + innov[t]
            Y = Y[burn:]
            shocks = shocks[burn:]
            df = pd.DataFrame(
                Y, columns=["y1", "y2", "y3"],
                index=pd.date_range("1980-01-01", periods=T, freq="QS"),
            )
            # Use the FIRST element of the structural shock vector as the LP shock
            irf = lib.lp_irf(
                df, shock=shocks[:, 0],
                horizon=6, lags=dgp["p"],
                shock_name="structural_shock_1",
            )
            for h in target_h:
                records_68[h]["lower"].append(irf.lower_68[h, 1, 0])
                records_68[h]["upper"].append(irf.upper_68[h, 1, 0])
                records_90[h]["lower"].append(irf.lower_90[h, 1, 0])
                records_90[h]["upper"].append(irf.upper_90[h, 1, 0])

        _assert_all_replications(records_68, records_90)

        for h in target_h:
            # LP recovers the (response, shock_1) entry; with shock_1 being
            # the structural shock to y1 (Cholesky-normalised), the true
            # response of y_j at horizon h is true_irf[h, j, 0] / L_true[0, 0].
            # Since L_true[0, 0] = std of u_1 = 1 here, the rescaling is 1.
            true_val = true_irf[h, 1, 0] / L_true[0, 0]
            cov_68 = _coverage_rate(
                true_val,
                np.asarray(records_68[h]["lower"]),
                np.asarray(records_68[h]["upper"]),
            )
            cov_90 = _coverage_rate(
                true_val,
                np.asarray(records_90[h]["lower"]),
                np.asarray(records_90[h]["upper"]),
            )
            # LP HAC bands can under-cover at long horizons in finite
            # samples (Montiel Olea & Plagborg-Møller 2021); these checks
            # are at short horizons (2 and 4), so the shared tolerance
            # applies.
            tol_pp = TOLERANCE_PP
            assert abs(cov_68 - 0.68) * 100 <= tol_pp, (
                f"LP HAC: at h={h}, empirical 68% coverage = "
                f"{cov_68:.3f} (target 0.68, tolerance ±{tol_pp / 100:.2f})"
            )
            assert abs(cov_90 - 0.90) * 100 <= tol_pp, (
                f"LP HAC: at h={h}, empirical 90% coverage = "
                f"{cov_90:.3f} (target 0.90, tolerance ±{tol_pp / 100:.2f})"
            )
