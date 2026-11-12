"""Tests for AFM (2023) higher-moment restrictions in sign_restricted_irf.

Validation strategy:

1. A threshold restriction must REJECT rotations whose implied
   structural-shock series violates the threshold.
2. The recovered shock series from an ACCEPTED rotation must satisfy
   the restriction (post-hoc check). The container stores only
   pointwise summaries across draws, so each check runs with
   ``target_accepted=1`` and fixed reduced-form parameters: the median
   is then that single accepted draw, and its impact vector b gives the
   shock series eps_t = u_t' Sigma_u^{-1} b, whose moment is recomputed
   here independently of the library's acceptance code.
3. Cross-shock comparisons (multi-shock mode) must enforce the
   pairwise inequality on the assigned shocks' moments, verified the
   same way for each assigned shock.
4. With "abs=True" the restriction applies to |moment|.
5. Validation errors fire on bad input (unknown moment, bad bounds,
   single-shock 'vs', etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _stats

import lib


# ---------------------------------------------------------------------------
# Synthetic data with non-Gaussian shocks (so moments matter)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nongaussian_var_data() -> pd.DataFrame:
    """Three-variable VAR(1) with heavily non-Gaussian innovations.

    Component 1: very leptokurtic + positively skewed (t_3 shifted right).
    Component 2: mildly leptokurtic + symmetric (t_8).
    Component 3: skewed (chi-squared centred).

    All three are non-Gaussian, so AFM-style restrictions can sharpen.
    """
    rng = np.random.default_rng(2024)
    T = 1000
    burn = 200
    A1 = np.array([[0.4, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.4]])
    L = np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [0.1, 0.2, 1.0]])

    eps1 = (rng.standard_t(df=3, size=T + burn) * np.sqrt(1 / 3)) + 0.4
    eps2 = rng.standard_t(df=8, size=T + burn) * np.sqrt(6 / 8)
    eps3 = (rng.chisquare(df=4, size=T + burn) - 4) / np.sqrt(8)   # centred chi^2_4
    eps = np.stack([eps1, eps2, eps3], axis=1)
    u = eps @ L.T

    Y = np.zeros((T + burn, 3))
    for t in range(1, T + burn):
        Y[t] = A1 @ Y[t - 1] + u[t]
    Y = Y[burn:]
    idx = pd.period_range("1980-01", periods=T, freq="M").to_timestamp()
    return pd.DataFrame(Y, columns=["y1", "y2", "y3"], index=idx)


def _single_accepted_draw(fit, seed, **kwargs):
    """Run ``sign_restricted_irf`` keeping exactly one accepted draw.

    With ``target_accepted=1`` and ``parameter_uncertainty=False`` the
    returned median is that one accepted rotation at the point-estimate
    reduced form, not a pointwise summary across draws.
    """
    irf = lib.sign_restricted_irf(
        fit, horizon=4, target_accepted=1, seed=seed,
        parameter_uncertainty=False, **kwargs,
    )
    assert irf.metadata["n_accepted"] == 1
    return irf


def _recovered_shock(fit, irf, shock_idx=0):
    """Structural shock series implied by an accepted impact vector.

    The impact vector is b = P q with P P' = Sigma_u, so the shock
    eps_t = q' P^{-1} u_t equals u_t' Sigma_u^{-1} b.
    """
    b = np.asarray(irf.median[0, :, shock_idx], dtype=float)
    return np.asarray(fit.residuals) @ np.linalg.solve(
        np.asarray(fit.Sigma_u), b,
    )


def _skew(eps):
    return float(_stats.skew(eps, bias=False))


def _excess_kurtosis(eps):
    return float(_stats.kurtosis(eps, fisher=True, bias=False))


# Seeds for the single-accepted-draw checks: several independent accepted
# rotations, each verified separately.
_POSTHOC_SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_moment_raises(self, nongaussian_var_data):
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        with pytest.raises(ValueError, match="unknown moment"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"x": {"y1": +1}},
                higher_moment_restrictions=[{"moment": "kurtosis"}],  # wrong; need "excess_kurtosis"
                target_accepted=5, max_rotations=200, seed=0,
                parameter_uncertainty=False,
            )

    def test_lower_greater_than_upper_raises(self, nongaussian_var_data):
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        with pytest.raises(ValueError, match="lower > upper"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"x": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "lower": 1.0, "upper": -1.0},
                ],
                target_accepted=5, max_rotations=200, seed=0,
                parameter_uncertainty=False,
            )

    def test_vs_on_single_shock_raises(self, nongaussian_var_data):
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        with pytest.raises(ValueError, match="cross-shock"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"x": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "vs": "other"},
                ],
                target_accepted=5, max_rotations=200, seed=0,
                parameter_uncertainty=False,
            )

    def test_vs_missing_comparison_raises(self, nongaussian_var_data):
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        with pytest.raises(ValueError, match="comparison"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
                higher_moment_restrictions=[
                    {"shock": "a", "moment": "skewness", "vs": "b"},
                ],
                target_accepted=5, max_rotations=2000, seed=0,
                parameter_uncertainty=False,
            )


# ---------------------------------------------------------------------------
# Single-shock threshold
# ---------------------------------------------------------------------------


class TestSingleShockThreshold:
    def test_accepted_shocks_satisfy_skew_lower(self, nongaussian_var_data):
        """Every accepted rotation's implied shock must have skew >= 0.2."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        # First, do an unconstrained sign-only fit to see the baseline acceptance
        irf_base = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1}},
            target_accepted=200, max_rotations=5000, seed=0,
            parameter_uncertainty=False,
        )
        base_n = irf_base.metadata["n_accepted"]

        # Now add a skewness threshold and verify (a) accepted draws satisfy it,
        # (b) acceptance count drops.
        irf_hm = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1}},
            higher_moment_restrictions=[
                {"moment": "skewness", "lower": 0.2},
            ],
            target_accepted=200, max_rotations=5000, seed=0,
            parameter_uncertainty=False,
        )
        # Acceptance rate must drop relative to sign-only (or equal if the
        # baseline already always satisfied skew >= 0.2)
        assert irf_hm.metadata["n_accepted"] <= base_n
        assert irf_hm.metadata["n_rejected_by_higher_moment"] > 0, (
            f"expected some HM rejections; got "
            f"n_rejected_by_higher_moment="
            f"{irf_hm.metadata['n_rejected_by_higher_moment']}"
        )
        assert "higher_moment" in irf_hm.metadata["identification"]
        # Post-hoc: recover the shock from single accepted draws and
        # recompute its skewness independently.
        for seed in _POSTHOC_SEEDS:
            irf_1 = _single_accepted_draw(
                fit, seed,
                sign_restrictions={"shock": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "lower": 0.2},
                ],
                max_rotations=5000,
            )
            skew = _skew(_recovered_shock(fit, irf_1))
            assert skew >= 0.2 - 1e-10, (
                f"seed={seed}: accepted shock has skewness {skew:.4f} < 0.2"
            )

    def test_unsatisfiable_threshold_raises(self, nongaussian_var_data):
        """A wildly unsatisfiable threshold should produce zero acceptances."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        with pytest.raises(RuntimeError, match="no rotations"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"shock": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "lower": 100.0},   # unsatisfiable
                ],
                target_accepted=200, max_rotations=500, seed=0,
                parameter_uncertainty=False,
            )

    def test_abs_flag_works(self, nongaussian_var_data):
        """abs=True should allow EITHER sign of the moment to satisfy a lower bound."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        # Threshold |skew| >= 0.2 (vs signed skew >= 0.2)
        irf_signed = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"shock": {"y1": 0}},   # no sign pin → more candidates
            higher_moment_restrictions=[
                {"moment": "skewness", "lower": 0.2},
            ],
            target_accepted=100, max_rotations=10_000, seed=0,
            parameter_uncertainty=False,
        )
        irf_abs = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"shock": {"y1": 0}},
            higher_moment_restrictions=[
                {"moment": "skewness", "lower": 0.2, "abs": True},
            ],
            target_accepted=100, max_rotations=10_000, seed=0,
            parameter_uncertainty=False,
        )
        # abs accepts both positive and negative skew → at least as many accepted
        assert (
            irf_abs.metadata["n_accepted"] >= irf_signed.metadata["n_accepted"]
        ), (
            f"abs ({irf_abs.metadata['n_accepted']}) should not accept fewer "
            f"than signed ({irf_signed.metadata['n_accepted']})"
        )


# ---------------------------------------------------------------------------
# Multi-shock: per-shock threshold + cross-shock comparison
# ---------------------------------------------------------------------------


class TestMultiShockHigherMoment:
    def test_per_shock_threshold_routes_correctly(self, nongaussian_var_data):
        """Per-shock thresholds in multi-shock mode require 'shock' key."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        # No 'shock' key in multi-shock mode → routing error
        with pytest.raises(ValueError, match="must include a 'shock' key"):
            lib.sign_restricted_irf(
                fit, horizon=4,
                sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "lower": 0.0},   # missing "shock"
                ],
                target_accepted=10, max_rotations=2000, seed=0,
                parameter_uncertainty=False,
            )

    def test_cross_shock_comparison_greater(self, nongaussian_var_data):
        """Cross-shock 'greater' comparison must enforce moment(a) >= moment(b)."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"a": {"y1": +1, "y2": 0}, "b": {"y2": +1}},
            higher_moment_restrictions=[
                {
                    "shock": "a", "moment": "excess_kurtosis",
                    "vs": "b", "comparison": "greater",
                },
            ],
            target_accepted=50, max_rotations=10_000, seed=0,
            parameter_uncertainty=False,
        )
        assert irf.metadata["n_accepted"] > 0
        # The metadata tag should reflect HM was active
        assert "higher_moment" in irf.metadata["identification"]
        # Post-hoc: recover both assigned shocks from single accepted draws
        # and check the pairwise inequality directly.
        for seed in _POSTHOC_SEEDS:
            irf_1 = _single_accepted_draw(
                fit, seed,
                sign_restrictions={"a": {"y1": +1, "y2": 0}, "b": {"y2": +1}},
                higher_moment_restrictions=[
                    {
                        "shock": "a", "moment": "excess_kurtosis",
                        "vs": "b", "comparison": "greater",
                    },
                ],
                max_rotations=10_000,
            )
            k_a = _excess_kurtosis(_recovered_shock(fit, irf_1, 0))
            k_b = _excess_kurtosis(_recovered_shock(fit, irf_1, 1))
            assert k_a >= k_b - 1e-10, (
                f"seed={seed}: kurtosis(a)={k_a:.4f} < kurtosis(b)={k_b:.4f}"
            )

    def test_cross_shock_comparison_less(self, nongaussian_var_data):
        """Cross-shock 'less' comparison must enforce moment(a) <= moment(b)."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
            higher_moment_restrictions=[
                {
                    "shock": "a", "moment": "skewness",
                    "vs": "b", "comparison": "less",
                },
            ],
            target_accepted=50, max_rotations=10_000, seed=0,
            parameter_uncertainty=False,
        )
        assert irf.metadata["n_accepted"] > 0
        for seed in _POSTHOC_SEEDS:
            irf_1 = _single_accepted_draw(
                fit, seed,
                sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
                higher_moment_restrictions=[
                    {
                        "shock": "a", "moment": "skewness",
                        "vs": "b", "comparison": "less",
                    },
                ],
                max_rotations=10_000,
            )
            s_a = _skew(_recovered_shock(fit, irf_1, 0))
            s_b = _skew(_recovered_shock(fit, irf_1, 1))
            assert s_a <= s_b + 1e-10, (
                f"seed={seed}: skewness(a)={s_a:.4f} > skewness(b)={s_b:.4f}"
            )

    def test_metadata_tag_with_hm(self, nongaussian_var_data):
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
            higher_moment_restrictions=[
                {"shock": "a", "moment": "skewness", "lower": -10.0, "upper": 10.0},
            ],
            target_accepted=20, max_rotations=2000, seed=0,
            parameter_uncertainty=False,
        )
        assert "higher_moment" in irf.metadata["identification"]
        assert "AFM 2023" in irf.metadata["method"]

    def test_combined_narrative_plus_higher_moment(self, nongaussian_var_data):
        """Composability: narrative + higher-moment together (both must hold)."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        # Pick a date in the effective sample
        date = fit.data.index[fit.p + 100]
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"a": {"y1": +1}, "b": {"y2": +1}},
            narrative_restrictions=[
                {"shock": "a", "date": date, "sign": +1},
            ],
            higher_moment_restrictions=[
                {"shock": "a", "moment": "excess_kurtosis", "lower": -10.0},
            ],
            target_accepted=10, max_rotations=10_000, seed=0,
            parameter_uncertainty=False,
        )
        assert irf.metadata["n_accepted"] > 0
        assert "narrative" in irf.metadata["identification"]
        assert "higher_moment" in irf.metadata["identification"]


# ---------------------------------------------------------------------------
# Post-hoc verification: shocks from accepted rotations satisfy the bound
# ---------------------------------------------------------------------------


class TestRecoveredShocksSatisfyBounds:
    def test_recovered_shock_skewness_above_threshold(self, nongaussian_var_data):
        """Recover the structural shock from single accepted rotations and
        verify each has skew >= lower bound."""
        fit = lib.fit_var(nongaussian_var_data, lags=1)
        # Use a moderate threshold the data CAN satisfy
        lower = 0.1
        irf = lib.sign_restricted_irf(
            fit, horizon=4,
            sign_restrictions={"shock": {"y1": +1}},
            higher_moment_restrictions=[
                {"moment": "skewness", "lower": lower},
            ],
            target_accepted=50, max_rotations=20_000, seed=0,
            parameter_uncertainty=False,
        )
        # The container does not store individual draws, and the pointwise
        # median across several accepted IRFs is generally not itself an
        # accepted rotation, so it cannot be used for the post-hoc check.
        # The multi-draw run only establishes that the bound binds.
        assert irf.metadata["n_accepted"] > 0
        assert irf.metadata["n_rejected_by_higher_moment"] > 0
        # Post-hoc check on single accepted draws, where the median IS the
        # accepted rotation.
        for seed in _POSTHOC_SEEDS:
            irf_1 = _single_accepted_draw(
                fit, seed,
                sign_restrictions={"shock": {"y1": +1}},
                higher_moment_restrictions=[
                    {"moment": "skewness", "lower": lower},
                ],
                max_rotations=20_000,
            )
            skew = _skew(_recovered_shock(fit, irf_1))
            assert skew >= lower - 1e-10, (
                f"seed={seed}: accepted shock has skewness {skew:.4f} < {lower}"
            )
