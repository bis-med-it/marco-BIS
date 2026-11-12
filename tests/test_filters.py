"""Tests for filters.py (trend-cycle decomposition).

Validation strategy:

1. **Additive decomposition** — for every filter, trend + cycle = input
   series at every index where the filter is defined.
2. **Cross-package** — HP filter matches ``statsmodels.tsa.filters.hp_filter``
   exactly (we wrap it); the Baxter-King and Christiano-Fitzgerald band-pass
   filters match ``statsmodels``' ``bkfilter`` / ``cffilter``.
3. **Theoretical properties** — Hamilton residuals are mean-zero; BN cycle
   uses the AR(p) companion-form formula; the KMW δ̄ restriction is the
   exact linear constraint Σφ = 1 - 1/sqrt(δ̄) (δ̄ = 0.25 ⇒ Σφ = -1),
   and at δ̄ below the unconstrained estimate it produces a LARGER cycle
   than unconstrained BN (the headline KMW result); the BK weights sum
   to zero (a constant series maps to a zero cycle); the one-sided HP trend
   equals the two-sided HP trend at the final sample point.
4. **Synthetic-DGP** — a known I(1) random walk has a trivial BN
   decomposition (cycle ≡ 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.filters.hp_filter import hpfilter as _sm_hpfilter
from statsmodels.tsa.filters.bk_filter import bkfilter as _sm_bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter as _sm_cffilter

import lib


@pytest.fixture(scope="module")
def quarterly_series() -> pd.Series:
    """A 200-quarter log-real-GDP-like series with trend + cycle structure."""
    rng = np.random.default_rng(0)
    T = 200
    t = np.arange(T)
    trend = 100.0 + 0.5 * t / 4   # linear trend at 2% per year
    cycle = 1.5 * np.sin(2 * np.pi * t / 32) + rng.standard_normal(T) * 0.3
    idx = pd.period_range("1970Q1", periods=T, freq="Q").to_timestamp()
    return pd.Series(trend + cycle, index=idx, name="y")


@pytest.fixture(scope="module")
def random_walk_series() -> pd.Series:
    rng = np.random.default_rng(1)
    T = 200
    eps = rng.standard_normal(T)
    return pd.Series(np.cumsum(eps), name="rw")


class TestHPFilter:
    def test_additive_decomposition(self, quarterly_series):
        result = lib.hp_filter(quarterly_series, lam=1600)
        recon = result.trend + result.cycle
        assert np.allclose(recon.values, quarterly_series.values, atol=1e-10)

    def test_matches_statsmodels(self, quarterly_series):
        result = lib.hp_filter(quarterly_series, lam=1600)
        cycle_sm, trend_sm = _sm_hpfilter(quarterly_series.astype(float), lamb=1600)
        assert np.allclose(result.cycle.values, cycle_sm.values, atol=1e-10)
        assert np.allclose(result.trend.values, trend_sm.values, atol=1e-10)

    def test_default_lambda_by_frequency(self, quarterly_series):
        r_q = lib.hp_filter(quarterly_series, frequency="quarterly")
        assert r_q.metadata["lam"] == 1600.0
        r_m = lib.hp_filter(quarterly_series, frequency="monthly")
        assert r_m.metadata["lam"] == 14400.0
        r_a = lib.hp_filter(quarterly_series, frequency="annual")
        assert r_a.metadata["lam"] == 6.25

    def test_rejects_nan(self, quarterly_series):
        bad = quarterly_series.copy()
        bad.iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.hp_filter(bad)


class TestHamiltonFilter:
    def test_additive_decomposition_where_defined(self, quarterly_series):
        result = lib.hamilton_filter(quarterly_series, h=8, p=4)
        recon = result.trend + result.cycle
        # First p + h - 1 = 11 observations are NaN
        mask = ~result.trend.isna()
        assert mask.sum() == len(quarterly_series) - (4 + 8 - 1)
        assert np.allclose(recon[mask].values, quarterly_series[mask].values,
                            atol=1e-10)

    def test_residuals_have_zero_mean(self, quarterly_series):
        """OLS residuals must have zero mean by construction."""
        result = lib.hamilton_filter(quarterly_series, h=8, p=4)
        cycle_valid = result.cycle.dropna()
        # Mean of OLS residuals when X includes a constant must be ~0
        assert abs(float(cycle_valid.mean())) < 1e-10

    def test_default_h_by_frequency(self, quarterly_series):
        r = lib.hamilton_filter(quarterly_series, frequency="quarterly")
        assert r.metadata["h"] == 8
        r = lib.hamilton_filter(quarterly_series, frequency="monthly")
        assert r.metadata["h"] == 24

    def test_invalid_parameters_rejected(self, quarterly_series):
        with pytest.raises(ValueError, match="p must be"):
            lib.hamilton_filter(quarterly_series, h=8, p=0)
        with pytest.raises(ValueError, match="h must be"):
            lib.hamilton_filter(quarterly_series, h=0, p=4)

    def test_too_short_series_rejected(self):
        short = pd.Series(np.arange(10).astype(float))
        with pytest.raises(ValueError, match="too short"):
            lib.hamilton_filter(short, h=8, p=4)


class TestBeveridgeNelson:
    def test_additive_decomposition_where_defined(self, quarterly_series):
        result = lib.beveridge_nelson(quarterly_series, ar_order=2)
        recon = result.trend + result.cycle
        mask = ~result.trend.isna()
        assert np.allclose(recon[mask].values, quarterly_series[mask].values,
                            atol=1e-10)

    def test_companion_eigenvalues_strictly_below_one(self, quarterly_series):
        """If the AR(p) is stationary, the companion-matrix eigenvalues must
        all have magnitude < 1; that's a sanity check on the AR fit."""
        result = lib.beveridge_nelson(quarterly_series, ar_order=2)
        assert result.metadata["max_companion_eigenvalue"] < 1.0

    def test_ar_order_selection_returns_valid_p(self, quarterly_series):
        result = lib.beveridge_nelson(quarterly_series, max_ar_order=8, ic="bic")
        assert 1 <= result.metadata["ar_order"] <= 8

    def test_random_walk_has_near_zero_cycle(self, random_walk_series):
        """A true random walk has Δy ~ WN, so AR(p) phi ≈ 0 and cycle ≈ 0
        (the BN trend ≈ the level itself)."""
        result = lib.beveridge_nelson(random_walk_series, ar_order=1)
        cycle_valid = result.cycle.dropna()
        # Cycle should be tiny relative to the level
        rel_cycle = float(np.std(cycle_valid) / np.std(random_walk_series))
        assert rel_cycle < 0.1

    def test_rejects_nan(self, quarterly_series):
        bad = quarterly_series.copy()
        bad.iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.beveridge_nelson(bad)


class TestKMW_BN:
    """KMW (2018) BN filter — the paper's δ̄ = ψ(1)² restriction.

    Regression contract: ``kmw_bn_filter`` imposes KMW's δ̄ = ψ(1)² via
    the exact LINEAR restriction Σφ = 1 - 1/sqrt(δ̄)
    (δ̄ = 0.25 ⇒ Σφ = -1, ψ(1) = 0.5), estimated by closed-form OLS on
    the Dickey-Fuller-substituted regression. An innovation-to-Var(Δy)
    ratio constraint — a different object — would instead amplify the
    cycle through near-unit-root persistence (Σφ → +1), qualitatively
    the opposite dynamics.
    """

    def test_runs_end_to_end(self, quarterly_series):
        result = lib.kmw_bn_filter(quarterly_series, delta=0.25, ar_order=4)
        assert result.method == "kmw_bn_filter"
        assert result.trend.notna().sum() > 0
        assert result.cycle.notna().sum() > 0

    def test_sum_phi_equals_kmw_restriction_exactly(self, quarterly_series):
        """The defining contract: Σφ = 1 - 1/sqrt(δ̄), exact (linear
        restriction imposed by substitution, not by an optimizer)."""
        for delta in (0.25, 0.5, 1.0, 2.0):
            kmw = lib.kmw_bn_filter(quarterly_series, delta=delta, ar_order=4)
            assert kmw.metadata["sum_phi"] == pytest.approx(
                1.0 - 1.0 / np.sqrt(delta), abs=1e-10)
        # delta = 0.25 is the canonical case: Σφ = -1, ψ(1) = 0.5
        kmw = lib.kmw_bn_filter(quarterly_series, delta=0.25, ar_order=4)
        assert kmw.metadata["sum_phi"] == pytest.approx(-1.0, abs=1e-10)
        assert kmw.metadata["psi_1"] == pytest.approx(0.5, abs=1e-10)

    def test_low_delta_cycle_is_larger_than_bn_on_gdp_like_dgp(self):
        """KMW's headline: on GDP-like data — where the unconstrained δ̂ is
        ≈ 2 and the classical BN cycle is tiny — imposing δ̄ = 0.25 produces
        a much larger-amplitude cycle. (This is an EMPIRICAL property of
        that setting, not a theorem: DGPs whose Δy is dominated by big
        oscillatory dynamics can invert it, so the test uses an I(1)
        AR(1)-growth process mirroring US log GDP.)"""
        rng = np.random.default_rng(4)
        T = 240
        dy = np.zeros(T)
        for t in range(1, T):
            dy[t] = 0.005 + 0.35 * dy[t - 1] + 0.007 * rng.standard_normal()
        y = pd.Series(100 + np.cumsum(dy),
                      index=pd.period_range("1960Q1", periods=T,
                                            freq="Q").to_timestamp(),
                      name="lgdp")
        bn = lib.beveridge_nelson(y, ar_order=4)
        kmw = lib.kmw_bn_filter(y, delta=0.25, ar_order=4)
        assert kmw.metadata["delta_unconstrained"] > 1.0   # the KMW setting
        assert kmw.cycle.std() > 1.5 * bn.cycle.std()

    def test_cycle_amplitude_monotone_decreasing_in_delta(self, quarterly_series):
        """Lower δ̄ = smaller trend-shock share = larger cycle: amplitude
        must fall monotonically as δ̄ rises."""
        stds = [lib.kmw_bn_filter(quarterly_series, delta=d, ar_order=4).cycle.std()
                for d in (0.05, 0.1, 0.25, 0.5)]
        assert all(stds[i] > stds[i + 1] for i in range(len(stds) - 1))

    def test_trend_plus_cycle_identity(self, quarterly_series):
        kmw = lib.kmw_bn_filter(quarterly_series, delta=0.25, ar_order=4)
        valid = kmw.trend.notna() & kmw.cycle.notna()
        np.testing.assert_allclose(
            (kmw.trend + kmw.cycle)[valid].values,
            quarterly_series[valid].values,
            atol=1e-10,
        )

    def test_constrained_ar_remains_stationary(self, quarterly_series):
        """With Σφ = -1 the AR stays away from the unit root; the companion
        max-modulus must be strictly inside the unit circle (the OLD
        mechanism drove it to ~0.99)."""
        kmw = lib.kmw_bn_filter(quarterly_series, delta=0.25, ar_order=4)
        assert kmw.metadata["max_companion_eigenvalue"] < 1.0

    def test_reports_unconstrained_delta(self, quarterly_series):
        kmw = lib.kmw_bn_filter(quarterly_series, delta=0.25, ar_order=4)
        assert np.isfinite(kmw.metadata["delta_unconstrained"])
        assert kmw.metadata["delta_unconstrained"] > 0

    def test_rejects_nan_input(self, quarterly_series):
        s = quarterly_series.copy()
        s.iloc[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.kmw_bn_filter(s, delta=0.25, ar_order=4)

    def test_rejects_invalid_delta(self, quarterly_series):
        for bad in (-0.1, 0.0):
            with pytest.raises(ValueError, match="delta must be"):
                lib.kmw_bn_filter(quarterly_series, delta=bad, ar_order=4)

    def test_rejects_too_short_series(self):
        s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2000", periods=3, freq="QS"))
        with pytest.raises(ValueError, match="too short"):
            lib.kmw_bn_filter(s, delta=0.25, ar_order=2)

    def test_ic_lag_selection_runs(self, quarterly_series):
        """Auto lag selection via BIC produces a usable result."""
        kmw = lib.kmw_bn_filter(quarterly_series, delta=0.25, ic="bic")
        assert isinstance(kmw.metadata["ar_order"], int)
        assert 1 <= kmw.metadata["ar_order"] <= 12


class TestOneSidedHP:
    """One-sided (expanding-window, real-time) HP filter."""

    def test_additive_decomposition_where_defined(self, quarterly_series):
        r = lib.one_sided_hp_filter(quarterly_series, lam=1600)
        mask = ~r.trend.isna()
        recon = (r.trend + r.cycle)[mask]
        assert np.allclose(recon.values, quarterly_series[mask].values, atol=1e-10)

    def test_last_point_matches_two_sided(self, quarterly_series):
        """At the final date the expanding window is the full sample, so the
        one-sided trend coincides with the two-sided HP trend there."""
        one = lib.one_sided_hp_filter(quarterly_series, lam=1600)
        two = lib.hp_filter(quarterly_series, lam=1600)
        assert np.isclose(one.trend.iloc[-1], two.trend.iloc[-1], atol=1e-8)

    def test_differs_from_two_sided_in_interior(self, quarterly_series):
        """Interior one-sided trend is not revised by future data, so it
        differs materially from the two-sided trend."""
        one = lib.one_sided_hp_filter(quarterly_series, lam=1600)
        two = lib.hp_filter(quarterly_series, lam=1600)
        assert abs(one.trend.iloc[100] - two.trend.iloc[100]) > 1e-3

    def test_warmup_is_nan(self, quarterly_series):
        r = lib.one_sided_hp_filter(quarterly_series, lam=1600, min_periods=4)
        assert r.trend.iloc[:3].isna().all()
        assert r.trend.iloc[3:].notna().all()
        assert r.metadata["two_sided"] is False

    def test_default_lambda_by_frequency(self, quarterly_series):
        assert lib.one_sided_hp_filter(quarterly_series, frequency="annual").metadata["lam"] == 6.25
        assert lib.one_sided_hp_filter(quarterly_series, frequency="monthly").metadata["lam"] == 14400.0

    def test_rejects_nan(self, quarterly_series):
        bad = quarterly_series.copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.one_sided_hp_filter(bad)

    def test_rejects_bad_min_periods(self, quarterly_series):
        with pytest.raises(ValueError, match="min_periods must be"):
            lib.one_sided_hp_filter(quarterly_series, min_periods=2)


class TestBKFilter:
    """Baxter-King (1999) symmetric band-pass filter."""

    def test_additive_decomposition_where_defined(self, quarterly_series):
        r = lib.bk_filter(quarterly_series, low=6, high=32, K=12)
        mask = ~r.trend.isna()
        assert mask.sum() == len(quarterly_series) - 2 * 12
        recon = (r.trend + r.cycle)[mask]
        assert np.allclose(recon.values, quarterly_series[mask].values, atol=1e-10)

    def test_matches_statsmodels(self, quarterly_series):
        r = lib.bk_filter(quarterly_series, low=6, high=32, K=12)
        cyc_sm = _sm_bkfilter(quarterly_series.astype(float), low=6, high=32, K=12)
        np.testing.assert_allclose(
            r.cycle.dropna().values, np.asarray(cyc_sm).ravel(), atol=1e-8
        )

    def test_weights_sum_to_zero_kills_constant(self):
        """Sum-to-zero weights => zero gain at frequency 0 => a constant maps
        to a (numerically) zero cycle."""
        const = pd.Series(np.full(100, 5.0), name="c")
        r = lib.bk_filter(const, low=6, high=32, K=12)
        assert np.allclose(r.cycle.dropna().values, 0.0, atol=1e-9)

    def test_defaults_by_frequency(self, quarterly_series):
        m = lib.bk_filter(quarterly_series, frequency="quarterly").metadata
        assert (m["low"], m["high"], m["K"]) == (6, 32, 12)

    def test_rejects_nan(self, quarterly_series):
        bad = quarterly_series.copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.bk_filter(bad)

    def test_too_short_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            lib.bk_filter(pd.Series(np.arange(20).astype(float)), K=12)

    def test_invalid_band_rejected(self, quarterly_series):
        with pytest.raises(ValueError, match="low < high"):
            lib.bk_filter(quarterly_series, low=32, high=6)


class TestCFFilter:
    """Christiano-Fitzgerald (2003) asymmetric band-pass filter."""

    def test_additive_decomposition_full_sample(self, quarterly_series):
        r = lib.cf_filter(quarterly_series, low=6, high=32)
        assert r.trend.notna().all()   # CF uses the full sample — no obs lost
        recon = r.trend + r.cycle
        assert np.allclose(recon.values, quarterly_series.values, atol=1e-10)

    def test_matches_statsmodels(self, quarterly_series):
        r = lib.cf_filter(quarterly_series, low=6, high=32, drift=True)
        cyc_sm, _ = _sm_cffilter(
            quarterly_series.astype(float), low=6, high=32, drift=True
        )
        np.testing.assert_allclose(
            r.cycle.values, np.asarray(cyc_sm).ravel(), atol=1e-10
        )

    def test_defaults_by_frequency(self, quarterly_series):
        m = lib.cf_filter(quarterly_series, frequency="monthly").metadata
        assert (m["low"], m["high"]) == (18, 96)
        assert m["two_sided"] is False

    def test_rejects_nan(self, quarterly_series):
        bad = quarterly_series.copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            lib.cf_filter(bad)

    def test_invalid_band_rejected(self, quarterly_series):
        with pytest.raises(ValueError, match="low < high"):
            lib.cf_filter(quarterly_series, low=40, high=10)


class TestFilterResult:
    def test_trend_cycle_index_must_match(self, quarterly_series):
        idx_short = quarterly_series.index[:-5]
        with pytest.raises(ValueError, match="same index"):
            lib.FilterResult(
                method="custom",
                trend=quarterly_series.copy(),
                cycle=pd.Series(np.zeros(len(idx_short)), index=idx_short),
            )

    def test_to_dict_is_json_friendly(self, quarterly_series):
        import json
        result = lib.hp_filter(quarterly_series, lam=1600)
        d = result.to_dict()
        # Should serialise without error
        s = json.dumps(d)
        assert "hp" in s


class TestBNStationarityGuard:
    """Acceptance check: the BN/KMW long-run forecast exists only for
    stationary growth dynamics; order selection must be unit-invariant."""

    def _level_series(self, phi: float, T: int = 150, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        dy = np.zeros(T)
        for t in range(1, T):
            dy[t] = phi * dy[t - 1] + rng.standard_normal()
        y = 100 + np.cumsum(dy)
        idx = pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp()
        return pd.Series(y, index=idx)

    def test_bn_rejects_explosive_growth(self):
        s = self._level_series(1.05)
        with pytest.raises(RuntimeError, match="stationary"):
            lib.beveridge_nelson(s, ar_order=1)

    def test_kmw_rejects_p1_with_default_delta(self):
        # sum(phi) = 1 - 1/sqrt(0.25) = -1; with p=1 that is phi_1 = -1
        # exactly — a negative unit root, so the BN sum does not converge.
        s = self._level_series(0.5)
        with pytest.raises(RuntimeError, match="stationary"):
            lib.kmw_bn_filter(s, ar_order=1)

    def test_bn_stationary_still_works(self):
        s = self._level_series(0.5)
        res = lib.beveridge_nelson(s, ar_order=1)
        assert np.isfinite(res.cycle.dropna()).all()

    def test_ar_order_selection_unit_invariant(self):
        s = self._level_series(0.6, T=120, seed=7)
        from lib.filters import _select_ar_order
        dy = s.diff().dropna().values
        orders = {c: _select_ar_order(dy * c, 12, "bic") for c in (0.01, 1.0, 100.0)}
        assert len(set(orders.values())) == 1, orders

    def test_bn_cycle_unit_invariant(self):
        s = self._level_series(0.6, T=120, seed=7)
        r1 = lib.beveridge_nelson(s, max_ar_order=12)
        r2 = lib.beveridge_nelson(s * 100.0, max_ar_order=12)
        a = r1.cycle.dropna()
        b = (r2.cycle / 100.0).dropna()
        assert np.allclose(a.values, b.values, rtol=1e-8)
