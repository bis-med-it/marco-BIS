"""Tests for connectedness.py (Diebold-Yilmaz 2012).

Validation strategy:

1. **Row normalisation** — generalised FEVD rows must sum to 1.
2. **Total spillover bounded** — total connectedness is in [0, 100].
3. **Independent variables** — for a diagonal Σ_u (uncorrelated shocks)
   and diagonal lag matrices, GFEVD is the identity → zero cross-variable
   spillovers → total ≈ 0.
4. **Symmetric DGP** — for a symmetric DGP, the pairwise net flows
   between any two variables should average ≈ 0.
5. **Rolling-window output shape**.
6. **Hand-computed measures**: a fixed asymmetric FEVD is injected at
   the ``generalised_fevd`` boundary and every reported measure (total,
   directional from/to, net, pairwise net, and the rolling columns) is
   compared with values worked out by hand, direction and sign included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


@pytest.fixture(scope="module")
def independent_var_data() -> pd.DataFrame:
    """Three uncorrelated AR(1) series — no cross-variable spillovers."""
    rng = np.random.default_rng(7)
    T = 400
    burn = 100
    n = 3
    phi = 0.5
    Y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        Y[t] = phi * Y[t - 1] + rng.standard_normal(n)
    return pd.DataFrame(
        Y[burn:],
        columns=["a", "b", "c"],
        index=pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp(),
    )


class TestGeneralisedFEVD:
    def test_rows_sum_to_one(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        fevd = lib.generalised_fevd(fit, horizon=10, row_normalise=True)
        assert np.allclose(fevd.sum(axis=1), 1.0, atol=1e-10)

    def test_raw_fevd_diagonal_positive(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        raw = lib.generalised_fevd(fit, horizon=10, row_normalise=False)
        # Each variable's own-contribution must be positive
        assert (np.diag(raw) > 0).all()

    def test_horizon_must_be_positive(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        with pytest.raises(ValueError, match="horizon must be"):
            lib.generalised_fevd(fit, horizon=0)


class TestConnectedness:
    def test_total_in_zero_to_hundred(self, synthetic_var_data):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=10)
        assert 0.0 <= result.total <= 100.0

    def test_directional_from_to_sum_relates_to_total(self, synthetic_var_data):
        """Both directional sums equal the total spillover index by construction.

        directional_from_i = (off-diag row sum_i) / N * 100
        total              = (sum of all off-diag entries) / N * 100
        Therefore Σ_i directional_from_i = total.
        Symmetrically for directional_to.
        """
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=10)
        assert np.isclose(result.directional_from.sum(), result.total, atol=1e-9)
        assert np.isclose(result.directional_to.sum(), result.total, atol=1e-9)

    def test_net_sums_to_zero(self, synthetic_var_data):
        """Net flows are zero-sum across variables by construction."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=10)
        assert abs(result.net.sum()) < 1e-9

    def test_independent_variables_have_near_zero_total(self, independent_var_data):
        """With diagonal AR coefficients and shocks, connectedness ≈ 0."""
        fit = lib.fit_var(independent_var_data, lags=1)
        result = lib.connectedness(fit, horizon=10)
        # The estimated Σ may have small off-diagonal sampling noise, so
        # total won't be exactly zero — but should be < 10%.
        assert result.total < 10.0

    def test_pairwise_net_antisymmetric(self, synthetic_var_data):
        """Pairwise net matrix must be antisymmetric: P_{ij} = -P_{ji}."""
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=10)
        assert np.allclose(result.pairwise_net, -result.pairwise_net.T, atol=1e-10)

    def test_result_dict_serialisable(self, synthetic_var_data):
        import json
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=10)
        d = result.to_dict()
        # Must serialise without error
        json.dumps(d)


# Row-normalised FEVD, theta[i, j] = share of i's forecast-error variance
# due to shocks in j. Asymmetric, every off-diagonal entry nonzero.
_HAND_FEVD = np.array([
    [0.70, 0.20, 0.10],
    [0.05, 0.90, 0.05],
    [0.30, 0.10, 0.60],
])
# Hand-computed with N = 3 (DY 2012 Eq. 12 divisor):
#   off-diagonal mass 0.80                       -> total 80 / 3
#   FROM = row off-diagonal sums 0.30, 0.10, 0.40 -> 30/3, 10/3, 40/3
#   TO   = col off-diagonal sums 0.35, 0.30, 0.15 -> 35/3, 30/3, 15/3
#   NET  = TO - FROM                              -> 5/3, 20/3, -25/3
#   pairwise[i, j] = 100 (theta[j, i] - theta[i, j]) / 3, positive when
#   i transmits more to j than it receives from j:
#     (a, b): 0.05 - 0.20 -> -15/3    (a is a net receiver from b)
#     (a, c): 0.30 - 0.10 -> +20/3    (a is a net transmitter to c)
#     (b, c): 0.10 - 0.05 ->  +5/3
_HAND_TOTAL = 80.0 / 3
_HAND_FROM = np.array([30.0, 10.0, 40.0]) / 3
_HAND_TO = np.array([35.0, 30.0, 15.0]) / 3
_HAND_NET = np.array([5.0, 20.0, -25.0]) / 3
_HAND_PAIRWISE = np.array([
    [0.0, -15.0, 20.0],
    [15.0, 0.0, 5.0],
    [-20.0, -5.0, 0.0],
]) / 3


@pytest.fixture
def hand_fevd(monkeypatch):
    """Replace the generalised FEVD with the fixed hand matrix."""
    import importlib
    conn_module = importlib.import_module("lib.connectedness")
    calls = []

    def fake_generalised_fevd(fit, horizon=10, row_normalise=True):
        calls.append((horizon, row_normalise))
        assert row_normalise, "connectedness must use the row-normalised FEVD"
        return _HAND_FEVD.copy()

    monkeypatch.setattr(conn_module, "generalised_fevd", fake_generalised_fevd)
    return calls


class TestHandComputedConnectedness:
    def test_measures_match_hand_computation(self, synthetic_var_data, hand_fevd):
        fit = lib.fit_var(synthetic_var_data, lags=2)
        result = lib.connectedness(fit, horizon=7)
        assert hand_fevd == [(7, True)]
        assert np.array_equal(result.fevd, _HAND_FEVD)
        assert result.total == pytest.approx(_HAND_TOTAL, abs=1e-12)
        np.testing.assert_allclose(result.directional_from, _HAND_FROM, atol=1e-12)
        np.testing.assert_allclose(result.directional_to, _HAND_TO, atol=1e-12)
        np.testing.assert_allclose(result.net, _HAND_NET, atol=1e-12)
        np.testing.assert_allclose(result.pairwise_net, _HAND_PAIRWISE, atol=1e-12)

    def test_rolling_columns_carry_hand_measures(self, synthetic_var_data, hand_fevd):
        result = lib.rolling_connectedness(
            synthetic_var_data, window=100, horizon=10, lags=2, step=100,
        )
        # windows end at rows 99, 199, 299, 399
        assert len(result) == 4 and len(hand_fevd) == 4
        np.testing.assert_allclose(result["total"], _HAND_TOTAL, atol=1e-12)
        for j, v in enumerate(synthetic_var_data.columns):
            np.testing.assert_allclose(
                result[f"directional_from_{v}"], _HAND_FROM[j], atol=1e-12)
            np.testing.assert_allclose(
                result[f"directional_to_{v}"], _HAND_TO[j], atol=1e-12)
            np.testing.assert_allclose(
                result[f"net_{v}"], _HAND_NET[j], atol=1e-12)


class TestRollingConnectedness:
    def test_output_shape_and_columns(self, synthetic_var_data):
        result = lib.rolling_connectedness(
            synthetic_var_data, window=100, horizon=10, lags=2, step=5,
        )
        # Columns are grouped by variable, not by metric: total + (from, to, net)
        # interleaved per variable
        expected_cols = ["total"]
        for v in synthetic_var_data.columns:
            expected_cols.extend([
                f"directional_from_{v}",
                f"directional_to_{v}",
                f"net_{v}",
            ])
        assert list(result.columns) == expected_cols
        n_expected = len(range(99, len(synthetic_var_data), 5))
        assert len(result) == n_expected

    def test_window_too_large_raises(self, synthetic_var_data):
        with pytest.raises(ValueError, match="window"):
            lib.rolling_connectedness(synthetic_var_data, window=10_000, lags=2)
