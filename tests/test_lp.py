"""Tests for local projections (lp.py).

Validation strategy:

1. **LP at h=0** with the same controls must equal the contemporaneous OLS
   coefficient on the shock — pure math.
2. **LP-IV first stage** must match a direct standalone OLS regression of
   the policy variable on the instrument plus controls.
3. **LP-IV scaling**: at h=0 with the policy variable as the response, the
   IRF must equal 1 (the structural shock IS the FFR shock, normalised).
4. **HAC bandwidth scales with horizon** via ``hac_lag_rule="horizon_plus_one"``.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import lib


class TestLP:
    def test_h0_equals_ols_with_same_controls(self,
                                                 synthetic_var_data, shock_series):
        """LP at h=0 must equal the OLS coefficient on the shock."""
        lags = 4
        irf = lib.lp_irf(
            data=synthetic_var_data, shock=shock_series,
            horizon=0, lags=lags, response_vars=["y1"], shock_name="z",
        )
        # Manual OLS at h=0
        controls, valid = lib.build_lp_controls(synthetic_var_data, lags)
        shock_v = shock_series[valid]
        X = add_constant(np.column_stack([shock_v, controls]))
        y = synthetic_var_data.values[valid, 0]
        manual_coef = OLS(y, X).fit().params[1]
        assert np.isclose(irf.median[0, 0, 0], manual_coef, atol=1e-10)

    def test_lp_irf_band_widths_consistent_with_se(self,
                                                      synthetic_var_data, shock_series):
        """68% band should be ~ 0.994σ wide each side; 90% should be ~ 1.645σ."""
        irf = lib.lp_irf(
            data=synthetic_var_data, shock=shock_series,
            horizon=5, lags=4, response_vars=["y1"],
            credible_intervals=(0.68, 0.90),
        )
        from scipy import stats as sp_stats
        z68 = sp_stats.norm.ppf(0.84)
        z90 = sp_stats.norm.ppf(0.95)
        # The ratio (upper_90 - upper_68) / (upper_68 - median) should be (z90 - z68) / z68
        med = irf.median[0, 0, 0]
        u68 = irf.upper_68[0, 0, 0]
        u90 = irf.upper_90[0, 0, 0]
        expected_ratio = (z90 - z68) / z68
        actual_ratio = (u90 - u68) / (u68 - med)
        assert np.isclose(actual_ratio, expected_ratio, atol=1e-6)

    def test_irf_container_shape_is_h_plus_1_vars_one_shock(self,
                                                                synthetic_var_data, shock_series):
        irf = lib.lp_irf(
            data=synthetic_var_data, shock=shock_series,
            horizon=12, lags=4,
        )
        assert irf.median.shape == (13, 3, 1)
        assert irf.shock_names == ["shock"]

    def test_response_var_subsetting_works(self,
                                              synthetic_var_data, shock_series):
        irf = lib.lp_irf(
            data=synthetic_var_data, shock=shock_series,
            horizon=5, lags=4,
            response_vars=["y2"],
        )
        assert irf.median.shape == (6, 1, 1)
        assert irf.variable_names == ["y2"]


class TestLPIV:
    def test_first_stage_matches_standalone_ols(self,
                                                   synthetic_var_data, shock_series):
        """The first-stage F and γ reported by lp_iv_irf must match a direct
        OLS of policy on instrument + controls."""
        # Treat "y1" as the policy variable and shock_series as the instrument
        lags = 4
        irf, fs = lib.lp_iv_irf(
            data=synthetic_var_data,
            instrument=shock_series,
            policy_var="y1",
            horizon=2,
            lags=lags,
        )
        controls, valid = lib.build_lp_controls(synthetic_var_data, lags)
        z_v = shock_series[valid]
        y1 = synthetic_var_data["y1"].values[valid]
        X = add_constant(np.column_stack([z_v, controls]))
        manual = OLS(y1, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        manual_F = float(manual.tvalues[1] ** 2)
        manual_gamma = float(manual.params[1])
        assert np.isclose(fs["gamma_hat"], manual_gamma, atol=1e-10)
        assert np.isclose(fs["F"], manual_F, atol=1e-8)

    def test_iv_response_of_policy_at_h0_is_one(self,
                                                   synthetic_var_data, shock_series):
        """At h=0, the LP-IV response of the policy variable to ITS OWN
        identified shock must equal 1 (the shock is normalised in terms of
        the policy variable's units)."""
        irf, fs = lib.lp_iv_irf(
            data=synthetic_var_data,
            instrument=shock_series,
            policy_var="y1",
            horizon=2, lags=4,
            response_vars=["y1"],
        )
        # IRF of y1 to y1-shock at h=0 should equal beta_RF_h0 / gamma
        # Now beta_RF_h0 is the same regression as the first stage but with
        # y1_{t+0} = y1_t as the LHS. So beta_RF_h0 == gamma, and the ratio is 1.
        assert np.isclose(irf.median[0, 0, 0], 1.0, atol=1e-10)

    def test_weak_instrument_flagged(self, synthetic_var_data):
        """A pure-noise instrument should have F well below 10."""
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(synthetic_var_data))
        irf, fs = lib.lp_iv_irf(
            data=synthetic_var_data,
            instrument=noise,
            policy_var="y1",
            horizon=2, lags=4,
        )
        assert fs["weak_instrument_F_lt_10"] is True
        assert fs["F"] < 10


# ---------------------------------------------------------------------------
# State-dependent LP (AG 2012 / RZ 2018)
# ---------------------------------------------------------------------------


def _sim_state_dependent_dgp(
    T: int = 400, b_E: float = 1.0, b_R: float = 2.0,
    state_p: float = 0.4, seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate (data, shock, state) where shock effect on y is b_E in
    state=0 and b_R in state=1.

    DGP: state_t ~ Bernoulli(state_p), shock_t ~ N(0, 1),
         y_t = 0.5 y_{t-1} + (b_E + (b_R - b_E) * state_t) * shock_t + eps_t.
    """
    rng = np.random.default_rng(seed)
    state = (rng.random(T) < state_p).astype(float)
    shock = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        eff = b_E + (b_R - b_E) * state[t]
        y[t] = 0.5 * y[t - 1] + eff * shock[t] + 0.5 * rng.standard_normal()
    idx = pd.period_range("1990-01", periods=T, freq="Q").to_timestamp()
    data = pd.DataFrame({"y": y}, index=idx)
    shock_s = pd.Series(shock, index=idx)
    state_s = pd.Series(state, index=idx)
    return data, shock_s, state_s


import pandas as pd     # imported here so the test fixtures above can use it


class TestStateDependentLP:
    def test_returns_irf_container_with_two_shocks(self):
        data, shock, state = _sim_state_dependent_dgp(T=200, seed=0)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=4, lags=2,
        )
        assert irf.median.shape == (5, 1, 2)
        assert irf.shock_names == ["shock_expansion", "shock_recession"]
        assert irf.variable_names == ["y"]

    def test_recovers_known_per_regime_coefficients(self):
        """h=0 expansion coefficient ≈ b_E, recession coefficient ≈ b_R."""
        data, shock, state = _sim_state_dependent_dgp(
            T=800, b_E=1.0, b_R=2.5, seed=7,
        )
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
        )
        b_E_hat = irf.median[0, 0, 0]
        b_R_hat = irf.median[0, 0, 1]
        assert abs(b_E_hat - 1.0) < 0.15, f"b_E_hat = {b_E_hat:.3f}"
        assert abs(b_R_hat - 2.5) < 0.20, f"b_R_hat = {b_R_hat:.3f}"

    def test_continuous_state_works(self):
        """Continuous state (smooth-transition weight) should also yield
        sensible per-regime coefficients."""
        rng = np.random.default_rng(0)
        T = 400
        z = rng.standard_normal(T)
        state = 1.0 / (1.0 + np.exp(-z))   # logistic, continuous
        shock = rng.standard_normal(T)
        b_E, b_R = 1.0, 2.0
        y = np.zeros(T)
        for t in range(1, T):
            eff = b_E + (b_R - b_E) * state[t]
            y[t] = 0.4 * y[t - 1] + eff * shock[t] + 0.4 * rng.standard_normal()
        idx = pd.period_range("1990-01", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame({"y": y}, index=idx)
        shock_s = pd.Series(shock, index=idx)
        state_s = pd.Series(state, index=idx)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock_s, state=state_s,
            horizon=2, lags=2,
        )
        # Recovered coefficients should bracket the DGP range
        b_E_hat = irf.median[0, 0, 0]
        b_R_hat = irf.median[0, 0, 1]
        assert b_E_hat < b_R_hat, (
            f"b_E_hat ({b_E_hat:.3f}) should be < b_R_hat ({b_R_hat:.3f})"
        )

    def test_bands_are_ordered(self):
        data, shock, state = _sim_state_dependent_dgp(T=400, seed=2)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=3, lags=2,
        )
        # Bands valid where median is finite
        mask = np.isfinite(irf.median)
        assert np.all(irf.lower_68[mask] <= irf.median[mask] + 1e-10)
        assert np.all(irf.median[mask] <= irf.upper_68[mask] + 1e-10)
        assert np.all(irf.lower_90[mask] <= irf.lower_68[mask] + 1e-10)
        assert np.all(irf.upper_68[mask] <= irf.upper_90[mask] + 1e-10)

    def test_custom_state_names_propagate(self):
        data, shock, state = _sim_state_dependent_dgp(T=200, seed=0)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
            state_names=("good_times", "bad_times"),
            shock_name="g",
        )
        assert irf.shock_names == ["g_good_times", "g_bad_times"]

    def test_metadata_identification_tag(self):
        data, shock, state = _sim_state_dependent_dgp(T=200, seed=0)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2,
        )
        assert irf.metadata["identification"] == "state_dependent_lp"
        assert "AG 2012" in irf.metadata["method"]
        assert "state_mean" in irf.metadata
        assert irf.metadata["interact_controls"] is True

    def test_interact_controls_false_changes_metadata(self):
        data, shock, state = _sim_state_dependent_dgp(T=200, seed=0)
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state,
            horizon=2, lags=2, interact_controls=False,
        )
        assert irf.metadata["interact_controls"] is False

    def test_constant_state_recovers_single_coefficient(self):
        """With state ≡ 0 everywhere, the expansion coefficient should equal
        the plain LP coefficient; the recession path is not identified and
        is returned as NaN with a warning."""
        rng = np.random.default_rng(0)
        T = 300
        shock = rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + 1.5 * shock[t] + 0.4 * rng.standard_normal()
        idx = pd.period_range("1990-01", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame({"y": y}, index=idx)
        shock_s = pd.Series(shock, index=idx)
        state_zero = pd.Series(np.zeros(T), index=idx)
        # With state always 0, the F * shock interaction is identically 0.
        # The documented contract: the expansion response is estimated
        # without the dead interaction column, the recession response is
        # left NaN, and a UserWarning reports the unidentified cells. No
        # exception is expected, so none is caught: a numerical result is
        # always subject to the coefficient check.
        with pytest.warns(UserWarning, match="not separately identified"):
            irf = lib.state_dependent_lp_irf(
                data=data, shock=shock_s, state=state_zero,
                horizon=2, lags=2,
            )
        assert abs(irf.median[0, 0, 0] - 1.5) < 0.20
        assert np.all(np.isfinite(irf.median[:, 0, 0]))
        assert np.all(np.isnan(irf.median[:, 0, 1]))
        assert irf.metadata["n_unidentified_cells"] == 3

    def test_mismatched_lengths_raise(self):
        data, _, state = _sim_state_dependent_dgp(T=200, seed=0)
        bad_shock = np.zeros(150)   # wrong length for array input
        # Either our length-check or pandas' Series-construction check
        # will fire; both indicate the misalignment to the caller.
        with pytest.raises(ValueError, match="length"):
            lib.state_dependent_lp_irf(
                data=data, shock=bad_shock, state=state,
                horizon=2, lags=2,
            )

    def test_response_vars_subset(self):
        rng = np.random.default_rng(0)
        T = 300
        state = (rng.random(T) < 0.4).astype(float)
        shock = rng.standard_normal(T)
        # Two response variables, both regime-dependent in different ways
        y1 = np.cumsum(shock + 0.5 * state)
        y2 = np.cumsum(0.5 * shock + state)
        idx = pd.period_range("1990-01", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame({"y1": y1, "y2": y2}, index=idx)
        irf = lib.state_dependent_lp_irf(
            data=data,
            shock=pd.Series(shock, index=idx),
            state=pd.Series(state, index=idx),
            horizon=2, lags=2,
            response_vars=["y2"],
        )
        assert irf.variable_names == ["y2"]
        assert irf.median.shape == (3, 1, 2)


class TestBLPIRF:
    """Tests for `lib.blp_irf` (Bayesian Local Projections, Ferreira,
    Miranda-Agrippino & Ricco 2025; 2021 working-paper design).

    Validation strategy:
    * shrinkage=0 → byte-identical to OLS LP
    * Impact (h=0) coefficient invariant to λ (prior mean = OLS LP at h=0)
    * Band shrinkage by 1/sqrt(1+λ) — algorithmic guarantee
    * Large λ → VAR-iterated IRF
    * Metadata tags reflect the BLP path
    * Validation errors
    """

    @staticmethod
    def _make_synthetic(seed=0, T=200, sparse=True):
        rng = np.random.default_rng(seed)
        A = np.array([[0.5, 0.1, 0.05], [0.0, 0.4, 0.0], [0.1, 0.0, 0.3]])
        y = np.zeros((T, 3))
        for t in range(1, T):
            y[t] = A @ y[t - 1] + rng.standard_normal(3) * 0.5
        data = pd.DataFrame(
            y, columns=["rgdp", "cpi", "ffr"],
            index=pd.date_range("1990-01-01", periods=T, freq="QS"),
        )
        if sparse:
            shock = np.zeros(T)
            shock[10] = 1.0
            shock[50] = -1.0
            shock += rng.standard_normal(T) * 0.1
        else:
            shock = rng.standard_normal(T)
        return data, shock

    def test_shrinkage_zero_matches_ols_lp(self):
        data, shock = self._make_synthetic()
        ols = lib.lp_irf(data, shock=shock, horizon=6, lags=2)
        blp = lib.blp_irf(data, shock=shock, horizon=6, lags=2, shrinkage=0.0)
        np.testing.assert_array_equal(blp.median, ols.median)
        np.testing.assert_array_equal(blp.lower_68, ols.lower_68)
        np.testing.assert_array_equal(blp.upper_68, ols.upper_68)

    def test_impact_coefficient_invariant_to_shrinkage(self):
        """h=0 coefficient = OLS LP at h=0 (= the impact loading used as
        the prior mean), so the convex combination at h=0 is unchanged."""
        data, shock = self._make_synthetic()
        blp1 = lib.blp_irf(data, shock=shock, horizon=4, lags=2, shrinkage=1.0)
        blp10 = lib.blp_irf(data, shock=shock, horizon=4, lags=2, shrinkage=10.0)
        blp100 = lib.blp_irf(data, shock=shock, horizon=4, lags=2, shrinkage=100.0)
        for response_idx in range(3):
            assert abs(blp1.median[0, response_idx, 0]
                       - blp10.median[0, response_idx, 0]) < 1e-10
            assert abs(blp1.median[0, response_idx, 0]
                       - blp100.median[0, response_idx, 0]) < 1e-10

    def test_band_shrinkage_factor(self):
        """Bands shrink by 1/sqrt(1+λ)."""
        data, shock = self._make_synthetic()
        ols = lib.lp_irf(data, shock=shock, horizon=6, lags=2)
        for lam in [1.0, 3.0, 9.0]:
            blp = lib.blp_irf(data, shock=shock, horizon=6, lags=2, shrinkage=lam)
            expected_ratio = 1.0 / np.sqrt(1.0 + lam)
            for h in range(1, 5):
                for j in range(3):
                    ols_hw = ols.upper_68[h, j, 0] - ols.median[h, j, 0]
                    blp_hw = blp.upper_68[h, j, 0] - blp.median[h, j, 0]
                    if abs(ols_hw) < 1e-10:
                        continue
                    actual_ratio = blp_hw / ols_hw
                    assert abs(actual_ratio - expected_ratio) < 1e-8

    def test_large_shrinkage_converges_to_var_irf(self):
        data, shock = self._make_synthetic(seed=2)
        var_fit = lib.fit_var(data, lags=2, trend="c")
        Phi = var_fit.ma_rep(6)
        from lib.lp import _blp_impact_loading
        loading = _blp_impact_loading(data, shock=shock, lags=2, extra_controls=None)
        var_irf = np.einsum("hjk,k->hj", Phi, loading)
        blp_big = lib.blp_irf(
            data, shock=shock, horizon=6, lags=2, shrinkage=1e6,
        )
        for h in range(7):
            for j in range(3):
                assert abs(blp_big.median[h, j, 0] - var_irf[h, j]) < 1e-3

    def test_metadata_tags_blp(self):
        data, shock = self._make_synthetic()
        blp = lib.blp_irf(data, shock=shock, horizon=4, lags=2, shrinkage=2.0)
        assert blp.metadata["method"] == "bayesian_local_projections"
        assert blp.metadata["identification"] == "bayesian_lp"
        assert blp.metadata["shrinkage"] == 2.0
        assert blp.metadata["var_lags"] == 2

    def test_negative_shrinkage_raises(self):
        data, shock = self._make_synthetic()
        with pytest.raises(ValueError, match="shrinkage"):
            lib.blp_irf(data, shock=shock, horizon=4, lags=2, shrinkage=-1.0)

    def test_response_vars_subset(self):
        data, shock = self._make_synthetic()
        blp = lib.blp_irf(
            data, shock=shock, horizon=4, lags=2, shrinkage=1.0,
            response_vars=["cpi"],
        )
        assert blp.variable_names == ["cpi"]
        assert blp.median.shape == (5, 1, 1)

    def test_var_lags_independent_of_lp_lags(self):
        data, shock = self._make_synthetic(T=300)
        blp = lib.blp_irf(
            data, shock=shock, horizon=4, lags=2, shrinkage=1.0, var_lags=4,
        )
        assert blp.metadata["n_lags"] == 2
        assert blp.metadata["var_lags"] == 4


class TestLagAugmentedLP:
    """Montiel Olea & Plagborg-Moller (2021) lag-augmented LP.

    One extra lag of everything (shock included in the lagged information
    set) + plain EHW (HC1) errors -- valid uniformly over persistence and
    long horizons, where the default NW(h+1)+normal bands under-cover.
    """

    @staticmethod
    def _persistent_dgp(T=400, seed=11):
        rng = np.random.default_rng(seed)
        shock = rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.95 * y[t - 1] + 0.5 * shock[t] + 0.3 * rng.standard_normal()
        data = pd.DataFrame({"y": y})
        return data, shock

    def test_runs_and_recovers_known_irf(self):
        data, shock = self._persistent_dgp()
        irf = lib.lp_irf(data, shock=shock, horizon=8, lags=2,
                         lag_augmented=True)
        # True IRF of y to a unit shock: 0.5 * 0.95^h
        true_h = 0.5 * 0.95 ** np.arange(9)
        est = irf.median[:, 0, 0]
        assert np.all(np.isfinite(est))
        assert np.max(np.abs(est - true_h)) < 0.25

    def test_metadata_and_se_regime(self):
        data, shock = self._persistent_dgp()
        aug = lib.lp_irf(data, shock=shock, horizon=6, lags=2,
                         lag_augmented=True)
        base = lib.lp_irf(data, shock=shock, horizon=6, lags=2)
        assert aug.metadata["lag_augmented"] is True
        assert "MOP 2021" in aug.metadata["cov_type"]
        assert aug.metadata["hac_lag_rule"] is None
        assert base.metadata["lag_augmented"] is False
        assert base.metadata["hac_lag_rule_effective"] == "horizon_plus_one"
        # Different control sets + different covariance estimators -> the
        # two runs must genuinely differ somewhere
        se_a = aug.upper_68 - aug.lower_68
        se_b = base.upper_68 - base.lower_68
        assert not np.allclose(se_a, se_b)

    def test_point_estimates_close_to_non_augmented_on_short_horizons(self):
        # The extra lag changes the control set slightly; on a well-specified
        # DGP the short-horizon point estimates should be close.
        data, shock = self._persistent_dgp()
        aug = lib.lp_irf(data, shock=shock, horizon=4, lags=2,
                         lag_augmented=True)
        base = lib.lp_irf(data, shock=shock, horizon=4, lags=2)
        assert np.max(np.abs(aug.median[:3, 0, 0] - base.median[:3, 0, 0])) < 0.1


class TestConstantControlGuard:
    """A constant-valued extra control must not displace the shock
    coefficient. statsmodels' ``add_constant`` defaults to
    ``has_constant='skip'``: when any regressor column is a non-zero
    constant it silently does NOT prepend the intercept, so ``params[1]``
    lands on a lagged control instead of the shock — a silent wrong IRF.
    lib passes ``has_constant='add'``; by Frisch-Waugh-Lovell the
    duplicated intercept leaves every other coefficient unchanged."""

    def _dgp(self, T=400, seed=7):
        import pandas as pd
        rng = np.random.default_rng(seed)
        shock = rng.standard_normal(T)
        y = np.zeros((T, 2))
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.array([0.4, 0.1]) * shock[t] \
                + 0.3 * rng.standard_normal(2)
        idx = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()
        return pd.DataFrame(y, columns=["y1", "y2"], index=idx), shock, idx

    def test_lp_irf_constant_extra_control_is_inert(self):
        import pandas as pd
        df, shock, idx = self._dgp()
        base = lib.lp_irf(data=df, shock=shock, response_vars=["y1"],
                          horizon=3, lags=2)
        cc = pd.DataFrame({"c5": np.full(len(df), 5.0)}, index=idx)
        guarded = lib.lp_irf(data=df, shock=shock, response_vars=["y1"],
                             horizon=3, lags=2, extra_controls=cc)
        np.testing.assert_allclose(
            guarded.median[:, 0, 0], base.median[:, 0, 0], atol=1e-8
        )


class TestStateLPRecessionSEClosedForm:
    """The recession coefficient AND its HAC standard error from the
    saturated interaction spec must equal a direct regression in the
    (1-F, F) parameterisation, where beta_R and se_R are read straight
    off the F*s column. Equality pins the full recession-SE formula
    Var(beta_E) + Var(beta_diff) + 2 Cov — dropping the covariance term
    gives a provably different number. (This is the closed-form check
    referenced by tests/test_conventions.py's state-LP band test.)"""

    def test_h0_recession_coef_and_se_match_direct_parameterisation(self):
        import pandas as pd
        from scipy import stats as sp_stats
        from lib.lp import build_lp_controls

        rng = np.random.default_rng(42)
        T = 500
        F = (rng.random(T) < 0.35).astype(float)
        s = rng.standard_normal(T)
        x = np.zeros(T)
        y = np.zeros(T)
        for t in range(1, T):
            eff = 1.0 * (1 - F[t]) + 2.0 * F[t]
            x[t] = 0.3 * x[t - 1] + rng.standard_normal()
            y[t] = 0.4 * y[t - 1] + 0.2 * x[t - 1] + eff * s[t] \
                + 0.5 * rng.standard_normal()
        idx = pd.period_range("1960Q1", periods=T, freq="Q").to_timestamp()
        data = pd.DataFrame({"y": y, "x": x}, index=idx)
        state = pd.Series(F, index=idx)

        irf = lib.state_dependent_lp_irf(
            data=data, shock=s, state=state, horizon=0, lags=2,
            response_vars=["y"],
        )
        z68 = sp_stats.norm.ppf(0.84)
        lib_bR = float(irf.median[0, 0, 1])
        lib_seR = float(irf.upper_68[0, 0, 1] - irf.median[0, 0, 1]) / z68

        # Direct (1-F, F) parameterisation on the identical sample
        Z, valid = build_lp_controls(data, 2)
        F_v, s_v = F[valid], s[valid]
        y_h = y[np.asarray(valid)]
        X_dir = np.column_stack([
            (1 - F_v), F_v, (1 - F_v) * s_v, F_v * s_v,
            (1 - F_v)[:, None] * Z, F_v[:, None] * Z,
        ])
        res = OLS(y_h, X_dir).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        np.testing.assert_allclose(lib_bR, res.params[3], atol=1e-9)
        np.testing.assert_allclose(lib_seR, res.bse[3], atol=1e-9)


class TestLPIVIdentificationGuard:
    """Acceptance checks: an instrument with no residual variation
    after projecting out intercept + controls must be rejected, and the
    check must be scale-invariant."""

    def test_constant_instrument_rejected(self, synthetic_var_data):
        z_const = np.ones(len(synthetic_var_data))
        with pytest.raises(RuntimeError, match="no variation"):
            lib.lp_iv_irf(
                data=synthetic_var_data, instrument=z_const,
                policy_var="y1", horizon=2, lags=2,
            )

    def test_instrument_spanned_by_controls_rejected(self,
                                                     synthetic_var_data,
                                                     shock_series):
        # Supply an exact copy of the instrument as an extra control: the
        # residualised instrument variation is zero.
        cc = pd.DataFrame({"copy_of_z": shock_series},
                          index=synthetic_var_data.index)
        with pytest.raises(RuntimeError, match="no variation"):
            lib.lp_iv_irf(
                data=synthetic_var_data, instrument=shock_series,
                policy_var="y1", horizon=2, lags=2, extra_controls=cc,
            )

    def test_rescaled_valid_instrument_still_passes(self,
                                                    synthetic_var_data,
                                                    shock_series):
        irf_base, fs_base = lib.lp_iv_irf(
            data=synthetic_var_data, instrument=shock_series,
            policy_var="y1", horizon=2, lags=2,
        )
        irf_scaled, fs_scaled = lib.lp_iv_irf(
            data=synthetic_var_data, instrument=shock_series * 1e8,
            policy_var="y1", horizon=2, lags=2,
        )
        # Identification unchanged: responses in policy units are identical
        assert np.allclose(irf_base.median, irf_scaled.median, rtol=1e-8)


class TestStateLPIdentificationGuards:
    """Acceptance check: unidentified regimes get NaN not pseudo-precision;
    time order comes from the calendar, never Series storage order."""

    def test_one_regime_shock_yields_nan_not_tight_band(self):
        rng = np.random.default_rng(3)
        T = 300
        idx = pd.date_range("1960-01-01", periods=T, freq="QS")
        state = (rng.uniform(size=T) < 0.3).astype(float)   # 1 = recession
        shock = rng.standard_normal(T) * state               # shocks ONLY in recession
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + 3.0 * shock[t] + rng.standard_normal() * 0.1
        data = pd.DataFrame({"y": y}, index=idx)
        # Every cell shares the same degenerate design, so the whole
        # exercise is unidentified and the function must refuse outright —
        # no cell may be published with a pseudo-precise band.
        with pytest.raises(RuntimeError, match="only one regime"):
            with pytest.warns(UserWarning, match="not separately identified"):
                lib.state_dependent_lp_irf(
                    data=data, shock=pd.Series(shock, index=idx),
                    state=pd.Series(state, index=idx), horizon=2, lags=2,
                )

    def test_two_regime_variation_unaffected_by_guard(self):
        data, shock, state = _sim_state_dependent_dgp(
            T=800, b_E=1.0, b_R=2.5, seed=7,
        )
        irf = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
        )
        assert irf.metadata["n_unidentified_cells"] == 0
        assert abs(irf.median[0, 0, 0] - 1.0) < 0.15

    def test_reversed_shock_series_storage_is_harmless(self):
        data, shock, state = _sim_state_dependent_dgp(
            T=400, b_E=1.0, b_R=2.5, seed=11,
        )
        base = lib.state_dependent_lp_irf(
            data=data, shock=shock, state=state, horizon=2, lags=2,
        )
        rev = lib.state_dependent_lp_irf(
            data=data, shock=shock.iloc[::-1], state=state, horizon=2, lags=2,
        )
        assert np.allclose(base.median, rev.median, equal_nan=True)


class TestStateLPZeroShockGuard:
    """Fix acceptance: an all-zero shock has no identifying variation in
    EITHER regime — the expansion-only fallback must not publish a
    spuriously precise zero response; the run must refuse outright."""

    def test_all_zero_shock_refused(self):
        rng = np.random.default_rng(5)
        T = 300
        idx = pd.date_range("1960-01-01", periods=T, freq="QS")
        state = (rng.uniform(size=T) < 0.4).astype(float)
        shock = np.zeros(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + rng.standard_normal() * 0.1
        data = pd.DataFrame({"y": y}, index=idx)
        with pytest.raises(RuntimeError, match="cell is identified"):
            with pytest.warns(UserWarning, match="not separately identified"):
                lib.state_dependent_lp_irf(
                    data=data, shock=pd.Series(shock, index=idx),
                    state=pd.Series(state, index=idx), horizon=2, lags=2,
                )
