"""Tests for stochastic_volatility.py (Kim-Shephard-Chib sampling).

Validation strategy:

1. **KSC table consistency** — probabilities sum to 1.
2. **Mixture indicators in valid range** — returned indicators are in [0, 6].
3. **Output shape** — draw_log_h_path returns (T,) array.
4. **DGP recovery** — simulate from a known SV DGP; the recovered h
   path should track the true h path (correlation > 0.5 in expectation).
"""

from __future__ import annotations

import numpy as np
import pytest

import lib


class TestKSCTable:
    def test_probs_sum_to_one(self):
        from lib.stochastic_volatility import KSC_PROBS
        assert np.isclose(KSC_PROBS.sum(), 1.0, atol=1e-4)

    def test_seven_components(self):
        from lib.stochastic_volatility import KSC_PROBS, KSC_MEANS, KSC_VARS
        assert len(KSC_PROBS) == 7
        assert len(KSC_MEANS) == 7
        assert len(KSC_VARS) == 7

    def test_variances_positive(self):
        from lib.stochastic_volatility import KSC_VARS
        assert np.all(KSC_VARS > 0)


class TestSampleMixtureIndicators:
    def test_indicators_in_valid_range(self):
        rng = np.random.default_rng(0)
        T = 100
        y_star = rng.standard_normal(T)
        h = np.zeros(T)
        s = lib.sample_mixture_indicators(y_star, h, rng)
        assert s.shape == (T,)
        assert s.min() >= 0
        assert s.max() <= 6


class TestDrawLogHPath:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        T = 50
        y = rng.standard_normal(T)
        h = lib.draw_log_h_path(y, sigma_h=0.2, seed=0)
        assert h.shape == (T,)
        assert np.all(np.isfinite(h))

    def test_dgp_recovery_correlated_with_truth(self):
        """Generate from a known SV DGP and check the recovered h is
        correlated with the truth. With 200 obs, we expect a positive
        correlation."""
        rng = np.random.default_rng(42)
        T = 200
        sigma_h_true = 0.3
        h_true = np.zeros(T)
        for t in range(1, T):
            h_true[t] = h_true[t - 1] + sigma_h_true * rng.standard_normal()
        eps = rng.standard_normal(T)
        y = np.exp(h_true / 2) * eps

        # Iterate a few times to let h converge
        h_est = np.zeros(T)
        for _ in range(5):
            h_est = lib.draw_log_h_path(
                y, sigma_h=sigma_h_true,
                h_current=h_est, seed=int(rng.integers(0, 2**31 - 1)),
            )
        corr = float(np.corrcoef(h_est, h_true)[0, 1])
        assert corr > 0.4, f"correlation between estimated and true h = {corr:.3f}"

    def test_sigma_h_small_means_smooth_h(self):
        """If σ_h is small, the sampled h should be smoother (smaller diff)
        than if σ_h is large."""
        rng = np.random.default_rng(0)
        T = 200
        eps = rng.standard_normal(T)
        h_true = 0.5 * np.sin(np.arange(T) / 20)
        y = np.exp(h_true / 2) * eps

        h_small = lib.draw_log_h_path(y, sigma_h=0.05, seed=0)
        h_large = lib.draw_log_h_path(y, sigma_h=0.5, seed=0)
        # h_small should have smaller first-differences on average
        assert np.std(np.diff(h_small)) < np.std(np.diff(h_large))


class TestLevelRecovery:
    """Level-recovery contracts on known-variance DGPs.

    The tabulated KSC means are centered, so the measurement intercept
    must be ``m_s + c_KSC`` (= m_s - 1.2704). A sign error in c_KSC
    would shift every log-volatility path by 2*1.2704 ~ 2.54 (variances
    ~13x off) while passing every correlation-based test -- hence these
    level assertions.
    """

    @staticmethod
    def _posterior_mean_h(y, n_iter=80, warm=30):
        from lib.stochastic_volatility import draw_log_h_path
        h, means = None, []
        for it in range(n_iter):
            h = draw_log_h_path(y, sigma_h=0.05, h_current=h, seed=1000 + it)
            if it >= warm:
                means.append(h.mean())
        return float(np.mean(means))

    def test_unit_variance_recovers_h_zero(self):
        rng = np.random.default_rng(11)
        y = rng.standard_normal(600)               # true h = 0
        h_bar = self._posterior_mean_h(y)
        assert abs(h_bar) < 0.5, (
            f"unit-variance data should give mean h ~ 0; got {h_bar:+.3f} "
            f"(a c_KSC sign error gives ~ -2.54)"
        )

    def test_known_level_recovered(self):
        rng = np.random.default_rng(11)
        y = np.exp(1.5 / 2) * rng.standard_normal(600)   # true h = 1.5
        h_bar = self._posterior_mean_h(y)
        assert abs(h_bar - 1.5) < 0.5, f"true h=1.5; got {h_bar:+.3f}"
