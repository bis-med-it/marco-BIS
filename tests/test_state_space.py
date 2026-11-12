"""Tests for state_space.py (Kalman wrapper + helpers).

What this file checks, on an AR(1) model in state-space form:

1. **Model construction**: dimension inference, shape validation and
   default zero intercepts.
2. **Output shapes** of the filter and smoother.
3. **Internal consistency of the likelihood**: ``ss_log_likelihood``
   returns the same value as ``kalman_filter_smooth().log_likelihood``.
   Both are library entry points, so this does not validate the Kalman
   likelihood itself against an independent reference.
4. **Smoother dominates filter at interior points**: smoothed states
   incorporate future information, so their covariance is <= the
   filtered state covariance.
5. **Missing data**: NaN observations give finite smoothed states with
   a larger smoothed covariance inside the gap.
6. **Simulation smoother centred**: averaging many simulation-smoother
   draws approaches the analytical smoothed mean.
7. **Mariano-Murasawa and flow weights**: exact values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


def _ar1_state_space(phi: float, sigma_eta: float, sigma_eps: float
                       ) -> lib.StateSpaceModel:
    """Build an AR(1) in state-space form.

    alpha_{t+1} = phi alpha_t + eta_t,  eta_t ~ N(0, sigma_eta^2)
    y_t = alpha_t + epsilon_t,           epsilon_t ~ N(0, sigma_eps^2)
    """
    return lib.StateSpaceModel(
        Z=np.array([[1.0]]),
        H=np.array([[sigma_eps ** 2]]),
        T=np.array([[phi]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma_eta ** 2]]),
        initial_state_mean=np.array([0.0]),
        initial_state_cov=np.array([[1e6]]),
    )


@pytest.fixture(scope="module")
def ar1_data() -> np.ndarray:
    """Simulate 200 observations from an AR(1) with phi=0.7, sigma_eta=1,
    plus measurement noise sigma_eps=0.5."""
    rng = np.random.default_rng(42)
    T = 200
    phi = 0.7
    alpha = np.zeros(T)
    for t in range(1, T):
        alpha[t] = phi * alpha[t - 1] + rng.standard_normal()
    eps = rng.standard_normal(T) * 0.5
    return (alpha + eps).reshape(-1, 1)


class TestStateSpaceModel:
    def test_dimension_inference(self):
        model = _ar1_state_space(0.5, 1.0, 0.5)
        assert model.n_state == 1
        assert model.n_obs == 1
        assert model.n_shock == 1

    def test_validates_shapes(self):
        with pytest.raises(ValueError, match="H must be"):
            lib.StateSpaceModel(
                Z=np.eye(1), H=np.eye(2),   # wrong
                T=np.eye(1), R=np.eye(1), Q=np.eye(1),
            )
        with pytest.raises(ValueError, match="T must be"):
            lib.StateSpaceModel(
                Z=np.eye(2), H=np.eye(2),
                T=np.eye(3),   # wrong
                R=np.eye(2), Q=np.eye(2),
            )

    def test_default_intercepts_are_zero(self):
        model = _ar1_state_space(0.5, 1.0, 0.5)
        assert np.allclose(model.state_intercept, 0.0)
        assert np.allclose(model.obs_intercept, 0.0)


class TestKalmanFilterSmooth:
    def test_output_shapes(self, ar1_data):
        model = _ar1_state_space(0.7, 1.0, 0.5)
        res = lib.kalman_filter_smooth(model, ar1_data)
        T = len(ar1_data)
        assert res.filtered_state.shape == (T, 1)
        assert res.filtered_cov.shape == (T, 1, 1)
        assert res.smoothed_state.shape == (T, 1)
        assert res.smoothed_cov.shape == (T, 1, 1)

    def test_smoother_covariance_le_filter(self, ar1_data):
        """Smoothing uses future info, so Cov(α_t | y_{1:T}) ≤ Cov(α_t | y_{1:t})
        for every interior t."""
        model = _ar1_state_space(0.7, 1.0, 0.5)
        res = lib.kalman_filter_smooth(model, ar1_data)
        T = len(ar1_data)
        # At t = T-1 the smoother == filter; at interior points smoother < filter
        for t in range(1, T - 1):
            assert res.smoothed_cov[t, 0, 0] <= res.filtered_cov[t, 0, 0] + 1e-10

    def test_log_likelihood_matches_filter(self, ar1_data):
        """log_likelihood() returns the same value as kalman_filter_smooth().log_likelihood."""
        model = _ar1_state_space(0.7, 1.0, 0.5)
        res = lib.kalman_filter_smooth(model, ar1_data)
        ll_only = lib.ss_log_likelihood(model, ar1_data)
        assert np.isclose(res.log_likelihood, ll_only, atol=1e-8)

    def test_missing_observation_handled(self, ar1_data):
        """Inserting NaN at a few timepoints should not crash and should
        produce smoother estimates that interpolate."""
        y_with_nan = ar1_data.copy()
        y_with_nan[50:55, 0] = np.nan
        model = _ar1_state_space(0.7, 1.0, 0.5)
        res = lib.kalman_filter_smooth(model, y_with_nan)
        # Smoothed states at the NaN gap should be finite
        assert np.all(np.isfinite(res.smoothed_state[50:55, 0]))
        # And the smoothed cov should be LARGER at the NaN gap (less info)
        avg_cov_around = (
            res.smoothed_cov[45:50, 0, 0].mean()
            + res.smoothed_cov[55:60, 0, 0].mean()
        ) / 2
        avg_cov_at_nan = res.smoothed_cov[50:55, 0, 0].mean()
        assert avg_cov_at_nan > avg_cov_around


class TestSimulationSmoother:
    def test_output_shape(self, ar1_data):
        model = _ar1_state_space(0.7, 1.0, 0.5)
        draws = lib.simulation_smoother(model, ar1_data, n_draws=50, seed=0)
        T = len(ar1_data)
        assert draws.shape == (50, T, 1)

    def test_draws_centred_on_smoothed_mean(self, ar1_data):
        """Averaging many draws should approach the analytical smoothed
        state. This is the defining property of the Durbin-Koopman
        simulation smoother."""
        model = _ar1_state_space(0.7, 1.0, 0.5)
        res = lib.kalman_filter_smooth(model, ar1_data)
        draws = lib.simulation_smoother(model, ar1_data, n_draws=500, seed=0)
        avg = draws.mean(axis=0)   # (T, n_state)
        # Tolerance: ~3 SDs of the empirical mean
        smoothed_sd = np.sqrt(res.smoothed_cov[:, 0, 0])
        tol = 5 * smoothed_sd / np.sqrt(500)   # 5σ around true smoother mean
        diff = np.abs(avg[:, 0] - res.smoothed_state[:, 0])
        assert np.all(diff < tol + 0.1), (
            f"Max diff = {diff.max():.4f}, expected within tolerance"
        )


class TestMarianoMurasawaWeights:
    def test_exact_values(self):
        w = lib.mariano_murasawa_weights()
        expected = np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])
        assert np.allclose(w, expected, atol=1e-15)

    def test_simple_flow_weights_sum_to_one(self):
        w = lib.simple_flow_weights(periods_per_obs=3)
        assert len(w) == 3
        assert np.isclose(w.sum(), 1.0)

    def test_simple_flow_arbitrary_k(self):
        for k in [2, 4, 12]:
            w = lib.simple_flow_weights(periods_per_obs=k)
            assert len(w) == k
            assert np.allclose(w, 1.0 / k)
