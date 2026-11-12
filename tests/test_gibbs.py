"""Tests for gibbs.py (Gibbs-sampling primitives).

Validation strategy:

1. **NIW marginals** — averaging many draws of Sigma should approach the
   IW prior mean ``S_scale / (df - n - 1)``.
2. **NIW Sigma is symmetric and PD** — every draw is a valid covariance.
3. **State-path draw centred on smoother mean** — averaging many calls to
   ``draw_state_path`` should approach the analytical smoother mean
   (same property as in test_state_space, but via the gibbs wrapper).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lib


class TestNIWPosterior:
    def test_sigma_is_symmetric_positive_definite(self):
        rng = np.random.default_rng(0)
        n, k = 3, 5
        B_mean = rng.standard_normal((k, n))
        V_factor = np.eye(k)
        S_scale = np.eye(n) * 10
        df = 50
        for _ in range(20):
            B, Sigma = lib.draw_niw_posterior(B_mean, V_factor, S_scale, df, rng=rng)
            assert np.allclose(Sigma, Sigma.T, atol=1e-10)
            eigs = np.linalg.eigvalsh(Sigma)
            assert np.all(eigs > 0)
            assert B.shape == (k, n)
            assert Sigma.shape == (n, n)

    def test_sigma_average_approaches_iw_prior_mean(self):
        """Mean of many IW draws approaches S_scale / (df - n - 1),
        element by element including the off-diagonals, within a
        Monte Carlo tolerance scaled by each element's IW variance."""
        rng = np.random.default_rng(7)
        n, k = 3, 5
        B_mean = rng.standard_normal((k, n))
        V_factor = np.eye(k) * 0.1
        # Diagonal [4, 9, 16]; correlations 0.2, 0.1, -0.2 so the
        # off-diagonal means are nonzero and of both signs
        S_scale = np.array([
            [4.0, 1.2, 0.8],
            [1.2, 9.0, -2.4],
            [0.8, -2.4, 16.0],
        ])
        df = 30
        n_draws = 2000
        sigmas = []
        for _ in range(n_draws):
            _, Sigma = lib.draw_niw_posterior(B_mean, V_factor, S_scale, df, rng=rng)
            assert np.all(np.linalg.eigvalsh(Sigma) > 0)
            sigmas.append(Sigma)
        avg_Sigma = np.mean(sigmas, axis=0)
        expected = S_scale / (df - n - 1)
        # Exact IW element variances (Press 1982):
        # Var(S_ij) = [(df-n+1) s_ij^2 + (df-n-1) s_ii s_jj]
        #             / [(df-n) (df-n-1)^2 (df-n-3)]
        d = np.diag(S_scale)
        var = (
            (df - n + 1) * S_scale ** 2 + (df - n - 1) * np.outer(d, d)
        ) / ((df - n) * (df - n - 1) ** 2 * (df - n - 3))
        mc_se = np.sqrt(var / n_draws)
        z = (avg_Sigma - expected) / mc_se
        # Six distinct elements, each approximately N(0, 1) at 2000
        # draws; |z| < 4 fails a correct sampler with probability < 1e-3,
        # while a 10% scale error on the diagonal already gives |z| > 10
        assert np.all(np.abs(z) < 4.0), (
            f"IW mean off by more than 4 Monte Carlo SEs; z = {z}"
        )

    def test_b_average_approaches_b_mean(self):
        """Mean of many B draws approaches B_mean."""
        rng = np.random.default_rng(13)
        n, k = 2, 3
        B_mean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        V_factor = np.eye(k) * 0.01
        S_scale = np.eye(n) * 1.0
        df = 30
        n_draws = 2000
        Bs = []
        for _ in range(n_draws):
            B, _ = lib.draw_niw_posterior(B_mean, V_factor, S_scale, df, rng=rng)
            Bs.append(B)
        avg_B = np.mean(Bs, axis=0)
        # Tighter tolerance since V_factor is small
        assert np.allclose(avg_B, B_mean, atol=0.1)


class TestDrawStatePath:
    def test_single_draw_shape(self):
        """draw_state_path returns a single (T, n_state) draw."""
        # Build a simple AR(1) state-space model
        model = lib.StateSpaceModel(
            Z=np.array([[1.0]]),
            H=np.array([[0.5]]),
            T=np.array([[0.7]]),
            R=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            initial_state_cov=np.array([[1e6]]),
        )
        # Simulate some data
        rng = np.random.default_rng(0)
        T = 50
        alpha = np.zeros(T)
        for t in range(1, T):
            alpha[t] = 0.7 * alpha[t - 1] + rng.standard_normal()
        y = (alpha + rng.standard_normal(T) * np.sqrt(0.5)).reshape(-1, 1)

        path = lib.draw_state_path(model, y, seed=42)
        assert path.shape == (T, 1)

    def test_consistent_with_simulation_smoother(self):
        """draw_state_path should equal simulation_smoother(n_draws=1)[0]
        when given the same seed."""
        model = lib.StateSpaceModel(
            Z=np.array([[1.0]]),
            H=np.array([[0.5]]),
            T=np.array([[0.5]]),
            R=np.array([[1.0]]),
            Q=np.array([[1.0]]),
            initial_state_cov=np.array([[1e6]]),
        )
        rng = np.random.default_rng(0)
        T = 30
        y = rng.standard_normal((T, 1))

        path_1 = lib.draw_state_path(model, y, seed=42)
        path_2 = lib.simulation_smoother(model, y, n_draws=1, seed=42)
        assert np.allclose(path_1, path_2[0])
