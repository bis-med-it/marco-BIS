"""
Gibbs-sampling utilities for VAR-family Bayesian models.

These primitives are used by:

* :mod:`lib.bvar` — Minnesota BVAR (already implemented inline; this module
  re-exports the same logic as standalone functions).
* :mod:`lib.mfvar` — mixed-frequency VAR Gibbs sampler.
* :mod:`lib.tvp` — TVP-VAR, with and without stochastic volatility.

Public API:

* :func:`draw_niw_posterior` — single draw from Normal-Inverse-Wishart
  posterior (Sigma from IW, then B|Sigma from matrix-normal).
* :func:`draw_state_path` — single Durbin-Koopman simulation-smoother draw
  for state-space models (thin wrapper around
  :func:`lib.state_space.simulation_smoother`).

For full BVAR fitting with stability filtering and metadata, use
:func:`lib.bvar.fit_minnesota_bvar` instead.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .state_space import StateSpaceModel, simulation_smoother


def draw_niw_posterior(
    B_mean: np.ndarray,
    V_factor: np.ndarray,
    S_scale: np.ndarray,
    df: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single draw from a Normal-Inverse-Wishart posterior.

    ``Sigma ~ IW(S_scale, df)``,
    ``B | Sigma ~ N(vec(B_mean), kron(Sigma, V_factor))``

    so ``B`` has dimensions ``(k, n)`` where ``k`` is the number of
    regressors and ``n`` is the number of equations.

    Parameters
    ----------
    B_mean : (k, n) ndarray
        Posterior mean of the coefficient matrix.
    V_factor : (k, k) ndarray
        Posterior covariance factor — typically ``(X_aug' X_aug)^{-1}``.
    S_scale : (n, n) ndarray
        IW scale matrix (sum of squared residuals + prior).
    df : int
        IW degrees of freedom (``ν_0 + T``).
    seed : int, optional
        Used only if ``rng`` is ``None``.
    rng : numpy.random.Generator, optional
        Pre-allocated RNG (preferred for tight Gibbs loops).

    Returns
    -------
    (B, Sigma) : tuple
        ``B`` shape ``(k, n)``, ``Sigma`` shape ``(n, n)``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    Sigma = sp_stats.invwishart.rvs(df=df, scale=S_scale, random_state=rng)
    L_sig = np.linalg.cholesky(Sigma)
    L_V = np.linalg.cholesky(V_factor)
    k, n = B_mean.shape
    Z = rng.standard_normal((k, n))
    B = B_mean + L_V @ Z @ L_sig.T
    return B, Sigma


def draw_state_path(
    model: StateSpaceModel,
    y: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Single Durbin-Koopman simulation-smoother draw of the state path.

    Thin wrapper around :func:`lib.state_space.simulation_smoother` for the
    common Gibbs use case of "given current parameters, draw one state
    path consistent with the observed data".

    Parameters
    ----------
    model : StateSpaceModel
    y : (T, n_obs) ndarray
        Observations (NaN allowed).
    seed : int, optional

    Returns
    -------
    ndarray of shape ``(T, n_state)`` — a single state-path draw.
    """
    draws = simulation_smoother(model, y, n_draws=1, seed=seed)
    return draws[0]
