"""
Stochastic volatility sampling via Kim-Shephard-Chib (1998) mixture algorithm.

Univariate SV model:
    y_t = exp(h_t / 2) * ε_t,    ε_t ~ N(0, 1)
    h_t = h_{t-1} + σ_h * ν_t,   ν_t ~ N(0, 1)

Posterior sampling of ``h_{1:T}`` proceeds in two steps per Gibbs iter:

1. **Mixture indicators**. Transform ``y*_t = log(y_t² + offset)``. By
   construction ``log ε_t² ~ log χ²(1)``. KSC (1998) approximate this
   density by a 7-component mixture of normals with the table at the
   top of this module. Given the current ``h`` path, sample the mixture
   indicator ``s_t`` for each ``t`` from its categorical posterior.

2. **State path**. Given mixture indicators, the model is linear-Gaussian:
       y*_t = h_t + (m_{s_t} + c_KSC) + e_t,   e_t ~ N(0, v²_{s_t})
       h_t = h_{t-1} + σ_h ν_t
   where ``c_KSC = E[log χ²(1)] ≈ -1.2704``. The tabulated means ``m_j``
   are *centered* (their probability-weighted mean is 0 — they
   approximate the shifted variable ``log χ²(1) - c_KSC``), so the
   measurement must *add* ``c_KSC`` back to restore the correct mean
   ``E[log ε²] = c_KSC`` (cf. Koop-Korobilis code: ``m_j - 1.2704``).
   Sample ``h_{1:T}`` via the Durbin-Koopman simulation smoother on
   this time-varying-H, time-varying-obs_intercept state-space.

Used by ``lib.tvp.fit_tvp_var_sv`` / ``fit_tvp_var_sv_primiceri`` and
Cogley-Sargent-style univariate SV applications.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# KSC (1998) Table 4 — 7-component mixture of normals
# approximating the log chi-squared(1) density
# ---------------------------------------------------------------------------

KSC_PROBS = np.array([
    0.00730, 0.10556, 0.00002, 0.04395,
    0.34001, 0.24566, 0.25750,
])
KSC_MEANS = np.array([
    -10.12999, -3.97281, -8.56686, 2.77786,
    0.61942, 1.79518, -1.08819,
])
KSC_VARS = np.array([
    5.79596, 2.61369, 5.17950, 0.16735,
    0.64009, 0.34023, 1.26261,
])
# E[log chi^2(1)] (constant offset to align the mixture mean with log ε²)
KSC_C = -1.2704


def sample_mixture_indicators(
    y_star: np.ndarray, h: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Sample mixture indicators ``s_t ∈ {0,...,6}`` from the full conditional.

    Per KSC (1998), given ``y*_t = h_t + (m_{s_t} + c_KSC) + e_t``,
    ``e_t ~ N(0, v²_{s_t})``, the posterior of ``s_t`` is
    proportional to ``π_i × N(y*_t; h_t + m_i + c_KSC, v²_i)``
    (the tabulated ``m_i`` are centered; ``c_KSC < 0`` restores the
    ``log χ²(1)`` mean).

    Returns
    -------
    ndarray of int, shape (T,)
        Sampled mixture indicators (zero-indexed).
    """
    T = len(y_star)
    n_components = len(KSC_PROBS)
    # log-density for each component, vectorised
    # diff[t, i] = y*_t - h_t - (m_i + c_KSC)
    diff = y_star[:, None] - h[:, None] - (KSC_MEANS + KSC_C)[None, :]
    log_lik = (
        np.log(KSC_PROBS)[None, :]
        - 0.5 * np.log(2 * np.pi * KSC_VARS)[None, :]
        - 0.5 * diff ** 2 / KSC_VARS[None, :]
    )
    # Subtract max for numerical stability before normalising
    log_lik -= log_lik.max(axis=1, keepdims=True)
    probs = np.exp(log_lik)
    probs /= probs.sum(axis=1, keepdims=True)
    # Sample categorical row by row
    cum = probs.cumsum(axis=1)
    u = rng.uniform(size=T)
    s = (u[:, None] >= cum).sum(axis=1)
    s = np.clip(s, 0, n_components - 1)
    return s


def draw_log_h_path(
    y: np.ndarray,
    sigma_h: float,
    h_init_mean: float = 0.0,
    h_init_cov: float = 10.0,
    h_current: np.ndarray | None = None,
    offset: float = 1e-6,
    seed: int | None = None,
) -> np.ndarray:
    """Sample the log-volatility path ``h_{1:T}`` via KSC (1998).

    Single Gibbs step. ``y`` is the residual series to which the SV
    model applies; we compute ``y*_t = log(y_t² + offset)`` internally.

    Parameters
    ----------
    y : (T,) ndarray
        Residual series. Treated as ``exp(h_t/2) * ε_t`` where ``ε_t ~ N(0,1)``.
    sigma_h : float
        Standard deviation of log-h innovations (assumed known for this call).
    h_init_mean, h_init_cov : float
        Prior on the first-period log-volatility ``h_1``.
    h_current : (T,) ndarray, optional
        Current estimate of ``h_{1:T}`` (used for the mixture-indicator
        conditional). Defaults to a zero vector.
    offset : float
        Small constant added to ``y_t²`` to avoid ``log(0)``. It is not
        scale-free: the default 1e-6 suits residuals of ordinary size
        (data in percent). With residuals of standard deviation near
        0.001 or below the constant dominates the small squared
        residuals and the estimated volatility path is flattened, so
        rescale the data or lower the offset. Primiceri (2005) uses
        0.001.
    seed : int, optional

    Returns
    -------
    (T,) ndarray
        Sampled ``log h_{1:T}`` path (i.e., ``h_t`` values themselves;
        note the variable is already log-variance, not log of variance).
    """
    from .state_space import StateSpaceModel, simulation_smoother

    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y).ravel().astype(float)
    T = len(y_arr)
    y_star = np.log(y_arr ** 2 + offset)

    if h_current is None:
        h_current = np.zeros(T)

    # Step 1: sample mixture indicators
    s = sample_mixture_indicators(y_star, h_current, rng)

    # Step 2: build time-varying state-space and sample h via simulation smoother
    obs_intercept = (KSC_MEANS[s] + KSC_C)[:, None]      # (T, 1)
    H_tv = KSC_VARS[s].reshape(T, 1, 1)                   # (T, 1, 1)

    model = StateSpaceModel(
        Z=np.array([[1.0]]),                              # time-invariant Z
        H=H_tv,
        T=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma_h ** 2]]),
        obs_intercept=obs_intercept,
        initial_state_mean=np.array([h_init_mean]),
        initial_state_cov=np.array([[h_init_cov]]),
    )
    draws = simulation_smoother(
        model, y_star.reshape(-1, 1), n_draws=1,
        seed=rng.integers(0, 2**31 - 1),
    )
    return draws[0, :, 0]
