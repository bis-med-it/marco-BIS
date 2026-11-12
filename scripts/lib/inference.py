"""
Inference utilities for VAR-based IRFs.

Two band-construction strategies:

* :func:`asymptotic_mc_draws` — Monte Carlo from the asymptotic distribution
  of the VAR coefficient matrix ``B``. Robust for I(1) systems where the
  residual bootstrap is known to give explosive paths.

* :func:`residual_bootstrap_draws` — classic IID residual bootstrap. Suitable
  for stationary VARs.

Both return arrays of shape ``(n_draws, H+1, n_vars, n_shocks)`` which can be
fed directly into :func:`IRFContainer.from_draws`.

Also exposes :func:`draw_parameter_samples` for routines (e.g., sign
restrictions) that need joint draws of ``(B, Sigma)`` without going all the
way to IRFs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .var import VARFit, extract_lag_matrices, ma_representation, _ensure_psd, _build_x_row


def draw_parameter_samples(
    fit: VARFit, n_draws: int, seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Joint draws of ``(B, Sigma)`` from the asymptotic distribution.

    For each draw:

    * ``vec(B) ~ N(vec(B_hat), Sigma_u ⊗ (X'X)^{-1})`` (equation-major
      stacking, Hamilton 1994 eq. 11.1.35 — see ``VARFit.vec_cov``)
    * ``Sigma`` held at ``Sigma_u`` (asymptotic covariance, classical).

    For a fully Bayesian normal-inverse-Wishart sampling, see ``bvar.py``.
    """
    rng = np.random.default_rng(seed)
    n = fit.n_vars
    k = fit.k
    XtX_inv = np.linalg.inv(fit.X_act.T @ fit.X_act)
    cov_vecB = np.kron(fit.Sigma_u, XtX_inv)
    cov_vecB = _ensure_psd(cov_vecB)
    L = np.linalg.cholesky(cov_vecB)
    vecB_hat = fit.B_hat.T.flatten()

    draws: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_draws):
        z = rng.standard_normal(len(vecB_hat))
        vecB_draw = vecB_hat + L @ z
        B_draw = vecB_draw.reshape(n, k).T   # (k, n)
        draws.append((B_draw, fit.Sigma_u))
    return draws


def asymptotic_mc_draws(
    fit: VARFit,
    horizon: int,
    n_draws: int,
    seed: int | None = None,
    structural_impact: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Monte Carlo IRF draws from the asymptotic distribution of ``B``.

    Parameters
    ----------
    fit : VARFit
    horizon : int
    n_draws : int
    seed : int, optional
    structural_impact : callable, optional
        ``f(Sigma) -> impact_matrix``. Defaults to Cholesky. The columns of
        the impact matrix are the structural impact vectors.

    Returns
    -------
    ndarray
        Shape ``(n_draws, H+1, n_vars, n_shocks)``.
    """
    if structural_impact is None:
        structural_impact = np.linalg.cholesky

    rng = np.random.default_rng(seed)
    n = fit.n_vars
    k = fit.k

    XtX_inv = np.linalg.inv(fit.X_act.T @ fit.X_act)
    cov_vecB = np.kron(fit.Sigma_u, XtX_inv)
    cov_vecB = _ensure_psd(cov_vecB)
    L = np.linalg.cholesky(cov_vecB)
    vecB_hat = fit.B_hat.T.flatten()

    P = structural_impact(fit.Sigma_u)
    n_shocks = P.shape[1]

    out = np.zeros((n_draws, horizon + 1, n, n_shocks))
    for b in range(n_draws):
        z = rng.standard_normal(len(vecB_hat))
        vecB = vecB_hat + L @ z
        B_d = vecB.reshape(n, k).T
        lag_mats = extract_lag_matrices(B_d, n, fit.p)
        Phi = ma_representation(lag_mats, horizon)
        out[b] = np.einsum("hij,jk->hik", Phi, P)
    return out


def residual_bootstrap_draws(
    fit: VARFit,
    horizon: int,
    n_draws: int,
    seed: int | None = None,
    structural_impact: Callable[[np.ndarray], np.ndarray] | None = None,
    require_stable: bool = True,
) -> np.ndarray:
    """Classic IID residual bootstrap for IRF bands.

    For each draw:

    1. Sample with replacement from the residuals, RESCALED by
       ``sqrt(T/(T-k))`` (Stine 1987): OLS residuals have second moment
       ``((T-k)/T) Sigma_hat``, so unrescaled resampling systematically
       shrinks the bootstrap world's innovation variance and mis-centres
       Cholesky-style IRF draws low by ~``(T-k)/T``.
    2. Reconstruct a bootstrap dataset ``Y*_t`` using ``B_hat`` and ``u*_t``,
       starting from the first ``p`` observations of the original sample.
    3. Re-estimate ``B`` and ``Sigma`` by OLS on the bootstrap sample,
       with ``Sigma`` on the SAME divisor as the point estimator so
       bootstrap and point conventions match: ``T_eff - k`` on the plain
       VAR path, ``T_eff`` (ML) on the VARX path (CONVENTIONS.md §3).
    4. Compute the orthogonalised IRFs at horizons 0..H.

    Discards unstable draws if ``require_stable`` is True.
    """
    if structural_impact is None:
        structural_impact = np.linalg.cholesky

    # Panel fits with a canonical-unit VARFit face (pooled, mean-group,
    # partial-pooling) carry one unit's data/residuals next to cross-unit
    # coefficients — resampling that single unit would misstate the panel's
    # sampling uncertainty.
    if getattr(fit, "units", None) and getattr(fit, "method", "") in (
        "pooled", "mean_group", "partial_pooling",
    ):
        raise NotImplementedError(
            f"residual_bootstrap_draws is not available for "
            f"{fit.method!r} panel fits: the fit's VARFit face holds a "
            "single canonical unit's data and residuals, so a single-VAR "
            "residual bootstrap would misstate the pooled sampling "
            "uncertainty. Use bands='none' (point estimates only), or a "
            "panel-specific bootstrap."
        )

    rng = np.random.default_rng(seed)
    n = fit.n_vars
    p = fit.p
    k = fit.k
    T_eff = fit.t_eff

    Y_full = fit.data.values
    initial_obs = Y_full[:p].copy()
    resid = fit.residuals
    B_hat = fit.B_hat

    # Determine deterministic + exog layout
    has_exog = fit.has_exog
    exog_arr = fit.exog.values if has_exog else None
    exog_lags = fit.exog_lags if has_exog else 0
    n_exog_cols = (fit.n_exog * (exog_lags + 1)) if has_exog else 0
    # Plain VAR layout: [lag1, ..., lagp, (const), (trend)]
    # VARX layout adds exog block at the end; total k columns.
    include_const = (k - n * p - n_exog_cols) >= 1
    include_trend = (k - n * p - n_exog_cols) >= 2

    P_point = structural_impact(fit.Sigma_u)
    n_shocks = P_point.shape[1]

    # Stine (1987) rescaling: resampled residuals must reproduce the
    # df-adjusted innovation variance, not the ML one (see docstring).
    resid_rescaled = resid * np.sqrt(T_eff / (T_eff - k))

    out: list[np.ndarray] = []
    attempts = 0
    max_attempts = n_draws * 4
    while len(out) < n_draws and attempts < max_attempts:
        attempts += 1

        u_star = resid_rescaled[rng.integers(0, T_eff, size=T_eff)]
        Y_star = np.zeros_like(Y_full, dtype=float)  # float even for integer input — otherwise every simulated row is truncated
        Y_star[:p] = initial_obs

        for t in range(p, p + T_eff):
            x_arr = np.asarray(_build_x_row(
                Y_star, t, p, include_const, include_trend, exog_arr, exog_lags,
            ))
            mean_t = x_arr @ B_hat
            Y_star[t] = mean_t + u_star[t - p]

        # Re-estimate on the bootstrap dataset (exog stays fixed)
        Y_act_b = Y_star[p:]
        X_rows = [
            _build_x_row(Y_star, t, p, include_const, include_trend, exog_arr, exog_lags)
            for t in range(p, p + T_eff)
        ]
        X_act_b = np.asarray(X_rows)

        try:
            B_d = np.linalg.solve(X_act_b.T @ X_act_b, X_act_b.T @ Y_act_b)
        except np.linalg.LinAlgError:
            continue
        E_b = Y_act_b - X_act_b @ B_d
        # Same divisor as the POINT estimator for this fit type: the plain
        # VAR path reports Sigma on T_eff - k (Lütkepohl; the previous ML
        # divisor stacked a second (T-k)/T shrink on the unrescaled
        # residuals, mis-centring the bands ~5-15%), while the VARX path
        # deliberately reports the ML divisor T_eff (var.py, CONVENTIONS §3).
        # Each bootstrap replication must estimate the SAME covariance
        # statistic as the point fit, or every Cholesky draw is scaled by
        # sqrt(T/(T-k)) relative to it.
        sigma_div = T_eff if has_exog else (T_eff - k)
        Sigma_d = (E_b.T @ E_b) / sigma_div

        lag_mats = extract_lag_matrices(B_d, n, p)
        if require_stable and np.max(np.abs(np.linalg.eigvals(
            _companion_from(lag_mats, n, p)
        ))) >= 1.0:
            continue

        try:
            P_d = structural_impact(Sigma_d)
        except np.linalg.LinAlgError:
            continue

        Phi = ma_representation(lag_mats, horizon)
        out.append(np.einsum("hij,jk->hik", Phi, P_d))

    if len(out) < n_draws:
        # Pad with the available draws (caller can warn)
        pass
    return np.stack(out, axis=0)


def _companion_from(lag_matrices: list[np.ndarray], n: int, p: int) -> np.ndarray:
    C = np.zeros((n * p, n * p))
    C[:n, :] = np.hstack(lag_matrices)
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return C


