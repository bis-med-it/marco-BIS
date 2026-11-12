"""
Factor-Augmented VAR (FAVAR) — Bernanke-Boivin-Eliasz (2005).

Two-step PC-based FAVAR:

1. Extract ``K`` principal components from a large information panel
   ``X_t`` (standardised columns).
2. If a slow/fast partition is provided, rotate the factors against the
   policy variable so the "slow" components are orthogonal to the
   contemporaneous policy shock (BBE Section II.C).
3. Estimate a VAR on the augmented state ``[F_rotated_t, Y_t]``.
4. Cholesky-identify the monetary policy shock (last in ordering).
5. Map IRFs of the augmented state back to the original ``X`` variables
   via loadings estimated on ``[F_t, Y_t]`` — BBE equation (2) includes
   the observed policy variable among the regressors, so panel variables
   (the federal funds rate above all) can respond on impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .var import VARFit, fit_var, extract_lag_matrices, ma_representation
from .identification import cholesky_irf
from .irf import IRFContainer


@dataclass
class FAVARFit:
    """Container for a fitted FAVAR.

    Attributes
    ----------
    factors : np.ndarray, shape (T, K)
        Rotated factor series.
    loadings : np.ndarray, shape (N, K+1)
        Estimated loadings from ``X_t = Λ^f F_t + λ^y Y_t + e_t``
        (BBE equation (2), on standardised X): the first ``K`` columns
        load on the factors, the last on the observed policy variable.
    means, stds : np.ndarray, shape (N,)
        Column means and standard deviations used to standardise ``X``.
    var_fit : VARFit
        VAR on the augmented state ``[F_rotated, Y]``.
    policy_var : str
        Name of the policy variable.
    info_panel_names : list of str
        Column names of the original information panel ``X``.
    """

    factors: np.ndarray
    loadings: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    var_fit: VARFit
    policy_var: str
    info_panel_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        return self.factors.shape[1]

    @property
    def n_info_vars(self) -> int:
        return len(self.info_panel_names)


def fit_favar(
    info_panel: pd.DataFrame,
    policy_series: pd.Series,
    n_factors: int = 3,
    lags: int = 13,
    slow_columns: Sequence[str] | None = None,
    trend: str = "c",
) -> FAVARFit:
    """Estimate a two-step FAVAR (Bernanke-Boivin-Eliasz 2005).

    Parameters
    ----------
    info_panel : pd.DataFrame
        Large information panel ``X_t``, ``T × N``.
    policy_series : pd.Series
        Policy variable ``Y_t`` (e.g., federal funds rate), length ``T``,
        aligned to ``info_panel.index``.
    n_factors : int, default 3
        Number of principal components to extract from ``X``.
    lags : int, default 13
        VAR lag order on the augmented state ``[F_rotated, Y]``.
    slow_columns : sequence of str, optional
        Names of columns in ``info_panel`` that are SLOW (don't react
        within the period to monetary policy). If provided, the BBE
        slow/fast rotation is applied. If ``None``, no rotation —
        the simple two-step FAVAR.
    trend : str, default "c"
        VAR trend setting.

    Returns
    -------
    FAVARFit
    """
    if len(info_panel) != len(policy_series):
        raise ValueError(
            f"info_panel ({len(info_panel)}) and policy_series "
            f"({len(policy_series)}) must have equal length"
        )
    if info_panel.isna().any().any():
        raise ValueError("info_panel contains NaN — drop or impute first")
    if policy_series.isna().any():
        raise ValueError("policy_series contains NaN")

    X = info_panel.values.astype(float)
    Y = np.asarray(policy_series).astype(float).ravel()
    N = X.shape[1]

    # Standardise the panel
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds < 1e-12] = 1.0
    X_std = (X - means) / stds

    # Extract factors via SVD
    F_unrotated, _ = _pc_factors(X_std, K=n_factors)

    # Apply slow/fast rotation if requested
    if slow_columns is not None:
        bad = [c for c in slow_columns if c not in info_panel.columns]
        if bad:
            raise ValueError(f"slow_columns not in info_panel: {bad!r}")
        slow_idx = [info_panel.columns.get_loc(c) for c in slow_columns]
        X_slow_std = X_std[:, slow_idx]
        F_slow_unrotated, _ = _pc_factors(X_slow_std, K=n_factors)
        F_rotated = _slow_fast_rotation(F_unrotated, F_slow_unrotated, Y)
    else:
        F_rotated = F_unrotated

    # Build augmented state [F_rotated, Y]
    state_arr = np.column_stack([F_rotated, Y])
    state_columns = [f"F{i + 1}" for i in range(n_factors)] + [policy_series.name or "Y"]
    state = pd.DataFrame(state_arr, index=info_panel.index, columns=state_columns)

    # Estimate loadings on the FULL augmented state [F, Y] — BBE eq (2)
    # includes the observed policy variable, so each panel variable carries
    # a direct loading on Y as well as on the factors. Shape (N, K+1).
    # The regressors are CENTERED first: X_std is demeaned and the factors
    # are ~mean-zero, but Y is in levels — regressing without an intercept
    # would let the loadings absorb Y's mean, making every projected IRF
    # depend on the policy variable's arbitrary zero point (adding 100 to
    # the rate must not change any response). Only the slopes are kept;
    # IRFs are responses to shocks, so the centering constant drops out.
    state_centered = state_arr - state_arr.mean(axis=0, keepdims=True)
    Lambda = np.linalg.lstsq(state_centered, X_std, rcond=None)[0].T
    var_fit = fit_var(state, lags=lags, trend=trend)

    metadata = {
        "method": "favar_two_step_pc",
        "n_factors": int(n_factors),
        "n_info_vars": N,
        "rotation": "slow_fast_BBE" if slow_columns is not None else "none",
        "n_lags": int(lags),
    }
    return FAVARFit(
        factors=F_rotated,
        loadings=Lambda,
        means=means,
        stds=stds,
        var_fit=var_fit,
        policy_var=policy_series.name or "Y",
        info_panel_names=list(info_panel.columns),
        metadata=metadata,
    )


def favar_irf(
    fit: FAVARFit, horizon: int, response_vars: Sequence[str] | None = None,
    bands: str = "parameter_draws", n_draws: int = 2000, seed: int | None = None,
) -> IRFContainer:
    """Compute IRFs of original information-panel variables to the monetary
    policy shock (last in the Cholesky ordering of the augmented state).

    Parameters
    ----------
    fit : FAVARFit
    horizon : int
        IRF horizon.
    response_vars : sequence of str, optional
        Subset of ``fit.info_panel_names`` for which to compute responses.
        Defaults to all panel variables.
    bands : {"none", "parameter_draws"}
    n_draws : int
    seed : int, optional

    Returns
    -------
    IRFContainer with shape ``(H+1, n_response, 1)`` (single MP shock).
    """
    if response_vars is None:
        response_vars = list(fit.info_panel_names)
    info_indices = [fit.info_panel_names.index(v) for v in response_vars]
    Lambda_sub = fit.loadings[info_indices]        # (n_response, K+1)
    stds_sub = fit.stds[info_indices]

    # Point estimate — project the FULL augmented-state IRF (factors AND
    # the observed policy variable) through the loadings, per BBE eq (2).
    irf_state = cholesky_irf(fit.var_fit, horizon=horizon, bands="none")
    policy_shock_idx = irf_state.n_shocks - 1
    state_irf_point = irf_state.median[:, :, policy_shock_idx]     # (H+1, K+1)
    median_arr = (state_irf_point @ Lambda_sub.T) * stds_sub       # (H+1, n_response)

    metadata = {
        **fit.metadata,
        "identification": "cholesky_on_augmented_state",
        "shock_name": f"{fit.policy_var} shock",
        "policy_shock_idx": int(policy_shock_idx),
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
    }

    if bands == "none":
        return IRFContainer.from_point(
            point=median_arr[:, :, None],
            variable_names=list(response_vars),
            shock_names=[f"{fit.policy_var} shock"],
            metadata=metadata,
        )

    if bands not in ("parameter_draws", "asymptotic_mc", "posterior"):
        raise ValueError(f"unknown bands method: {bands!r}")

    # Project each parameter draw through Lambda BEFORE quantiling — otherwise
    # negative Lambda entries swap the bands.
    params = fit.var_fit.parameter_draws(n_draws=n_draws, seed=seed)
    n_state = fit.var_fit.n_vars
    draws_proj = []
    for B_d, Sigma_d in params:
        try:
            P_d = np.linalg.cholesky(Sigma_d)
        except np.linalg.LinAlgError:
            continue
        lag_mats = extract_lag_matrices(B_d, n_state, fit.var_fit.p)
        Phi_d = ma_representation(lag_mats, horizon)
        state_irf_d = np.einsum("hij,jk->hik", Phi_d, P_d)         # (H+1, n_state, n_state)
        state_irf_slice = state_irf_d[:, :, policy_shock_idx]      # (H+1, K+1)
        proj = (state_irf_slice @ Lambda_sub.T) * stds_sub         # (H+1, n_response)
        draws_proj.append(proj[:, :, None])
    if not draws_proj:
        raise RuntimeError("no usable parameter draws for FAVAR bands")
    draws_arr = np.stack(draws_proj, axis=0)
    return IRFContainer.from_draws(
        draws=draws_arr,
        variable_names=list(response_vars),
        shock_names=[f"{fit.policy_var} shock"],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pc_factors(X_std: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract the first ``K`` principal components from a standardised
    panel ``X_std`` of shape ``(T, N)``.

    Returns ``(F, Lambda)`` where ``F`` is ``(T, K)`` and ``Lambda`` is
    ``(N, K)`` with ``X_std ≈ F Λ'``.

    Normalisation: ``F' F / T = I_K`` (PC convention).
    """
    T, N = X_std.shape
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    # X = U S V'; F = U_K * sqrt(T), Lambda = V_K * S_K / sqrt(T)
    sqrt_T = np.sqrt(T)
    F = U[:, :K] * sqrt_T
    Lambda = (Vt[:K, :].T * S[:K]) / sqrt_T
    return F, Lambda


def _slow_fast_rotation(
    F: np.ndarray, F_slow: np.ndarray, Y: np.ndarray,
) -> np.ndarray:
    """Slow/fast rotation following BBE (2005), Section II.C.

    For each factor in ``F``, regress it on ``F_slow`` and ``Y`` (policy
    variable), then subtract the part explained by ``Y`` — yielding the
    "cleaned" factor that is contemporaneously orthogonal to the policy
    variable.

    Returns the rotated ``(T, K)`` factor matrix.
    """
    T, K = F.shape
    X = np.column_stack([np.ones(T), F_slow, Y])
    # Index of Y in the design matrix
    y_idx = 1 + F_slow.shape[1]

    F_rotated = np.empty_like(F)
    for k in range(K):
        beta, *_ = np.linalg.lstsq(X, F[:, k], rcond=None)
        # Subtract the contribution of Y from F[:, k]
        F_rotated[:, k] = F[:, k] - beta[y_idx] * Y
    return F_rotated
