# =============================================================================
# EXPERIMENTAL
# The estimators in this file are less established than the core VAR code.
# They rest on the library's own tests on simulated data. Treat results with
# caution and confirm important ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
"""
Time-varying parameter VAR (Primiceri 2005 style).

Implements a TVP-VAR with time-varying coefficients via Gibbs sampling,
in two variants: ``fit_tvp_var`` (constant Sigma, homoskedastic shocks)
and ``fit_tvp_var_sv_primiceri`` (the full Primiceri 2005 model with
stochastic volatility via the Kim-Shephard-Chib mixture algorithm).

Model
-----
    y_t = X_t' β_t + u_t,         u_t ~ N(0, Σ)
    β_t = β_{t-1} + ν_t,          ν_t ~ N(0, Q)

where ``y_t`` is ``n × 1``, ``X_t = [y_{t-1}', ..., y_{t-p}', 1]'`` is
``k × 1``, ``β_t`` is ``k × n`` (one column per equation), and the
state vector is ``vec(β_t)`` with dimension ``nk``.

Gibbs sampler:
  1. Given Σ, Q: draw β_{1:T} from the time-varying state-space using
     the Carter-Kohn / Durbin-Koopman simulation smoother.
  2. Given β_{1:T}, Σ: draw Q from an inverse-Wishart conditional on
     the implied β innovations.
  3. Given β_{1:T}, Q: draw Σ from an inverse-Wishart conditional on
     the implied y residuals.

Public API:

* :class:`TVPVARSpec` — variables, lags, training-sample size, etc.
* :class:`TVPVARFit` — holds Gibbs draws of β_t paths, Σ, Q.
* :func:`fit_tvp_var` — Gibbs sampler.

Notes
-----
* This is the **homoskedastic TVP-VAR** (no stochastic volatility). The
  full Primiceri 2005 model with time-varying Σ_t via Kim-Shephard-Chib
  mixture sampling is ``fit_tvp_var_sv_primiceri`` in this module.
* The Gibbs sampler is computationally heavy: each iteration runs a
  Kalman simulation smoother on an ``n*k``-dimensional state. Use small
  ``n_iter`` for prototyping; budget for ~1-5 seconds per 100 iterations
  on a 3-variable VAR(2).
* The training sample sets the priors and also stays in the estimation
  sample (Primiceri 2005 drops it from estimation), so paths are
  reported for all ``T - lags`` dates.
* Several priors differ from the paper (see ``fit_tvp_var_sv_primiceri``
  and CONVENTIONS.md §4c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .state_space import StateSpaceModel, simulation_smoother
from .var import (
    VARFit, build_design_matrices, extract_lag_matrices,
    ma_representation, max_eigenvalue_modulus,
)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class TVPVARSpec:
    """TVP-VAR specification.

    Parameters
    ----------
    lags : int
        VAR lag order.
    training_sample : int, default 40
        Number of initial observations used to construct the prior on
        ``β_0`` and ``Q``. Primiceri (2005) uses a 10-year training
        sample (40 quarterly obs).
    Q_prior_scale : float, default 1e-4
        Hyperparameter scaling the inverse-Wishart prior on ``Q``. This is
        ``k_Q²`` in Primiceri's (2005, p. 831) notation: his ``k_Q = 0.01``,
        so ``k_Q² = 1e-4`` — the default. Smaller
        = tighter prior (less time-variation); larger = looser (more drift).
    Q_prior_df_factor : int, default 1
        Multiplicative factor on the IW degrees-of-freedom for Q.
        Higher = tighter prior. Primiceri uses ``df_Q = training_sample``.
    Sigma_prior_df : int, optional
        IW degrees-of-freedom for Σ prior. Defaults to ``n + 2`` (just
        proper).
    """

    lags: int
    training_sample: int = 40
    Q_prior_scale: float = 1e-4
    Q_prior_df_factor: int = 1
    Sigma_prior_df: int | None = None

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise ValueError(f"lags must be ≥ 1, got {self.lags}")
        if self.training_sample <= 0:
            raise ValueError(
                f"training_sample must be positive, got {self.training_sample}"
            )


# ---------------------------------------------------------------------------
# Fit object
# ---------------------------------------------------------------------------


@dataclass
class TVPVARFit(VARFit):
    """Fitted TVP-VAR.

    Stores Gibbs draws of the time-varying parameter path and constant
    covariances Q (drift) and Σ (residual). Inherits :class:`VARFit` for
    compatibility with identification routines: ``B_hat`` holds the
    posterior-mean coefficients AT THE LAST PERIOD (so e.g.
    ``cholesky_irf`` traces a counterfactual constant-coefficient VAR
    with the most-recent parameters).

    Attributes
    ----------
    spec : TVPVARSpec
    beta_path_draws : (n_draws, T_eff, k, n) ndarray
        Gibbs draws of ``β_t`` for each post-burn-in iteration.
    Sigma_draws : (n_draws, n, n) ndarray
        Gibbs draws of Σ.
    Q_draws : (n_draws, k*n, k*n) ndarray
        Gibbs draws of Q (the coefficient-drift covariance).
    beta_path_posterior_mean : (T_eff, k, n) ndarray
    """

    spec: TVPVARSpec = None
    beta_path_draws: np.ndarray = field(default=None, repr=False)
    Sigma_draws: np.ndarray = field(default=None, repr=False)
    Q_draws: np.ndarray = field(default=None, repr=False)
    beta_path_posterior_mean: np.ndarray = field(default=None, repr=False)

    @property
    def beta_at(self) -> "callable":
        """Closure returning the posterior-mean β at any index ``t``."""
        def _at(t: int) -> np.ndarray:
            return self.beta_path_posterior_mean[t]
        return _at

    def parameter_draws(
        self, n_draws: int, seed: int | None = None,
        require_stable: bool = False, eig_tol: float = 1.01,
        burn_in: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return draws of (β at last period, Σ) — these can flow into
        the identification routines as if for a constant-parameter VAR
        with the most-recent coefficients."""
        if self.beta_path_draws is None:
            raise RuntimeError("TVPVARFit has no stored Gibbs draws")
        rng = np.random.default_rng(seed)
        n_avail = len(self.beta_path_draws)
        if n_draws > n_avail:
            raise ValueError(
                f"requested {n_draws} draws but only {n_avail} are stored"
            )
        idx = rng.choice(n_avail, size=n_draws, replace=False)
        out = []
        for i in idx:
            B = self.beta_path_draws[i][-1]   # last-period β
            S = self.Sigma_draws[i]
            if require_stable:
                lag_mats = extract_lag_matrices(B, self.n_vars, self.p)
                if max_eigenvalue_modulus(lag_mats) >= eig_tol:
                    continue
            out.append((B, S))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_design_per_t(Y: np.ndarray, p: int, include_const: bool = True
                        ) -> np.ndarray:
    """Build the regressor matrix at each effective time step.

    Returns shape ``(T_eff, k)`` where ``T_eff = T - p`` and
    ``k = n * p + (1 if include_const else 0)``.

    Row ``t`` is ``[y_{t}', y_{t-1}', ..., y_{t-p+1}', 1]`` (the
    regressors that predict ``y_{t+p}`` in the standard layout
    matching ``var.build_design_matrices``).
    """
    _, X_act = build_design_matrices(Y, p, include_const=include_const)
    return X_act


def _build_time_varying_state_space(
    T_eff: int, n: int, k: int, X_act: np.ndarray, Sigma: np.ndarray, Q: np.ndarray,
    beta_init_mean: np.ndarray, beta_init_cov: np.ndarray,
) -> StateSpaceModel:
    """Construct the time-varying state-space for the β path.

    State: ``α_t = vec(β_t)`` of dimension ``nk`` — column-stacked (Fortran
    order), so the first ``k`` entries are equation 1's coefficients,
    matching ``beta_0_mean = B_ols.flatten(order="F")`` and the prior
    covariance ``kron(Σ, (X'X)^{-1})``.
    Transition: ``α_t = α_{t-1} + η_t``, ``η_t ~ N(0, Q)``.
    Measurement: ``y_t = (I_n ⊗ X_t') α_t + u_t``, ``u_t ~ N(0, Σ)`` —
    the identity-first Kronecker is what maps the column-stacked
    ``vec(β_t)`` to ``β_t' x_t`` (Primiceri 2005, eq. 1's X_t').

    The design ``Z_t`` is time-varying because it depends on ``X_t``.
    """
    # Z_t: (n, nk) = I_n ⊗ X_t'   (block-diagonal x_t' per equation)
    Z = np.zeros((T_eff, n, n * k))
    for t in range(T_eff):
        Z[t] = np.kron(np.eye(n), X_act[t].reshape(1, k))

    T_mat = np.eye(n * k)
    R = np.eye(n * k)

    return StateSpaceModel(
        Z=Z, H=Sigma, T=T_mat, R=R, Q=Q,
        initial_state_mean=beta_init_mean,
        initial_state_cov=beta_init_cov,
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_tvp_var(
    data: pd.DataFrame,
    spec: TVPVARSpec,
    n_iter: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    seed: int | None = None,
    trend: str = "c",
    verbose: bool = False,
) -> TVPVARFit:
    """Estimate a TVP-VAR (homoskedastic) via Gibbs sampling.

    Parameters
    ----------
    data : pd.DataFrame
    spec : TVPVARSpec
    n_iter, burn_in, thin : int
    seed : int, optional
    trend : str, default "c"
    verbose : bool

    Returns
    -------
    TVPVARFit

    Notes
    -----
    See module docstring for the model and algorithm. The training
    sample (first ``spec.training_sample`` observations) is used to
    construct the prior for ``β_0`` (OLS on training data) and for Q
    (Primiceri 2005 convention: IW scale = k_Q² · τ · V_OLS^train).
    """
    if data.isna().any().any():
        raise ValueError("data contains NaN")
    rng = np.random.default_rng(seed)
    Y = data.values.astype(float)
    T, n = Y.shape
    p = spec.lags
    include_const = trend == "c"
    n_train = spec.training_sample

    if T <= n_train + p + 5:
        raise ValueError(
            f"data length {T} too short for training_sample={n_train}, lags={p}"
        )

    # --- 1. Build design ---
    Y_act, X_act = build_design_matrices(Y, p, include_const=include_const)
    T_eff = Y_act.shape[0]
    k = X_act.shape[1]
    nk = n * k

    # --- 2. Training-sample prior for β_0 and Q ---
    train_end = n_train
    Y_train = Y_act[:train_end]
    X_train = X_act[:train_end]
    XtX_train = X_train.T @ X_train
    XtX_train_inv = np.linalg.inv(XtX_train)
    beta_ols_train = XtX_train_inv @ X_train.T @ Y_train    # (k, n)
    resid_train = Y_train - X_train @ beta_ols_train
    Sigma_ols_train = (resid_train.T @ resid_train) / max(train_end - k, 1)

    beta_0_mean = beta_ols_train.flatten(order="F")          # vec(β) (nk,)
    V_ols_train = np.kron(Sigma_ols_train, XtX_train_inv)    # V(B_OLS^train)
    beta_0_cov = 4 * V_ols_train       # Primiceri's looser prior on β_0 only

    # Q prior: Inverse-Wishart with scale k_Q² · τ · V_OLS (Primiceri 2005,
    # p. 831 / Appendix C). The scale multiplies by τ = n_train and uses
    # V_OLS; the factor 4 belongs to the β_0 prior only. Pinned by
    # TestTVPPriorConventions.
    Q_prior_scale = spec.Q_prior_scale * V_ols_train * n_train
    Q_prior_df = max(spec.Q_prior_df_factor * n_train, nk + 2)

    # Σ prior: IW with df = Sigma_prior_df, scale = Sigma_ols_train
    Sigma_prior_df = spec.Sigma_prior_df or (n + 2)
    Sigma_prior_scale = Sigma_ols_train * Sigma_prior_df

    # --- 3. Initialise Gibbs from OLS-on-training-sample point estimates ---
    Sigma_curr = Sigma_ols_train.copy()
    Q_curr = Q_prior_scale / max(Q_prior_df - nk - 1, 1)    # IW prior mean

    n_kept = len(range(burn_in, n_iter, thin))  # matches the storage schedule below
    if n_kept <= 0:
        raise ValueError("n_iter must exceed burn_in")

    beta_path_store = np.zeros((n_kept, T_eff, k, n))
    Sigma_store = np.zeros((n_kept, n, n))
    Q_store = np.zeros((n_kept, nk, nk))

    stored = 0
    for it in range(n_iter):
        # --- 3a. Draw β path via simulation smoother ---
        ss_model = _build_time_varying_state_space(
            T_eff=T_eff, n=n, k=k,
            X_act=X_act, Sigma=Sigma_curr, Q=Q_curr,
            beta_init_mean=beta_0_mean, beta_init_cov=beta_0_cov,
        )
        try:
            draws = simulation_smoother(
                ss_model, Y_act, n_draws=1,
                seed=rng.integers(0, 2**31 - 1),
            )
        except np.linalg.LinAlgError:
            # Q or Sigma may have become near-singular; reset to previous
            continue
        alpha_path = draws[0]                  # (T_eff, nk)
        # Reshape to (T_eff, k, n)
        beta_path = alpha_path.reshape(T_eff, n, k).transpose(0, 2, 1)

        # --- 3b. Draw Q | β, Sigma ---
        # β innovations: ν_t = β_t - β_{t-1}, for t = 1..T_eff-1
        nu = np.diff(alpha_path, axis=0)                    # (T_eff-1, nk)
        S_Q = nu.T @ nu + Q_prior_scale
        df_Q = nu.shape[0] + Q_prior_df
        # atleast_2d: SciPy squeezes a 1-D inverse-Wishart draw to a scalar
        Q_curr = np.atleast_2d(sp_stats.invwishart.rvs(
            df=df_Q, scale=S_Q, random_state=rng,
        ))

        # --- 3c. Draw Σ | β, data ---
        # Compute residuals u_t = y_t - X_t' β_t at each t
        u = np.zeros((T_eff, n))
        for t in range(T_eff):
            u[t] = Y_act[t] - X_act[t] @ beta_path[t]
        S_Sigma = u.T @ u + Sigma_prior_scale
        df_Sigma = T_eff + Sigma_prior_df
        Sigma_curr = np.atleast_2d(sp_stats.invwishart.rvs(
            df=df_Sigma, scale=S_Sigma, random_state=rng,
        ))

        # --- Store ---
        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_path_store[stored] = beta_path
            Sigma_store[stored] = Sigma_curr
            Q_store[stored] = Q_curr
            stored += 1

        if verbose and (it + 1) % 50 == 0:
            print(f"      Gibbs iter {it + 1}/{n_iter} "
                  f"(stored: {stored}/{n_kept})")

    # --- 4. Aggregate ---
    # Drop zero-padded slots from failed (LinAlgError-skipped) draws —
    # averaging over preallocated zeros silently shrinks posterior means
    # toward zero and misreports the draw count.
    if stored == 0:
        raise RuntimeError(
            "all post-burn-in Gibbs draws failed (LinAlgError) — no posterior")
    if stored < n_kept:
        import warnings
        warnings.warn(
            f"{n_kept - stored} of {n_kept} post-burn-in draws failed "
            f"(LinAlgError) and were dropped; posterior summaries use the "
            f"{stored} successful draws.", UserWarning, stacklevel=2)
        beta_path_store = beta_path_store[:stored]
        Sigma_store = Sigma_store[:stored]
        Q_store = Q_store[:stored]
        n_kept = stored
    beta_path_mean = beta_path_store.mean(axis=0)            # (T_eff, k, n)
    Sigma_mean = Sigma_store.mean(axis=0)
    beta_last_mean = beta_path_mean[-1]

    # Build VARFit base info using last-period β
    resid_at_last = Y_act - np.einsum("tk,kn->tn", X_act, beta_last_mean)

    fit_info = {
        "trend": trend,
        "method": "tvp_var_gibbs",
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_draws_stored": int(n_kept),
        "training_sample": n_train,
        "Q_prior_scale": float(spec.Q_prior_scale),
        "Sigma_prior_df": int(Sigma_prior_df),
    }

    return TVPVARFit(
        data=data,
        p=p,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=beta_last_mean,
        Sigma_u=Sigma_mean,
        residuals=resid_at_last,
        variable_names=list(data.columns),
        fit_info=fit_info,
        spec=spec,
        beta_path_draws=beta_path_store,
        Sigma_draws=Sigma_store,
        Q_draws=Q_store,
        beta_path_posterior_mean=beta_path_mean,
    )


# ---------------------------------------------------------------------------
# TVP-VAR with diagonal stochastic volatility
# ---------------------------------------------------------------------------


def fit_tvp_var_sv(
    data: pd.DataFrame,
    spec: TVPVARSpec,
    n_iter: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    sigma_h_init: float = 0.2,
    sigma_h_prior_a: float = 3.0,
    sigma_h_prior_b: float = 0.01,
    seed: int | None = None,
    trend: str = "c",
    verbose: bool = False,
) -> TVPVARFit:
    """TVP-VAR with diagonal stochastic volatility.

    Model:
        y_t = X_t' β_t + u_t,    u_t ~ N(0, Σ_t)
        β_t = β_{t-1} + ν_t,     ν_t ~ N(0, Q)
        Σ_t = diag(exp(h_{1t}), ..., exp(h_{nt}))
        h_{j,t} = h_{j,t-1} + σ_{h,j} * ξ_{j,t}

    This is a SIMPLIFIED Primiceri (2005). Full Primiceri also has a
    time-varying contemporaneous-impact matrix A_t; here we force Σ_t
    diagonal (no cross-equation correlations), isolating SV from the
    A_t complication. For time-varying A_t, see
    :func:`fit_tvp_var_sv_primiceri` and its documented qualifications.

    Per-iteration Gibbs:
      1. Build time-varying state-space with current Σ_t; draw β path.
      2. Compute residuals; for each equation j, draw log h_{j,1:T} via
         the KSC mixture algorithm (:func:`lib.draw_log_h_path`).
      3. Draw σ²_{h,j} from IG given log h innovations.
      4. Draw Q from IW given β innovations.

    Parameters
    ----------
    data : pd.DataFrame
    spec : TVPVARSpec
    n_iter, burn_in, thin : int
    sigma_h_init : float, default 0.2
    sigma_h_prior_a, sigma_h_prior_b : float
        IG prior on σ²_h. The default IG(3, 0.01) has mean 0.005 and
        favours slowly moving volatility.
    seed : int, optional
    trend : str
    verbose : bool

    Returns
    -------
    TVPVARFit
        ``Sigma_draws[i]`` holds the time-averaged Σ for draw ``i``. The
        full time-varying log h path lives in
        ``fit_info["log_h_path_posterior_mean"]``.
    """
    from .stochastic_volatility import draw_log_h_path

    if data.isna().any().any():
        raise ValueError("data contains NaN")
    rng = np.random.default_rng(seed)
    Y = data.values.astype(float)
    T, n = Y.shape
    p = spec.lags
    include_const = trend == "c"
    n_train = spec.training_sample

    if T <= n_train + p + 5:
        raise ValueError(
            f"data too short for training_sample={n_train}, lags={p}"
        )

    Y_act, X_act = build_design_matrices(Y, p, include_const=include_const)
    T_eff = Y_act.shape[0]
    k = X_act.shape[1]
    nk = n * k

    Y_train = Y_act[:n_train]
    X_train = X_act[:n_train]
    XtX_train_inv = np.linalg.inv(X_train.T @ X_train)
    beta_ols_train = XtX_train_inv @ X_train.T @ Y_train
    resid_train = Y_train - X_train @ beta_ols_train
    Sigma_ols_train = (resid_train.T @ resid_train) / max(n_train - k, 1)

    beta_0_mean = beta_ols_train.flatten(order="F")
    V_ols_train = np.kron(Sigma_ols_train, XtX_train_inv)    # V(B_OLS^train)
    beta_0_cov = 4 * V_ols_train       # Primiceri's looser prior on β_0 only
    # Q prior: IW scale = k_Q² · τ · V_OLS (Primiceri 2005 p. 831). The
    # factor 4 belongs to the β_0 prior only, NOT the Q scale. Pinned by TestTVPPriorConventions.
    Q_prior_scale = spec.Q_prior_scale * V_ols_train * n_train
    Q_prior_df = max(spec.Q_prior_df_factor * n_train, nk + 2)

    # Initialise log h to log of OLS-residual variances
    log_h_curr = np.tile(
        np.log(np.maximum(np.diag(Sigma_ols_train), 1e-8)), (T_eff, 1),
    )
    Q_curr = Q_prior_scale / max(Q_prior_df - nk - 1, 1)
    sigma_h_curr = np.full(n, sigma_h_init)

    n_kept = len(range(burn_in, n_iter, thin))  # matches the storage schedule below
    if n_kept <= 0:
        raise ValueError("n_iter must exceed burn_in")

    beta_path_store = np.zeros((n_kept, T_eff, k, n))
    log_h_store = np.zeros((n_kept, T_eff, n))
    Q_store = np.zeros((n_kept, nk, nk))
    sigma_h_store = np.zeros((n_kept, n))

    stored = 0
    for it in range(n_iter):
        # 1. Time-varying Σ_t = diag(exp(h_t))
        H_tv = np.zeros((T_eff, n, n))
        for t in range(T_eff):
            H_tv[t] = np.diag(np.exp(log_h_curr[t]))

        Z = np.zeros((T_eff, n, nk))
        for t in range(T_eff):
            Z[t] = np.kron(np.eye(n), X_act[t].reshape(1, k))

        from .state_space import StateSpaceModel, simulation_smoother
        ss_model = StateSpaceModel(
            Z=Z, H=H_tv,
            T=np.eye(nk), R=np.eye(nk), Q=Q_curr,
            initial_state_mean=beta_0_mean,
            initial_state_cov=beta_0_cov,
        )
        try:
            draws = simulation_smoother(
                ss_model, Y_act, n_draws=1,
                seed=rng.integers(0, 2**31 - 1),
            )
        except np.linalg.LinAlgError:
            continue
        alpha_path = draws[0]
        beta_path = alpha_path.reshape(T_eff, n, k).transpose(0, 2, 1)

        # 2. Residuals + per-equation KSC update
        u = np.zeros((T_eff, n))
        for t in range(T_eff):
            u[t] = Y_act[t] - X_act[t] @ beta_path[t]
        for j in range(n):
            log_h_curr[:, j] = draw_log_h_path(
                u[:, j], sigma_h=sigma_h_curr[j],
                h_current=log_h_curr[:, j],
                seed=int(rng.integers(0, 2**31 - 1)),
            )

        # 3. Per-equation σ²_h from IG
        for j in range(n):
            dh = np.diff(log_h_curr[:, j])
            a_post = sigma_h_prior_a + len(dh) / 2.0
            b_post = sigma_h_prior_b + 0.5 * float(dh @ dh)
            sigma2_h_j = 1.0 / sp_stats.gamma.rvs(
                a=a_post, scale=1.0 / b_post, random_state=rng,
            )
            sigma_h_curr[j] = max(np.sqrt(sigma2_h_j), 1e-4)

        # 4. Q from IW
        nu = np.diff(alpha_path, axis=0)
        S_Q = nu.T @ nu + Q_prior_scale
        df_Q = nu.shape[0] + Q_prior_df
        # atleast_2d: SciPy squeezes a 1-D inverse-Wishart draw to a scalar
        Q_curr = np.atleast_2d(sp_stats.invwishart.rvs(
            df=df_Q, scale=S_Q, random_state=rng,
        ))

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_path_store[stored] = beta_path
            log_h_store[stored] = log_h_curr
            Q_store[stored] = Q_curr
            sigma_h_store[stored] = sigma_h_curr
            stored += 1

        if verbose and (it + 1) % 50 == 0:
            print(f"      TVP-VAR-SV Gibbs {it + 1}/{n_iter} "
                  f"(stored: {stored}/{n_kept})")

    # Drop zero-padded slots from failed draws (see fit_tvp_var note).
    if stored == 0:
        raise RuntimeError(
            "all post-burn-in Gibbs draws failed (LinAlgError) — no posterior")
    if stored < n_kept:
        import warnings
        warnings.warn(
            f"{n_kept - stored} of {n_kept} post-burn-in draws failed "
            f"(LinAlgError) and were dropped; posterior summaries use the "
            f"{stored} successful draws.", UserWarning, stacklevel=2)
        beta_path_store = beta_path_store[:stored]
        log_h_store = log_h_store[:stored]
        Q_store = Q_store[:stored]
        sigma_h_store = sigma_h_store[:stored]
        n_kept = stored
    beta_path_mean = beta_path_store.mean(axis=0)
    log_h_mean = log_h_store.mean(axis=0)
    beta_last_mean = beta_path_mean[-1]

    # Time-averaged Σ for each stored draw and across draws
    Sigma_store_avg = np.zeros((n_kept, n, n))
    for i in range(n_kept):
        for t in range(T_eff):
            Sigma_store_avg[i] += np.diag(np.exp(log_h_store[i, t]))
        Sigma_store_avg[i] /= T_eff
    Sigma_avg = Sigma_store_avg.mean(axis=0)
    resid_at_last = Y_act - np.einsum("tk,kn->tn", X_act, beta_last_mean)

    fit_info = {
        "trend": trend,
        "method": "tvp_var_sv_gibbs_diagonal",
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_draws_stored": int(n_kept),
        "training_sample": n_train,
        "Q_prior_scale": float(spec.Q_prior_scale),
        "log_h_path_posterior_mean": log_h_mean.tolist(),
        "sigma_h_posterior_mean": sigma_h_store.mean(axis=0).tolist(),
    }

    return TVPVARFit(
        data=data,
        p=p,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=beta_last_mean,
        Sigma_u=Sigma_avg,
        residuals=resid_at_last,
        variable_names=list(data.columns),
        fit_info=fit_info,
        spec=spec,
        beta_path_draws=beta_path_store,
        Sigma_draws=Sigma_store_avg,
        Q_draws=Q_store,
        beta_path_posterior_mean=beta_path_mean,
    )


# ---------------------------------------------------------------------------
# Full Primiceri TVP-VAR with time-varying A_t
# ---------------------------------------------------------------------------


def _Sigma_from_A_h(A: np.ndarray, log_h: np.ndarray) -> np.ndarray:
    """Σ = A^{-1} diag(exp(h)) A^{-T}.

    ``A`` is lower triangular with 1s on the diagonal, shape ``(n, n)``.
    ``log_h`` is ``(n,)``.
    """
    A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
    D = np.diag(np.exp(log_h))
    return A_inv @ D @ A_inv.T


def fit_tvp_var_sv_primiceri(
    data: pd.DataFrame,
    spec: TVPVARSpec,
    n_iter: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    sigma_h_init: float = 0.2,
    sigma_h_prior_a: float = 3.0,
    sigma_h_prior_b: float = 0.01,
    S_prior_scale: float = 1e-4,
    seed: int | None = None,
    trend: str = "c",
    verbose: bool = False,
) -> TVPVARFit:
    """Full Primiceri (2005) TVP-VAR — time-varying β_t, A_t, and h_t.

    Model:
        y_t = X_t' β_t + u_t,                        u_t ~ N(0, Σ_t)
        Σ_t = A_t^{-1} diag(exp(h_t)) A_t^{-T}
        β_t = β_{t-1} + ν_t                          ν_t ~ N(0, Q)
        a_{j,t} = a_{j,t-1} + ζ_{j,t}                ζ_{j,t} ~ N(0, S_j)
        h_{j,t} = h_{j,t-1} + σ_{h,j} ξ_{j,t}

    ``A_t`` is lower triangular with 1s on the diagonal. Free elements
    are organised into row-blocks of sizes 1, 2, ..., n-1.

    Per-iteration Gibbs:
      1. Build state-space with current Σ_t; draw β path.
      2. Compute u_t = y_t - X_t β_t; draw row-j elements of A_t for
         each j ≥ 1 via a per-row state-space.
      3. w_t = A_t u_t. Draw log h_{j,t} per equation via KSC.
      4. Draw Q (IW), S_j (IW per row), σ²_{h,j} (IG).

    Returns
    -------
    TVPVARFit
        Time-averaged Σ in ``Sigma_u``. Full time-varying paths in
        ``fit_info``: ``log_h_path_posterior_mean`` (T_eff × n),
        ``A_path_posterior_mean`` (T_eff × n × n).

    Notes
    -----
    **Block ordering.** Primiceri's original 2005 sampler drew the
    volatilities conditional on mixture indicators from the previous
    sweep. Del Negro and Primiceri (2015, REStud corrigendum) correct
    this by drawing the indicators after the coefficient and covariance
    blocks and before the volatilities. This implementation uses that
    corrected order: the indicators are redrawn in every sweep, after
    the (β, A) blocks and immediately before the volatilities, and
    nothing else conditions on them.

    **Prior deviations from Primiceri (2005)** — simpler, documented
    choices; calibrate via ``TVPVARSpec`` / kwargs where exposed:

    * ``W`` (volatility-drift covariance) is forced DIAGONAL with
      per-equation IG(3, 0.01) priors on σ²_{h,j}, vs Primiceri's joint
      IW(k_W²·(1+n)·I, 1+n).
    * ``S_j`` priors are IW(``S_prior_scale``·τ·I_j, df=τ), vs Primiceri's
      IW(k_S²·(1+j)·V(Â_j,OLS), 1+j). The scale here is a fixed number,
      not scaled by the data, so check that it suits the units of the
      data. The default ``S_prior_scale=1e-4`` gives a prior mean for
      the drift variance of about 1e-4; a value of 0.01 produces visible
      drift in the covariance states even on data with constant
      covariances.
    * The A-block initial state is N(0, 10·I) and ``h_0 ~ N(0, 10)``,
      vs Primiceri's training-sample-OLS-centred priors.

    The Q prior follows Primiceri exactly (k_Q²·τ·V_OLS; see
    CONVENTIONS.md §4c).

    **Training sample.** The first ``training_sample`` observations set
    the priors and also remain in the estimation sample, so the returned
    paths cover all ``T - lags`` dates. Primiceri (2005) drops them from
    estimation.

    **Scale of the data.** The volatility sampler adds 1e-6 inside the
    logarithm (the paper uses 0.001). With residuals of standard
    deviation near 0.001 or below the estimated volatility path is
    flattened, so express the data in percent.
    """
    from .stochastic_volatility import draw_log_h_path
    from .state_space import StateSpaceModel, simulation_smoother

    if data.isna().any().any():
        raise ValueError("data contains NaN")
    rng = np.random.default_rng(seed)
    Y = data.values.astype(float)
    T, n = Y.shape
    p = spec.lags
    include_const = trend == "c"
    n_train = spec.training_sample

    if T <= n_train + p + 5:
        raise ValueError(
            f"data too short for training_sample={n_train}, lags={p}"
        )

    Y_act, X_act = build_design_matrices(Y, p, include_const=include_const)
    T_eff = Y_act.shape[0]
    k = X_act.shape[1]
    nk = n * k

    # Training priors
    Y_train = Y_act[:n_train]
    X_train = X_act[:n_train]
    XtX_train_inv = np.linalg.inv(X_train.T @ X_train)
    beta_ols_train = XtX_train_inv @ X_train.T @ Y_train
    resid_train = Y_train - X_train @ beta_ols_train
    Sigma_ols_train = (resid_train.T @ resid_train) / max(n_train - k, 1)

    beta_0_mean = beta_ols_train.flatten(order="F")
    V_ols_train = np.kron(Sigma_ols_train, XtX_train_inv)    # V(B_OLS^train)
    beta_0_cov = 4 * V_ols_train       # Primiceri's looser prior on β_0 only
    # Q prior: IW scale = k_Q² · τ · V_OLS (Primiceri 2005 p. 831). The
    # factor 4 belongs to the β_0 prior only, NOT the Q scale. Pinned by TestTVPPriorConventions.
    Q_prior_scale = spec.Q_prior_scale * V_ols_train * n_train
    Q_prior_df = max(spec.Q_prior_df_factor * n_train, nk + 2)

    # Initial A: rescale Σ_OLS Cholesky to unit-diagonal lower-triangular A
    L_OLS = np.linalg.cholesky(Sigma_ols_train)
    inv_L = np.linalg.solve(L_OLS, np.eye(n))
    A_init = np.diag(1.0 / np.diag(inv_L)) @ inv_L
    np.fill_diagonal(A_init, 1.0)
    h_init = np.log(np.maximum(np.diag(A_init @ Sigma_ols_train @ A_init.T), 1e-8))

    A_curr = np.tile(A_init, (T_eff, 1, 1))
    log_h_curr = np.tile(h_init, (T_eff, 1))
    Q_curr = Q_prior_scale / max(Q_prior_df - nk - 1, 1)
    sigma_h_curr = np.full(n, sigma_h_init)
    S_priors_scale = [
        np.atleast_2d(S_prior_scale * np.eye(j) * n_train) for j in range(1, n)
    ]
    S_priors_df = [n_train for _ in range(n - 1)]
    S_curr = [
        np.atleast_2d(S_priors_scale[i] / max(S_priors_df[i] - i - 1, 1))
        for i in range(n - 1)
    ]

    n_kept = len(range(burn_in, n_iter, thin))  # matches the storage schedule below
    if n_kept <= 0:
        raise ValueError("n_iter must exceed burn_in")

    beta_path_store = np.zeros((n_kept, T_eff, k, n))
    A_path_store = np.zeros((n_kept, T_eff, n, n))
    log_h_store = np.zeros((n_kept, T_eff, n))
    Sigma_store = np.zeros((n_kept, n, n))
    Q_store = np.zeros((n_kept, nk, nk))
    sigma_h_store = np.zeros((n_kept, n))

    stored = 0
    for it in range(n_iter):
        # 1. β path
        Sigma_path = np.zeros((T_eff, n, n))
        for t in range(T_eff):
            Sigma_path[t] = _Sigma_from_A_h(A_curr[t], log_h_curr[t])
        Z = np.zeros((T_eff, n, nk))
        for t in range(T_eff):
            Z[t] = np.kron(np.eye(n), X_act[t].reshape(1, k))
        ss_beta = StateSpaceModel(
            Z=Z, H=Sigma_path,
            T=np.eye(nk), R=np.eye(nk), Q=Q_curr,
            initial_state_mean=beta_0_mean,
            initial_state_cov=beta_0_cov,
        )
        try:
            draws = simulation_smoother(
                ss_beta, Y_act, n_draws=1,
                seed=rng.integers(0, 2**31 - 1),
            )
        except np.linalg.LinAlgError:
            continue
        alpha_path = draws[0]
        beta_path = alpha_path.reshape(T_eff, n, k).transpose(0, 2, 1)

        # 2. u_t and per-row A_t
        u = np.zeros((T_eff, n))
        for t in range(T_eff):
            u[t] = Y_act[t] - X_act[t] @ beta_path[t]
        for j in range(1, n):
            Z_j = (-u[:, :j]).reshape(T_eff, 1, j)
            H_j = np.exp(log_h_curr[:, j]).reshape(T_eff, 1, 1)
            ss_alpha = StateSpaceModel(
                Z=Z_j, H=H_j,
                T=np.eye(j), R=np.eye(j),
                Q=S_curr[j - 1],
                initial_state_cov=np.eye(j) * 10.0,
            )
            try:
                alpha_draws = simulation_smoother(
                    ss_alpha, u[:, j:j + 1], n_draws=1,
                    seed=rng.integers(0, 2**31 - 1),
                )
            except np.linalg.LinAlgError:
                continue
            alpha_j_path = alpha_draws[0]
            for t in range(T_eff):
                A_curr[t, j, :j] = alpha_j_path[t]

        # 3. w_t and log h per equation
        w = np.zeros_like(u)
        for t in range(T_eff):
            w[t] = A_curr[t] @ u[t]
        for j in range(n):
            log_h_curr[:, j] = draw_log_h_path(
                w[:, j], sigma_h=sigma_h_curr[j],
                h_current=log_h_curr[:, j],
                seed=int(rng.integers(0, 2**31 - 1)),
            )

        # 4. σ²_h per equation
        for j in range(n):
            dh = np.diff(log_h_curr[:, j])
            a_post = sigma_h_prior_a + len(dh) / 2.0
            b_post = sigma_h_prior_b + 0.5 * float(dh @ dh)
            sigma2_h_j = 1.0 / sp_stats.gamma.rvs(
                a=a_post, scale=1.0 / b_post, random_state=rng,
            )
            sigma_h_curr[j] = max(np.sqrt(sigma2_h_j), 1e-4)

        # 5. Q
        nu = np.diff(alpha_path, axis=0)
        S_Q = nu.T @ nu + Q_prior_scale
        df_Q = nu.shape[0] + Q_prior_df
        # atleast_2d: SciPy squeezes a 1-D inverse-Wishart draw to a scalar
        Q_curr = np.atleast_2d(sp_stats.invwishart.rvs(
            df=df_Q, scale=S_Q, random_state=rng,
        ))

        # 6. S_j per row
        for j in range(1, n):
            alpha_j = A_curr[:, j, :j]
            dalpha = np.diff(alpha_j, axis=0)
            S_S = dalpha.T @ dalpha + S_priors_scale[j - 1]
            df_S = dalpha.shape[0] + S_priors_df[j - 1]
            try:
                S_curr[j - 1] = np.atleast_2d(sp_stats.invwishart.rvs(
                    df=df_S, scale=S_S, random_state=rng,
                ))
            except np.linalg.LinAlgError:
                pass

        if it >= burn_in and (it - burn_in) % thin == 0:
            beta_path_store[stored] = beta_path
            A_path_store[stored] = A_curr.copy()
            log_h_store[stored] = log_h_curr
            sigma_h_store[stored] = sigma_h_curr
            Q_store[stored] = Q_curr
            Sigma_avg_t = np.zeros((n, n))
            for t in range(T_eff):
                Sigma_avg_t += _Sigma_from_A_h(A_curr[t], log_h_curr[t])
            Sigma_store[stored] = Sigma_avg_t / T_eff
            stored += 1

        if verbose and (it + 1) % 50 == 0:
            print(f"      Full-Primiceri Gibbs {it + 1}/{n_iter} "
                  f"(stored: {stored}/{n_kept})")

    # Drop zero-padded slots from failed draws (see fit_tvp_var note).
    if stored == 0:
        raise RuntimeError(
            "all post-burn-in Gibbs draws failed (LinAlgError) — no posterior")
    if stored < n_kept:
        import warnings
        warnings.warn(
            f"{n_kept - stored} of {n_kept} post-burn-in draws failed "
            f"(LinAlgError) and were dropped; posterior summaries use the "
            f"{stored} successful draws.", UserWarning, stacklevel=2)
        beta_path_store = beta_path_store[:stored]
        A_path_store = A_path_store[:stored]
        log_h_store = log_h_store[:stored]
        sigma_h_store = sigma_h_store[:stored]
        Q_store = Q_store[:stored]
        Sigma_store = Sigma_store[:stored]
        n_kept = stored
    beta_path_mean = beta_path_store.mean(axis=0)
    A_path_mean = A_path_store.mean(axis=0)
    log_h_mean = log_h_store.mean(axis=0)
    beta_last_mean = beta_path_mean[-1]
    Sigma_avg = Sigma_store.mean(axis=0)
    resid_at_last = Y_act - np.einsum("tk,kn->tn", X_act, beta_last_mean)

    fit_info = {
        "trend": trend,
        "method": "tvp_var_sv_full_primiceri",
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_draws_stored": int(n_kept),
        "training_sample": n_train,
        "Q_prior_scale": float(spec.Q_prior_scale),
        "S_prior_scale": float(S_prior_scale),
        "log_h_path_posterior_mean": log_h_mean.tolist(),
        "A_path_posterior_mean": A_path_mean.tolist(),
        "sigma_h_posterior_mean": sigma_h_store.mean(axis=0).tolist(),
    }

    return TVPVARFit(
        data=data,
        p=p,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=beta_last_mean,
        Sigma_u=Sigma_avg,
        residuals=resid_at_last,
        variable_names=list(data.columns),
        fit_info=fit_info,
        spec=spec,
        beta_path_draws=beta_path_store,
        Sigma_draws=Sigma_store,
        Q_draws=Q_store,
        beta_path_posterior_mean=beta_path_mean,
    )
