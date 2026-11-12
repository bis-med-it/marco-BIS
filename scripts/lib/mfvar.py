# =============================================================================
# EXPERIMENTAL
# The estimator in this file is less established than the core VAR code.
# It rests on the library's own tests on simulated data. Treat results with
# caution and confirm important ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
"""
Mixed-frequency Bayesian VAR (Schorfheide-Song 2015 / Bańbura-Modugno 2014).

A monthly-base VAR(p) where some variables are observed at quarterly
frequency. Quarterly observations enter the measurement equation as
Mariano-Murasawa (2003) flow-aggregation weights of five consecutive
monthly latent states.

Estimation: Gibbs sampling
  1. Given current (B, Sigma): build the state-space, draw a monthly
     state path conditional on observed mixed-frequency data via the
     Durbin-Koopman simulation smoother.
  2. Given the drawn state path: treat the (now-complete) monthly data
     as observed and draw (B, Sigma) from the NIW posterior of a
     Minnesota BVAR.
  3. Iterate, store post-burn-in draws.

Restrictions in the current implementation:
* Quarterly variables aggregate via Mariano-Murasawa weights
  ``[1/3, 2/3, 1, 2/3, 1/3]`` (log-growth flow form). Other aggregations
  (simple averaging for level flows, point-in-time for stocks) can be
  added by extending :func:`_build_measurement_design`.
* VAR lag order ``p ≤ state_lags`` (default 5) so the 5-month
  aggregation window fits in the companion state.
* Monthly-base data only (so each timestamp is a month).
* Units. The weights assume that a quarterly variable is a
  quarter-on-quarter growth rate and its monthly counterpart a
  month-on-month growth rate, both NOT annualised (for example
  ``100 * dlog``), with the quarterly value placed in the last month of
  the quarter. Annualised rates or log levels are accepted without error
  and give a wrong monthly series. The aggregation identity still holds
  in that case, so it does not reveal the mistake. Monthly variables may
  be in any units.
* Forecasts from the fitted object start from the posterior-mean monthly
  path, so their bands leave out the uncertainty about the latent
  monthly values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .bvar import MinnesotaPrior, build_minnesota_dummies
from .state_space import StateSpaceModel, simulation_smoother, mariano_murasawa_weights
from .var import (
    VARFit, build_design_matrices, extract_lag_matrices,
    ma_representation, max_eigenvalue_modulus,
)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass
class MFVARSpec:
    """Spec for a mixed-frequency VAR.

    Parameters
    ----------
    monthly_vars : list of str
        Variable names observed every month.
    quarterly_vars : list of str
        Variable names observed only at end-of-quarter months (March,
        June, September, December). Other months must contain NaN in
        the data DataFrame for these columns.
    lags : int
        VAR lag order. Must be ≤ ``state_lags`` (default 5).
    state_lags : int, default 5
        Number of lags carried in the state for the Mariano-Murasawa
        aggregation window. Five is the standard for quarterly-from-monthly
        log-growth-flow aggregation.
    """

    monthly_vars: list[str]
    quarterly_vars: list[str]
    lags: int
    state_lags: int = 5

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise ValueError(f"lags must be ≥ 1, got {self.lags}")
        if self.lags > self.state_lags:
            raise ValueError(
                f"lags ({self.lags}) cannot exceed state_lags ({self.state_lags})"
            )
        if not self.monthly_vars and not self.quarterly_vars:
            raise ValueError("at least one monthly or quarterly variable required")

    @property
    def n_monthly(self) -> int:
        return len(self.monthly_vars)

    @property
    def n_quarterly(self) -> int:
        return len(self.quarterly_vars)

    @property
    def n_vars(self) -> int:
        return self.n_monthly + self.n_quarterly

    @property
    def variable_order(self) -> list[str]:
        """Canonical ordering: monthlies first, then quarterlies."""
        return list(self.monthly_vars) + list(self.quarterly_vars)


# ---------------------------------------------------------------------------
# State-space construction
# ---------------------------------------------------------------------------


def _initial_state_prior(
    Y_obs: np.ndarray, spec: MFVARSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior on the first companion state, i.e. on the months before the sample.

    The first state stacks ``y_0`` and the ``L - 1`` months before it. Those
    months are not observed, so they need a prior. It is centred on each
    variable's first observation, on the monthly scale, with a variance that
    grows with the distance from the sample as for a random walk:
    ``y_{-k} ~ N(m_i, (k + 1) s_i^2)``.

    * Monthly variable: ``m_i`` is the first observed value and ``s_i^2`` the
      variance of its month-to-month changes.
    * Quarterly variable: ``m_i`` is the first quarterly value divided by the
      sum of the aggregation weights (the constant monthly value consistent
      with it) and ``s_i^2`` the variance of the quarterly values divided by
      the sum of squared weights.

    A prior centred on zero is wrong in the units of the data. The sampled
    pre-sample values enter the parameter regression, and values far from
    the data act as high-leverage outliers that push the coefficients on
    the longer lags to zero. Centring on the first observation makes the
    estimator invariant to adding a constant to a variable.

    Returns ``(mean, cov)`` for a state of dimension ``n * L``.
    """
    n, L, n_M = spec.n_vars, spec.state_lags, spec.n_monthly
    w = mariano_murasawa_weights()
    m = np.zeros(n)
    s2 = np.ones(n)
    for i in range(n):
        col = Y_obs[:, i]
        obs = col[~np.isnan(col)]
        if obs.size == 0:
            continue
        if i < n_M:
            m[i] = obs[0]
            d = np.diff(col)
            d = d[~np.isnan(d)]
            v = float(np.var(d)) if d.size > 1 else float(np.var(obs))
        else:
            m[i] = obs[0] / float(w.sum())
            v = float(np.var(obs)) / float(np.sum(w ** 2))
        if np.isfinite(v) and v > 0:
            s2[i] = v
    mean = np.tile(m, L)
    cov = np.diag(np.concatenate([(lag + 1) * s2 for lag in range(L)]))
    return mean, cov


def _build_state_space(
    spec: MFVARSpec, B: np.ndarray, Sigma: np.ndarray,
    include_const: bool = True,
    initial_state_mean: np.ndarray | None = None,
    initial_state_cov: np.ndarray | None = None,
) -> StateSpaceModel:
    """Build the state-space representation for the current MFVARSpec + parameters.

    State vector: ``α_t = [y_t', y_{t-1}', ..., y_{t-(L-1)}']'``
    where ``L = spec.state_lags`` and ``y_t`` is the (latent) monthly value
    vector ordered by ``spec.variable_order``.

    Measurement equation includes one row per variable. For monthly
    variables, the row picks the current value. For quarterly variables,
    the row computes the Mariano-Murasawa aggregation of the past 5
    monthly states. NaN rows in observed data are handled natively by
    statsmodels' Kalman filter.
    """
    n = spec.n_vars
    L = spec.state_lags
    p = spec.lags
    state_dim = n * L

    # --- Transition matrix T (state_dim × state_dim) ---
    # Top n rows: VAR(p) coefficients
    T = np.zeros((state_dim, state_dim))
    lag_mats = extract_lag_matrices(B, n, p)
    for j in range(p):
        T[:n, j * n:(j + 1) * n] = lag_mats[j]
    # Lower identity rows shift the lagged values
    for j in range(L - 1):
        T[(j + 1) * n:(j + 2) * n, j * n:(j + 1) * n] = np.eye(n)

    # --- Selection matrix R (state_dim × n) ---
    R = np.zeros((state_dim, n))
    R[:n, :] = np.eye(n)

    # --- State covariance Q (n × n) ---
    Q = Sigma

    # --- State intercept c (state_dim,) ---
    state_intercept = np.zeros(state_dim)
    if include_const:
        # The constant is the last column of B (the (k, n) coefficient matrix)
        const_row = B[-1, :]    # (n,)
        state_intercept[:n] = const_row

    # --- Measurement design Z (n × state_dim) ---
    Z = _build_measurement_design(spec)

    # --- Measurement noise H ---
    # Zero noise for exact aggregation; add tiny ridge for numerical stability
    H = np.eye(n) * 1e-10

    # --- Initial conditions ---
    # The estimator passes the prior built by _initial_state_prior (centred
    # on the first observations). The N(0, I) fallback is kept only for
    # direct calls with data that are centred near zero.
    if initial_state_cov is None:
        initial_state_cov = np.eye(state_dim) * 1.0

    return StateSpaceModel(
        Z=Z, H=H, T=T, R=R, Q=Q,
        state_intercept=state_intercept,
        initial_state_mean=initial_state_mean,
        initial_state_cov=initial_state_cov,
    )


def _build_measurement_design(spec: MFVARSpec) -> np.ndarray:
    """Construct the time-invariant Z matrix mapping state to observations.

    For monthly variables: identity row selecting the current value.
    For quarterly variables: Mariano-Murasawa weighted sum of the past
    5 monthly states.

    Layout: variable ``i`` in the data corresponds to row ``i`` of Z.
    Monthly variables come first (indices 0 .. n_M-1), then quarterly
    variables (indices n_M .. n-1).
    """
    n = spec.n_vars
    n_M = spec.n_monthly
    L = spec.state_lags
    state_dim = n * L
    Z = np.zeros((n, state_dim))

    # Monthly variables
    for i in range(n_M):
        Z[i, i] = 1.0    # selects current value of variable i

    # Quarterly variables
    if spec.n_quarterly > 0:
        w = mariano_murasawa_weights()    # [1/3, 2/3, 1, 2/3, 1/3]
        if L < len(w):
            raise ValueError(
                f"state_lags={L} too small for Mariano-Murasawa "
                f"(needs at least {len(w)})"
            )
        for q_idx in range(spec.n_quarterly):
            var_idx = n_M + q_idx
            for lag in range(len(w)):
                Z[var_idx, lag * n + var_idx] = w[lag]

    return Z


# ---------------------------------------------------------------------------
# MFVAR fit object
# ---------------------------------------------------------------------------


@dataclass
class MFVARFit(VARFit):
    """Fitted MF-VAR.

    Inherits :class:`VARFit` for compatibility with the identification
    routines (``cholesky_irf``, etc.). The ``B_hat`` and ``Sigma_u``
    fields hold the POSTERIOR MEAN of the Gibbs draws. The full draws
    are stored separately and ``parameter_draws`` is overridden to return
    them directly.

    Attributes
    ----------
    spec : MFVARSpec
    prior : MinnesotaPrior
    gibbs_B_draws : (n_draws, k, n) ndarray
    gibbs_Sigma_draws : (n_draws, n, n) ndarray
    state_path_posterior_mean : (T, n) ndarray
        Posterior mean of the latent monthly state path (useful for
        nowcasting and inspecting interpolation of quarterly variables).
    """

    spec: MFVARSpec = None
    prior: MinnesotaPrior = field(default_factory=MinnesotaPrior)
    gibbs_B_draws: np.ndarray = field(default=None, repr=False)
    gibbs_Sigma_draws: np.ndarray = field(default=None, repr=False)
    state_path_posterior_mean: np.ndarray = field(default=None, repr=False)

    def parameter_draws(
        self, n_draws: int, seed: int | None = None,
        require_stable: bool = False, eig_tol: float = 1.01,
        burn_in: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return saved Gibbs draws (not fresh asymptotic draws).

        ``BVARFit.parameter_draws`` would re-sample from the NIW
        posterior; here we have actual Gibbs draws and just resample
        from them.
        """
        if self.gibbs_B_draws is None:
            raise RuntimeError("MFVARFit has no stored Gibbs draws")
        rng = np.random.default_rng(seed)
        n_avail = len(self.gibbs_B_draws)
        if n_draws > n_avail:
            raise ValueError(
                f"requested {n_draws} draws but only {n_avail} are stored; "
                f"re-run fit_mfvar_bayesian with larger n_iter"
            )
        idx = rng.choice(n_avail, size=n_draws, replace=False)
        out = []
        for i in idx:
            B = self.gibbs_B_draws[i]
            Sigma = self.gibbs_Sigma_draws[i]
            if require_stable:
                lag_mats = extract_lag_matrices(B, self.n_vars, self.p)
                if max_eigenvalue_modulus(lag_mats) >= eig_tol:
                    continue
            out.append((B, Sigma))
        return out


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_mfvar_bayesian(
    data: pd.DataFrame,
    spec: MFVARSpec,
    prior: MinnesotaPrior | None = None,
    n_iter: int = 1000,
    burn_in: int = 200,
    thin: int = 1,
    seed: int | None = None,
    trend: str = "c",
    verbose: bool = False,
) -> MFVARFit:
    """Estimate a mixed-frequency BVAR via Gibbs sampling.

    Parameters
    ----------
    data : pd.DataFrame
        Monthly-indexed DataFrame containing all monthly and quarterly
        variables. Quarterly columns must have NaN at intra-quarter
        months (only end-of-quarter months — March/June/September/
        December — should contain the quarterly value). Quarterly
        variables must be quarter-on-quarter growth rates, not
        annualised (see the module notes on units).
    spec : MFVARSpec
    prior : MinnesotaPrior, optional
        Defaults to ``MinnesotaPrior()``. SYMMETRIC (natural-conjugate)
        path only — an asymmetric prior raises ``NotImplementedError``:
        the Gibbs sampler holds the prior's dummy observations fixed
        across iterations, which the per-equation path cannot do.
    n_iter : int, default 1000
        Total Gibbs iterations (including burn-in).
    burn_in : int, default 200
        Iterations to discard before storing draws.
    thin : int, default 1
        Keep every ``thin``-th post-burn-in draw.
    seed : int, optional
    trend : str, default "c"
    verbose : bool, default False
        Print progress every 100 iterations.

    Returns
    -------
    MFVARFit

    Notes
    -----
    Gibbs coherence: the Minnesota dummies are built ONCE from the
    initial interpolated completion and held fixed across iterations
    (a prior rebuilt from each latent completion would be
    data-dependent), and the parameter block regresses over the SAME
    transitions the simulation smoother scores — the sampled prehistory
    rows from the first companion state are prepended before the BVAR
    step, so the two blocks are conditionals of one joint model.

    Start of the sample: the months before the first observation enter
    the first companion state. Their prior is centred on each variable's
    first observation, with a variance that grows with the distance from
    the sample (see ``_initial_state_prior``), so adding a constant to a
    variable leaves the lag coefficients unchanged.
    """
    prior = prior or MinnesotaPrior()
    rng = np.random.default_rng(seed)

    # Validate
    var_order = spec.variable_order
    missing = [v for v in var_order if v not in data.columns]
    if missing:
        raise ValueError(f"data missing required columns: {missing}")

    # Reorder data
    Y_obs = data[var_order].values.astype(float)   # (T, n) with NaN in quarterly slots
    T = Y_obs.shape[0]
    n = spec.n_vars
    p = spec.lags
    if T <= p + 5:
        raise ValueError(f"data too short ({T}) for lags={p} and 5-lag aggregation")

    # Initial state path: simple interpolation of NaN values
    Y_init = pd.DataFrame(Y_obs, columns=var_order).interpolate(
        method="linear", limit_direction="both"
    ).values

    # Initial parameter draw: standard Minnesota BVAR on the interpolated data.
    # The Minnesota dummies are built ONCE here, from the initial interpolated
    # completion, and reused in every Gibbs iteration below — rebuilding them
    # from each latent completion would make the prior depend on the sampled
    # data and invalidate the sampler.
    from .bvar import fit_minnesota_bvar
    if prior.asymmetric:
        raise NotImplementedError(
            "fit_mfvar_bayesian requires the symmetric (natural-conjugate) "
            "Minnesota prior: the Gibbs sampler holds the prior's dummy "
            "observations fixed across iterations, which the asymmetric "
            "(per-equation) path does not support"
        )
    init_df = pd.DataFrame(Y_init, columns=var_order)
    fixed_dummies = build_minnesota_dummies(
        Y_init, p, prior, include_const=(trend == "c"))
    bvar0 = fit_minnesota_bvar(init_df, lags=p, prior=prior, trend=trend,
                               dummies=fixed_dummies)
    B_curr = bvar0.B_hat.copy()
    Sigma_curr = bvar0.Sigma_u.copy()

    # Prior on the months before the sample: centred on the first
    # observations, fixed across iterations (see _initial_state_prior).
    init_mean, init_cov = _initial_state_prior(Y_obs, spec)

    # Storage for post-burn-in draws
    n_kept = len(range(burn_in, n_iter, thin))  # matches the storage schedule below
    if n_kept <= 0:
        raise ValueError("n_iter must exceed burn_in")
    k = bvar0.k

    B_store = np.zeros((n_kept, k, n))
    Sigma_store = np.zeros((n_kept, n, n))
    state_path_sum = np.zeros((T, n))
    state_path_count = 0

    # Gibbs iterations
    stored = 0
    for it in range(n_iter):
        # 1. Draw state path given current (B, Sigma)
        ss_model = _build_state_space(spec, B_curr, Sigma_curr,
                                       include_const=(trend == "c"),
                                       initial_state_mean=init_mean,
                                       initial_state_cov=init_cov)
        draws = simulation_smoother(ss_model, Y_obs, n_draws=1,
                                     seed=rng.integers(0, 2**31 - 1))
        state_path = draws[0]              # (T, state_dim)
        Y_complete = state_path[:, :n]    # (T, n) — current monthly values

        # 2. Draw (B, Sigma) given completed Y — with the FIXED dummies, so
        # the prior does not move with the latent completion.
        #
        # The parameter regression must cover the SAME transitions the
        # smoother scored: state_0 -> state_1 onward. state_path[0] carries
        # the sampled prehistory (lagged blocks y_{-1}, ..., y_{-(p-1)}), so
        # prepend those rows before fitting — otherwise the first p-1
        # transition contributions are dropped and the two Gibbs blocks are
        # conditionals of DIFFERENT joint models (the omitted terms depend
        # on both the parameters and the latent values).
        if p > 1:
            prehistory = np.vstack(
                [state_path[0, j * n:(j + 1) * n] for j in range(p - 1, 0, -1)]
            )
            Y_ext = np.vstack([prehistory, Y_complete])
        else:
            Y_ext = Y_complete
        complete_df = pd.DataFrame(Y_ext, columns=var_order)
        bvar_iter = fit_minnesota_bvar(complete_df, lags=p, prior=prior, trend=trend,
                                       dummies=fixed_dummies)
        param_draws = bvar_iter.parameter_draws(
            n_draws=1, seed=rng.integers(0, 2**31 - 1),
            require_stable=False,
        )
        B_curr, Sigma_curr = param_draws[0]

        # Store
        if it >= burn_in and (it - burn_in) % thin == 0:
            B_store[stored] = B_curr
            Sigma_store[stored] = Sigma_curr
            state_path_sum += Y_complete
            state_path_count += 1
            stored += 1

        if verbose and (it + 1) % 100 == 0:
            print(f"      Gibbs iter {it + 1}/{n_iter} "
                  f"(stored: {stored}/{n_kept})")

    # Aggregate posterior means
    B_post_mean = B_store.mean(axis=0)
    Sigma_post_mean = Sigma_store.mean(axis=0)
    state_path_mean = state_path_sum / max(state_path_count, 1)

    # Build a VARFit-compatible object using the posterior-mean parameters
    # plus the imputed (posterior-mean) state path for diagnostics
    Y_eff = state_path_mean
    state_df = pd.DataFrame(Y_eff, columns=var_order, index=data.index)
    Y_act, X_act = build_design_matrices(Y_eff, p, include_const=(trend == "c"))
    resid = Y_act - X_act @ B_post_mean

    fit_info = {
        "trend": trend,
        "method": "mfvar_bayesian_gibbs",
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_draws_stored": n_kept,
        "prior": {
            "lambda_1": prior.lambda_1,
            "lambda_2": prior.lambda_2,
            "lambda_3": prior.lambda_3,
            "lambda_4": prior.lambda_4,
            "mu": prior.mu,
            "delta": prior.delta,
        },
    }

    return MFVARFit(
        data=state_df,
        p=p,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=B_post_mean,
        Sigma_u=Sigma_post_mean,
        residuals=resid,
        variable_names=list(var_order),
        fit_info=fit_info,
        spec=spec,
        prior=prior,
        gibbs_B_draws=B_store,
        gibbs_Sigma_draws=Sigma_store,
        state_path_posterior_mean=state_path_mean,
    )
