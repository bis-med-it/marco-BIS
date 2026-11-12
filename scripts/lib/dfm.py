"""
Bayesian dynamic factor model with arbitrary missing data, after
Bańbura & Modugno (2014). BM estimate the model by maximum likelihood
with a modified EM algorithm; the Gibbs sampler here is this
repository's substitution (see the banbura_modugno_2014_bdfm README).

Model:
    y_t = Λ f_t + e_t,            e_t ~ N(0, diag(R))
    f_t = Φ f_{t-1} + u_t,        u_t ~ N(0, Q)

with ``y_t`` an ``n × 1`` vector of observed variables (possibly with
NaN values for missing observations / ragged edge), ``f_t`` an ``r × 1``
vector of latent factors, ``Λ`` the ``n × r`` loadings, ``R`` a diagonal
``n × n`` idiosyncratic covariance, ``Φ`` an ``r × r`` factor AR(1)
transition, and ``Q`` an ``r × r`` factor-innovation covariance.

Gibbs sampler:
  1. Given ``(Λ, R, Φ, Q)``: draw the latent factor path ``f_{1:T}`` via
     the Durbin-Koopman simulation smoother on the state-space form.
  2. Given ``f_{1:T}``: draw each row of ``Λ`` (univariate regression
     of ``y_j`` on factors) and each diagonal element of ``R``
     (inverse-gamma posterior on the residual variance).
  3. Given ``f_{1:T}``: draw ``Φ`` (matrix-normal on factor AR) and
     ``Q`` (inverse-Wishart on factor innovations).

NaN values in ``y`` are handled natively by ``statsmodels``' Kalman filter
during the simulation-smoother step. In steps 2-3 we work with the
drawn complete factors so the parameter conditionals are standard.

Initialization uses principal components on a NaN-imputed copy of the data,
following the BM 2014 convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .state_space import StateSpaceModel, simulation_smoother
from .factors import _pc_factors


# ---------------------------------------------------------------------------
# Specification + fit object
# ---------------------------------------------------------------------------


@dataclass
class DFMSpec:
    """Bayesian DFM specification.

    Parameters
    ----------
    n_factors : int
        Number of latent factors ``r``.
    standardize : bool, default True
        Whether to z-score each column of ``y`` before estimation.
    """

    n_factors: int
    standardize: bool = True

    def __post_init__(self) -> None:
        if self.n_factors < 1:
            raise ValueError(f"n_factors must be ≥ 1, got {self.n_factors}")


@dataclass
class DFMFit:
    """Fitted Bayesian DFM.

    Attributes
    ----------
    spec : DFMSpec
    factor_path_posterior_mean : (T, r) ndarray
    loadings_posterior_mean : (n, r) ndarray
    factor_AR_posterior_mean : (r, r) ndarray
    Q_posterior_mean : (r, r) ndarray
    R_posterior_mean : (n,) ndarray
        Diagonal idiosyncratic variances.
    variable_names : list[str]
    means, stds : (n,) ndarrays
        Used for standardisation (only meaningful if ``spec.standardize`` was True).
    """

    spec: DFMSpec
    factor_path_posterior_mean: np.ndarray
    loadings_posterior_mean: np.ndarray
    factor_AR_posterior_mean: np.ndarray
    Q_posterior_mean: np.ndarray
    R_posterior_mean: np.ndarray
    variable_names: list[str]
    means: np.ndarray
    stds: np.ndarray
    factor_path_draws: np.ndarray = field(default=None, repr=False)
    # Posterior mean of the common component Λ F' accumulated WITHIN each
    # draw, shape (T, n), on the standardised scale when spec.standardize.
    # Use THIS for imputation/reconstruction: factors and loadings are
    # identified only up to sign/rotation across draws, so
    # mean(F) @ mean(Λ)' is neither the posterior-mean common component nor
    # invariant to equivalent relabelings (equivalent draws (F, Λ) and
    # (−F, −Λ) would cancel).
    common_component_posterior_mean: np.ndarray = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        return self.spec.n_factors

    @property
    def n_vars(self) -> int:
        return len(self.variable_names)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_dfm_bayesian(
    data: pd.DataFrame,
    spec: DFMSpec,
    n_iter: int = 500,
    burn_in: int = 100,
    thin: int = 1,
    seed: int | None = None,
    verbose: bool = False,
) -> DFMFit:
    """Estimate a Bayesian DFM via Gibbs sampling.

    Parameters
    ----------
    data : pd.DataFrame
        Observations. NaN entries are treated as missing (Kalman handles
        them via standard missing-observation conventions).
    spec : DFMSpec
    n_iter : int
    burn_in : int
    thin : int
    seed : int, optional
    verbose : bool

    Returns
    -------
    DFMFit
        For imputation or reconstruction use
        ``common_component_posterior_mean`` (accumulated WITHIN each
        draw) — ``factor_path_posterior_mean @
        loadings_posterior_mean.T`` is not invariant to per-draw
        sign/rotation relabelling of the factors.
    """
    if data.shape[0] < 20:
        raise ValueError(f"data too short ({data.shape[0]} rows)")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be pd.DataFrame, got {type(data).__name__}")

    rng = np.random.default_rng(seed)
    Y_raw = data.values.astype(float)
    T, n = Y_raw.shape
    r = spec.n_factors

    # --- Standardisation ---
    # Use column-wise mean and std, ignoring NaNs
    means = np.nanmean(Y_raw, axis=0)
    stds = np.nanstd(Y_raw, axis=0, ddof=1)
    stds[stds < 1e-12] = 1.0
    if spec.standardize:
        Y_std = (Y_raw - means) / stds
    else:
        Y_std = Y_raw.copy()

    # --- Initialise factors via PC on NaN-imputed copy ---
    Y_imp = pd.DataFrame(Y_std, columns=data.columns).interpolate(
        method="linear", limit_direction="both"
    ).fillna(0).values
    F_init, Lambda_init = _pc_factors(Y_imp, K=r)

    # --- Initial parameter values ---
    Lambda_curr = Lambda_init.copy()                                  # (n, r)
    # Idiosyncratic variances: from residuals of PC fit
    eps_init = Y_imp - F_init @ Lambda_init.T
    R_curr = np.var(eps_init, axis=0, ddof=1)                         # (n,)
    R_curr = np.maximum(R_curr, 1e-6)
    # Factor AR(1) initial: OLS on factor levels
    Phi_curr = _ols_var1(F_init)                                       # (r, r)
    # Factor innovation covariance
    F_resid = F_init[1:] - F_init[:-1] @ Phi_curr.T
    Q_curr = np.atleast_2d(np.cov(F_resid.T, ddof=1))                  # (r, r)
    if Q_curr.shape != (r, r):
        Q_curr = np.eye(r) * float(Q_curr.flatten()[0])

    n_kept = len(range(burn_in, n_iter, thin))  # matches the storage schedule below
    if n_kept <= 0:
        raise ValueError("n_iter must exceed burn_in")

    F_store = np.zeros((n_kept, T, r))
    Lambda_store = np.zeros((n_kept, n, r))
    common_sum = np.zeros((T, n))   # per-draw Λ F' accumulation (sign-invariant)
    R_store = np.zeros((n_kept, n))
    Phi_store = np.zeros((n_kept, r, r))
    Q_store = np.zeros((n_kept, r, r))

    # --- Priors (weakly informative) ---
    # Λ rows: N(0, I_r * 10)
    Lambda_prior_var = 10.0
    # R diag: IG(a=1, b=1) — diffuse
    R_prior_a, R_prior_b = 1.0, 1.0
    # Φ: N(0, I * 10), Q: IW(df = r + 2, scale = I_r)
    Phi_prior_var = 10.0
    Q_prior_df = r + 2
    Q_prior_scale = np.eye(r) * 0.1

    stored = 0
    for it in range(n_iter):
        # --- Step 1: draw factors via simulation smoother ---
        ss_model = _build_dfm_state_space(
            Lambda=Lambda_curr, R_diag=R_curr, Phi=Phi_curr, Q=Q_curr,
        )
        try:
            draws = simulation_smoother(
                ss_model, Y_std, n_draws=1,
                seed=rng.integers(0, 2**31 - 1),
            )
        except np.linalg.LinAlgError:
            continue
        F_curr = draws[0]                                              # (T, r)

        # --- Step 2: draw Λ_j and R_j for each j ---
        for j in range(n):
            mask = ~np.isnan(Y_std[:, j])
            if mask.sum() < r + 2:
                continue
            F_obs = F_curr[mask]
            y_obs = Y_std[mask, j]
            # Posterior for Λ_j with N(0, prior_var * I) prior, N(0, R_j) likelihood
            FtF = F_obs.T @ F_obs
            Fty = F_obs.T @ y_obs
            V_inv = FtF / R_curr[j] + np.eye(r) / Lambda_prior_var
            V = np.linalg.inv(V_inv)
            m = V @ Fty / R_curr[j]
            try:
                L = np.linalg.cholesky(V)
            except np.linalg.LinAlgError:
                continue
            Lambda_curr[j] = m + L @ rng.standard_normal(r)

            # Posterior for R_j: IG(a_post, b_post)
            resid = y_obs - F_obs @ Lambda_curr[j]
            a_post = R_prior_a + mask.sum() / 2.0
            b_post = R_prior_b + 0.5 * float(resid @ resid)
            R_curr[j] = max(1.0 / sp_stats.gamma.rvs(
                a=a_post, scale=1.0 / b_post, random_state=rng,
            ), 1e-8)

        # --- Step 3: draw Φ and Q ---
        # Φ from regression of f_t on f_{t-1}
        F_lhs = F_curr[1:]
        F_rhs = F_curr[:-1]
        FtF = F_rhs.T @ F_rhs
        FtF_inv = np.linalg.inv(
            FtF + np.eye(r) / Phi_prior_var,
        )
        # B = (F_rhs'F_rhs + prior)^{-1} F_rhs'F_lhs is the REGRESSION matrix
        # (regressors x equations), i.e. B = Phi' for f_t = Phi f_{t-1}.
        B_mean = FtF_inv @ F_rhs.T @ F_lhs             # (r, r) = Phi'
        # Matric-normal draw of B: row-cov FtF_inv (regressors), col-cov Q
        L_FtF_inv = np.linalg.cholesky(FtF_inv)
        try:
            L_Q = np.linalg.cholesky(Q_curr)
        except np.linalg.LinAlgError:
            L_Q = np.eye(r) * 0.1
        Z = rng.standard_normal((r, r))
        B_draw = B_mean + L_FtF_inv @ Z @ L_Q.T
        # Store Phi itself (transpose of the regression matrix) — matching
        # _ols_var1's orientation and the downstream consumers
        # (transition T=Phi; residual F_lhs - F_rhs @ Phi').
        Phi_curr = B_draw.T

        # Q from IW. The conditional given Φ must also carry the Φ prior's
        # contribution: B = Φ' ~ MN(0, V0, Q) with V0 = Phi_prior_var·I_r has
        # COLUMN covariance Q, so p(Q | Φ, F) picks up B'V0⁻¹B in the scale
        # and r extra degrees of freedom — dropping them mis-centres Q.
        F_resid_new = F_lhs - F_rhs @ Phi_curr.T
        B_curr = Phi_curr.T
        S_Q = (F_resid_new.T @ F_resid_new + Q_prior_scale
               + (B_curr.T @ B_curr) / Phi_prior_var)
        df_Q = (T - 1) + Q_prior_df + r
        # atleast_2d: SciPy squeezes a 1-D inverse-Wishart draw to a scalar,
        # which would break the next iteration's StateSpaceModel (one-factor case)
        Q_curr = np.atleast_2d(sp_stats.invwishart.rvs(
            df=df_Q, scale=S_Q, random_state=rng,
        ))

        # --- Store ---
        if it >= burn_in and (it - burn_in) % thin == 0:
            F_store[stored] = F_curr
            Lambda_store[stored] = Lambda_curr
            R_store[stored] = R_curr
            Phi_store[stored] = Phi_curr
            Q_store[stored] = Q_curr
            # Reconstruct the common component WITHIN the draw — the only
            # average that is invariant to per-draw sign/rotation relabels
            common_sum += F_curr @ Lambda_curr.T
            stored += 1

        if verbose and (it + 1) % 50 == 0:
            print(f"      DFM Gibbs {it + 1}/{n_iter} "
                  f"(stored: {stored}/{n_kept})")

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
        F_store = F_store[:stored]
        Lambda_store = Lambda_store[:stored]
        R_store = R_store[:stored]
        Phi_store = Phi_store[:stored]
        Q_store = Q_store[:stored]
        n_kept = stored

    return DFMFit(
        spec=spec,
        factor_path_posterior_mean=F_store.mean(axis=0),
        common_component_posterior_mean=common_sum / n_kept,
        loadings_posterior_mean=Lambda_store.mean(axis=0),
        factor_AR_posterior_mean=Phi_store.mean(axis=0),
        Q_posterior_mean=Q_store.mean(axis=0),
        R_posterior_mean=R_store.mean(axis=0),
        variable_names=list(data.columns),
        means=means,
        stds=stds,
        factor_path_draws=F_store,
        metadata={
            "method": "bayesian_dfm_gibbs",
            "n_iter": n_iter,
            "burn_in": burn_in,
            "thin": thin,
            "n_draws_stored": int(n_kept),
            "standardize": bool(spec.standardize),
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_dfm_state_space(
    Lambda: np.ndarray, R_diag: np.ndarray, Phi: np.ndarray, Q: np.ndarray,
) -> StateSpaceModel:
    """Construct the state-space for the DFM at current parameter values.

    State α_t = f_t (r,).
    """
    r = Lambda.shape[1]
    return StateSpaceModel(
        Z=Lambda,
        H=np.diag(R_diag),
        T=Phi,
        R=np.eye(r),
        Q=Q,
        initial_state_cov=np.eye(r) * 10.0,
    )


def _ols_var1(F: np.ndarray) -> np.ndarray:
    """OLS VAR(1) on factor matrix F (T, r). Returns the AR matrix Φ (r, r)."""
    T, r = F.shape
    if T < 2:
        return np.eye(r) * 0.5
    F_lhs = F[1:]
    F_rhs = F[:-1]
    FtF = F_rhs.T @ F_rhs
    if np.abs(np.linalg.det(FtF)) < 1e-12:
        return np.eye(r) * 0.5
    return np.linalg.solve(FtF, F_rhs.T @ F_lhs).T
