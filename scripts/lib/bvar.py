"""
Bayesian VAR with Minnesota prior, estimated via the dummy-observation
(Theil mixed-estimation) approach.

The Minnesota prior shrinks the VAR coefficients toward a random-walk
representation for each variable. Implementation follows the standard form
in Bańbura, Giannone & Reichlin (2010) and Del Negro & Schorfheide (2011):
construct dummy observations encoding the prior, stack them on top of the
data, and run OLS on the augmented system. The result is the posterior
mean of the Normal-Inverse-Wishart posterior.

Public API:

* :class:`MinnesotaPrior` — prior hyperparameters.
* :func:`build_minnesota_dummies` — dummy-observation matrices encoding the prior.
* :class:`BVARFit` — subclass of ``VARFit`` whose ``parameter_draws()``
  samples from the NIW posterior instead of the asymptotic distribution.
* :func:`fit_minnesota_bvar` — factory.

Once you have a ``BVARFit``, plug it straight into the existing
identification routines:

>>> bvar = fit_minnesota_bvar(data, lags=4, prior=MinnesotaPrior(lambda_1=0.1))
>>> irf  = cholesky_irf(bvar, horizon=40, bands="parameter_draws", n_draws=2000)

The ``cholesky_irf`` call detects the ``BVARFit`` and uses posterior draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .var import (
    VARFit,
    build_design_matrices,
    extract_lag_matrices,
    max_eigenvalue_modulus,
)


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------


@dataclass
class MinnesotaPrior:
    """Hyperparameters for the Minnesota / Sims-Zha-style Bayesian VAR prior.

    Defaults follow ``config.yaml::bvar.minnesota`` and the
    Bańbura-Giannone-Reichlin (2010) convention.

    Parameters
    ----------
    lambda_1 : float
        Overall tightness. Smaller = tighter shrinkage to the random walk.
        Default 0.1; BGR (2010) suggest tighter values for larger systems.
    lambda_2 : float
        Cross-variable tightness. Cross-variable lag coefficients are shrunk
        ``lambda_2`` times tighter than own-variable lags. Default 0.5.

        **Only takes effect when** ``asymmetric=True``. Under the default
        symmetric / natural-conjugate BGR (2010) formulation the prior
        covariance has Kronecker structure ``Sigma ⊗ V_0``, where
        ``V_0 = (X_d'X_d)^{-1}`` is the regressor covariance factor shared
        by all equations (its inverse is the prior precision), so an
        asymmetric own-vs-cross
        scaling cannot be encoded and ``lambda_2`` is a no-op there.
    lambda_3 : float
        Lag decay. Standard deviation of lag-``l`` coefficients scales as ``1/l^lambda_3``.
        Default 1.0 (the standard Minnesota choice).
    lambda_4 : float
        Constant-term tightness. Larger = looser. Default 100 (effectively flat).
    mu : float
        Strength of the sum-of-coefficients prior (Doan-Litterman-Sims 1984).
        Encodes the prior that variables are persistent and their lag
        coefficients sum to (close to) the unit root. ``mu = 0`` disables.
        Default 1.0. Each row is scaled by ``rw_deltas``, so a variable
        with ``rw_deltas = 0`` gets no sum-of-coefficients row.
    delta : float
        Strength of the dummy-initial-observation prior (single unit root
        prior — Sims 1992; Sims-Zha 1998). ``delta = 0`` disables.
        Default 1.0.
    rw_deltas : sequence of float, optional
        Prior means for the own-lag-1 coefficient of each variable. Default
        ``[1, 1, ..., 1]`` (random walk for every variable). Use 0 for
        stationary I(0) variables (white-noise prior).
    asymmetric : bool, default False
        If ``True``, fit via equation-by-equation Ridge with per-equation
        precision matrices — the textbook Litterman (1986) / Sims-Zha (1998)
        prior, where cross-variable coefficients are shrunk by an extra
        ``lambda_2`` factor relative to own-variable ones. This is the
        intended semantics of ``lambda_2``.

        Tradeoffs: the asymmetric mode treats ``Sigma_u`` as a fixed plug-in
        (residual covariance from the per-equation fits) rather than drawing
        it from an Inverse-Wishart posterior, so :meth:`BVARFit.parameter_draws`
        in this mode samples ``beta`` from per-equation Gaussian posteriors
        with ``Sigma_u`` held constant. :func:`log_marginal_data_density`
        is not defined in this mode and raises ``NotImplementedError``.
    lambda_bex : float, default 0.001
        Block-exogeneity shrinkage. When ``fit_minnesota_bvar``
        is called with ``exogenous_block``, the prior std on an excluded
        variable's lags in an exogenous equation is multiplied by ``lambda_bex``
        — pinning that coefficient at ~0. Smaller = stricter exclusion.
    """

    lambda_1: float = 0.1
    lambda_2: float = 0.5
    lambda_3: float = 1.0
    lambda_4: float = 100.0
    mu: float = 1.0
    delta: float = 1.0
    rw_deltas: Sequence[float] | None = None
    asymmetric: bool = False
    lambda_bex: float = 0.001


# ---------------------------------------------------------------------------
# Dummy-observation construction
# ---------------------------------------------------------------------------


def _presample_ar_sigmas(
    Y: np.ndarray, p: int, presample_length: int | None = None,
) -> np.ndarray:
    """Per-variable AR(p) residual standard deviations.

    Following Sims-Zha (1998) and BGR (2010): for each variable, fit a
    univariate AR(p) on a presample of length ``presample_length`` (default
    ``max(4*p + 10, T // 4)``) and take the residual standard deviation.

    The presample is the first chunk of the data so that the dummy scaling
    encodes "prior information" before the bulk of the sample. If the data
    are too short for the default presample, the routine falls back to the
    AR(p) residual on the entire series.
    """
    T, n = Y.shape
    if presample_length is None:
        presample_length = min(max(4 * p + 10, T // 4), T)
    if presample_length <= p + 1:
        presample_length = T   # fall back to full-sample AR(p)

    sigmas = np.empty(n)
    for i in range(n):
        y = Y[:presample_length, i]
        T_pre = len(y)
        # Build AR(p) design matrix
        Y_act = y[p:]
        X_rows = []
        for t in range(p, T_pre):
            row = [y[t - lag] for lag in range(1, p + 1)] + [1.0]
            X_rows.append(row)
        X_act = np.asarray(X_rows)
        if X_act.shape[0] <= X_act.shape[1]:
            # Insufficient observations — fall back to series std
            sigmas[i] = max(np.std(y, ddof=1), 1e-10)
            continue
        beta, *_ = np.linalg.lstsq(X_act, Y_act, rcond=None)
        resid = Y_act - X_act @ beta
        sigmas[i] = max(np.std(resid, ddof=1), 1e-10)
    return sigmas


def build_minnesota_dummies(
    Y: np.ndarray,
    p: int,
    prior: MinnesotaPrior,
    include_const: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct dummy-observation matrices encoding the Minnesota prior.

    The returned ``(Yd, Xd)`` are stacked on top of the actual ``(Y, X)``
    matrices in the Theil mixed-estimation step. Layout of ``Xd`` matches
    ``var.build_design_matrices`` exactly (lags then constant).

    Note on ``lambda_2``: the symmetric (natural-conjugate) dummy form has no
    explicit cross-variable knob — cross-variable scaling arises only through
    the sigma_i/sigma_j ratios of the Kronecker structure, and ``lambda_2`` is
    inert here. It becomes active in the asymmetric (Litterman) path.
    """
    T, n = Y.shape
    if T <= p:
        raise ValueError(f"Need T > p; got T={T}, p={p}")

    sigmas = _presample_ar_sigmas(Y, p)
    y_bar = Y[:p].mean(axis=0)
    rw_deltas = (
        np.ones(n)
        if prior.rw_deltas is None
        else np.asarray(prior.rw_deltas, dtype=float)
    )
    if rw_deltas.shape != (n,):
        raise ValueError(f"rw_deltas must have shape ({n},), got {rw_deltas.shape}")

    n_const = 1 if include_const else 0
    k = n * p + n_const

    blocks_Y: list[np.ndarray] = []
    blocks_X: list[np.ndarray] = []

    # (a) Own-lag tightness — Sims-Zha (1998) / BGR (2010) convention.
    #
    # Prior:    A_{ii}^{lag} ~ N(rw_delta_i [if lag==1] or 0, prior_var)
    # prior_var = (lambda_1 / lag^lambda_3)^2   (own-lag)
    #
    # In dummy-observation form (Theil mixed estimation), an OLS row
    # (y_d, x_d) with x_d having a single non-zero entry s at column c says:
    #   "beta_c has prior precision s^2 and prior mean y_d / s"
    # So the per-equation scale s on equation i is:
    #   s = sigma_i / (lambda_1 / lag^lambda_3) = sigma_i * lag^lambda_3 / lambda_1
    # The sigma_i factor scales the dummy to the equation's residual variance
    # so the dummies enter on the same footing as actual observations.
    #
    # Convention: smaller lambda_1 = TIGHTER prior (s is large, dominates data).
    for lag in range(1, p + 1):
        Yd = np.zeros((n, n))
        Xd = np.zeros((n, k))
        for i in range(n):
            scale = sigmas[i] * (lag ** prior.lambda_3) / prior.lambda_1
            target_mean = rw_deltas[i] if lag == 1 else 0.0
            Yd[i, i] = scale * target_mean
            Xd[i, (lag - 1) * n + i] = scale
        blocks_Y.append(Yd)
        blocks_X.append(Xd)

    # (b) Sigma-prior block — BGR (2010, JAE) eq. 5, middle block: n rows
    # with Yd = diag(sigma), Xd = 0. No coefficient matrix can fit these
    # rows, so the dummy-residual scale satisfies S_0 >= diag(sigma^2) > 0 —
    # this is what makes the implied inverse-Wishart prior on Sigma PROPER
    # (nu_0 >= n) and anchors its scale. Without it, the exact random-walk
    # B fits every other dummy row exactly under the default rw_deltas = 1,
    # S_0 = 0, and the closed-form marginal data density is undefined.
    # These rows do NOT move the posterior mean of B (Xd = 0 contributes
    # nothing to X'X or X'Y); they enter only the Sigma scale and df.
    blocks_Y.append(np.diag(sigmas))
    blocks_X.append(np.zeros((n, k)))

    # Cross-variable tightness lambda_2: inert in this symmetric path (see
    # docstring note); the asymmetric Litterman path activates it.

    # (c) Constant: loose prior. Same convention — larger lambda_4 = looser.
    # The dummy adds an observation whose precision on the constant is
    # 1/lambda_4^2, so large lambda_4 means tiny precision = essentially flat.
    if include_const:
        Yd_c = np.zeros((1, n))
        Xd_c = np.zeros((1, k))
        Xd_c[0, -1] = 1.0 / prior.lambda_4
        blocks_Y.append(Yd_c)
        blocks_X.append(Xd_c)

    # (d) Sum-of-coefficients prior. Each row is scaled by the variable's
    # random-walk marker rw_deltas[i] (Banbura, Giannone and Reichlin
    # 2010): the belief that the own-lag coefficients sum to one belongs
    # to variables given a random-walk prior mean, and the row is left out
    # for a variable marked stationary (rw_deltas[i] = 0). Without this
    # the two parts of the prior contradict each other for such a
    # variable and the implied prior scale of its error variance is
    # inflated. With the default rw_deltas = 1 nothing changes.
    if prior.mu > 0:
        soc_rows = np.diag(y_bar * rw_deltas)[rw_deltas != 0.0] * prior.mu
        if soc_rows.shape[0]:
            Xd_s = np.zeros((soc_rows.shape[0], k))
            for lag in range(p):
                Xd_s[:, lag * n:(lag + 1) * n] = soc_rows
            blocks_Y.append(soc_rows)
            blocks_X.append(Xd_s)

    # (e) Dummy-initial-observation (single unit-root prior)
    if prior.delta > 0:
        Yd_d = (y_bar * prior.delta).reshape(1, -1)
        Xd_d = np.zeros((1, k))
        for lag in range(p):
            Xd_d[0, lag * n:(lag + 1) * n] = y_bar * prior.delta
        if include_const:
            Xd_d[0, -1] = prior.delta
        blocks_Y.append(Yd_d)
        blocks_X.append(Xd_d)

    return np.vstack(blocks_Y), np.vstack(blocks_X)


# ---------------------------------------------------------------------------
# BVAR fit object
# ---------------------------------------------------------------------------


@dataclass
class BVARFit(VARFit):
    """BVAR fit. Same interface as :class:`VARFit` plus posterior info.

    ``B_hat`` is the posterior mean of the coefficient matrix.
    ``Sigma_u`` is the posterior mean (symmetric mode) or residual-covariance
    plug-in (asymmetric mode) of the residual covariance.

    Parameter draws come from the Normal-Inverse-Wishart posterior in
    symmetric mode, or from per-equation Gaussian posteriors with Sigma
    fixed at the plug-in in asymmetric mode — call :meth:`parameter_draws`.
    """

    prior: MinnesotaPrior = field(default_factory=MinnesotaPrior)
    S_post: np.ndarray = field(default=None, repr=False)       # (n, n)
    df_post: int = 0
    XtX_inv: np.ndarray = field(default=None, repr=False)      # (k, k)
    n_dummies: int = 0
    n_aug: int = 0
    asymmetric: bool = False
    XtX_inv_list: list[np.ndarray] | None = field(default=None, repr=False)
    sigmas: np.ndarray | None = field(default=None, repr=False)

    def parameter_draws(
        self,
        n_draws: int,
        seed: int | None = None,
        require_stable: bool = True,
        eig_tol: float = 1.01,
        burn_in: int = 0,
        max_attempts_factor: int = 4,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Draw ``(B, Sigma)`` from the posterior.

        Symmetric mode (default — natural-conjugate NIW):

        * ``Sigma ~ IW(S_post, df_post)``
        * ``B | Sigma ~ N(B_post, Sigma ⊗ (X'X)^{-1})``

        Asymmetric mode (Litterman / Sims-Zha; ``prior.asymmetric=True``):

        * ``Sigma`` is held fixed at the plug-in ``Sigma_u``
        * Each equation ``i``: ``beta_i ~ N(B_hat[:, i], sigma_i^2 (X_aug' X_aug)^{-1}_i)``

        If ``require_stable``, draws with companion-matrix eigenvalues above
        ``eig_tol`` are rejected and resampled (``eig_tol = 1.01`` lets near-
        unit-root draws through, which is standard for level VARs with
        sum-of-coefficients and dummy-initial-obs priors).

        Returns a list of ``(B, Sigma)`` tuples of length ``n_draws``.
        """
        rng = np.random.default_rng(seed)
        n = self.n_vars
        k = self.k

        if self.asymmetric:
            if self.XtX_inv_list is None or self.sigmas is None:
                raise RuntimeError("Asymmetric BVARFit missing per-equation state — re-fit.")
            return self._asymmetric_draws(
                rng, n_draws, require_stable, eig_tol, burn_in, max_attempts_factor,
            )

        if self.S_post is None or self.XtX_inv is None:
            raise RuntimeError("BVARFit is missing posterior state — re-fit.")

        L_xtx = np.linalg.cholesky(self.XtX_inv)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        attempts = 0
        max_attempts = max(n_draws * max_attempts_factor, n_draws + 1)
        burn_remaining = burn_in

        while len(out) < n_draws and attempts < max_attempts:
            attempts += 1
            try:
                # atleast_2d: SciPy squeezes a 1-D inverse-Wishart draw to a
                # scalar, which matrix Cholesky would reject — a one-variable
                # BVAR would then discard every draw.
                Sigma_d = np.atleast_2d(sp_stats.invwishart.rvs(
                    df=self.df_post, scale=self.S_post,
                    random_state=rng,
                ))
                L_sig = np.linalg.cholesky(Sigma_d)
            except (np.linalg.LinAlgError, ValueError):
                continue

            Z = rng.standard_normal((k, n))
            B_d = self.B_hat + L_xtx @ Z @ L_sig.T

            if burn_remaining > 0:
                burn_remaining -= 1
                continue
            if require_stable:
                lag_mats = extract_lag_matrices(B_d, n, self.p)
                if max_eigenvalue_modulus(lag_mats) >= eig_tol:
                    continue
            out.append((B_d, Sigma_d))

        if len(out) < n_draws:
            import warnings
            warnings.warn(
                f"BVARFit.parameter_draws: only {len(out)} stable draws found "
                f"after {attempts} attempts (requested {n_draws}). "
                f"Consider widening eig_tol, raising max_attempts_factor, or "
                f"loosening the prior."
            )
        return out

    def _asymmetric_draws(
        self, rng, n_draws, require_stable, eig_tol, burn_in, max_attempts_factor,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n = self.n_vars
        k = self.k
        # Per-equation Cholesky factors of sigma_i^2 * XtX_inv_i (the conditional
        # posterior covariance of beta_i given fixed sigma_i^2).
        L_list = []
        for i in range(n):
            cov_i = self.sigmas[i] ** 2 * self.XtX_inv_list[i]
            L_list.append(np.linalg.cholesky(cov_i))

        out: list[tuple[np.ndarray, np.ndarray]] = []
        attempts = 0
        max_attempts = max(n_draws * max_attempts_factor, n_draws + 1)
        burn_remaining = burn_in

        while len(out) < n_draws and attempts < max_attempts:
            attempts += 1
            B_d = np.empty_like(self.B_hat)
            for i in range(n):
                z = rng.standard_normal(k)
                B_d[:, i] = self.B_hat[:, i] + L_list[i] @ z

            if burn_remaining > 0:
                burn_remaining -= 1
                continue
            if require_stable:
                lag_mats = extract_lag_matrices(B_d, n, self.p)
                if max_eigenvalue_modulus(lag_mats) >= eig_tol:
                    continue
            out.append((B_d, self.Sigma_u.copy()))

        if len(out) < n_draws:
            import warnings
            warnings.warn(
                f"BVARFit.parameter_draws (asymmetric): only {len(out)} stable "
                f"draws found after {attempts} attempts (requested {n_draws}). "
                f"Consider widening eig_tol or raising max_attempts_factor."
            )
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_block_exogeneity(
    exogenous_block: list[str], names: list[str],
) -> dict[int, set[int]]:
    """Resolve an exogenous-variable list to ``{equation_idx: {excluded var idx}}``.

    Each variable in ``exogenous_block`` is treated as exogenous to all the OTHER
    (non-listed) variables: its equation excludes the lags of every variable not
    in the block — the standard small-open-economy structure (the foreign block
    does not respond to domestic lags).
    """
    missing = [v for v in exogenous_block if v not in names]
    if missing:
        raise ValueError(
            f"exogenous_block variables not in data columns: {missing}; "
            f"available: {names}"
        )
    exo_idx = [names.index(v) for v in exogenous_block]
    excluded = {j for j in range(len(names)) if j not in exo_idx}
    if not excluded:
        raise ValueError(
            "exogenous_block lists every variable — nothing would be excluded; "
            "leave at least one (domestic) variable out of the block"
        )
    return {i: set(excluded) for i in exo_idx}


def fit_minnesota_bvar(
    data: pd.DataFrame,
    lags: int,
    prior: MinnesotaPrior | None = None,
    trend: str = "c",
    obs_scale: np.ndarray | None = None,
    exogenous_block: list[str] | None = None,
    dummies: tuple[np.ndarray, np.ndarray] | None = None,
) -> BVARFit:
    """Estimate a BVAR with Minnesota prior via dummy observations.

    Two estimation modes, selected by ``prior.asymmetric``:

    * ``False`` (default) — natural-conjugate symmetric prior (BGR 2010).
      Dummy observations stacked once; OLS on the augmented system gives
      the NIW posterior. Closed-form ``log_marginal_data_density`` is
      available. ``lambda_2`` is a no-op in this mode (see the
      :class:`MinnesotaPrior` docstring for the architectural reason).
    * ``True`` — Litterman (1986) / Sims-Zha (1998) asymmetric prior via
      equation-by-equation Ridge with per-equation precision matrices.
      Cross-variable coefficients are shrunk by an extra ``lambda_2``
      factor. ``Sigma_u`` is the residual covariance from the per-equation
      fits, treated as fixed for parameter draws.

    Parameters
    ----------
    data : pd.DataFrame
    lags : int
    prior : MinnesotaPrior, optional
        Defaults to ``MinnesotaPrior()``.
    trend : str
        ``"c"`` (constant) or ``"n"`` (no constant). ``"ct"`` not yet supported.
    obs_scale : array-like of shape ``(T_eff,)``, optional
        Per-observation scale factor ``s_t`` for the post-lag effective
        sample (i.e., aligned with ``Y_act`` rows, not the full data
        index). The model becomes ``y_t = c + sum_l A_l y_{t-l} + s_t * u_t``
        with ``u_t ~ N(0, Sigma)`` — equivalent to GLS with diagonal weight
        matrix ``diag(1/s_t^2)``. Data rows ``y_t`` and ``x_t`` are scaled
        by ``1/s_t`` before stacking with dummies; the dummies retain
        their original weight (they encode the prior, not data). Higher
        ``s_t`` ⇒ this observation contributes less to the posterior; an
        ``s_t = ∞`` observation is effectively dropped.

        Used by Lenza-Primiceri (2022) "How to Estimate a VAR after March
        2020" to down-weight COVID-period innovations without dropping
        them. ``obs_scale=None`` (default) recovers the standard fit.
        Symmetric (natural-conjugate) path only — the asymmetric and
        block-exogeneity routes reject it (their reported covariance
        would not match the stated scaled model).

        **MDD note**: ``log_marginal_data_density`` returns the MDD on the
        SCALED data. To compare across different ``obs_scale`` choices,
        add the change-of-variables Jacobian ``-n_vars * sum_t log(s_t)``.
    exogenous_block : list of str, optional
        Variable names forming an EXOGENOUS block (e.g. the foreign block in a
        small open economy). Each listed variable's equation excludes the lags
        of every variable NOT in the block — its prior std on those coefficients
        is shrunk by ``prior.lambda_bex`` toward zero, so domestic shocks do not
        feed back into the foreign block. Block exogeneity needs per-equation
        priors, so this always uses the asymmetric (Litterman) path regardless of
        ``prior.asymmetric``.
    dummies : (Yd, Xd) tuple of np.ndarray, optional
        Pre-built Minnesota dummy observations to use INSTEAD of rebuilding
        them from ``data``. Needed when the same prior must stay fixed while
        the data changes — e.g. the MF-VAR Gibbs sampler, where rebuilding
        the dummies from each latent completion would make the prior
        data-dependent and invalidate the sampler. Symmetric
        (natural-conjugate) path only.

    Returns
    -------
    BVARFit
    """
    if data.isna().any().any():
        raise ValueError("data contains NaN; drop or impute before fitting")
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")
    if trend not in ("c", "n"):
        raise ValueError(f"trend must be 'c' or 'n' for BVAR; got {trend!r}")
    prior = prior or MinnesotaPrior()

    if obs_scale is not None:
        obs_scale = np.asarray(obs_scale, dtype=float).ravel()
        if np.any(obs_scale <= 0):
            raise ValueError(
                f"obs_scale must be strictly positive (these are sd-scale "
                f"factors, not variances); got min={obs_scale.min():.3e}"
            )
        expected_T = data.shape[0] - lags
        if obs_scale.shape[0] != expected_T:
            raise ValueError(
                f"obs_scale must have length T_eff = T - lags = {expected_T} "
                f"(aligned with Y_act rows), got {obs_scale.shape[0]}"
            )

    if dummies is not None and (exogenous_block is not None or prior.asymmetric):
        raise ValueError(
            "pre-built dummies are only supported on the symmetric "
            "(natural-conjugate) path — the asymmetric/block-exogenous paths "
            "build per-equation precisions, not stacked dummy observations"
        )
    if obs_scale is not None and (exogenous_block is not None or prior.asymmetric):
        raise NotImplementedError(
            "obs_scale is only supported on the symmetric (natural-conjugate) "
            "path: the asymmetric/block-exogeneity estimator fits coefficients "
            "on scaled rows but would report Sigma_u on the unscaled residuals, "
            "so its covariance and parameter draws would not belong to the "
            "stated model y_t = X_t B + s_t u_t."
        )
    if exogenous_block is not None:
        block_exo_excluded = _resolve_block_exogeneity(
            exogenous_block, list(data.columns))
        return _fit_minnesota_asymmetric(
            data, lags, prior, trend, obs_scale=obs_scale,
            block_exo_excluded=block_exo_excluded,
        )
    if prior.asymmetric:
        return _fit_minnesota_asymmetric(data, lags, prior, trend, obs_scale=obs_scale)
    return _fit_minnesota_symmetric(data, lags, prior, trend, obs_scale=obs_scale,
                                    dummies=dummies)


def _fit_minnesota_symmetric(
    data: pd.DataFrame, lags: int, prior: MinnesotaPrior, trend: str,
    obs_scale: np.ndarray | None = None,
    dummies: tuple[np.ndarray, np.ndarray] | None = None,
) -> BVARFit:
    """Natural-conjugate BGR (2010) symmetric prior via dummy observations."""
    Y = data.values
    include_const = trend == "c"

    # Actual data design
    Y_act, X_act = build_design_matrices(Y, lags, include_const=include_const)

    # Pre-scale data rows by 1/s_t (GLS); dummies retain original weight.
    if obs_scale is not None:
        w = (1.0 / obs_scale)[:, None]
        Y_act_w = Y_act * w
        X_act_w = X_act * w
    else:
        Y_act_w = Y_act
        X_act_w = X_act

    # Dummies — pre-built ones keep the prior fixed when the data varies
    # (MF-VAR Gibbs); otherwise build from this data's presample.
    if dummies is not None:
        Yd, Xd = dummies
    else:
        Yd, Xd = build_minnesota_dummies(Y, lags, prior, include_const=include_const)
    n_dummies = Yd.shape[0]

    # Augmented system
    Y_aug = np.vstack([Yd, Y_act_w])
    X_aug = np.vstack([Xd, X_act_w])
    n_aug = Y_aug.shape[0]

    # Posterior mean = OLS on augmented system
    XtX_aug = X_aug.T @ X_aug
    XtX_aug_inv = np.linalg.inv(XtX_aug)
    B_post = XtX_aug_inv @ X_aug.T @ Y_aug

    # Posterior residuals and IW scale
    E = Y_aug - X_aug @ B_post
    S_post = E.T @ E
    df_post = n_aug - X_aug.shape[1]
    # Posterior mean of Sigma under IW(S_post, df_post) is S_post/(df-n-1),
    # not S_post/df (which understates Sigma by ~(n+1)/df proportionally).
    Sigma_post_mean = S_post / max(df_post - Y_aug.shape[1] - 1, 1)

    fit_info = {
        "trend": trend,
        "method": "minnesota_bvar_dummy_observations",
        "n_dummies": int(n_dummies),
        "n_aug": int(n_aug),
        "df_post": int(df_post),
        "prior": _prior_to_dict(prior),
        "obs_scale_present": obs_scale is not None,
    }

    # Residuals reported on the ORIGINAL data scale (not the scaled one),
    # for interpretation. The IW posterior `Sigma_post_mean` is on the
    # SCALED innovations (u_t in y_t = c + ... + s_t u_t).
    resid_actual = Y_act - X_act @ B_post

    return BVARFit(
        data=data,
        p=lags,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=B_post,
        Sigma_u=Sigma_post_mean,
        residuals=resid_actual,
        variable_names=list(data.columns),
        fit_info=fit_info,
        prior=prior,
        S_post=S_post,
        df_post=int(df_post),
        XtX_inv=XtX_aug_inv,
        n_dummies=int(n_dummies),
        n_aug=int(n_aug),
        asymmetric=False,
    )


def _fit_minnesota_asymmetric(
    data: pd.DataFrame, lags: int, prior: MinnesotaPrior, trend: str,
    obs_scale: np.ndarray | None = None,
    block_exo_excluded: dict[int, set[int]] | None = None,
) -> BVARFit:
    """Litterman (1986) / Sims-Zha (1998) asymmetric prior via per-equation Ridge.

    The prior std on coefficient ``A_{ij}(l)`` (equation ``i``, regressor
    ``j``, lag ``l``) is:

    * Own-lag (``j == i``): ``lambda_1 / l**lambda_3``
    * Cross-lag (``j != i``): ``(sigma_i / sigma_j) * lambda_1 * lambda_2 / l**lambda_3``

    The ``sigma_i`` factors are presample univariate-AR residual stds.
    Each equation gets its own augmented OLS system; ``Sigma_u`` is the
    residual covariance from the per-equation fits.
    """
    Y = data.values
    include_const = trend == "c"
    T, n = Y.shape
    if T <= lags:
        raise ValueError(f"Need T > lags; got T={T}, lags={lags}")

    sigmas = _presample_ar_sigmas(Y, lags)
    y_bar = Y[:lags].mean(axis=0)

    rw_deltas = (
        np.ones(n)
        if prior.rw_deltas is None
        else np.asarray(prior.rw_deltas, dtype=float)
    )
    if rw_deltas.shape != (n,):
        raise ValueError(f"rw_deltas must have shape ({n},), got {rw_deltas.shape}")

    Y_act, X_act = build_design_matrices(Y, lags, include_const=include_const)
    n_const = 1 if include_const else 0
    k = n * lags + n_const

    # Pre-scale data rows by 1/s_t for GLS-style heteroskedasticity (LP 2022).
    # Dummies (prior info) keep their original weight.
    if obs_scale is not None:
        w = (1.0 / obs_scale)[:, None]
        Y_act_w = Y_act * w
        X_act_w = X_act * w
    else:
        Y_act_w = Y_act
        X_act_w = X_act

    # Shared dummies (constant, SoC, DIO) — apply per-equation via column slice.
    blocks_Y_shared: list[np.ndarray] = []
    blocks_X_shared: list[np.ndarray] = []

    if include_const:
        Yd_c = np.zeros((1, n))
        Xd_c = np.zeros((1, k))
        Xd_c[0, -1] = 1.0 / prior.lambda_4
        blocks_Y_shared.append(Yd_c)
        blocks_X_shared.append(Xd_c)

    # Sum-of-coefficients rows scaled by rw_deltas, as in
    # build_minnesota_dummies (left out for variables marked stationary).
    if prior.mu > 0:
        soc_rows = np.diag(y_bar * rw_deltas)[rw_deltas != 0.0] * prior.mu
        if soc_rows.shape[0]:
            Xd_s = np.zeros((soc_rows.shape[0], k))
            for lag in range(lags):
                Xd_s[:, lag * n:(lag + 1) * n] = soc_rows
            blocks_Y_shared.append(soc_rows)
            blocks_X_shared.append(Xd_s)

    if prior.delta > 0:
        Yd_d = (y_bar * prior.delta).reshape(1, -1)
        Xd_d = np.zeros((1, k))
        for lag in range(lags):
            Xd_d[0, lag * n:(lag + 1) * n] = y_bar * prior.delta
        if include_const:
            Xd_d[0, -1] = prior.delta
        blocks_Y_shared.append(Yd_d)
        blocks_X_shared.append(Xd_d)

    if blocks_Y_shared:
        Yd_shared = np.vstack(blocks_Y_shared)
        Xd_shared = np.vstack(blocks_X_shared)
    else:
        Yd_shared = np.zeros((0, n))
        Xd_shared = np.zeros((0, k))
    n_shared = Yd_shared.shape[0]

    # Per-equation coefficient prior + Ridge fit
    n_coef_dummies = lags * n
    n_aug = n_coef_dummies + n_shared + Y_act.shape[0]

    B_hat = np.zeros((k, n))
    XtX_inv_list: list[np.ndarray] = []

    for i in range(n):
        Yd_coef = np.zeros(n_coef_dummies)
        Xd_coef = np.zeros((n_coef_dummies, k))
        excluded_i = block_exo_excluded.get(i, ()) if block_exo_excluded else ()
        row = 0
        for lag in range(1, lags + 1):
            for j in range(n):
                if j == i:
                    scale = sigmas[i] * (lag ** prior.lambda_3) / prior.lambda_1
                    target_mean = rw_deltas[i] if lag == 1 else 0.0
                else:
                    scale = (
                        sigmas[j] * (lag ** prior.lambda_3)
                        / (prior.lambda_1 * prior.lambda_2)
                    )
                    target_mean = 0.0
                if j in excluded_i:
                    # Block exogeneity: variable j's lags are excluded from
                    # equation i — shrink the prior std by lambda_bex so the
                    # coefficient is pinned at ~0 (block exogeneity).
                    scale = scale / prior.lambda_bex
                    target_mean = 0.0
                Yd_coef[row] = scale * target_mean
                Xd_coef[row, (lag - 1) * n + j] = scale
                row += 1

        y_aug_i = np.concatenate([Yd_coef, Yd_shared[:, i], Y_act_w[:, i]])
        X_aug_i = np.vstack([Xd_coef, Xd_shared, X_act_w])

        XtX_i = X_aug_i.T @ X_aug_i
        try:
            XtX_inv_i = np.linalg.inv(XtX_i)
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                f"Singular augmented design for equation {i} in asymmetric "
                f"Minnesota fit (sigmas={sigmas}, k={k}). Consider loosening "
                f"the prior or shortening lags."
            ) from err
        B_hat[:, i] = XtX_inv_i @ X_aug_i.T @ y_aug_i
        XtX_inv_list.append(XtX_inv_i)

    resid_actual = Y_act - X_act @ B_hat
    Sigma_u = resid_actual.T @ resid_actual / max(Y_act.shape[0] - k, 1)

    fit_info = {
        "trend": trend,
        "method": "minnesota_bvar_asymmetric_ridge",
        "n_dummies_per_eq": int(n_coef_dummies),
        "n_shared_dummies": int(n_shared),
        "n_aug": int(n_aug),
        "presample_sigmas": [float(s) for s in sigmas],
        "prior": _prior_to_dict(prior),
        "obs_scale_present": obs_scale is not None,
        "block_exogeneity": (
            {int(i): sorted(int(j) for j in js)
             for i, js in block_exo_excluded.items()}
            if block_exo_excluded else None
        ),
    }

    return BVARFit(
        data=data,
        p=lags,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=B_hat,
        Sigma_u=Sigma_u,
        residuals=resid_actual,
        variable_names=list(data.columns),
        fit_info=fit_info,
        prior=prior,
        S_post=None,
        df_post=0,
        XtX_inv=None,
        n_dummies=int(n_coef_dummies + n_shared),
        n_aug=int(n_aug),
        asymmetric=True,
        XtX_inv_list=XtX_inv_list,
        sigmas=sigmas,
    )


def _prior_to_dict(prior: MinnesotaPrior) -> dict:
    return {
        "lambda_1": prior.lambda_1,
        "lambda_2": prior.lambda_2,
        "lambda_3": prior.lambda_3,
        "lambda_4": prior.lambda_4,
        "mu": prior.mu,
        "delta": prior.delta,
        "rw_deltas": (
            None if prior.rw_deltas is None
            else list(map(float, prior.rw_deltas))
        ),
        "asymmetric": prior.asymmetric,
    }


# ---------------------------------------------------------------------------
# GLP 2015 — marginal data density and hyperparameter optimisation
# ---------------------------------------------------------------------------


def log_marginal_data_density(fit: "BVARFit") -> float:
    """Log marginal data density for a Minnesota BVAR with dummy-observation
    NIW conjugate prior (Giannone-Lenza-Primiceri 2015, eq. 5).

    For a BVAR estimated via dummy observations, the log MDD has the closed
    form

        log p(Y|θ) = -nT/2 log(π)
                   + (n/2) (log|X_aug'X_aug|^{-1} - log|X_d'X_d|^{-1})
                   + (ν_0/2) log|S_0| - (ν_post/2) log|S_post|
                   + log Γ_n(ν_post/2) - log Γ_n(ν_0/2)

    where the prior moments ``(V_0, β_0, S_0, ν_0)`` are recovered from the
    dummy observations alone, and the posterior moments are computed on
    the augmented (dummy + actual) system. This is the GLP (2015) /
    Karlsson (2013) formula.

    Parameters
    ----------
    fit : BVARFit

    Returns
    -------
    float
        Log marginal data density. Higher values indicate hyperparameters
        that better fit the observed data.

    Notes
    -----
    Values are comparable across hyperparameters for the same data and
    lag length. They are NOT comparable across lag lengths when the fits
    are given the same data frame: a fit with ``lags=p`` explains the
    observations from ``p + 1`` on, so each lag length explains a
    different set of rows and the comparison favours the longest lag. To
    compare lag lengths, drop leading rows so that every fit explains the
    same observations (for a maximum lag ``P``, pass ``data.iloc[P - p:]``).
    """
    from scipy.special import gammaln
    if fit.asymmetric:
        raise NotImplementedError(
            "log_marginal_data_density is not defined for the asymmetric "
            "Litterman / Sims-Zha prior (it conditions on plug-in sigmas, "
            "so the closed-form NIW MDD does not apply). For MDD-based "
            "hyperparameter selection, fit with prior.asymmetric=False."
        )
    if fit.S_post is None or fit.XtX_inv is None:
        raise RuntimeError("BVARFit missing posterior state — re-fit.")

    n = fit.n_vars
    T = fit.t_eff
    k = fit.k
    prior = fit.prior

    # Reconstruct the dummy observations and their OLS-on-dummies-only quantities
    Y_full = fit.data.values
    include_const = k > n * fit.p
    Yd, Xd = build_minnesota_dummies(Y_full, fit.p, prior, include_const=include_const)
    n_dummies = Yd.shape[0]
    if n_dummies <= k:
        # Improper prior — MDD is undefined
        return -np.inf

    # Prior moments from dummies alone (OLS on dummy system)
    XdtXd = Xd.T @ Xd
    try:
        XdtXd_inv = np.linalg.inv(XdtXd)
    except np.linalg.LinAlgError:
        return -np.inf
    beta_0 = XdtXd_inv @ Xd.T @ Yd
    resid_0 = Yd - Xd @ beta_0
    S_0 = resid_0.T @ resid_0
    nu_0 = n_dummies - k

    # Posterior moments (already on the augmented system in fit)
    S_post = fit.S_post
    nu_post = fit.df_post   # = nu_0 + T

    # log determinant terms
    try:
        sign_d, logdet_XdtXd = np.linalg.slogdet(XdtXd)
        sign_a, logdet_XaXa = np.linalg.slogdet(np.linalg.inv(fit.XtX_inv))
        sign_S0, logdet_S0 = np.linalg.slogdet(S_0)
        sign_Sp, logdet_Sp = np.linalg.slogdet(S_post)
    except np.linalg.LinAlgError:
        return -np.inf
    if min(sign_d, sign_a, sign_S0, sign_Sp) <= 0:
        return -np.inf

    # Multivariate gamma function log
    log_gamma_n_post = _log_multigamma(nu_post / 2.0, n)
    log_gamma_n_0 = _log_multigamma(nu_0 / 2.0, n)

    log_mdd = (
        # Matric-variate-t normalizing constant is pi^{-nT/2}: the 2's cancel
        # between the Gaussian and IW normalizers (Zellner 1971; Karlsson
        # 2013 par. 3.3; GLP 2015 eq. 5). 2*pi understates the absolute MDD
        # by (T/2)log2 per variable and biases lag-order comparisons.
        -n * T / 2.0 * np.log(np.pi)
        + (n / 2.0) * (logdet_XdtXd - logdet_XaXa)
        + (nu_0 / 2.0) * logdet_S0
        - (nu_post / 2.0) * logdet_Sp
        + log_gamma_n_post - log_gamma_n_0
    )
    return float(log_mdd)


def _log_multigamma(a: float, p: int) -> float:
    """Logarithm of the multivariate gamma function Γ_p(a)."""
    from scipy.special import gammaln
    if a <= (p - 1) / 2.0:
        return -np.inf   # outside the domain where Γ_p is defined
    out = p * (p - 1) / 4.0 * np.log(np.pi)
    for j in range(1, p + 1):
        out += gammaln(a + (1.0 - j) / 2.0)
    return float(out)


def optimize_minnesota_hyperparameters(
    data: pd.DataFrame,
    lags: int,
    trend: str = "c",
    params_to_optimize: list[str] | None = None,
    initial: MinnesotaPrior | None = None,
    bounds: dict | None = None,
    method: str = "Nelder-Mead",
    verbose: bool = False,
) -> tuple[MinnesotaPrior, dict]:
    """Maximise the marginal data density over the Minnesota hyperparameters.

    GLP (2015) cast hyperparameter selection as Bayesian model selection
    over the prior tightness parameters. With a NIW conjugate prior the
    marginal data density has a closed form (:func:`log_marginal_data_density`),
    and we can find the maximum via numerical optimisation.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series for the VAR.
    lags : int
        VAR lag order.
    trend : str, default "c"
    params_to_optimize : list of str, optional
        Subset of ``{"lambda_1", "lambda_2", "lambda_3", "lambda_4", "mu", "delta"}``
        to vary. Defaults to ``["lambda_1"]`` (GLP's headline single-
        hyperparameter case). The remaining parameters are held at the
        ``initial`` values.
    initial : MinnesotaPrior, optional
        Initial values; defaults to ``MinnesotaPrior()``.
    bounds : dict, optional
        Per-parameter ``(low, high)`` bounds. Defaults to
        ``{"lambda_1": (1e-4, 5.0), "lambda_2": (0.01, 5.0), ...}``.
    method : str, default "Nelder-Mead"
        scipy.optimize method.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    MinnesotaPrior
        Optimised prior.
    dict
        Diagnostics: ``best_mdd``, ``initial_mdd``, ``n_evaluations``,
        ``optimized_params``.
    """
    from scipy.optimize import minimize

    if params_to_optimize is None:
        params_to_optimize = ["lambda_1"]
    valid = {"lambda_1", "lambda_2", "lambda_3", "lambda_4", "mu", "delta"}
    bad = [p for p in params_to_optimize if p not in valid]
    if bad:
        raise ValueError(f"unknown params_to_optimize: {bad!r}")

    default_bounds = {
        "lambda_1": (1e-4, 5.0),
        "lambda_2": (0.01, 5.0),
        "lambda_3": (0.1, 5.0),
        "lambda_4": (1.0, 1e6),
        "mu": (0.0, 5.0),
        "delta": (0.0, 5.0),
    }
    if bounds is None:
        bounds = {}
    use_bounds = {**default_bounds, **bounds}
    bounds_list = [use_bounds[p] for p in params_to_optimize]

    initial = initial or MinnesotaPrior()
    if initial.asymmetric:
        raise ValueError(
            "optimize_minnesota_hyperparameters requires prior.asymmetric=False "
            "(MDD-based selection uses the closed-form NIW marginal data "
            "density, which is not defined for the asymmetric prior)."
        )
    init_vec = np.array([getattr(initial, p) for p in params_to_optimize])

    # Hyperparameters are optimised in levels; out-of-bounds values are penalised
    def _vec_to_prior(vec: np.ndarray) -> MinnesotaPrior:
        kwargs = {
            "lambda_1": initial.lambda_1,
            "lambda_2": initial.lambda_2,
            "lambda_3": initial.lambda_3,
            "lambda_4": initial.lambda_4,
            "mu": initial.mu,
            "delta": initial.delta,
            "rw_deltas": initial.rw_deltas,
        }
        for name, val in zip(params_to_optimize, vec):
            kwargs[name] = float(val)
        return MinnesotaPrior(**kwargs)

    initial_mdd = log_marginal_data_density(
        fit_minnesota_bvar(data, lags=lags, prior=initial, trend=trend)
    )
    if verbose:
        print(f"      initial MDD = {initial_mdd:.4f} at {init_vec}")

    n_calls = {"count": 0}

    def _neg_mdd(vec: np.ndarray) -> float:
        # Penalty for going out of bounds
        for v, (lo, hi) in zip(vec, bounds_list):
            if v < lo or v > hi:
                return 1e12
        prior = _vec_to_prior(vec)
        try:
            fit = fit_minnesota_bvar(data, lags=lags, prior=prior, trend=trend)
            mdd = log_marginal_data_density(fit)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return 1e12
        if not np.isfinite(mdd):
            return 1e12
        n_calls["count"] += 1
        if verbose and n_calls["count"] % 10 == 0:
            print(f"      eval {n_calls['count']}: MDD={mdd:.4f} at "
                  f"{ {p: float(v) for p, v in zip(params_to_optimize, vec)} }")
        return -mdd

    result = minimize(
        _neg_mdd, init_vec, method=method,
        options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 500},
    )
    optimal_prior = _vec_to_prior(result.x)
    best_mdd = -result.fun

    diag = {
        "best_mdd": float(best_mdd),
        "initial_mdd": float(initial_mdd),
        "improvement": float(best_mdd - initial_mdd),
        "n_evaluations": int(n_calls["count"]),
        "converged": bool(result.success),
        "method": method,
        "optimized_params": {
            p: float(getattr(optimal_prior, p)) for p in params_to_optimize
        },
    }
    return optimal_prior, diag


# ===========================================================================
# Steady-state (mean-adjusted) BVAR -- Villani (2009)
# ===========================================================================
#
# Model (constant deterministic term d_t = 1):
#     y_t = Psi + sum_{j=1..p} A_j (y_{t-j} - Psi) + e_t,   e_t ~ N(0, Sigma)
# The series revert to the steady state Psi (their unconditional mean), with an
# INDEPENDENT NORMAL prior on Psi. Headline use: anchor a variable's long-run
# mean to a known value (e.g. an inflation target) rather than letting the
# sample mean drive long-horizon forecasts. Estimated by the three-block Gibbs
# sampler of Villani (2009):
#   (1) Psi   | A, Sigma   -- Normal (GLS; de-meaning multiplier R = I - sum A_j)
#   (2) Sigma | A, Psi     -- inverse-Wishart on the de-meaned residuals
#   (3) A     | Sigma, Psi -- Normal with a Minnesota prior on the de-meaned VAR


def _sym(M):
    """Symmetrise (guards Cholesky / sampling against tiny asymmetry)."""
    return 0.5 * (M + M.T)


@dataclass
class SteadyStatePrior:
    """Villani (2009) mean-adjusted (steady-state) BVAR prior.

    A Minnesota prior on the autoregressive coefficients (deviation-from-mean
    form) plus an INDEPENDENT NORMAL prior on the steady state ``Psi`` -- the
    unconditional mean each variable reverts to.

    Parameters
    ----------
    psi_mean : sequence of float, shape (n,)
        Prior mean of the steady state, one per endogenous variable. For a
        central-bank application this is where you anchor the long-run mean
        (e.g. the inflation target for the inflation equation).
    psi_sd : sequence of float, shape (n,)
        Prior standard deviation of the steady state, per variable. Small =>
        the long-run mean is pinned near ``psi_mean``; large => the data
        (sample mean) dominate. Must be > 0.
    lambda_1 : float, default 0.2
        Minnesota overall tightness on the AR coefficients.
    lambda_3 : float, default 1.0
        Minnesota lag decay (standard deviation of lag-l coefficients scales 1 / l**lambda_3).
    rw_deltas : sequence of float, optional
        Prior means for the own-lag-1 AR coefficient of each variable. Default
        ``[1, 1, ...]`` (random-walk centre, as in the standard Minnesota prior).
    sigma_prior_df : int, optional
        Degrees of freedom of the (proper, weakly informative) inverse-Wishart
        prior on Sigma. Default ``n + 2``; prior scale is the identity.
    """

    psi_mean: Sequence[float]
    psi_sd: Sequence[float]
    lambda_1: float = 0.2
    lambda_3: float = 1.0
    rw_deltas: Sequence[float] | None = None
    sigma_prior_df: int | None = None


def _minnesota_dev_prior(s, n, p, lambda_1, lambda_3, rw_deltas):
    """Minnesota prior mean (n*p, n) and diagonal variance (n*p, n) for the AR
    coefficients in deviation form (no intercept). Layout matches a regression
    of z_t on [z_{t-1}, ..., z_{t-p}]: row (l-1)*n + j is variable j at lag l."""
    np_ = n * p
    b0 = np.zeros((np_, n))
    var = np.zeros((np_, n))
    for i in range(n):
        b0[i, i] = rw_deltas[i]              # own lag-1 coefficient mean
        for l in range(1, p + 1):
            base = (lambda_1 / (l ** lambda_3)) ** 2
            for j in range(n):
                r = (l - 1) * n + j
                var[r, i] = base if j == i else base * (s[i] ** 2) / (s[j] ** 2)
    return b0, var


def _ss_to_reduced_form(A_stack, psi, n, p, k):
    """Map (deviation-form AR stack, steady state) to a standard reduced-form
    B_hat with layout [lag1, ..., lagp, const]: the implied intercept is
    c = (I - sum_l A_l) Psi."""
    B = np.zeros((k, n))
    B[: n * p, :] = A_stack
    sumA = sum(A_stack[l * n:(l + 1) * n, :].T for l in range(p))  # sum_l A_l (n,n)
    c = (np.eye(n) - sumA) @ np.asarray(psi).ravel()
    B[n * p, :] = c                          # const row (trend='c' => one det row)
    return B


def _steady_state_gibbs(Y, p, prior, n_draws, burn_in, seed):
    """Three-block Villani (2009) Gibbs (constant deterministic term, q=1).

    Returns (A_draws (D, n*p, n), psi_draws (D, n), sigma_draws (D, n, n))."""
    from scipy.stats import invwishart
    rng = np.random.default_rng(seed)
    T_full, n = Y.shape
    T = T_full - p
    s = _presample_ar_sigmas(Y, p)
    rw = (np.ones(n) if prior.rw_deltas is None
          else np.asarray(prior.rw_deltas, dtype=float))
    b0, v0 = _minnesota_dev_prior(s, n, p, prior.lambda_1, prior.lambda_3, rw)
    b0_vec = b0.flatten(order="F")
    v0_inv = np.diag(1.0 / v0.flatten(order="F"))
    psi0 = np.asarray(prior.psi_mean, dtype=float)
    psi_sd = np.asarray(prior.psi_sd, dtype=float)
    if psi0.shape != (n,) or psi_sd.shape != (n,):
        raise ValueError(f"psi_mean and psi_sd must have shape ({n},)")
    if np.any(psi_sd <= 0):
        raise ValueError("psi_sd entries must be > 0")
    Lam0_inv = np.diag(1.0 / psi_sd ** 2)
    df0 = (n + 2) if prior.sigma_prior_df is None else int(prior.sigma_prior_df)
    S0 = np.eye(n)

    def design(Z):
        lhs = Z[p:]
        lags = np.column_stack([Z[p - l: T_full - l] for l in range(1, p + 1)])
        return lhs, lags

    A = np.zeros((n * p, n))
    A[:n, :] = np.diag(rw)
    Sigma = np.atleast_2d(np.cov(Y.T))
    Psi = Y.mean(axis=0)

    A_draws, psi_draws, sigma_draws = [], [], []
    for it in range(burn_in + n_draws):
        Sigma_inv = np.linalg.inv(Sigma)
        # block 1: Psi | A, Sigma  -- x_t = y_t - sum_l A_l y_{t-l} = R Psi + e_t
        lhs, lags = design(Y)
        x = lhs - lags @ A
        sumA = sum(A[l * n:(l + 1) * n, :].T for l in range(p))
        R = np.eye(n) - sumA
        prec = Lam0_inv + T * (R.T @ Sigma_inv @ R)
        cov = np.linalg.inv(_sym(prec))
        mean = cov @ (Lam0_inv @ psi0 + R.T @ Sigma_inv @ x.sum(axis=0))
        Psi = rng.multivariate_normal(mean, _sym(cov))
        # de-mean
        Z = Y - Psi
        zlhs, zlags = design(Z)
        # block 2: Sigma | A, Psi  -- IW on de-meaned residuals
        E = zlhs - zlags @ A
        Sigma = np.atleast_2d(invwishart.rvs(df=df0 + T, scale=_sym(S0 + E.T @ E),
                                             random_state=rng))
        Sigma_inv = np.linalg.inv(Sigma)
        # block 3: A | Sigma, Psi  -- Minnesota Normal posterior
        ZtZ = zlags.T @ zlags
        prec_a = v0_inv + np.kron(Sigma_inv, ZtZ)
        cov_a = np.linalg.inv(_sym(prec_a))
        rhs_a = v0_inv @ b0_vec + (zlags.T @ zlhs @ Sigma_inv).flatten(order="F")
        a_vec = rng.multivariate_normal(cov_a @ rhs_a, _sym(cov_a))
        A = a_vec.reshape((n * p, n), order="F")
        if it >= burn_in:
            A_draws.append(A.copy())
            psi_draws.append(Psi.copy())
            sigma_draws.append(Sigma.copy())
    return np.array(A_draws), np.array(psi_draws), np.array(sigma_draws)


@dataclass
class SteadyStateBVARFit(VARFit):
    """Mean-adjusted (steady-state) BVAR fit. A standard reduced-form VARFit (so
    Cholesky/sign IRFs, FEVD and forecasts work unchanged) whose coefficients
    were estimated under the Villani (2009) steady-state prior, plus the
    posterior steady state ``psi`` and the raw Gibbs draws."""

    psi: np.ndarray = field(default=None, repr=False)          # (n, 1) steady state
    prior: SteadyStatePrior | None = None
    a_draws: np.ndarray = field(default=None, repr=False)      # (D, n*p, n)
    psi_draws: np.ndarray = field(default=None, repr=False)    # (D, n)
    sigma_draws: np.ndarray = field(default=None, repr=False)  # (D, n, n)

    @property
    def steady_state(self):
        """Posterior-mean unconditional mean of each variable (length n)."""
        return self.psi.ravel()

    def parameter_draws(self, n_draws, seed=None, require_stable=True,
                        eig_tol=1.01, burn_in=0, **_):
        """Return ``(B, Sigma)`` pairs from the stored posterior draws, mapped to
        standard reduced-form coefficient matrices."""
        from .var import max_eigenvalue_modulus
        rng = np.random.default_rng(seed)
        n, p, k = self.n_vars, self.p, self.k
        M = self.a_draws.shape[0]
        idx = (np.arange(M) if n_draws == M
               else rng.integers(0, M, size=n_draws))
        out = []
        for kk in idx:
            A = self.a_draws[kk]
            B = _ss_to_reduced_form(A, self.psi_draws[kk], n, p, k)
            if require_stable:
                lagmats = [A[l * n:(l + 1) * n, :].T for l in range(p)]
                if max_eigenvalue_modulus(lagmats) >= eig_tol:
                    continue
            out.append((B, self.sigma_draws[kk]))
        return out


def fit_steady_state_bvar(data, lags, prior, n_draws=2000, burn_in=1000, seed=None):
    """Fit the Villani (2009) mean-adjusted / steady-state BVAR by Gibbs.

    Parameters
    ----------
    data : pandas.DataFrame
        Endogenous series (a constant is the only deterministic term).
    lags : int
    prior : SteadyStatePrior
    n_draws, burn_in : int
        Post-burn draws kept, and burn-in discarded.
    seed : int, optional

    Returns
    -------
    SteadyStateBVARFit
        Posterior-mean reduced-form VAR (IRF/forecast-ready) plus ``.psi`` /
        ``.steady_state`` and the raw draws.
    """
    from .var import fit_var
    Y = np.asarray(data.values, dtype=float)
    n = Y.shape[1]
    A_draws, psi_draws, sigma_draws = _steady_state_gibbs(
        Y, lags, prior, n_draws, burn_in, seed)
    A_hat = A_draws.mean(0)
    psi_hat = psi_draws.mean(0)
    Sigma_hat = sigma_draws.mean(0)
    base = fit_var(data, lags=lags, trend="c")
    B_hat = _ss_to_reduced_form(A_hat, psi_hat, n, lags, base.X_act.shape[1])
    residuals = base.Y_act - base.X_act @ B_hat
    return SteadyStateBVARFit(
        data=base.data, p=lags, Y_act=base.Y_act, X_act=base.X_act,
        B_hat=B_hat, Sigma_u=Sigma_hat, residuals=residuals,
        variable_names=list(base.variable_names),
        fit_info={"method": "steady_state_bvar (Villani 2009)",
                  "n_draws": int(n_draws), "burn_in": int(burn_in)},
        psi=psi_hat.reshape(n, 1), prior=prior,
        a_draws=A_draws, psi_draws=psi_draws, sigma_draws=sigma_draws,
    )
