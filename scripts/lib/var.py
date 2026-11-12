"""
Reduced-form VAR utilities.

Thin layer over ``statsmodels.tsa.api.VAR`` plus the linear-algebra primitives
(companion form, MA representation, stability) shared by the identification,
inference and forecasting modules.

Per DEVELOPMENT.md: prefer established packages — this module uses
``statsmodels`` for the OLS fit and only adds the bits ``statsmodels`` doesn't
expose cleanly (vec-coefficient distribution, MA-rep with arbitrary lags,
flexible MC draws used by identification routines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as _StatsmodelsVAR


# ---------------------------------------------------------------------------
# Pure linear-algebra helpers (no model state)
# ---------------------------------------------------------------------------


def _build_x_row(
    Y: np.ndarray, t: int, p: int,
    include_const: bool, include_trend: bool,
    exog: np.ndarray | None = None, exog_lags: int = 0,
) -> list[float]:
    """Build a single X_act row at time ``t`` (0-indexed in full Y).

    Layout (matches :func:`build_design_matrices`):
    ``[lag1, lag2, ..., lagp, (const), (trend), (exog_lag0, ..., exog_lagq)]``.
    """
    row: list[float] = []
    for lag in range(1, p + 1):
        row.extend(Y[t - lag])
    if include_const:
        row.append(1.0)
    if include_trend:
        row.append(float(t - p + 1))
    if exog is not None:
        for lag in range(0, exog_lags + 1):
            row.extend(exog[t - lag])
    return row


def build_design_matrices(
    Y: np.ndarray, p: int,
    include_const: bool = True, include_trend: bool = False,
    exog: np.ndarray | None = None, exog_lags: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``Y_act, X_act`` for a VAR(p) (with optional exogenous block).

    ``Y_act`` has shape ``(T-p, n)``; ``X_act`` has shape
    ``(T-p, n*p + c + d + k*(q+1))`` where ``c = 1`` if ``include_const``
    else 0, ``d = 1`` if ``include_trend`` else 0, ``k`` is the number of
    exogenous variables (0 if ``exog`` is None), and ``q = exog_lags``.

    Column layout: ``[lag1_var1, ..., lag1_varn, lag2_var1, ..., lagp_varn,
    (const), (trend), exog_lag0, exog_lag1, ..., exog_lagq]``. Trend values
    are ``1, 2, ..., T - p`` over the effective sample. NOTE: statsmodels'
    ``trend="ct"`` numbers the trend ``p+1, ..., T`` instead (JMulTi
    convention) — a pure intercept reparameterisation: slopes, lag
    matrices, residuals and Sigma_u are identical, and the constants
    relate as ``const_here = const_sm + p * slope``. Internally consistent
    everywhere the trend is rebuilt (bootstrap included).

    Parameters
    ----------
    exog : np.ndarray, optional
        Exogenous regressors aligned with ``Y`` row-for-row, shape
        ``(T, k)``. When ``None`` the design matrix has no exog block.
    exog_lags : int
        Number of lags of ``exog`` to include in addition to the
        contemporaneous value (i.e., total exog columns = ``k * (q + 1)``).
        Must satisfy ``exog_lags <= p`` so that the same ``T-p`` row drop
        suffices.
    """
    T, n = Y.shape
    if T <= p:
        raise ValueError(f"Need T > p, got T={T}, p={p}")
    if exog is not None:
        if exog.ndim != 2:
            raise ValueError(f"exog must be 2-D (T, k); got shape {exog.shape}")
        if exog.shape[0] != T:
            raise ValueError(
                f"exog rows ({exog.shape[0]}) must equal Y rows ({T})"
            )
        if exog_lags < 0:
            raise ValueError(f"exog_lags must be >= 0; got {exog_lags}")
        if exog_lags > p:
            raise ValueError(
                f"exog_lags ({exog_lags}) must be <= p ({p}); larger exog "
                "lags would require a deeper row drop."
            )
    Y_act = Y[p:]
    rows = [
        _build_x_row(Y, t, p, include_const, include_trend, exog, exog_lags)
        for t in range(p, T)
    ]
    return Y_act, np.asarray(rows)


def extract_lag_matrices(B: np.ndarray, n: int, p: int) -> list[np.ndarray]:
    """Given the stacked coefficient matrix ``B`` (k, n), return the ``p`` lag
    matrices ``A_1, ..., A_p`` of shape ``(n, n)``.

    ``B`` is assumed to be the OLS solution ``B = (X'X)^{-1} X'Y`` so its rows
    are ordered as [lag1_var1, lag1_var2, ..., lag2_var1, ...] (matching
    ``build_design_matrices``). Returned matrices are ``A_j[i, k]`` = effect of
    variable k at lag j on variable i.
    """
    return [B[lag * n:(lag + 1) * n, :].T for lag in range(p)]


def companion_matrix(lag_matrices: list[np.ndarray]) -> np.ndarray:
    """Build the ``np × np`` companion matrix from a list of ``A_j`` matrices."""
    p = len(lag_matrices)
    n = lag_matrices[0].shape[0]
    C = np.zeros((n * p, n * p))
    C[:n, :] = np.hstack(lag_matrices)
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return C


def max_eigenvalue_modulus(lag_matrices: list[np.ndarray]) -> float:
    """Largest eigenvalue modulus of the companion matrix. <1 = stationary."""
    return float(np.max(np.abs(np.linalg.eigvals(companion_matrix(lag_matrices)))))


def ma_representation(lag_matrices: list[np.ndarray], horizon: int) -> np.ndarray:
    """MA coefficient matrices ``Phi_0, ..., Phi_H``.

    HORIZON CONVENTION:

      Returns shape ``(H+1, n, n)`` indexed ``Phi[0..H]`` (H+1 matrices).
      ``Phi[h]`` is the MA coefficient at lag ``h`` (response to a unit
      reduced-form innovation ``h`` periods ago). NOT to be confused with
      the "H-step-ahead forecast horizon" used by FEVD.

      Relationship:
        ``y_t = sum_{h>=0} Phi_h u_{t-h}`` (Wold decomposition)
        ``y_{t+H} - E[y_{t+H}|t] = sum_{h=0..H-1} Phi_h u_{t+H-h}``  (H-step forecast error)

      So an "IRF at horizon h" uses ``Phi[h]`` (or ``Phi[h] @ S`` for the
      structural IRF), but an "H-step FEVD" sums ``h=0..H-1`` of
      ``Phi[h] Sigma Phi[h]'`` — H terms, NOT H+1.

    RECURSION:

      ``Phi[0] = I_n``
      ``Phi[h] = sum_{j=1..min(h,p)} A_j Phi[h-j]``
    """
    p = len(lag_matrices)
    n = lag_matrices[0].shape[0]
    Phi = np.zeros((horizon + 1, n, n))
    Phi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        for j in range(min(h, p)):
            Phi[h] += lag_matrices[j] @ Phi[h - j - 1]
    return Phi


# ---------------------------------------------------------------------------
# OLS VAR fit (statsmodels wrapper)
# ---------------------------------------------------------------------------


@dataclass
class VARFit:
    """Holder for a fitted reduced-form VAR.

    Designed to be the common input to identification routines (Cholesky, sign
    restrictions, proxy SVAR). Carries enough state to (a) recompute MA reps,
    (b) draw from the asymptotic coefficient distribution for MC bands, and
    (c) generate residual-bootstrap samples.
    """

    data: pd.DataFrame
    p: int
    Y_act: np.ndarray           # (T_eff, n)
    X_act: np.ndarray           # (T_eff, k)
    B_hat: np.ndarray           # (k, n) OLS coefficients
    # Residual covariance. Plain VAR path: statsmodels' Lütkepohl T_eff-K
    # divisor (where K = regressors per equation). VARX path: ML divisor
    # T_eff (no Bessel correction). See CONVENTIONS.md §3 for details.
    Sigma_u: np.ndarray         # (n, n)
    residuals: np.ndarray       # (T_eff, n)
    variable_names: list[str]
    fit_info: dict[str, Any]
    exog: pd.DataFrame | None = None   # VARX exogenous block (None = no exog)
    exog_lags: int = 0                 # lags of exog (0 = contemporaneous only)
    exog_names: list[str] = field(default_factory=list)

    @property
    def has_exog(self) -> bool:
        return self.exog is not None

    @property
    def n_exog(self) -> int:
        return 0 if self.exog is None else self.exog.shape[1]

    @property
    def n_vars(self) -> int:
        return self.Y_act.shape[1]

    @property
    def t_eff(self) -> int:
        return self.Y_act.shape[0]

    @property
    def k(self) -> int:
        return self.X_act.shape[1]

    def lag_matrices(self) -> list[np.ndarray]:
        return extract_lag_matrices(self.B_hat, self.n_vars, self.p)

    def exog_coefs(self) -> np.ndarray:
        """Return the exogenous-block coefficient matrix, shape ``(k*(q+1), n)``.

        Empty array (shape ``(0, n)``) when ``has_exog`` is False.
        """
        if not self.has_exog:
            return np.empty((0, self.n_vars))
        n_exog_cols = self.n_exog * (self.exog_lags + 1)
        # Layout: [lags | const | trend | exog]. Exog block is the last
        # n_exog_cols rows of B_hat.
        return self.B_hat[-n_exog_cols:, :]

    def stable(self, tol: float = 1.0) -> bool:
        return max_eigenvalue_modulus(self.lag_matrices()) < tol

    def ma_rep(self, horizon: int) -> np.ndarray:
        return ma_representation(self.lag_matrices(), horizon)

    def vec_cov(self) -> np.ndarray:
        """Asymptotic covariance of ``vec(B_hat)``: ``Sigma_u ⊗ (X'X)^{-1}``.

        Equation-major stacking (Hamilton 1994, eq. 11.1.35): the vec here
        is ``B_hat.T.flatten()`` — equation 1's coefficients first. Under
        the column-stacking vec of ``B_hat`` itself the kron order would be
        reversed; the pairing used here is pinned against statsmodels'
        standard errors in ``tests/test_var.py``.
        """
        XtX_inv = np.linalg.inv(self.X_act.T @ self.X_act)
        return np.kron(self.Sigma_u, XtX_inv)

    def parameter_draws(
        self,
        n_draws: int,
        seed: int | None = None,
        require_stable: bool = False,
        eig_tol: float = 1.0,
        burn_in: int = 0,
        include_sigma_uncertainty: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Draws of ``(B, Sigma)`` from the asymptotic distribution.

        Default for plain (frequentist) VARs. ``BVARFit`` overrides this to
        sample from the Normal-Inverse-Wishart posterior. Identification
        routines call this generic interface and get the right uncertainty
        automatically.

        Parameters
        ----------
        n_draws : int
        seed : int, optional
        require_stable : bool
            Reject draws with companion eigenvalues at or above ``eig_tol``.
            Default False (asymptotic theory does not require stability).
        eig_tol : float
            Eigenvalue ceiling when ``require_stable`` is True.
        burn_in : int
            Discard the first ``burn_in`` GENERATED draws (before the
            stability filter is applied). Default 0.
        include_sigma_uncertainty : bool, default True
            If True (default), each draw samples ``Sigma`` from an
            inverse-Wishart RECENTRED on the point estimate: scale =
            (T-k-n-1) Sigma_hat, df = T-k, so that E[Sigma_draw] =
            Sigma_hat exactly. (The textbook flat-prior posterior is
            IW(U'U, T-k), whose mean sits (T-k)/(T-k-n-1) above
            Sigma_hat — an O((n+1)/T) recentring, documented in
            CONVENTIONS.md §4.) ``Sigma`` is drawn jointly with ``B``. The resulting ``B`` draw is then
            conditioned on the drawn ``Sigma``: ``vec(B) ~ N(vec(B_hat),
            Sigma_d ⊗ (X'X)^{-1})``. This is the textbook joint asymptotic
            distribution and gives nominal frequentist coverage for IRFs
            that depend on ``Sigma`` (e.g., Cholesky's ``L = chol(Sigma)``).

            If False, ``Sigma`` is held fixed at ``Sigma_hat`` and only
            ``B`` is sampled; bands for IRFs that depend on ``Sigma``
            (e.g. Cholesky) are then too narrow.

        Notes
        -----
        When ``T - k - n - 1`` is not strictly positive (very small
        sample / very large system), the IW posterior mean is undefined;
        the routine falls back to fixed-Sigma sampling and emits a
        warning.
        """
        rng = np.random.default_rng(seed)
        n, k = self.n_vars, self.k
        XtX_inv = np.linalg.inv(self.X_act.T @ self.X_act)
        L_xtx = np.linalg.cholesky(_ensure_psd(XtX_inv))
        vecB_hat = self.B_hat.T.flatten()
        Sigma_hat = np.asarray(self.Sigma_u)

        # Determine whether IW sampling is well-defined
        df_iw = self.t_eff - k
        iw_scale_factor = df_iw - n - 1
        if include_sigma_uncertainty and iw_scale_factor <= 0:
            import warnings
            warnings.warn(
                f"VARFit.parameter_draws: T-k-n-1 = {iw_scale_factor} ≤ 0, "
                f"IW posterior mean undefined. Falling back to fixed-Sigma "
                f"sampling (under-covers Cholesky-style IRFs).",
                UserWarning, stacklevel=2,
            )
            include_sigma_uncertainty = False

        if not include_sigma_uncertainty:
            # Legacy path: Sigma fixed at point estimate; only B varies
            cov_vecB = _ensure_psd(self.vec_cov())
            L = np.linalg.cholesky(cov_vecB)
            out: list[tuple[np.ndarray, np.ndarray]] = []
            burn_remaining = burn_in
            max_attempts = max(4 * n_draws, n_draws + 100) + burn_in
            attempts = 0
            while len(out) < n_draws:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"parameter_draws: only {len(out)}/{n_draws} stable "
                        f"draws after {attempts} attempts (eig_tol="
                        f"{eig_tol}). The fitted system is likely explosive "
                        f"— loosen eig_tol or drop require_stable."
                    )
                attempts += 1
                z = rng.standard_normal(len(vecB_hat))
                B_d = (vecB_hat + L @ z).reshape(n, k).T
                if burn_remaining > 0:
                    burn_remaining -= 1
                    continue
                if require_stable:
                    lag_mats = extract_lag_matrices(B_d, n, self.p)
                    if max_eigenvalue_modulus(lag_mats) >= eig_tol:
                        continue
                out.append((B_d, Sigma_hat))
            return out

        # Joint asymptotic path: draw Sigma from IW, then B conditionally
        from scipy import stats as _sp_stats
        scale = iw_scale_factor * Sigma_hat
        out = []
        burn_remaining = burn_in
        attempts = 0
        max_attempts = max(n_draws * 4, n_draws + 100)
        while len(out) < n_draws and attempts < max_attempts:
            attempts += 1
            try:
                Sigma_d = _sp_stats.invwishart.rvs(
                    df=df_iw, scale=scale, random_state=rng,
                )
                L_sig = np.linalg.cholesky(Sigma_d)
            except (np.linalg.LinAlgError, ValueError):
                continue
            # B | Sigma ~ N(B_hat, Sigma_d ⊗ (X'X)^{-1}).
            # Matrix-form sampler with Z ~ N(0, I) of shape (k, n):
            #   B_d = B_hat + L_xtx @ Z @ L_sig'
            # satisfies vec(B_d - B_hat) = (L_sig ⊗ L_xtx) vec(Z) ~
            #   N(0, Sigma ⊗ (X'X)^{-1}) — matches `vec_cov`. Avoids
            # forming the (n*k, n*k) Kronecker product per draw.
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
                f"VARFit.parameter_draws: only {len(out)} draws after "
                f"{attempts} attempts (requested {n_draws}).",
                UserWarning, stacklevel=2,
            )
        return out


def fit_var(
    data: pd.DataFrame, lags: int, trend: str = "c",
    exog: pd.DataFrame | None = None, exog_lags: int = 0,
) -> VARFit:
    """Estimate a reduced-form VAR(p) (or VARX) by OLS.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series. Columns become ``variable_names``.
    lags : int
        Lag order for endogenous variables.
    trend : str
        ``"c"`` for constant, ``"ct"`` for constant + linear trend,
        ``"n"`` for none. Mirrors ``statsmodels`` convention.
    exog : pd.DataFrame, optional
        Exogenous regressors aligned with ``data`` row-for-row. Same
        DatetimeIndex as ``data`` required. Passing ``None`` (default)
        reproduces the plain VAR fit byte-identically.
    exog_lags : int, default 0
        Number of lags of ``exog`` to include in addition to the
        contemporaneous value. Must satisfy ``exog_lags <= lags``.

    Notes
    -----
    When ``exog`` is non-None, the fit bypasses statsmodels and computes
    OLS + AIC/BIC manually. Information criteria use the convention
    ``AIC = T*log|Sigma| + 2*K`` (Lutkepohl 4.3) with ``K`` total
    coefficients (including exog block). statsmodels' VAR doesn't
    natively support lagged-exog blocks, so consistency with the
    plain-VAR information criteria is not guaranteed when ``exog`` is
    used.
    """
    if data.isna().any().any():
        raise ValueError("data contains NaN; drop or impute before fitting")
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    include_const = trend in ("c", "ct")
    include_trend = trend == "ct"
    Y = data.values

    if exog is None:
        # Plain VAR path — keep statsmodels-backed AIC/BIC.
        # Pass a plain ndarray: statsmodels only needs the values here
        # (sigma_u / ICs / llf), and a freq-less DatetimeIndex triggers a
        # ValueWarning per call (13+ per diagnostics run).
        model = _StatsmodelsVAR(np.asarray(data))
        results = model.fit(maxlags=lags, ic=None, trend=trend)
        Y_act, X_act = build_design_matrices(
            Y, lags, include_const=include_const, include_trend=include_trend,
        )
        B_hat = np.linalg.solve(X_act.T @ X_act, X_act.T @ Y_act)
        resid = Y_act - X_act @ B_hat
        Sigma_u = results.sigma_u
        fit_info = {
            "trend": trend,
            "aic": float(results.aic),
            "bic": float(results.bic),
            "hqic": float(results.hqic),
            "llf": float(results.llf),
            "nobs": int(results.nobs),
            "k_ar": int(results.k_ar),
            "has_exog": False,
        }
        return VARFit(
            data=data, p=lags, Y_act=Y_act, X_act=X_act,
            B_hat=B_hat, Sigma_u=Sigma_u, residuals=resid,
            variable_names=list(data.columns), fit_info=fit_info,
        )

    # VARX path — direct OLS, manual information criteria.
    if exog.isna().any().any():
        raise ValueError("exog contains NaN; drop or impute before fitting")
    if not exog.index.equals(data.index):
        raise ValueError("exog index must match data index exactly")
    if exog_lags < 0:
        raise ValueError(f"exog_lags must be >= 0; got {exog_lags}")
    if exog_lags > lags:
        raise ValueError(
            f"exog_lags ({exog_lags}) must be <= lags ({lags})"
        )

    exog_arr = exog.values
    Y_act, X_act = build_design_matrices(
        Y, lags, include_const=include_const, include_trend=include_trend,
        exog=exog_arr, exog_lags=exog_lags,
    )
    B_hat = np.linalg.solve(X_act.T @ X_act, X_act.T @ Y_act)
    resid = Y_act - X_act @ B_hat
    T_eff = Y_act.shape[0]
    n = Y_act.shape[1]
    Sigma_u = (resid.T @ resid) / T_eff      # ML divisor (the PLAIN path uses T_eff - k; see CONVENTIONS §3)
    sign, logdet = np.linalg.slogdet(Sigma_u)
    K = X_act.shape[1] * n                    # total parameters (k columns x n equations)
    # IC convention on this path: T-scaled form (T·ln|Σ_ML| + penalty·K),
    # which equals exactly T_eff × statsmodels' per-T VARX criteria; the
    # llf below is the standard concentrated Gaussian likelihood and is
    # IDENTICAL to statsmodels'. Monotone-equivalent for lag selection at
    # fixed T, but don't compare IC levels across the plain-VAR and VARX
    # paths (different Σ divisor AND different IC scaling).
    llf = -0.5 * T_eff * (n * np.log(2 * np.pi) + logdet + n)
    aic = float(T_eff * logdet + 2 * K)
    bic = float(T_eff * logdet + np.log(T_eff) * K)
    hqic = float(T_eff * logdet + 2 * np.log(np.log(T_eff)) * K)

    fit_info = {
        "trend": trend,
        "aic": aic,
        "bic": bic,
        "hqic": hqic,
        "llf": float(llf),
        "nobs": int(T_eff),
        "k_ar": int(lags),
        "has_exog": True,
        "n_exog": int(exog.shape[1]),
        "exog_lags": int(exog_lags),
    }
    return VARFit(
        data=data, p=lags, Y_act=Y_act, X_act=X_act,
        B_hat=B_hat, Sigma_u=Sigma_u, residuals=resid,
        variable_names=list(data.columns), fit_info=fit_info,
        exog=exog, exog_lags=exog_lags,
        exog_names=list(exog.columns),
    )


def select_lag_order(data: pd.DataFrame, maxlags: int, criterion: str = "bic",
                     trend: str = "c") -> int:
    """Lag-order selection by an information criterion via statsmodels.

    ``criterion`` is one of ``"aic" | "bic" | "hqic" | "fpe"``. Common
    Hannan-Quinn spellings (``"hq"``, ``"hannan_quinn"``, ``"hannan-quinn"``)
    are accepted as aliases for ``"hqic"`` — the skills and ``config.yaml``
    abbreviate it "HQ".
    """
    aliases = {"hq": "hqic", "hannan_quinn": "hqic", "hannan-quinn": "hqic"}
    criterion = aliases.get(str(criterion).lower(), str(criterion).lower())
    # Plain ndarray: avoids statsmodels' no-frequency ValueWarning (the
    # index is not used by anything read from this fit).
    model = _StatsmodelsVAR(np.asarray(data))
    sel = model.select_order(maxlags=maxlags, trend=trend)
    crit = {"aic": sel.aic, "bic": sel.bic, "hqic": sel.hqic, "fpe": sel.fpe}
    if criterion not in crit:
        raise ValueError(f"criterion must be one of {list(crit)}, got {criterion!r}")
    return int(crit[criterion])


# ---------------------------------------------------------------------------
# Linear-algebra helpers (shared across modules)
# ---------------------------------------------------------------------------


def _ensure_psd(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Add a small identity ridge if needed to make ``M`` positive definite.

    Lives here (rather than in ``inference.py``) so both ``var`` and
    ``inference`` can use it without inducing a circular import. The
    leading underscore signals "internal — not part of the public API".
    """
    eig_min = float(np.linalg.eigvalsh(M).min())
    if eig_min < 0:
        return M + np.eye(M.shape[0]) * (abs(eig_min) + tol)
    return M
