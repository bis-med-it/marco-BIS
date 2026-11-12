"""Panel VAR estimation — Canova-Ciccarelli (2013) family.

The Canova-Ciccarelli "Panel VAR Models: A Survey" (2013) covers a
taxonomy of methods for VARs estimated on cross-sectional panels (N
units × T periods each). This module implements four estimators from
that taxonomy:

* :func:`fit_pooled_var` — pooled VAR with unit fixed effects. Single
  shared lag coefficients across all units; per-unit intercepts.

* :func:`fit_mean_group` — Pesaran-Smith (1995) mean-group estimator.
  Fit each unit separately, average lag coefficients across units.

* :func:`fit_partial_pooling_var` — Bayesian partial-pooling
  (hierarchical Minnesota with shrinkage to a common mean).

* :func:`fit_global_var` — stacked cross-country VAR with cross-unit
  interdependence (Canova-Ciccarelli §3.2, estimated with the
  large-scale-VAR shrinkage of §6.1). It does not implement the
  weighted-foreign-variable GVAR of Dees et al. (2007).
  Country j's lagged variables can affect country i's contemporaneous
  variables. Stacks the panel into a single VAR of dimension
  ``n_units × n_vars_per_unit`` with Minnesota shrinkage that tightens
  cross-unit blocks toward zero by an additional ``lambda_cross_unit``
  factor.

CURRENT SCOPE:

  All four estimators support the standard VAR(p) + Cholesky /
  proxy-SVAR / sign-restriction identification machinery via the
  unified :class:`PanelVARFit` container, which subclasses
  :class:`VARFit` so it works with all existing identification
  routines.

Not supported:

  * Exchangeable-prior PVAR (estimate common mean from a subset, shrink
    remaining units toward it).
  * Time-varying coefficient PVAR.
  * Factor-augmented PVAR (common factor + idiosyncratic component).


INPUT FORMAT:

  Panel data accepted as either:
    * dict ``{unit_id: DataFrame}`` where each DataFrame has identical
      columns (the n endogenous variables) and a DatetimeIndex.
    * Long-format DataFrame with a `unit_col` column identifying units;
      gets converted via groupby.

  Convention: all units must have the SAME variable columns. Sample
  windows can differ across units (the function aligns each unit's data
  individually). T per unit must satisfy T > p for the unit to be
  usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .var import (
    VARFit, build_design_matrices, extract_lag_matrices,
    max_eigenvalue_modulus, ma_representation,
)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class PanelVARFit(VARFit):
    """Holder for a fitted panel VAR.

    Subclasses :class:`VARFit` so it can drop into any existing
    identification routine (Cholesky, sign restrictions, proxy SVAR).

    EXTENSIONS BEYOND VARFit:

      * :attr:`panel_data` — the original per-unit data dict.
      * :attr:`units` — list of unit identifiers.
      * :attr:`unit_intercepts` — dict ``{unit: np.ndarray (n,)}`` of
        unit-specific constants.
      * :attr:`unit_residuals` — dict ``{unit: np.ndarray (T_i-p, n)}``
        of per-unit residuals.
      * :attr:`per_unit_B`, ``per_unit_Sigma_u`` — populated only by
        mean-group; ``None`` for pooled.
      * :attr:`method` — ``"pooled"`` or ``"mean_group"``.

    INHERITED FROM VARFit BEHAVIOUR:

      * :attr:`B_hat` — for pooled fits this is the stacked
        ``(n*p + 1, n)`` matrix with the LAST row being a placeholder
        intercept (set to the mean of unit intercepts, so :func:`lag_matrices`
        returns the pooled lag block correctly).
      * :attr:`Sigma_u` — for pooled fits, the pooled residual
        covariance ``(1 / sum_i T_eff_i) * sum_i U_i' U_i``. For
        mean-group, the average of per-unit Sigma_u.
    """

    panel_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    units: list[str] = field(default_factory=list)
    unit_intercepts: dict[str, np.ndarray] = field(default_factory=dict)
    unit_residuals: dict[str, np.ndarray] = field(default_factory=dict)
    per_unit_B: dict[str, np.ndarray] | None = None
    per_unit_Sigma_u: dict[str, np.ndarray] | None = None
    method: str = "pooled"

    # Methods whose VARFit face carries a single canonical unit's data,
    # regressors and residuals purely for interface compatibility. Generic
    # single-VAR inference on that face would misstate panel uncertainty.
    _CANONICAL_UNIT_METHODS = ("pooled", "mean_group", "partial_pooling")

    @property
    def n_units(self) -> int:
        return len(self.units)

    def parameter_draws(self, *args, **kwargs):
        """Reject generic asymptotic draws for canonical-unit panel fits.

        For pooled, mean-group and partial-pooling fits, ``Y_act`` /
        ``X_act`` / ``residuals`` hold ONE canonical unit's sample (kept
        only so identification routines can run) while ``B_hat`` pools
        information across all units — so the generic VARFit asymptotic
        covariance neither matches the estimator nor its sample size.
        These fits support point-estimate IRFs only (``bands="none"``);
        interval inference needs a panel-specific covariance or a panel
        bootstrap (resampling units, or time within units), which this
        module does not implement. Global-VAR fits estimate one genuine
        stacked system and keep the inherited behaviour; hierarchical
        fits override this with posterior draws.
        """
        if self.method in self._CANONICAL_UNIT_METHODS:
            raise NotImplementedError(
                f"parameter_draws is not available for {self.method!r} panel "
                "fits: the fit's VARFit face holds a single canonical unit's "
                "regressors and residuals, so generic asymptotic draws would "
                "misstate the pooled sampling uncertainty. Use bands='none' "
                "(point estimates only), or a panel-specific bootstrap."
            )
        return super().parameter_draws(*args, **kwargs)

    def unit_lag_matrices(self) -> dict[str, list[np.ndarray]]:
        """Per-unit lag matrices (mean-group only)."""
        if self.per_unit_B is None:
            raise ValueError(
                "unit_lag_matrices is only defined for mean-group fits; "
                "pooled fits share coefficients across units"
            )
        return {
            u: extract_lag_matrices(B, self.n_vars, self.p)
            for u, B in self.per_unit_B.items()
        }

    def cross_unit_lag_block(
        self,
        unit_to: str,
        unit_from: str,
        lag: int = 1,
    ) -> np.ndarray:
        """Return the ``K × K`` lag-``lag`` block ``B_{i,j}^{(lag)}``
        for global-VAR fits.

        Captures the contemporaneous effect of ``unit_from``'s variables
        at time ``t - lag`` on ``unit_to``'s variables at time ``t``.

        Parameters
        ----------
        unit_to : str
            Receiving unit identifier (the "i" in ``B_{ij}``).
        unit_from : str
            Source unit identifier (the "j").
        lag : int, default 1
            Lag index (1, 2, ..., p).

        Returns
        -------
        (K, K) np.ndarray where K is the number of variables per unit.

        Raises
        ------
        ValueError if this is not a global-VAR fit (i.e., the underlying
        VAR's variable count does not equal ``n_units * K`` for some K).
        """
        if self.method != "global_var":
            raise ValueError(
                f"cross_unit_lag_block is only defined for global-VAR fits; "
                f"this fit has method={self.method!r}"
            )
        K = self.fit_info["n_vars_per_unit"]
        if unit_to not in self.units:
            raise ValueError(
                f"unknown unit_to={unit_to!r}; expected one of {self.units}"
            )
        if unit_from not in self.units:
            raise ValueError(
                f"unknown unit_from={unit_from!r}; expected one of {self.units}"
            )
        if not (1 <= lag <= self.p):
            raise ValueError(
                f"lag must be in [1, {self.p}]; got {lag}"
            )
        i = self.units.index(unit_to)
        j = self.units.index(unit_from)
        # extract_lag_matrices returns lag_mats[l] of shape (N, N), N = n_units * K,
        # with Y_t = c + sum_l lag_mats[l] @ Y_{t-l-1}; the block
        # [i*K:(i+1)*K, j*K:(j+1)*K] is the effect of unit j at t-lag on unit i at t.
        lag_mats = extract_lag_matrices(self.B_hat, self.n_vars, self.p)
        full_lag_l = lag_mats[lag - 1]   # (N, N)
        return full_lag_l[i * K : (i + 1) * K, j * K : (j + 1) * K].copy()

    def average_cross_unit_spillover(self, lag: int = 1) -> float:
        """Frobenius norm of cross-unit blocks averaged over distinct
        (unit_to, unit_from) pairs at the named lag, normalised by the
        Frobenius norm of own-unit blocks at the same lag.

        Useful summary statistic: 0 = no cross-unit spillovers
        (block-diagonal); 1 = cross-unit blocks as large as own-unit.

        Raises ValueError for non-global-VAR fits.
        """
        if self.method != "global_var":
            raise ValueError(
                f"average_cross_unit_spillover is only defined for global-VAR "
                f"fits; this fit has method={self.method!r}"
            )
        cross_norms = []
        own_norms = []
        for ui in self.units:
            for uj in self.units:
                block = self.cross_unit_lag_block(ui, uj, lag=lag)
                fro = float(np.linalg.norm(block, ord="fro"))
                if ui == uj:
                    own_norms.append(fro)
                else:
                    cross_norms.append(fro)
        if not cross_norms or not own_norms:
            return 0.0
        return float(np.mean(cross_norms)) / max(float(np.mean(own_norms)), 1e-12)


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

def _normalise_panel_input(
    panel: dict[str, pd.DataFrame] | pd.DataFrame,
    unit_col: str = "unit",
) -> dict[str, pd.DataFrame]:
    """Convert long-format DataFrame to dict-of-DataFrames if needed.

    Returns a dict mapping unit_id -> DataFrame, where each DataFrame
    has identical columns (the endogenous variables) and a DatetimeIndex.
    """
    if isinstance(panel, dict):
        return {str(k): v.copy() for k, v in panel.items()}
    if isinstance(panel, pd.DataFrame):
        if unit_col not in panel.columns:
            raise ValueError(
                f"Long-format panel input must have a '{unit_col}' column; "
                f"got columns {list(panel.columns)}"
            )
        return {
            str(u): grp.drop(columns=[unit_col]).copy()
            for u, grp in panel.groupby(unit_col)
        }
    raise TypeError(
        f"panel must be dict or DataFrame; got {type(panel).__name__}"
    )


def _validate_panel(
    panel: dict[str, pd.DataFrame], lags: int,
) -> tuple[list[str], list[str]]:
    """Validate panel structure. Returns (units_kept, variable_names)."""
    if not panel:
        raise ValueError("panel is empty")
    var_names: list[str] | None = None
    for u, df in panel.items():
        if var_names is None:
            var_names = list(df.columns)
        elif list(df.columns) != var_names:
            raise ValueError(
                f"unit {u!r} has columns {list(df.columns)}, expected {var_names}"
            )
        if df.isna().any().any():
            raise ValueError(f"unit {u!r} has NaN values; impute or drop")
    # Drop units with T <= p
    kept = [u for u, df in panel.items() if len(df) > lags]
    if len(kept) < len(panel):
        dropped = [u for u in panel if u not in kept]
        import warnings
        warnings.warn(
            f"Dropped {len(dropped)} unit(s) with insufficient observations "
            f"(T <= p={lags}): {dropped}",
            UserWarning, stacklevel=3,
        )
    if not kept:
        raise ValueError(f"No units have T > p={lags}")
    return kept, var_names


# ---------------------------------------------------------------------------
# Pooled VAR with unit fixed effects
# ---------------------------------------------------------------------------

def fit_pooled_var(
    panel: dict[str, pd.DataFrame] | pd.DataFrame,
    lags: int,
    trend: str = "c",
    unit_col: str = "unit",
) -> PanelVARFit:
    """Fit a pooled VAR with unit-specific intercepts (fixed effects).

    Model:

        Y_{i, t} = c_i + B_1 Y_{i, t-1} + ... + B_p Y_{i, t-p} + u_{i, t}

    where (B_1, ..., B_p) is shared across all units and c_i varies.
    All units' data are stacked and OLS estimated jointly with unit
    dummies for the intercepts. Residuals are pooled to estimate
    Sigma_u with divisor sum_i (T_i - p).

    Parameters
    ----------
    panel : dict[str, DataFrame] or DataFrame
        Either a dict of per-unit DataFrames, or a long-format DataFrame
        with a `unit_col` column.
    lags : int
        Lag order, applied uniformly across units.
    trend : str
        ``"c"`` (constant, default — gives unit-specific intercepts),
        ``"n"`` (no constant — useful when data is pre-demeaned), or
        ``"ct"`` (constant + linear trend; trend is also unit-specific).
    unit_col : str
        Column name identifying units in long-format input. Ignored
        when input is a dict.

    Returns
    -------
    PanelVARFit
        POINT-ESTIMATE ONLY for interval inference: the fit's VARFit
        face carries one canonical unit's sample for interface
        compatibility, so generic ``parameter_draws`` and the residual
        bootstrap are rejected (they would misstate the pooled sampling
        uncertainty). Use ``bands="none"`` downstream.
    """
    panel = _normalise_panel_input(panel, unit_col=unit_col)
    units, var_names = _validate_panel(panel, lags)
    n = len(var_names)
    p = lags

    include_const = trend in ("c", "ct")
    include_trend = trend == "ct"
    n_det_per_unit = (1 if include_const else 0) + (1 if include_trend else 0)

    # Build stacked design matrices.
    # Layout of X_aug per unit: [lag1_var1, ..., lagp_varn, unit_dummies(N_units * n_det_per_unit)]
    # The unit dummies replace the single constant column with N_units columns
    # (one per unit-by-deterministic combination), so the lag block is
    # shared across units.
    Y_blocks = []
    X_lag_blocks = []
    unit_idx_per_row: list[int] = []
    T_eff_per_unit: dict[str, int] = {}

    for i_u, u in enumerate(units):
        Y_u = panel[u].values
        # build_design_matrices with include_const=False gives us just the lag block.
        Y_act_u, X_lag_u = build_design_matrices(
            Y_u, p, include_const=False, include_trend=False,
        )
        Y_blocks.append(Y_act_u)
        X_lag_blocks.append(X_lag_u)
        T_eff_per_unit[u] = Y_act_u.shape[0]
        unit_idx_per_row.extend([i_u] * Y_act_u.shape[0])

    Y_stack = np.vstack(Y_blocks)                 # (sum T_eff, n)
    X_lag_stack = np.vstack(X_lag_blocks)         # (sum T_eff, n*p)
    T_total = Y_stack.shape[0]
    n_units = len(units)

    # Unit dummies for intercepts (and optionally trends)
    det_cols = []
    if include_const:
        for i_u in range(n_units):
            col = np.array([1.0 if idx == i_u else 0.0 for idx in unit_idx_per_row])
            det_cols.append(col)
    if include_trend:
        # Per-unit linear trend
        for i_u, u in enumerate(units):
            T_eff_u = T_eff_per_unit[u]
            col = np.zeros(T_total)
            mask = np.array(unit_idx_per_row) == i_u
            col[mask] = np.arange(1, T_eff_u + 1, dtype=float)
            det_cols.append(col)

    if det_cols:
        D = np.column_stack(det_cols)               # (T_total, N * n_det_per_unit)
        X_stack = np.hstack([X_lag_stack, D])
    else:
        X_stack = X_lag_stack

    # OLS on stacked system
    B_full = np.linalg.solve(X_stack.T @ X_stack, X_stack.T @ Y_stack)
    # B_full shape: (n*p + N*n_det_per_unit, n)

    # Separate lag block and deterministic block
    B_lag = B_full[: n * p, :]                      # (n*p, n) — shared lag coefficients
    B_det = B_full[n * p:, :]                       # deterministic block

    # Compute residuals
    U_stack = Y_stack - X_stack @ B_full

    # Per-unit residuals + intercepts
    unit_residuals: dict[str, np.ndarray] = {}
    unit_intercepts: dict[str, np.ndarray] = {}
    cursor = 0
    for i_u, u in enumerate(units):
        T_eff_u = T_eff_per_unit[u]
        unit_residuals[u] = U_stack[cursor:cursor + T_eff_u].copy()
        cursor += T_eff_u
        if include_const:
            unit_intercepts[u] = B_det[i_u].copy()
        else:
            unit_intercepts[u] = np.zeros(n)

    # Pooled residual covariance
    Sigma_u = (U_stack.T @ U_stack) / T_total

    # Construct a VARFit-compatible B_hat: stack [B_lag, mean(intercepts)] so
    # that VARFit.lag_matrices() returns the pooled lag block and downstream
    # identification routines can use the fit. The mean intercept is a
    # placeholder for IRF visualisation; per-unit intercepts live in
    # unit_intercepts.
    if include_const:
        mean_intercept = np.mean([unit_intercepts[u] for u in units], axis=0)
        B_hat = np.vstack([B_lag, mean_intercept.reshape(1, -1)])
    else:
        B_hat = B_lag

    # Take the first unit's data as the canonical .data for VARFit interface
    # compatibility (used by some identification routines for sample length).
    canonical_data = panel[units[0]]
    canonical_Y_act = Y_blocks[0]
    canonical_X_act = np.hstack([
        X_lag_blocks[0],
        np.ones((X_lag_blocks[0].shape[0], 1)) if include_const else np.zeros((X_lag_blocks[0].shape[0], 0)),
    ]) if include_const else X_lag_blocks[0]

    sign, logdet = np.linalg.slogdet(Sigma_u)
    fit_info = {
        "method": "pooled_var_with_unit_fixed_effects",
        "n_units": n_units,
        "T_per_unit": dict(T_eff_per_unit),
        "T_total": T_total,
        "trend": trend,
        "logdet_Sigma_u": float(logdet),
        "has_exog": False,
        "n_params": int(B_full.size),
    }

    return PanelVARFit(
        data=canonical_data,
        p=p,
        Y_act=canonical_Y_act,
        X_act=canonical_X_act,
        B_hat=B_hat,
        Sigma_u=Sigma_u,
        residuals=unit_residuals[units[0]],   # canonical unit's residuals
        variable_names=var_names,
        fit_info=fit_info,
        panel_data=panel,
        units=units,
        unit_intercepts=unit_intercepts,
        unit_residuals=unit_residuals,
        per_unit_B=None,
        per_unit_Sigma_u=None,
        method="pooled",
    )


# ---------------------------------------------------------------------------
# Mean-group estimator (Pesaran-Smith 1995)
# ---------------------------------------------------------------------------

def fit_mean_group(
    panel: dict[str, pd.DataFrame] | pd.DataFrame,
    lags: int,
    trend: str = "c",
    unit_col: str = "unit",
) -> PanelVARFit:
    """Pesaran-Smith (1995) mean-group estimator.

    Fit a VAR independently per unit; average lag coefficients across
    units. The "average VAR" produced is used for IRF analysis; per-unit
    coefficients are preserved in ``per_unit_B``.

    Average:
        B_bar = (1/N) sum_i B_i
        Sigma_bar = (1/N) sum_i Sigma_u_i

    The mean-group estimator is consistent under the assumption that
    coefficients are random draws from a common distribution (B_i = B_bar
    + v_i, E[v_i] = 0). Less efficient than pooled estimation under true
    homogeneity, but more robust under heterogeneity.

    Parameters: same as :func:`fit_pooled_var`. Like the pooled fit, the
    returned ``PanelVARFit`` is POINT-ESTIMATE ONLY for interval
    inference (generic draws / residual bootstrap are rejected).
    """
    from .var import fit_var

    panel = _normalise_panel_input(panel, unit_col=unit_col)
    units, var_names = _validate_panel(panel, lags)
    n = len(var_names)

    per_unit_fits: dict[str, "VARFit"] = {}
    for u in units:
        per_unit_fits[u] = fit_var(panel[u], lags=lags, trend=trend)

    # Average lag block (n*p rows) across units.
    # Convert per-unit Sigma_u to ML (1/T_eff) to match the other panel
    # estimators (pooled and partial-pooling). fit_var returns
    # statsmodels' Lütkepohl T_eff-K Sigma_u on its plain-VAR path; rescale.
    per_unit_B = {u: f.B_hat for u, f in per_unit_fits.items()}
    per_unit_Sigma = {}
    for u, f in per_unit_fits.items():
        sigma_lutkepohl = np.asarray(f.Sigma_u)
        # Rescale Lütkepohl T_eff-K back to ML divisor T_eff
        sigma_ml = sigma_lutkepohl * (f.t_eff - f.k) / f.t_eff
        per_unit_Sigma[u] = sigma_ml
    B_bar = np.mean([B for B in per_unit_B.values()], axis=0)
    Sigma_bar = np.mean([np.asarray(s) for s in per_unit_Sigma.values()], axis=0)

    # Unit intercepts (per-unit constants if trend in (c, ct))
    include_const = trend in ("c", "ct")
    unit_intercepts: dict[str, np.ndarray] = {}
    unit_residuals: dict[str, np.ndarray] = {}
    for u, f in per_unit_fits.items():
        unit_residuals[u] = f.residuals
        if include_const:
            # B_hat layout: [lag1..lagp, const, (trend)]. Constant is at row n*p.
            unit_intercepts[u] = f.B_hat[n * lags].copy()
        else:
            unit_intercepts[u] = np.zeros(n)

    # Canonical-unit reference for VARFit-interface compatibility
    canonical_unit = units[0]
    canonical_fit = per_unit_fits[canonical_unit]

    fit_info = {
        "method": "mean_group",
        "n_units": len(units),
        "T_per_unit": {u: int(f.t_eff) for u, f in per_unit_fits.items()},
        "trend": trend,
        "has_exog": False,
        "n_params": int(sum(f.B_hat.size for f in per_unit_fits.values())),
    }

    return PanelVARFit(
        data=canonical_fit.data,
        p=lags,
        Y_act=canonical_fit.Y_act,
        X_act=canonical_fit.X_act,
        B_hat=B_bar,
        Sigma_u=Sigma_bar,
        residuals=canonical_fit.residuals,
        variable_names=var_names,
        fit_info=fit_info,
        panel_data=panel,
        units=units,
        unit_intercepts=unit_intercepts,
        unit_residuals=unit_residuals,
        per_unit_B=per_unit_B,
        per_unit_Sigma_u=per_unit_Sigma,
        method="mean_group",
    )


# ---------------------------------------------------------------------------
# Bayesian partial-pooling PVAR (Lindley-Smith / Canova-Ciccarelli)
# ---------------------------------------------------------------------------

def fit_partial_pooling_var(
    panel: dict[str, pd.DataFrame] | pd.DataFrame,
    lags: int,
    trend: str = "c",
    gamma: float = 1.0,
    common_mean: "np.ndarray | str" = "pooled",
    unit_col: str = "unit",
) -> PanelVARFit:
    """Bayesian partial-pooling panel VAR — hierarchical Ridge.

    Uniform-precision Ridge shrinkage of per-unit VARs toward a common
    mean. For each unit ``i``, the posterior coefficient matrix is
    computed equation-by-equation via Ridge regression with the target
    ``B_bar``:

        B_i_post = (X_i'X_i + gamma * I)^{-1} (X_i'X_i @ B_i_ols + gamma * B_bar)

    .. note::

       This is *not* a full hierarchical prior. It uses a single scalar
       shrinkage precision (``gamma * I``) on the deviation ``B_i - B_bar``,
       with the common mean ``B_bar`` FIXED (not learned). For the proper
       exchangeable-prior version — where the common mean ``delta_bar`` and the
       per-coefficient cross-unit dispersion ``tau^2`` (pooling strength) are
       LEARNED by a Gibbs sampler (Canova-Ciccarelli 2013) — use
       :func:`fit_hierarchical_panel_var`.

    where ``B_i_ols`` is the per-unit OLS estimate and ``gamma >= 0`` is
    the shrinkage hyperparameter:

      * ``gamma = 0``: no shrinkage; reduces to per-unit OLS (i.e.,
        :func:`fit_mean_group`'s per-unit fits exactly).
      * ``gamma = inf``: full shrinkage; every unit's coefficients
        collapse to ``B_bar``.
      * Intermediate ``gamma``: partial pooling — bias-variance trade-off.

    The "common mean" ``B_bar`` defaults to the pooled-OLS estimate
    (:func:`fit_pooled_var`). Alternatives:

      * ``common_mean="zero"`` — shrink toward 0 (white-noise prior)
      * ``common_mean="mean_group"`` — shrink toward the simple average
        of per-unit OLS B_i (Pesaran-Smith)
      * ``common_mean=np.ndarray`` — explicit (k, n) matrix

    Per-unit residual covariance is the OLS residual covariance from
    the partial-pooling fit (no shrinkage on Sigma). Aggregate
    ``Sigma_u`` is the average of per-unit Sigma_i.

    Parameters
    ----------
    panel : dict[str, DataFrame] or DataFrame
        Same input format as :func:`fit_pooled_var`.
    lags : int
        Lag order, applied uniformly across units.
    trend : str
        ``"c"``, ``"ct"``, or ``"n"``. Per-unit intercepts.
    gamma : float, default 1.0
        Shrinkage strength. Larger = tighter pooling. Must be >= 0.
    common_mean : np.ndarray or str
        Target for shrinkage. See description above.
    unit_col : str
        Column name identifying units in long-format input.

    Returns
    -------
    PanelVARFit with ``method="partial_pooling"`` and ``per_unit_B``
    populated with the partial-pooling estimates. POINT-ESTIMATE ONLY
    for interval inference (generic ``parameter_draws`` / residual
    bootstrap are rejected — the VARFit face carries one canonical
    unit's sample); use :func:`fit_hierarchical_panel_var` for posterior
    draws.
    """
    from .var import fit_var

    if gamma < 0:
        raise ValueError(f"gamma must be >= 0; got {gamma}")

    panel = _normalise_panel_input(panel, unit_col=unit_col)
    units, var_names = _validate_panel(panel, lags)
    n = len(var_names)

    # Determine B_bar
    if isinstance(common_mean, str):
        if common_mean == "pooled":
            pooled = fit_pooled_var(panel, lags=lags, trend=trend, unit_col=unit_col)
            # Use pooled.B_hat (shape (n*p + n_det, n))
            B_bar = pooled.B_hat.copy()
        elif common_mean == "zero":
            # We need to know k = X_act.shape[1] first; fit one unit to get it
            tmp_fit = fit_var(panel[units[0]], lags=lags, trend=trend)
            B_bar = np.zeros_like(tmp_fit.B_hat)
        elif common_mean == "mean_group":
            mg = fit_mean_group(panel, lags=lags, trend=trend, unit_col=unit_col)
            B_bar = mg.B_hat.copy()
        else:
            raise ValueError(
                f"common_mean string must be 'pooled', 'zero', or 'mean_group'; "
                f"got {common_mean!r}"
            )
    elif isinstance(common_mean, np.ndarray):
        B_bar = common_mean.astype(float).copy()
    else:
        raise TypeError(
            f"common_mean must be a string or np.ndarray; "
            f"got {type(common_mean).__name__}"
        )

    # Fit per-unit OLS first to validate shapes, then apply Ridge shrinkage
    per_unit_B: dict[str, np.ndarray] = {}
    per_unit_Sigma: dict[str, np.ndarray] = {}
    unit_intercepts: dict[str, np.ndarray] = {}
    unit_residuals: dict[str, np.ndarray] = {}
    include_const = trend in ("c", "ct")

    for u in units:
        fit_i = fit_var(panel[u], lags=lags, trend=trend)
        X_i = fit_i.X_act
        Y_i = fit_i.Y_act
        # Validate B_bar shape compatibility
        if B_bar.shape != fit_i.B_hat.shape:
            raise ValueError(
                f"common_mean shape {B_bar.shape} doesn't match per-unit "
                f"B_hat shape {fit_i.B_hat.shape}"
            )
        # Ridge with target B_bar:
        #   B_post = (X'X + gamma * I)^{-1} (X'Y + gamma * B_bar)
        XtX = X_i.T @ X_i
        k = XtX.shape[0]
        regulariser = gamma * np.eye(k)
        B_post = np.linalg.solve(
            XtX + regulariser,
            X_i.T @ Y_i + gamma * B_bar,
        )
        per_unit_B[u] = B_post
        # Recompute residuals and Sigma_u at the partial-pooling point
        resid_i = Y_i - X_i @ B_post
        Sigma_i = (resid_i.T @ resid_i) / X_i.shape[0]
        per_unit_Sigma[u] = Sigma_i
        unit_residuals[u] = resid_i
        if include_const:
            # Constant is at row n*lags
            unit_intercepts[u] = B_post[n * lags].copy()
        else:
            unit_intercepts[u] = np.zeros(n)

    # Aggregate B as the simple average of per-unit B_post; Sigma similarly.
    B_avg = np.mean([B for B in per_unit_B.values()], axis=0)
    Sigma_avg = np.mean([np.asarray(s) for s in per_unit_Sigma.values()], axis=0)

    # Canonical-unit reference (for VARFit interface compatibility)
    canonical_unit = units[0]
    canonical_fit = fit_var(panel[canonical_unit], lags=lags, trend=trend)

    fit_info = {
        "method": "partial_pooling",
        "gamma": float(gamma),
        "common_mean": (
            common_mean if isinstance(common_mean, str) else "ndarray"
        ),
        "n_units": len(units),
        "T_per_unit": {u: int(fit_var(panel[u], lags=lags, trend=trend).t_eff)
                       for u in units},
        "trend": trend,
        "has_exog": False,
    }

    return PanelVARFit(
        data=canonical_fit.data,
        p=lags,
        Y_act=canonical_fit.Y_act,
        X_act=canonical_fit.X_act,
        B_hat=B_avg,
        Sigma_u=Sigma_avg,
        residuals=canonical_fit.residuals,
        variable_names=var_names,
        fit_info=fit_info,
        panel_data=panel,
        units=units,
        unit_intercepts=unit_intercepts,
        unit_residuals=unit_residuals,
        per_unit_B=per_unit_B,
        per_unit_Sigma_u=per_unit_Sigma,
        method="partial_pooling",
    )


# ---------------------------------------------------------------------------
# Stacked cross-country VAR (full cross-unit interdependence) — Canova-
# Ciccarelli §3.2 / §6.1; not a Dees et al. (2007) GVAR.
# ---------------------------------------------------------------------------

def fit_global_var(
    panel: dict[str, pd.DataFrame] | pd.DataFrame,
    lags: int,
    trend: str = "c",
    lambda_1: float = 0.1,
    lambda_2: float = 0.5,
    lambda_cross_unit: float = 0.1,
    lambda_3: float = 1.0,
    lambda_4: float = 100.0,
    unit_col: str = "unit",
) -> PanelVARFit:
    """Stacked cross-country VAR with cross-unit interdependence
    (Canova-Ciccarelli §3.2, with the large-scale-VAR shrinkage of §6.1).
    Not the weighted-foreign-variable GVAR of Dees et al. (2007), which
    links country models through trade-weighted foreign variables.

    Stacks all units' variables into a single VAR of dimension
    ``N = n_units * K`` (where ``K`` is the number of variables per
    unit) and fits with Minnesota-style shrinkage that tightens
    cross-unit blocks toward zero. The model:

    .. math::

       Y_{i,t} = c_i + \\sum_{j=1}^{n_{\\text{units}}}
                  \\sum_{p'=1}^{p} B_{ij}^{(p')} Y_{j, t-p'} + u_{i,t}

    where ``B_{ii}^{(p')}`` is the own-unit lag block (textbook within-
    country dynamics) and ``B_{ij}^{(p')}`` for ``i ≠ j`` captures
    cross-unit spillovers.

    Parameter count: ``N² * p`` lag coefficients + ``N`` intercepts (if
    ``trend != "n"``). For ``n_units = 4`` countries × ``K = 4`` variables
    × ``p = 4`` lags, that is 1024 + 16 = 1040 — heavily overparameterised
    on typical macro samples (T = 100-500), so Minnesota shrinkage is
    essential.

    Minnesota prior on coefficient ``B_{ij}^{(p'), kl}`` (effect of unit
    ``j``'s variable ``l`` at lag ``p'`` on unit ``i``'s variable ``k``):

    .. math::

       \\text{Var}\\!\\left(B_{ij}^{(p'), kl}\\right)
         = \\left(\\frac{\\lambda_{\\text{eff}}}{(p')^{\\lambda_3}}
                   \\cdot \\frac{\\sigma_{ik}}{\\sigma_{jl}}\\right)^2

    with ``λ_eff`` differing by block type:

    +---------------------------+-------------------------------------------------+
    | Block type                | λ_eff                                           |
    +===========================+=================================================+
    | Own-unit, own-variable    | λ₁                                              |
    +---------------------------+-------------------------------------------------+
    | Own-unit, cross-variable  | λ₁ · λ₂                                         |
    +---------------------------+-------------------------------------------------+
    | Cross-unit, own-variable  | λ₁ · λ_cross_unit                               |
    +---------------------------+-------------------------------------------------+
    | Cross-unit, cross-variab. | λ₁ · λ₂ · λ_cross_unit                          |
    +---------------------------+-------------------------------------------------+

    Prior mean: random walk on own-own at lag 1
    (``B_{ii}^{(1), kk} = 1``), zero elsewhere.

    Fit equation-by-equation via Ridge regression with the per-equation
    prior precision matrix:

    .. math::

       \\hat\\beta_{eq} = \\left(X^\\top X + \\Phi^{-1}\\right)^{-1}
                      \\left(X^\\top y_{eq} + \\Phi^{-1} \\mu\\right)

    where ``Φ = diag(prior_var)`` and ``μ`` is the prior mean.

    Parameters
    ----------
    panel : dict[str, DataFrame] or DataFrame
        Per-unit data. All units MUST have identical DatetimeIndex —
        align via ``pd.concat([df_us, df_jp], axis=1).dropna()`` upstream
        if the windows differ.
    lags : int
        Lag order, applied uniformly.
    trend : str
        ``"c"``, ``"ct"``, or ``"n"``. Constants are per-equation
        (i.e., for each of the ``N = n_units × K`` stacked variables).
    lambda_1 : float
        Overall tightness on own-lag, own-variable coefficients.
    lambda_2 : float
        Cross-variable tightness (within unit). Larger = looser
        cross-variable spillovers.
    lambda_cross_unit : float
        Cross-unit tightness. Larger = looser cross-unit spillovers.
        At ``lambda_cross_unit = 0`` cross-unit blocks are exactly zero
        (block-diagonal global VAR = mean-group with shared sigmas).
        At ``lambda_cross_unit = 1`` cross-unit blocks have the same
        prior tightness as cross-variable own-unit blocks.
        Typical: ``0.01-0.2`` for genuine cross-country spillovers.
    lambda_3 : float
        Lag decay exponent. Distant-lag prior variance scales as
        ``1 / (lag^lambda_3)^2``.
    lambda_4 : float
        Intercept-tightness multiplier. The prior std on each constant
        is ``lambda_1 * lambda_4`` (loose; default = 10 makes the
        intercept essentially unrestricted).
    unit_col : str
        Column name identifying units in long-format input.

    Returns
    -------
    PanelVARFit with ``method="global_var"``. The underlying VAR is on
    ``N = n_units * K`` stacked variables (named ``"{unit}_{var}"``).
    Access cross-unit blocks via :meth:`PanelVARFit.cross_unit_lag_block`
    or the summary statistic :meth:`PanelVARFit.average_cross_unit_spillover`.

    All identification routines (cholesky_irf, sign_restricted_irf,
    proxy_svar_irf) work on the returned fit because it subclasses VARFit
    and presents an N-variable VAR(p) interface.

    References
    ----------
    Dees, S., di Mauro, F., Pesaran, M.H. and Smith, L.V. (2007).
    "Exploring the international linkages of the euro area: a global VAR
    analysis." Journal of Applied Econometrics, 22(1), 1-38.
    Canova, F. and Ciccarelli, M. (2013). "Panel VAR models: a survey."
    VAR Models in Macroeconomics — New Developments and Applications,
    Advances in Econometrics 32, 205-246. §3.2, §6.1 (GVARs: §6.4).
    """
    from .bvar import _presample_ar_sigmas

    if lambda_1 < 0 or lambda_2 < 0 or lambda_cross_unit < 0 or lambda_3 < 0:
        raise ValueError(
            "all lambda hyperparameters must be >= 0; "
            f"got lambda_1={lambda_1}, lambda_2={lambda_2}, "
            f"lambda_cross_unit={lambda_cross_unit}, lambda_3={lambda_3}"
        )
    if lambda_4 <= 0:
        raise ValueError(f"lambda_4 must be > 0; got {lambda_4}")

    panel = _normalise_panel_input(panel, unit_col=unit_col)
    units, var_names = _validate_panel(panel, lags)
    K = len(var_names)
    n_units = len(units)
    N = n_units * K

    # Time alignment check
    canonical_index = panel[units[0]].index
    for u in units[1:]:
        if not panel[u].index.equals(canonical_index):
            raise ValueError(
                f"unit {u!r} has DatetimeIndex differing from {units[0]!r}; "
                f"global VAR requires all units on a common index. Align "
                f"via pd.concat([df_a, df_b], axis=1).dropna() first."
            )

    # Stack columns: order is (unit_0_var_0, unit_0_var_1, ..., unit_n-1_var_K-1)
    stacked_var_names: list[str] = []
    for u in units:
        for v in var_names:
            stacked_var_names.append(f"{u}_{v}")
    Y_stack = np.hstack([panel[u].values for u in units])    # (T, N)
    T = Y_stack.shape[0]
    T_eff = T - lags

    # Per-equation presample residual sigmas (Sims-Zha / BGR convention)
    sigmas = _presample_ar_sigmas(Y_stack, lags)              # (N,)

    # Design matrices: just the lag block (constant added separately)
    Y_act, X_lag = build_design_matrices(
        Y_stack, lags, include_const=False, include_trend=False,
    )
    include_const = trend in ("c", "ct")
    include_trend = trend == "ct"
    n_det = (1 if include_const else 0) + (1 if include_trend else 0)
    if n_det > 0:
        det_cols = []
        if include_const:
            det_cols.append(np.ones(T_eff))
        if include_trend:
            det_cols.append(np.arange(1, T_eff + 1, dtype=float))
        D = np.column_stack(det_cols)
        X_act = np.hstack([X_lag, D])
    else:
        X_act = X_lag
    P = X_act.shape[1]      # N*lags + n_det

    # Equation-by-equation Ridge with per-equation prior
    B_hat = np.zeros((P, N))
    XtX = X_act.T @ X_act
    XtY = X_act.T @ Y_act
    for eq in range(N):
        unit_eq = eq // K
        var_eq = eq % K
        prior_var = np.empty(P)
        prior_mean = np.zeros(P)
        # Lag-block coefficients
        for lag_idx in range(lags):
            p_prime = lag_idx + 1
            decay = p_prime ** lambda_3
            for col in range(N):
                unit_col_ = col // K
                var_col_ = col % K
                # Determine effective lambda
                if unit_col_ == unit_eq:
                    if var_col_ == var_eq:
                        lam_eff = lambda_1                              # own
                    else:
                        lam_eff = lambda_1 * lambda_2                   # own-unit cross-var
                else:
                    if var_col_ == var_eq:
                        lam_eff = lambda_1 * lambda_cross_unit          # cross-unit own-var
                    else:
                        lam_eff = (
                            lambda_1 * lambda_2 * lambda_cross_unit
                        )                                                # cross-unit cross-var
                idx = lag_idx * N + col
                std = (lam_eff / decay) * sigmas[eq] / sigmas[col]
                prior_var[idx] = max(std ** 2, 1e-20)
                # Prior mean: random walk on own-own at lag 1
                if (
                    lag_idx == 0
                    and unit_col_ == unit_eq
                    and var_col_ == var_eq
                ):
                    prior_mean[idx] = 1.0
        # Deterministic-block coefficients (loose prior)
        det_start = N * lags
        if include_const:
            prior_var[det_start] = (lambda_1 * lambda_4 * sigmas[eq]) ** 2
            prior_mean[det_start] = 0.0
        if include_trend:
            prior_var[det_start + (1 if include_const else 0)] = (
                lambda_1 * lambda_4 * sigmas[eq]
            ) ** 2
            prior_mean[det_start + (1 if include_const else 0)] = 0.0

        Phi_inv_diag = 1.0 / prior_var                           # (P,)
        # Solve the scaled normal equations
        #   (X'X / sigma_eq^2 + diag(Phi_inv)) beta
        #       = X'y / sigma_eq^2 + diag(Phi_inv) * mu
        # The Gaussian likelihood's precision is X'X / sigma_eq^2; adding the
        # absolute coefficient prior to UNSCALED X'X would implicitly give
        # every equation unit innovation variance, making the shrinkage (and
        # therefore the coefficients) change with the data's measurement
        # units. sigma_eq is the presample AR noise scale used by the prior.
        s2 = float(sigmas[eq] ** 2)
        A = XtX / s2 + np.diag(Phi_inv_diag)
        b = XtY[:, eq] / s2 + Phi_inv_diag * prior_mean
        B_hat[:, eq] = np.linalg.solve(A, b)

    # Residuals and Sigma_u
    U = Y_act - X_act @ B_hat                                    # (T_eff, N)
    Sigma_u = (U.T @ U) / T_eff

    # Build per-unit views into the stacked fit
    unit_intercepts: dict[str, np.ndarray] = {}
    unit_residuals: dict[str, np.ndarray] = {}
    for i_u, u in enumerate(units):
        eq_slice = slice(i_u * K, (i_u + 1) * K)
        if include_const:
            unit_intercepts[u] = B_hat[N * lags, eq_slice].copy()
        else:
            unit_intercepts[u] = np.zeros(K)
        unit_residuals[u] = U[:, eq_slice].copy()

    sign, logdet = np.linalg.slogdet(Sigma_u)
    fit_info = {
        "method": "global_var",
        "n_units": n_units,
        "n_vars_per_unit": K,
        "T_per_unit": {u: T_eff for u in units},
        "T_eff": T_eff,
        "trend": trend,
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "lambda_cross_unit": float(lambda_cross_unit),
        "lambda_3": float(lambda_3),
        "lambda_4": float(lambda_4),
        "logdet_Sigma_u": float(logdet),
        "has_exog": False,
        "n_params": int(B_hat.size),
        "presample_sigmas": sigmas.tolist(),
        "unit_var_layout": stacked_var_names,
    }

    # Canonical .data for VARFit interface: the stacked DataFrame
    canonical_data = pd.DataFrame(
        Y_stack, columns=stacked_var_names, index=canonical_index,
    )

    return PanelVARFit(
        data=canonical_data,
        p=lags,
        Y_act=Y_act,
        X_act=X_act,
        B_hat=B_hat,
        Sigma_u=Sigma_u,
        residuals=U,
        variable_names=stacked_var_names,
        fit_info=fit_info,
        panel_data=panel,
        units=units,
        unit_intercepts=unit_intercepts,
        unit_residuals=unit_residuals,
        per_unit_B=None,
        per_unit_Sigma_u=None,
        method="global_var",
    )


# ===========================================================================
# Hierarchical (random-coefficient) Bayesian panel VAR -- Canova-Ciccarelli
# ===========================================================================
# Completes fit_partial_pooling_var's simplified Ridge with the proper
# exchangeable prior: beta_i ~ N(delta_bar, diag(tau^2)), where the common mean
# delta_bar and the per-coefficient cross-unit dispersion tau^2 are LEARNED from
# the data by a Gibbs sampler -- so the amount of pooling is estimated rather
# than fixed by a single scalar gamma. A DIAGONAL Sigma_delta keeps the
# posterior well-posed for any number of units (a full kn x kn inverse-Wishart
# would need N >= kn units). This is the hierarchical model of
# Canova-Ciccarelli (2013). A common alternative scales a FIXED Minnesota
# diagonal by ONE learned scalar; this learns a per-coefficient diagonal tau^2
# (a richer diagonal -- own and cross lags can pool differently). Neither uses
# a full Sigma_delta.


def _symm(M):
    return 0.5 * (M + M.T)


@dataclass
class HierarchicalPanelPrior:
    """Exchangeable random-coefficient prior for fit_hierarchical_panel_var.

    ``beta_i ~ N(delta_bar, diag(tau^2))`` across units. The common mean
    ``delta_bar`` and the per-coefficient cross-unit dispersion ``tau^2``
    (pooling strength) are learned:

      * small ``tau_j^2`` => coefficient j is strongly pooled toward delta_bar_j
      * large ``tau_j^2`` => coefficient j is unit-specific

    Parameters
    ----------
    tau_a0, tau_b0 : float
        Inverse-gamma hyperprior (shape, scale) on each ``tau_j^2``. Weakly
        informative defaults (2.0, 0.5). The default scale is large for VAR
        coefficients (a prior cross-unit standard deviation of about 0.4 to
        0.7), so with the defaults the estimator pools little and stays close
        to unit-by-unit estimates; lower ``tau_b0`` for more pooling (0.01
        corresponds to a standard deviation near 0.1).
    delta_bar_sd : float
        Standard deviation of the (loose) Normal hyperprior on each common-mean
        coefficient ``delta_bar_j``. Default 10.0.
    sigma_prior_df : int, optional
        Inverse-Wishart df for the per-unit residual covariance ``Sigma_i``.
        Default ``n + 2`` (proper, weak); the scale is the identity.
    """

    tau_a0: float = 2.0
    tau_b0: float = 0.5
    delta_bar_sd: float = 10.0
    sigma_prior_df: int | None = None


@dataclass
class HierarchicalPanelVARFit(PanelVARFit):
    """Hierarchical panel VAR fit. ``per_unit_B`` holds the shrunk unit-specific
    coefficients; ``common_mean_B`` the learned common dynamics (delta_bar) and
    ``pooling_tau2`` the learned per-coefficient cross-unit variance (small =>
    strongly pooled). The aggregate VARFit view (``B_hat``) is the common mean,
    so Cholesky/sign IRFs and FEVD describe the representative panel VAR."""

    common_mean_B: np.ndarray = field(default=None, repr=False)    # (k, n)
    pooling_tau2: np.ndarray = field(default=None, repr=False)     # (k, n)
    delta_draws: np.ndarray = field(default=None, repr=False)      # (D, k*n)
    sigma_agg_draws: np.ndarray = field(default=None, repr=False)  # (D, n, n)

    def parameter_draws(self, n_draws, seed=None, require_stable=True,
                        eig_tol=1.01, burn_in=0, **_):
        """(B, Sigma) draws for the COMMON (representative) panel VAR, from the
        stored hierarchical posterior draws of delta_bar and the aggregate
        Sigma."""
        from .var import max_eigenvalue_modulus
        rng = np.random.default_rng(seed)
        n, p, k = self.n_vars, self.p, self.k
        M = self.delta_draws.shape[0]
        idx = np.arange(M) if n_draws == M else rng.integers(0, M, size=n_draws)
        out = []
        for kk in idx:
            B = self.delta_draws[kk].reshape((k, n), order="F")
            if require_stable:
                if max_eigenvalue_modulus(extract_lag_matrices(B, n, p)) >= eig_tol:
                    continue
            out.append((B, self.sigma_agg_draws[kk]))
        return out


# =============================================================================
# EXPERIMENTAL
# The estimator below is less established than the core VAR code.
# It rests on the library's own tests on simulated data. Treat results with
# caution and confirm important ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
def fit_hierarchical_panel_var(panel, lags, trend="c", prior=None,
                               n_draws=2000, burn_in=1000, seed=None,
                               unit_col="unit"):
    """Hierarchical random-coefficient Bayesian panel VAR (Canova-Ciccarelli).

    Each unit ``i`` has its own VAR coefficients ``beta_i = vec(B_i)``, with the
    exchangeable prior ``beta_i ~ N(delta_bar, diag(tau^2))``. A Gibbs sampler
    jointly draws the unit coefficients ``beta_i``, the per-unit residual
    covariances ``Sigma_i``, the common mean ``delta_bar``, and the
    per-coefficient pooling variances ``tau^2`` (so the degree of pooling is
    estimated). This is the proper hierarchical upgrade of
    :func:`fit_partial_pooling_var`'s scalar-gamma Ridge.

    Parameters
    ----------
    panel : dict[str, DataFrame] or long DataFrame
        Same input format as the other panel fits.
    lags : int
    trend : str, default "c"
    prior : HierarchicalPanelPrior, optional
    n_draws, burn_in : int
    seed : int, optional
    unit_col : str

    Returns
    -------
    HierarchicalPanelVARFit with ``method="hierarchical"``, ``per_unit_B`` shrunk
    toward ``common_mean_B``, and ``pooling_tau2`` reporting the learned
    cross-unit dispersion per coefficient.
    """
    from scipy.stats import invwishart, invgamma
    from .var import fit_var

    if prior is None:
        prior = HierarchicalPanelPrior()
    panel = _normalise_panel_input(panel, unit_col=unit_col)
    units, var_names = _validate_panel(panel, lags)
    n = len(var_names)
    rng = np.random.default_rng(seed)

    designs, B_ols, Sig = {}, {}, {}
    for u in units:
        fit_i = fit_var(panel[u], lags=lags, trend=trend)
        designs[u] = (fit_i.X_act, fit_i.Y_act, fit_i.X_act.T @ fit_i.X_act,
                      fit_i.X_act.T @ fit_i.Y_act, fit_i.X_act.shape[0])
        B_ols[u] = fit_i.B_hat.copy()
        Sig[u] = np.atleast_2d(fit_i.Sigma_u).copy()
    k = next(iter(designs.values()))[0].shape[1]
    kn = k * n
    N = len(units)
    df0 = (n + 2) if prior.sigma_prior_df is None else int(prior.sigma_prior_df)
    S0 = np.eye(n)
    lam0 = prior.delta_bar_sd ** 2

    beta_stack = np.column_stack([B_ols[u].flatten(order="F") for u in units]).T  # (N, kn)
    delta_bar = beta_stack.mean(axis=0)
    tau2 = beta_stack.var(axis=0) + 1e-6

    sumB = {u: np.zeros((k, n)) for u in units}
    sumSig = {u: np.zeros((n, n)) for u in units}
    sum_tau2 = np.zeros(kn)
    delta_draws = np.empty((n_draws, kn))
    sigma_agg_draws = np.empty((n_draws, n, n))

    for it in range(burn_in + n_draws):
        betas = np.empty((N, kn))
        sig_sum = np.zeros((n, n))
        for ui, u in enumerate(units):
            X, Y, XtX, XtY, Ti = designs[u]
            Sinv = np.linalg.inv(Sig[u])
            prec = np.diag(1.0 / tau2) + np.kron(Sinv, XtX)
            cov = np.linalg.inv(_symm(prec))
            rhs = delta_bar / tau2 + (XtY @ Sinv).flatten(order="F")
            beta_i = rng.multivariate_normal(cov @ rhs, _symm(cov))
            betas[ui] = beta_i
            Bi_u = beta_i.reshape((k, n), order="F")
            E = Y - X @ Bi_u
            Sig[u] = np.atleast_2d(invwishart.rvs(df=df0 + Ti,
                                                  scale=_symm(S0 + E.T @ E),
                                                  random_state=rng))
            sig_sum += Sig[u]
            if it >= burn_in:
                sumB[u] += Bi_u
                sumSig[u] += Sig[u]
        # common mean delta_bar | betas, tau2  (per coefficient Normal; delta0=0)
        post_var = 1.0 / (N / tau2 + 1.0 / lam0)
        delta_bar = (post_var * (betas.sum(axis=0) / tau2)
                     + np.sqrt(post_var) * rng.standard_normal(kn))
        # pooling tau2_j | betas, delta_bar  (inverse-gamma per coefficient)
        ssq = ((betas - delta_bar) ** 2).sum(axis=0)
        tau2 = invgamma.rvs(a=prior.tau_a0 + N / 2.0,
                            scale=prior.tau_b0 + 0.5 * ssq,
                            size=kn, random_state=rng)
        if it >= burn_in:
            sum_tau2 += tau2
            delta_draws[it - burn_in] = delta_bar
            sigma_agg_draws[it - burn_in] = sig_sum / N

    per_unit_B = {u: sumB[u] / n_draws for u in units}
    per_unit_Sigma = {u: sumSig[u] / n_draws for u in units}
    common_mean_B = delta_draws.mean(axis=0).reshape((k, n), order="F")
    pooling_tau2 = (sum_tau2 / n_draws).reshape((k, n), order="F")
    Sigma_agg = np.mean([per_unit_Sigma[u] for u in units], axis=0)

    u0 = units[0]
    X0, Y0 = designs[u0][0], designs[u0][1]
    resid0 = Y0 - X0 @ common_mean_B
    return HierarchicalPanelVARFit(
        data=panel[u0], p=lags, Y_act=Y0, X_act=X0,
        B_hat=common_mean_B, Sigma_u=Sigma_agg, residuals=resid0,
        variable_names=list(var_names),
        fit_info={"method": "hierarchical_panel (Canova-Ciccarelli)",
                  "n_units": N, "n_draws": int(n_draws), "burn_in": int(burn_in),
                  # Required by forecast consumers: a missing trend field is
                  # defaulted to "c", which would misread a no-constant fit's
                  # last lag-coefficient row as an intercept.
                  "trend": trend},
        panel_data=panel, units=list(units),
        per_unit_B=per_unit_B, per_unit_Sigma_u=per_unit_Sigma,
        method="hierarchical",
        common_mean_B=common_mean_B, pooling_tau2=pooling_tau2,
        delta_draws=delta_draws, sigma_agg_draws=sigma_agg_draws,
    )
