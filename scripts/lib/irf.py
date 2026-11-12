"""
Standardised IRF and FEVD containers.

Every estimation skill (VAR, BVAR, LP, sign-restricted VAR, etc.) produces an
``IRFContainer`` so downstream consumers (results-presentation, benchmarking,
LP-vs-VAR comparison) read a single format.

Internal storage: numpy arrays of shape ``(H+1, n_vars, n_shocks)`` where
``axis 1`` is the responding variable and ``axis 2`` is the shock. This matches
the convention used in ``lib.report.VARReport.add_irf``.

Export via ``to_contract_dict()`` produces the dict-of-dicts format documented in
``DEVELOPMENT.md`` (Interface contracts → Standardised IRF structure).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class IRFContainer:
    """Standardised impulse-response container.

    Parameters
    ----------
    median, lower_68, upper_68, lower_90, upper_90 : ndarray
        Shape ``(H+1, n_vars, n_shocks)``. ``median[h, i, j]`` is the median
        response of variable ``i`` to shock ``j`` at horizon ``h``.
    variable_names : list[str]
        Names of the response variables, ordered to match axis 1.
    shock_names : list[str]
        Names of the shocks, ordered to match axis 2.
    metadata : dict
        Free-form metadata. Recommended keys: ``method``, ``identification``,
        ``sample_start``, ``sample_end``, ``n_lags``, ``n_draws``, ``seed``.
    """

    median: np.ndarray
    lower_68: np.ndarray
    upper_68: np.ndarray
    lower_90: np.ndarray
    upper_90: np.ndarray
    variable_names: list[str]
    shock_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = [self.median, self.lower_68, self.upper_68, self.lower_90, self.upper_90]
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1:
            raise ValueError(f"IRF arrays must share a shape; got {shapes}")
        h_plus_one, n_vars, n_shocks = arrays[0].shape
        if n_vars != len(self.variable_names):
            raise ValueError(
                f"axis 1 of IRF arrays is {n_vars} but variable_names has "
                f"{len(self.variable_names)}"
            )
        if n_shocks != len(self.shock_names):
            raise ValueError(
                f"axis 2 of IRF arrays is {n_shocks} but shock_names has "
                f"{len(self.shock_names)}"
            )

    @property
    def horizon(self) -> int:
        return self.median.shape[0] - 1

    @property
    def horizons(self) -> np.ndarray:
        return np.arange(self.median.shape[0])

    @property
    def n_vars(self) -> int:
        return self.median.shape[1]

    @property
    def n_shocks(self) -> int:
        return self.median.shape[2]

    @classmethod
    def from_draws(
        cls,
        draws: np.ndarray,
        variable_names: list[str],
        shock_names: list[str],
        metadata: dict[str, Any] | None = None,
        ci_levels: tuple[float, float] = (0.68, 0.90),
    ) -> "IRFContainer":
        """Build a container by taking quantiles across a draw axis.

        ``draws`` has shape ``(n_draws, H+1, n_vars, n_shocks)``. Median plus
        68%/90% credible intervals are computed along ``axis 0``.
        """
        if draws.ndim != 4:
            raise ValueError(f"draws must be 4-D; got shape {draws.shape}")
        c68, c90 = ci_levels
        lo68, hi68 = 50 - 100 * c68 / 2, 50 + 100 * c68 / 2
        lo90, hi90 = 50 - 100 * c90 / 2, 50 + 100 * c90 / 2
        return cls(
            median=np.median(draws, axis=0),
            lower_68=np.percentile(draws, lo68, axis=0),
            upper_68=np.percentile(draws, hi68, axis=0),
            lower_90=np.percentile(draws, lo90, axis=0),
            upper_90=np.percentile(draws, hi90, axis=0),
            variable_names=list(variable_names),
            shock_names=list(shock_names),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_point(
        cls,
        point: np.ndarray,
        variable_names: list[str],
        shock_names: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> "IRFContainer":
        """Build a container with no uncertainty (bands equal to the point)."""
        return cls(
            median=point.copy(),
            lower_68=point.copy(),
            upper_68=point.copy(),
            lower_90=point.copy(),
            upper_90=point.copy(),
            variable_names=list(variable_names),
            shock_names=list(shock_names),
            metadata=dict(metadata or {}),
        )

    def select(self, shocks: list[str] | None = None,
               variables: list[str] | None = None) -> "IRFContainer":
        """Return a sub-container restricted to chosen shocks / variables."""
        if shocks is None:
            shock_idx = list(range(self.n_shocks))
            new_shock_names = list(self.shock_names)
        else:
            shock_idx = [self.shock_names.index(s) for s in shocks]
            new_shock_names = list(shocks)
        if variables is None:
            var_idx = list(range(self.n_vars))
            new_var_names = list(self.variable_names)
        else:
            var_idx = [self.variable_names.index(v) for v in variables]
            new_var_names = list(variables)

        def _slice(arr: np.ndarray) -> np.ndarray:
            return arr[:, var_idx, :][:, :, shock_idx]

        return IRFContainer(
            median=_slice(self.median),
            lower_68=_slice(self.lower_68),
            upper_68=_slice(self.upper_68),
            lower_90=_slice(self.lower_90),
            upper_90=_slice(self.upper_90),
            variable_names=new_var_names,
            shock_names=new_shock_names,
            metadata=dict(self.metadata),
        )

    def to_contract_dict(self) -> dict[str, dict[str, dict[str, list]]]:
        """Export to the dict-of-dicts format documented in DEVELOPMENT.md.

        Returns
        -------
        ``{shock_name: {variable_name: {"horizon": [...], "median": [...],
          "lower_68": [...], "upper_68": [...], "lower_90": [...],
          "upper_90": [...]}}}``
        """
        horizons = self.horizons.tolist()
        out: dict[str, dict[str, dict[str, list]]] = {}
        for j, shock in enumerate(self.shock_names):
            out[shock] = {}
            for i, var in enumerate(self.variable_names):
                out[shock][var] = {
                    "horizon": horizons,
                    "median": self.median[:, i, j].tolist(),
                    "lower_68": self.lower_68[:, i, j].tolist(),
                    "upper_68": self.upper_68[:, i, j].tolist(),
                    "lower_90": self.lower_90[:, i, j].tolist(),
                    "upper_90": self.upper_90[:, i, j].tolist(),
                }
        return out

    def save_snapshot(self, path: str | Path) -> Path:
        """Save a JSON snapshot for regression-testing / replication archives.

        Contains a coarse fingerprint (rounded to 6 decimals) of the IRFs and
        metadata. Designed to detect numerical drift between library versions.
        """
        path = Path(path)
        snapshot = {
            "metadata": self.metadata,
            "variable_names": self.variable_names,
            "shock_names": self.shock_names,
            "horizons": self.horizons.tolist(),
            "irf": {
                "median": np.round(self.median, 6).tolist(),
                "lower_68": np.round(self.lower_68, 6).tolist(),
                "upper_68": np.round(self.upper_68, 6).tolist(),
                "lower_90": np.round(self.lower_90, 6).tolist(),
                "upper_90": np.round(self.upper_90, 6).tolist(),
            },
        }
        path.write_text(json.dumps(snapshot, indent=2))
        return path

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "IRFContainer":
        """Reconstruct an :class:`IRFContainer` from a ``save_snapshot`` file.

        The inverse of :meth:`save_snapshot` (up to the snapshot's 6-decimal
        rounding). Useful for comparing a new run against an archived
        replication's IRFs without re-estimating — e.g. overlaying a variant
        on the canonical baseline in a robustness section.
        """
        snap = json.loads(Path(path).read_text())
        irf = snap["irf"]
        return cls(
            median=np.asarray(irf["median"], dtype=float),
            lower_68=np.asarray(irf["lower_68"], dtype=float),
            upper_68=np.asarray(irf["upper_68"], dtype=float),
            lower_90=np.asarray(irf["lower_90"], dtype=float),
            upper_90=np.asarray(irf["upper_90"], dtype=float),
            variable_names=list(snap["variable_names"]),
            shock_names=list(snap["shock_names"]),
            metadata=dict(snap.get("metadata") or {}),
        )


@dataclass
class FEVDContainer:
    """Standardised forecast-error variance-decomposition container.

    Internal: shares of variable variance attributable to each shock at each
    horizon. ``shares[h, i, j]`` = share of variable ``i``'s forecast-error
    variance at horizon ``h`` attributable to shock ``j``. Rows sum to 1.
    """

    median: np.ndarray             # (H+1, n_vars, n_shocks)
    lower_68: np.ndarray
    upper_68: np.ndarray
    variable_names: list[str]
    shock_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_draws(
        cls,
        draws: np.ndarray,
        variable_names: list[str],
        shock_names: list[str],
        metadata: dict[str, Any] | None = None,
        ci_level: float = 0.68,
    ) -> "FEVDContainer":
        lo, hi = 50 - 100 * ci_level / 2, 50 + 100 * ci_level / 2
        return cls(
            median=np.median(draws, axis=0),
            lower_68=np.percentile(draws, lo, axis=0),
            upper_68=np.percentile(draws, hi, axis=0),
            variable_names=list(variable_names),
            shock_names=list(shock_names),
            metadata=dict(metadata or {}),
        )

    def to_contract_dict(self) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
        """Export to DEVELOPMENT.md FEVD contract: by horizon → shock → var → stats."""
        out: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
        for h in range(self.median.shape[0]):
            out[int(h)] = {}
            for j, shock in enumerate(self.shock_names):
                out[h][shock] = {}
                for i, var in enumerate(self.variable_names):
                    out[h][shock][var] = {
                        "median": float(self.median[h, i, j]),
                        "lower_68": float(self.lower_68[h, i, j]),
                        "upper_68": float(self.upper_68[h, i, j]),
                    }
        return out


def structural_fevd(
    fit: "Any",
    structural_impact: np.ndarray,
    horizon: int,
    shock_names: list[str] | None = None,
) -> np.ndarray:
    """Structural forecast-error variance decomposition for an identified system.

    RETURN SHAPE: a single ``(n_vars, n_shocks)`` matrix at the requested
    horizon — NOT an IRF-style ``(horizon, n, m)`` path. Loop over horizons
    (or use ``VARReport.add_standard_sections``, which does) for a table.

    HORIZON CONVENTION (textbook, Lutkepohl §2.2.2 / Hamilton §11.5):

      The ``H``-step-ahead forecast error is ``sum_{s=0..H-1} Phi_s u_{t+H-s}``,
      so its variance sums ``H`` terms (NOT H+1). ``horizon=1`` returns the
      impact-period decomposition; ``horizon=8`` returns the 2-year-ahead
      decomposition for quarterly data.

    FORMULA:

      FEVD_{k, j}(H) = sum_{s=0..H-1} (Phi_s @ S[:, j])_k^2
                     / sum_{s=0..H-1} (Phi_s Sigma_u Phi_s')_{k, k}

    where ``S`` is the structural impact matrix (full or partial). When
    ``S`` is partial (n, m) with m < n, the m identified-shock shares
    don't sum to 1 along rows; the residual is the share attributable to
    unidentified shocks.

    **SHOCK-VARIANCE CONVENTION**: this formula implicitly assumes the
    identified structural shocks have IDENTITY covariance (Σ_ε = I).
    This is true by construction for Cholesky (S = chol(Σ_u),
    S S' = Σ_u) and sign-restricted (S = chol(Σ_u) Q for orthogonal Q)
    identification. It is NOT true for proxy-SVAR with the default
    ``normalization="unit_policy"`` (where S is rescaled so policy
    variable jumps by 1 unit — the implied shock has variance ≠ 1).
    For proxy-SVAR FEVD analysis, either (a) use
    ``normalization="one_sd"`` and accept that shock variance still
    depends on cross-variable structure, or (b) compute FEVD shares
    by hand using an estimated σ²_ε.

    Parameters
    ----------
    fit : VARFit
        Must expose ``ma_rep(horizon)``, ``Sigma_u``, ``n_vars``,
        ``variable_names``.
    structural_impact : np.ndarray
        Shape ``(n, m)`` with ``1 <= m <= n``. A 1-D length-n array is
        accepted and treated as ``m = 1``.
    horizon : int
        Number of forecast steps (>=1).
    shock_names : list[str], optional
        Length ``m``; validated but otherwise unused (the return is a bare
        ndarray — attach names downstream, e.g. in ``FEVDContainer`` or
        the report layer).

    Returns
    -------
    np.ndarray
        Shape ``(n_vars, m)``. Element ``[k, j]`` = share of variable
        ``k``'s ``H``-step-ahead forecast-error variance attributable to
        identified structural shock ``j``.

    Notes
    -----
    Returns a single matrix at the specified horizon (mirrors
    ``lib.generalised_fevd``'s API). To get a profile across multiple
    horizons, loop:

        shares_by_h = {h: structural_fevd(fit, S, h) for h in [1, 4, 8, 12]}
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    S = np.asarray(structural_impact, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    n = fit.n_vars
    if S.shape[0] != n:
        raise ValueError(
            f"structural_impact must have {n} rows (one per variable); "
            f"got {S.shape}"
        )
    m = S.shape[1]
    if m < 1 or m > n:
        raise ValueError(f"identified shocks m must satisfy 1 <= m <= {n}; got {m}")
    if shock_names is not None and len(shock_names) != m:
        raise ValueError(
            f"shock_names must have length {m}; got {len(shock_names)}"
        )

    Phi = fit.ma_rep(horizon - 1)                 # shape (H, n, n) — Phi_0..Phi_{H-1}
    Sigma_u = np.asarray(fit.Sigma_u)

    # Numerator: sum_{s=0..H-1} (Phi_s S)[k, j]^2
    irf_struct = np.einsum("hij,jk->hik", Phi, S)  # (H, n, m)
    numer = (irf_struct ** 2).sum(axis=0)          # (n, m)

    # Denominator: sum_{s=0..H-1} (Phi_s Sigma Phi_s')[k, k]
    denom = np.einsum("hij,jk,hlk->il", Phi, Sigma_u, Phi)  # (n, n), but we only want diagonal
    denom_diag = np.diag(denom)[:, None]           # (n, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = numer / denom_diag
    return shares


@dataclass
class HistoricalDecompositionContainer:
    """Standardised historical-decomposition container.

    See also :func:`structural_shocks` for recovering the shock series
    alone, without the full decomposition.

    Decomposes each observed variable's path into:

    * ``deterministic_plus_initial`` — contribution of intercept, trend, and
      the propagated effect of pre-sample initial conditions.
    * ``shocks`` — dict ``{shock_name: DataFrame (T_eff x n_vars)}`` with the
      cumulative contribution of each identified structural shock.
    * ``other_shocks`` — DataFrame ``(T_eff x n_vars)``, the residual
      contribution from un-identified structural shocks. ``None`` when
      identification is full (n_shocks == n_vars).

    By construction:

        data == deterministic_plus_initial
                + sum_j shocks[j]
                + other_shocks   (if partial identification)

    up to numerical precision.
    """

    data: "pd.DataFrame"
    deterministic_plus_initial: "pd.DataFrame"
    shocks: dict[str, "pd.DataFrame"]
    other_shocks: "pd.DataFrame | None"
    variable_names: list[str]
    shock_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def reconstruction_error(self) -> "pd.Series":
        """Maximum |reconstruction error| per variable. Should be ~1e-10."""
        recon = self.deterministic_plus_initial.copy()
        for df in self.shocks.values():
            recon = recon + df
        if self.other_shocks is not None:
            recon = recon + self.other_shocks
        return (self.data - recon).abs().max()


def _reject_time_varying_fit(fit, helper_name: str) -> None:
    """Historical attribution assumes CONSTANT coefficients.

    A TVP fit's VARFit face deliberately carries last-period coefficients
    and the residuals implied by them (a documented final-parameter
    counterfactual for IRF tracing). Feeding that face into historical
    decomposition or shock recovery would book coefficient DRIFT as
    historical innovations and propagate them with the last period's
    dynamics — and the reconstruction identity cannot catch it, because
    the deterministic/initial component is defined as the remainder.
    """
    method = str((getattr(fit, "fit_info", {}) or {}).get("method", ""))
    if (getattr(fit, "beta_path_mean", None) is not None
            or getattr(fit, "beta_path_draws", None) is not None
            or method.startswith("tvp")):
        raise NotImplementedError(
            f"{helper_name} does not support time-varying-parameter fits: "
            f"the fit's constant-coefficient face uses LAST-period "
            f"coefficients for the whole sample, so the result would be a "
            f"final-parameter counterfactual presented as history. A "
            f"period-specific decomposition is required for TVP models."
        )


def structural_shocks(
    fit: "Any",
    structural_impact: np.ndarray,
    shock_names: list[str] | None = None,
) -> "pd.DataFrame":
    """Recover the identified structural shock series from a fitted VAR.

    Given reduced-form residuals ``u_t`` and a structural impact matrix
    ``S`` (full ``(n, n)`` or partial ``(n, m)``), returns the implied
    structural shocks as a ``(T_eff, m)`` DataFrame indexed like the
    effective sample.

    Full identification (``m == n``): ``eps_t = S^{-1} u_t``. Partial
    identification: the GLS projection ``eps = (S' Sigma_u^{-1} S)^{-1}
    S' Sigma_u^{-1} u`` — the same convention
    :func:`historical_decomposition` uses (NOT the Euclidean
    pseudo-inverse, which is wrong for non-orthogonal columns).

    With a correctly scaled impact matrix the recovered shocks have
    unit variance up to sampling error — a cheap sanity check.

    Raises ``NotImplementedError`` for time-varying-parameter fits:
    their constant-coefficient face uses last-period coefficients for
    the whole sample, so the "shocks" would mix coefficient drift with
    innovations (a final-parameter counterfactual, not history).
    """
    import pandas as pd  # local — avoid hard dep at module import time

    _reject_time_varying_fit(fit, "structural_shocks")
    S = np.asarray(structural_impact, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    n = fit.n_vars
    if S.shape[0] != n:
        raise ValueError(
            f"structural_impact must have {n} rows (one per variable); got {S.shape}"
        )
    m = S.shape[1]
    if m < 1 or m > n:
        raise ValueError(f"identified shocks m must satisfy 1 <= m <= {n}; got {m}")

    u = np.asarray(fit.residuals)
    if m == n:
        eps = np.linalg.solve(S, u.T).T
    else:
        Sigma_u = np.asarray(fit.Sigma_u, dtype=float)
        Sigma_inv_S = np.linalg.solve(Sigma_u, S)
        gram = S.T @ Sigma_inv_S
        eps = np.linalg.solve(gram, Sigma_inv_S.T @ u.T).T

    names = list(shock_names) if shock_names is not None else [
        f"shock_{j}" for j in range(m)
    ]
    if len(names) != m:
        raise ValueError(f"shock_names must have length {m}; got {len(names)}")
    index = getattr(fit, "data", None)
    index = index.index[fit.p:] if index is not None else None
    return pd.DataFrame(eps, columns=names, index=index)


def historical_decomposition(
    fit: "Any",
    structural_impact: np.ndarray,
    shock_names: list[str] | None = None,
    other_label: str = "other_shocks",
) -> HistoricalDecompositionContainer:
    """Historical decomposition of observed Y into structural-shock contributions.

    Given a fitted VAR + a structural impact matrix S (full ``(n, n)`` or
    partial ``(n, m)`` for ``m < n``), decomposes each variable's observed
    path into cumulative contributions of each identified shock plus a
    deterministic-+-initial-conditions component (and, when partial, an
    'other shocks' residual).

    Algorithm:
      * Reduced-form residuals u_t = Y_t - (intercept + trend + lag terms).
      * Structural shocks for the identified columns: the GLS projection
        eps^id = (S' Sigma_u^{-1} S)^{-1} S' Sigma_u^{-1} u (NOT the
        Euclidean pinv, which is wrong for non-orthogonal columns).
        When S is square invertible, this reduces to S^{-1} u.
      * For each identified shock j, the contribution to variable k at time
        t is the convolution ``sum_{tau<=t} (Phi_{t-tau} s_j)_k * eps_{j, tau}``.
      * When S is partial, the residual u_other_t = u_t - S eps^id_t is
        propagated as a single 'other_shocks' contribution.

    Parameters
    ----------
    fit : VARFit
        Must expose ``Y_act`` (T_eff, n), ``residuals``, ``ma_rep(horizon)``,
        ``data``, ``p``, ``n_vars``, ``variable_names``.
    structural_impact : np.ndarray
        Shape ``(n, m)`` with ``1 <= m <= n``. A 1-D array of length n is
        accepted and treated as a single shock column.
    shock_names : list[str], optional
        Length ``m``. Defaults to ``["shock_0", ..., "shock_{m-1}"]``.
    other_label : str
        Name used in metadata for the un-identified residual block. Default
        ``"other_shocks"``.

    Returns
    -------
    HistoricalDecompositionContainer

    Raises
    ------
    NotImplementedError
        For time-varying-parameter fits — the constant-coefficient face
        uses last-period coefficients for the whole sample, so the
        decomposition would book coefficient drift as historical shocks
        (and the reconstruction identity cannot detect it, because the
        deterministic/initial component is defined as the remainder).
    """
    import pandas as pd  # local — avoid hard dep at module import time

    _reject_time_varying_fit(fit, "historical_decomposition")
    S = np.asarray(structural_impact, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    n = fit.n_vars
    if S.shape[0] != n:
        raise ValueError(
            f"structural_impact must have {n} rows (one per variable); got {S.shape}"
        )
    m = S.shape[1]
    if m < 1 or m > n:
        raise ValueError(f"identified shocks m must satisfy 1 <= m <= {n}; got {m}")

    Y_act = fit.Y_act                                  # (T_eff, n)
    u = fit.residuals                                  # (T_eff, n)
    T_eff = Y_act.shape[0]

    # Recover identified structural shocks
    if m == n:
        eps = np.linalg.solve(S, u.T).T               # (T_eff, n)
        u_other = None
    else:
        # Partial identification: with unit-variance structural shocks and a
        # full impact matrix B = [S, S_perp] satisfying B B' = Sigma_u, the
        # identified shocks are eps = S' Sigma_u^{-1} u (the first m rows of
        # B^{-1} = B' Sigma_u^{-1}). The Euclidean pseudo-inverse
        # (S'S)^{-1} S' u is correct ONLY if S's columns are orthogonal in the
        # Euclidean metric -- generally false (e.g. a Cholesky column), and it
        # cross-contaminates the per-shock and other_shocks contributions.
        Sigma_u = np.asarray(fit.Sigma_u, dtype=float)
        Sigma_inv_S = np.linalg.solve(Sigma_u, S)     # (n, m)
        # GLS form handles rescaled columns (non-unit shock size) too:
        # eps = (S' Sigma^{-1} S)^{-1} S' Sigma^{-1} u
        gram = S.T @ Sigma_inv_S                      # (m, m)
        eps = np.linalg.solve(gram, Sigma_inv_S.T @ u.T).T   # (T_eff, m)
        u_other = u - eps @ S.T                       # (T_eff, n)

    # MA representation for the full sample
    Phi = fit.ma_rep(T_eff - 1)                       # (T_eff, n, n)
    # Pre-compute Phi @ s_j for each shock j: shape (T_eff, n, m)
    irf_struct = np.einsum("hij,jk->hik", Phi, S)     # (T_eff, n, m)

    # Convolution: contrib[t, :, j] = sum_{tau=0}^{t} irf_struct[t-tau, :, j] * eps[tau, j]
    contrib = np.zeros((T_eff, n, m))
    for j in range(m):
        for t in range(T_eff):
            contrib[t, :, j] = irf_struct[: t + 1, :, j].T @ eps[: t + 1, j][::-1]

    if u_other is not None:
        other_contrib = np.zeros((T_eff, n))
        for t in range(T_eff):
            other_contrib[t, :] = np.einsum(
                "hij,hj->i", Phi[: t + 1], u_other[: t + 1][::-1]
            )
    else:
        other_contrib = None

    # Residual = data minus all shock contributions = deterministic + initial conditions
    sum_shocks = contrib.sum(axis=2)
    if other_contrib is not None:
        sum_shocks = sum_shocks + other_contrib
    det_plus_init = Y_act - sum_shocks

    # Wrap into DataFrames
    if hasattr(fit, "data") and hasattr(fit.data, "index"):
        idx = fit.data.index[fit.p:]
    else:
        idx = pd.RangeIndex(T_eff)

    if shock_names is None:
        shock_names = [f"shock_{j}" for j in range(m)]
    if len(shock_names) != m:
        raise ValueError(f"shock_names length {len(shock_names)} != m={m}")

    var_names = list(fit.variable_names)
    return HistoricalDecompositionContainer(
        data=pd.DataFrame(Y_act, index=idx, columns=var_names),
        deterministic_plus_initial=pd.DataFrame(
            det_plus_init, index=idx, columns=var_names
        ),
        shocks={
            name: pd.DataFrame(contrib[:, :, j], index=idx, columns=var_names)
            for j, name in enumerate(shock_names)
        },
        other_shocks=(
            pd.DataFrame(other_contrib, index=idx, columns=var_names)
            if other_contrib is not None else None
        ),
        variable_names=var_names,
        shock_names=list(shock_names),
        metadata={
            "n_identified_shocks": int(m),
            "n_vars": int(n),
            "T_eff": int(T_eff),
            "other_label": other_label,
            "identification_complete": bool(m == n),
        },
    )
