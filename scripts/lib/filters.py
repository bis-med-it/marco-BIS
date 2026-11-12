"""
Univariate trend-cycle filters.

Each filter takes a time series and returns a :class:`FilterResult` carrying
the trend, the cycle, and metadata about the method and parameters used.

Trend + cycle = original series (additive decomposition), with some
filters leaving NaN at the sample edges where the filter is not defined.

Filters implemented:

* :func:`hp_filter` — Hodrick-Prescott (two-sided), via ``statsmodels``.
* :func:`one_sided_hp_filter` — real-time (expanding-window) HP filter.
* :func:`hamilton_filter` — Hamilton (2018) regression-based filter.
* :func:`beveridge_nelson` — Beveridge-Nelson (1981) decomposition for an
  AR(p) on the first-differenced series.
* :func:`kmw_bn_filter` — Kamber-Morley-Wong (2018) BN imposing the
  paper's signal-to-noise ratio δ = ψ(1)² via the exact linear
  restriction Σφ = 1 - 1/sqrt(δ̄) (closed-form constrained OLS).
* :func:`bk_filter` — Baxter-King (1999) symmetric fixed-length band-pass
  filter.
* :func:`cf_filter` — Christiano-Fitzgerald (2003) asymmetric (random-walk)
  full-sample band-pass filter, via ``statsmodels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter as _sm_hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter as _sm_cffilter


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Trend-cycle decomposition result.

    Attributes
    ----------
    method : str
        Name of the filter (for reporting / logging).
    trend : pd.Series
        Trend (permanent) component, indexed like the input series.
    cycle : pd.Series
        Cyclical (transitory) component, indexed like the input series.
    metadata : dict
        Method-specific parameters and diagnostics.

    Invariant: ``trend + cycle == input_series`` element-wise (with NaN
    propagation at the sample edges if the filter is not defined there).
    """

    method: str
    trend: pd.Series
    cycle: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.trend, pd.Series):
            raise TypeError(f"trend must be pd.Series, got {type(self.trend).__name__}")
        if not isinstance(self.cycle, pd.Series):
            raise TypeError(f"cycle must be pd.Series, got {type(self.cycle).__name__}")
        if not self.trend.index.equals(self.cycle.index):
            raise ValueError("trend and cycle must share the same index")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "method": self.method,
            "trend": self.trend.values.tolist(),
            "cycle": self.cycle.values.tolist(),
            "index": [str(i) for i in self.trend.index],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Hodrick-Prescott
# ---------------------------------------------------------------------------


_HP_LAMBDA_DEFAULTS = {
    "quarterly": 1600.0,
    "monthly": 14400.0,
    "annual": 6.25,
}


def hp_filter(
    series: pd.Series, lam: float | None = None, frequency: str = "quarterly",
) -> FilterResult:
    """Hodrick-Prescott filter (two-sided).

    Parameters
    ----------
    series : pd.Series
        Input series.
    lam : float, optional
        Smoothing parameter. If ``None``, uses canonical defaults:
        1600 quarterly, 14400 monthly, 6.25 annual.
    frequency : {"quarterly", "monthly", "annual"}
        Used to pick the default ``lam`` if not supplied.

    Returns
    -------
    FilterResult with trend + cycle = series.

    Notes
    -----
    Hamilton (2018) shows the HP filter introduces spurious dynamic
    relations and end-of-sample distortions. Prefer ``hamilton_filter``
    or ``kmw_bn_filter`` unless replicating earlier HP-based work.
    """
    if lam is None:
        if frequency not in _HP_LAMBDA_DEFAULTS:
            raise ValueError(
                f"frequency must be one of {list(_HP_LAMBDA_DEFAULTS)} "
                f"when lam is not provided"
            )
        lam = _HP_LAMBDA_DEFAULTS[frequency]
    if series.isna().any():
        raise ValueError("HP filter does not accept NaN — drop or interpolate first")

    cycle, trend = _sm_hpfilter(series.astype(float), lamb=lam)
    return FilterResult(
        method="hp",
        trend=pd.Series(trend.values, index=series.index, name=f"{series.name}_trend"),
        cycle=pd.Series(cycle.values, index=series.index, name=f"{series.name}_cycle"),
        metadata={"lam": float(lam), "frequency": frequency, "two_sided": True},
    )


# ---------------------------------------------------------------------------
# Hamilton (2018) regression filter
# ---------------------------------------------------------------------------


_HAMILTON_H_DEFAULTS = {"quarterly": 8, "monthly": 24, "annual": 2}


def hamilton_filter(
    series: pd.Series, h: int | None = None, p: int = 4,
    frequency: str = "quarterly",
) -> FilterResult:
    """Hamilton (2018) regression-based filter.

    Decomposition: at each time ``t + h`` for ``t ≥ p - 1``, regress
    ``y_{t+h}`` on a constant and ``(y_t, y_{t-1}, ..., y_{t-p+1})``.
    The fitted value is the trend; the residual is the cycle.

    For the first ``p + h - 1`` observations the trend and cycle are
    undefined (NaN).

    Parameters
    ----------
    series : pd.Series
    h : int, optional
        Forward horizon for the regression. Default by frequency:
        8 (quarterly), 24 (monthly), 2 (annual) — Hamilton's recommendation.
    p : int, default 4
        Number of lags in the regression (Hamilton recommends 4).
    frequency : {"quarterly", "monthly", "annual"}
        Used to pick the default ``h`` if not supplied.

    Returns
    -------
    FilterResult with ``trend + cycle = series`` for indices where defined.

    References
    ----------
    Hamilton, J. D. (2018). "Why You Should Never Use the Hodrick-Prescott
    Filter." Review of Economics and Statistics, 100(5), 831-843.
    """
    if h is None:
        if frequency not in _HAMILTON_H_DEFAULTS:
            raise ValueError(
                f"frequency must be one of {list(_HAMILTON_H_DEFAULTS)} "
                f"when h is not provided"
            )
        h = _HAMILTON_H_DEFAULTS[frequency]
    if p < 1:
        raise ValueError(f"p must be ≥ 1, got {p}")
    if h < 1:
        raise ValueError(f"h must be ≥ 1, got {h}")

    y = series.astype(float).values
    n = len(y)
    if n < p + h + 1:
        raise ValueError(
            f"Series too short ({n}) for p={p}, h={h}; need at least {p + h + 1}"
        )

    # For each forecast date s = t + h with t in [p-1, n-h-1]:
    #   regress y[s] on (1, y[t], y[t-1], ..., y[t-p+1])
    # Equivalently, regress y[s] on lags h, h+1, ..., h+p-1 of itself.
    rhs_cols = []
    valid_s = []
    for t in range(p - 1, n - h):
        s = t + h
        rhs_cols.append([1.0] + [y[t - j] for j in range(p)])
        valid_s.append(s)
    X = np.asarray(rhs_cols)
    y_fwd = y[np.asarray(valid_s)]

    beta, *_ = np.linalg.lstsq(X, y_fwd, rcond=None)
    fitted = X @ beta
    resid = y_fwd - fitted

    trend = pd.Series(np.full(n, np.nan), index=series.index,
                      name=f"{series.name}_trend")
    cycle = pd.Series(np.full(n, np.nan), index=series.index,
                      name=f"{series.name}_cycle")
    trend.iloc[valid_s] = fitted
    cycle.iloc[valid_s] = resid

    return FilterResult(
        method="hamilton",
        trend=trend, cycle=cycle,
        metadata={
            "h": int(h), "p": int(p), "frequency": frequency,
            "n_obs_used": len(valid_s),
            "n_obs_dropped": int(p + h - 1),
        },
    )


# ---------------------------------------------------------------------------
# Beveridge-Nelson (1981)
# ---------------------------------------------------------------------------


def beveridge_nelson(
    series: pd.Series, ar_order: int | None = None, max_ar_order: int = 12,
    ic: str = "bic",
) -> FilterResult:
    """Beveridge-Nelson decomposition via AR(p) on first differences.

    The series is assumed to be I(1). Fit AR(p) to ``Δy_t`` (with p chosen
    by the information criterion if not specified). The BN cycle is the
    deviation of the current level from its long-run forecast:

        c_t = -e_1' A (I - A)^{-1} X_t

    where ``X_t = [x_t, x_{t-1}, ..., x_{t-p+1}]'`` with ``x = Δy - μ``,
    and ``A`` is the companion matrix of the AR(p) on ``x``.

    Parameters
    ----------
    series : pd.Series
        Level series (assumed I(1) — e.g. log real GDP).
    ar_order : int, optional
        AR lag order. If ``None``, selected by ``ic`` from 1..max_ar_order.
    max_ar_order : int, default 12
        Upper bound for IC-based lag selection.
    ic : {"bic", "aic"}
        Information criterion for lag selection.

    Returns
    -------
    FilterResult with trend (permanent) + cycle (transitory) = series
    (additive identity holds for indices where the AR is defined).

    Raises
    ------
    RuntimeError
        If the fitted AR growth dynamics are not stationary (companion
        spectral radius >= 1) — the Beveridge-Nelson long-run forecast
        does not converge there, and a nonsingular ``(I - A)`` alone
        does not establish convergence.

    References
    ----------
    Beveridge, S. and Nelson, C. R. (1981). "A new approach to decomposition
    of economic time series into permanent and transitory components with
    particular attention to measurement of the 'business cycle'." Journal
    of Monetary Economics, 7(2), 151-174.
    """
    if series.isna().any():
        raise ValueError("BN filter does not accept NaN — drop or interpolate first")
    y = series.astype(float).values
    n = len(y)
    if n < 5:
        raise ValueError("series too short for BN decomposition")

    # First differences
    dy = np.diff(y)
    T = len(dy)

    # Lag-order selection
    if ar_order is None:
        ar_order = _select_ar_order(dy, max_ar_order, ic)
    p = int(ar_order)
    if p < 1:
        raise ValueError(f"ar_order must be ≥ 1, got {p}")
    if T <= p + 2:
        raise ValueError(f"series too short for AR({p})")

    # AR(p) regression: dy_t = c + sum phi_j dy_{t-j} + e_t
    phi, c = _fit_ar_p(dy, p)
    mu = c / (1.0 - phi.sum())   # unconditional mean of dy

    # Companion form on x = dy - mu (zero-mean AR(p))
    A = np.zeros((p, p))
    A[0, :] = phi
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    # The BN trend is the long-run conditional forecast, which EXISTS only
    # when the growth dynamics are stationary. Invertibility of (I - A)
    # alone excludes just a root at +1 — an explosive or negative-unit root
    # still passes it while the forecast sum diverges or oscillates.
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius >= 1.0:
        raise RuntimeError(
            f"bn_filter: the fitted AR({p}) growth dynamics are not "
            f"stationary (companion spectral radius {radius:.4f} >= 1), so "
            f"the Beveridge-Nelson long-run forecast does not converge. "
            f"Check the differencing/transform of the input series or the "
            f"lag order."
        )
    eye_minus_A = np.eye(p) - A
    try:
        inv_eye_minus_A = np.linalg.inv(eye_minus_A)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"AR({p}) gives a singular (I - A); series may have a near-unit root."
        ) from exc

    # BN cycle: c_t = -e1' * A * (I - A)^{-1} * X_t  (KMW 2018, eq. 4)
    weight_vec = -np.eye(p)[0] @ A @ inv_eye_minus_A   # (p,)

    # Build cycle for t = p-1, ..., T-1 (indices into dy)
    # Then map to indices in the original level series: y[t+1] corresponds
    # to dy[t]. The cycle at y[t+1] uses x_t = dy[t] - mu and lags.
    cycle_arr = np.full(n, np.nan)
    x = dy - mu
    for t in range(p - 1, T):
        X_t = x[t::-1][:p]   # [x_t, x_{t-1}, ..., x_{t-p+1}]
        c_val = float(weight_vec @ X_t)
        cycle_arr[t + 1] = c_val
    trend_arr = y - cycle_arr

    return FilterResult(
        method="beveridge_nelson",
        trend=pd.Series(trend_arr, index=series.index,
                        name=f"{series.name}_trend"),
        cycle=pd.Series(cycle_arr, index=series.index,
                        name=f"{series.name}_cycle"),
        metadata={
            "ar_order": p,
            "ic": ic,
            "mu": float(mu),
            "phi": phi.tolist(),
            "max_companion_eigenvalue": float(
                np.max(np.abs(np.linalg.eigvals(A)))
            ),
        },
    )


# ---------------------------------------------------------------------------
# Kamber-Morley-Wong (2018): BN with low signal-to-noise constraint
# ---------------------------------------------------------------------------


def kmw_bn_filter(
    series: pd.Series,
    delta: float = 0.25,
    ar_order: int | None = None,
    max_ar_order: int = 12,
    ic: str = "bic",
) -> FilterResult:
    """Kamber-Morley-Wong (2018) Beveridge-Nelson filter.

    Imposes KMW's signal-to-noise ratio ``δ ≡ σ²_Δτ / σ²_e = ψ(1)²`` —
    the variance of trend shocks relative to the AR forecast-error
    variance — on the AR(p) for ``Δy_t``. Since ``ψ(1) = 1/(1 - Σφ_j)``,
    fixing ``δ = δ̄`` is the LINEAR restriction

        Σφ_j = 1 - 1/sqrt(δ̄)

    (e.g. ``δ̄ = 0.25`` ⇒ ``Σφ = -1``), imposed exactly via the
    Dickey-Fuller-style substitution ``φ_1 = s̄ - Σ_{j≥2} φ_j``: regress
    ``Δy_t - s̄ Δy_{t-1}`` on a constant and ``(Δy_{t-j} - Δy_{t-1})``
    for ``j = 2..p`` by OLS — closed form, no optimizer (KMW eq. 6, §II.B).
    A low ``δ̄`` shrinks the trend-shock share, producing the paper's
    headline large-amplitude, NBER-aligned output gap.

    Parameters
    ----------
    series : pd.Series
        Level series (assumed I(1) — e.g. log real GDP).
    delta : float, default 0.25
        KMW's δ̄ — the imposed trend-shock-to-forecast-error variance
        ratio. Must be > 0. Unconstrained US-GDP estimates are typically
        ≈ 2; KMW's US application selects δ̄ = 0.24 automatically by
        maximising the amplitude-to-noise ratio (their eq. 8 — not
        implemented here; 0.25 is a rounded convenience default). A
        ratio well below 1 forces the intuitive, large-amplitude gap.
        δ̄ = 1 imposes Σφ = 0 (unit long-run multiplier).
    ar_order : int, optional
        AR lag order. If ``None``, selected by ``ic`` from
        1..max_ar_order on the *unconstrained* AR. KMW use p = 12 for US
        quarterly GDP. Note ``ar_order=1`` cannot satisfy the constraint
        stationarily for ``delta <= 0.25`` (``delta=0.25`` forces
        ``phi_1 = -1`` exactly, a negative unit root) — the function
        raises rather than returning a divergent decomposition.
    max_ar_order : int, default 12
        Upper bound for IC-based lag selection.
    ic : {"bic", "aic"}
        Information criterion for lag selection.

    Returns
    -------
    FilterResult
        ``trend + cycle = series`` (where the AR is defined). ``metadata``
        carries ``delta_target``, ``sum_phi`` (= 1 - 1/sqrt(δ̄), exact),
        ``psi_1``, ``delta_unconstrained``, ``phi``,
        ``phi_unconstrained``, ``ar_order``, ``mu``, and
        ``max_companion_eigenvalue``.

    Raises
    ------
    RuntimeError
        If the CONSTRAINED AR dynamics are not stationary (companion
        spectral radius >= 1): the constraint pins the SUM of the
        coefficients, not stationarity, and the BN long-run forecast
        does not converge for a unit or explosive root.

    References
    ----------
    Kamber, G., Morley, J., Wong, B. (2018). "Intuitive and Reliable
    Estimates of the Output Gap from a Beveridge-Nelson Filter." Review
    of Economics and Statistics, 100(3), 550-566.
    """
    if series.isna().any():
        raise ValueError("KMW BN filter does not accept NaN — drop or interpolate first")
    if not delta > 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    y = series.astype(float).values
    n = len(y)
    if n < 5:
        raise ValueError("series too short for KMW BN decomposition")

    dy = np.diff(y)
    T = len(dy)

    if ar_order is None:
        ar_order = _select_ar_order(dy, max_ar_order, ic)
    p = int(ar_order)
    if p < 1:
        raise ValueError(f"ar_order must be >= 1, got {p}")
    if T <= p + 2:
        raise ValueError(f"series too short for AR({p})")

    # Unconstrained OLS AR(p) baseline (for the reported comparison)
    phi_uncon, c_uncon = _fit_ar_p(dy, p)
    psi1_uncon = 1.0 / (1.0 - phi_uncon.sum())
    delta_uncon = psi1_uncon ** 2

    # KMW restriction: Σφ = s̄ = 1 - 1/sqrt(δ̄), imposed exactly by the
    # substitution φ_1 = s̄ - Σ_{j≥2} φ_j. OLS on the transformed model:
    #   Δy_t - s̄ Δy_{t-1} = c + Σ_{j=2}^p φ_j (Δy_{t-j} - Δy_{t-1}) + e_t
    s_bar = 1.0 - 1.0 / np.sqrt(delta)
    lhs = []
    rhs = []
    for t in range(p, T):
        lhs.append(dy[t] - s_bar * dy[t - 1])
        row = [dy[t - j] - dy[t - 1] for j in range(2, p + 1)] + [1.0]
        rhs.append(row)
    X_c = np.asarray(rhs)
    y_c = np.asarray(lhs)
    coefs, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
    phi_rest = coefs[: p - 1] if p > 1 else np.array([])
    c = float(coefs[-1])
    phi = np.empty(p)
    phi[0] = s_bar - phi_rest.sum()
    if p > 1:
        phi[1:] = phi_rest

    # Build BN decomposition with the constrained phi. (I - A) is never
    # singular here: 1 - Σφ = 1/sqrt(δ̄) > 0 by construction — but that only
    # excludes a root at +1. The constraint pins the SUM of the coefficients,
    # not stationarity: with p=1 and δ̄=0.25 it forces φ₁ = -1 exactly (a
    # negative unit root), and the long-run forecast does not converge.
    mu = c / (1.0 - phi.sum())
    A = np.zeros((p, p))
    A[0, :] = phi
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius >= 1.0:
        raise RuntimeError(
            f"kmw_bn_filter: the constrained AR({p}) growth dynamics are not "
            f"stationary (companion spectral radius {radius:.4f} >= 1), so "
            f"the Beveridge-Nelson long-run forecast does not converge. "
            f"Increase ar_order (p=1 cannot satisfy the delta-bar constraint "
            f"stationarily) or choose a different delta."
        )
    inv_eye_minus_A = np.linalg.inv(np.eye(p) - A)
    weight_vec = -np.eye(p)[0] @ A @ inv_eye_minus_A

    cycle_arr = np.full(n, np.nan)
    x = dy - mu
    for t in range(p - 1, T):
        X_t = x[t::-1][:p]
        cycle_arr[t + 1] = float(weight_vec @ X_t)
    trend_arr = y - cycle_arr

    return FilterResult(
        method="kmw_bn_filter",
        trend=pd.Series(trend_arr, index=series.index,
                        name=f"{series.name}_trend"),
        cycle=pd.Series(cycle_arr, index=series.index,
                        name=f"{series.name}_cycle"),
        metadata={
            "ar_order": p,
            "ic": ic,
            "delta_target": float(delta),
            "sum_phi": float(phi.sum()),
            "psi_1": float(1.0 / (1.0 - phi.sum())),
            "delta_unconstrained": float(delta_uncon),
            "mu": float(mu),
            "phi": phi.tolist(),
            "phi_unconstrained": phi_uncon.tolist(),
            "max_companion_eigenvalue": float(
                np.max(np.abs(np.linalg.eigvals(A)))
            ),
        },
    )


# ---------------------------------------------------------------------------
# One-sided (real-time) Hodrick-Prescott
# ---------------------------------------------------------------------------


def one_sided_hp_filter(
    series: pd.Series,
    lam: float | None = None,
    frequency: str = "quarterly",
    min_periods: int = 4,
) -> FilterResult:
    """One-sided (real-time) Hodrick-Prescott filter.

    Runs the two-sided HP filter on the expanding window ``y[:t+1]`` and
    keeps only the last fitted trend point ``τ_{t|t}`` — so the trend at
    time ``t`` uses information up to ``t`` only (Stock-Watson 1999). This
    is the real-time analogue of :func:`hp_filter`: it never revises past
    trend estimates as new data arrive, which makes it appropriate for
    pseudo-real-time output-gap / recession-signal exercises where the
    two-sided HP's end-of-sample revisions are a problem.

    The first ``min_periods - 1`` observations are undefined (NaN) — the
    expanding window is too short to filter.

    Parameters
    ----------
    series : pd.Series
        Input series.
    lam : float, optional
        Smoothing parameter; same canonical defaults as :func:`hp_filter`
        (1600 quarterly, 14400 monthly, 6.25 annual).
    frequency : {"quarterly", "monthly", "annual"}
        Used to pick the default ``lam`` if not supplied.
    min_periods : int, default 4
        Minimum expanding-window length before a trend point is reported.

    Returns
    -------
    FilterResult with ``trend + cycle = series`` for ``t >= min_periods-1``.
    By construction ``trend.iloc[-1]`` equals the last point of the
    two-sided :func:`hp_filter` applied to the full sample.

    Notes
    -----
    Costs ``O(T)`` HP solves (one per expanding window). For typical macro
    samples (``T`` up to a few thousand) this is sub-second.

    References
    ----------
    Stock, J. H. and Watson, M. W. (1999). "Forecasting Inflation." Journal
    of Monetary Economics, 44(2), 293-335 (one-sided HP as a real-time
    trend). Hodrick, R. J. and Prescott, E. C. (1997).
    """
    if lam is None:
        if frequency not in _HP_LAMBDA_DEFAULTS:
            raise ValueError(
                f"frequency must be one of {list(_HP_LAMBDA_DEFAULTS)} "
                f"when lam is not provided"
            )
        lam = _HP_LAMBDA_DEFAULTS[frequency]
    if series.isna().any():
        raise ValueError(
            "one-sided HP filter does not accept NaN — drop or interpolate first"
        )
    if min_periods < 3:
        raise ValueError(f"min_periods must be >= 3, got {min_periods}")
    y = series.astype(float).values
    n = len(y)
    if n < min_periods:
        raise ValueError(f"series too short ({n}) for min_periods={min_periods}")

    trend_arr = np.full(n, np.nan)
    for t in range(min_periods - 1, n):
        _, tr = _sm_hpfilter(y[: t + 1], lamb=lam)
        trend_arr[t] = float(np.asarray(tr).ravel()[-1])
    cycle_arr = y - trend_arr

    return FilterResult(
        method="one_sided_hp",
        trend=pd.Series(trend_arr, index=series.index, name=f"{series.name}_trend"),
        cycle=pd.Series(cycle_arr, index=series.index, name=f"{series.name}_cycle"),
        metadata={
            "lam": float(lam),
            "frequency": frequency,
            "two_sided": False,
            "min_periods": int(min_periods),
            "n_obs_dropped": int(min_periods - 1),
        },
    )


# ---------------------------------------------------------------------------
# Band-pass filters (Baxter-King 1999, Christiano-Fitzgerald 2003)
# ---------------------------------------------------------------------------


# Business-cycle pass-band (period bounds) and BK truncation, by frequency.
# Quarterly: 6-32 quarters (1.5-8 years, Baxter-King 1999); scaled elsewhere.
_BANDPASS_PERIOD_DEFAULTS = {
    "quarterly": (6, 32),
    "monthly": (18, 96),
    "annual": (2, 8),
}
_BK_TRUNCATION_DEFAULTS = {"quarterly": 12, "monthly": 36, "annual": 3}


def bk_filter(
    series: pd.Series,
    low: int | None = None,
    high: int | None = None,
    K: int | None = None,
    frequency: str = "quarterly",
) -> FilterResult:
    """Baxter-King (1999) band-pass filter (symmetric, fixed length).

    Extracts fluctuations with periodicity between ``low`` and ``high`` (in
    units of the series' frequency) using a symmetric ``2K+1``-term moving
    average that approximates the ideal band-pass filter. The weights are
    constrained to sum to zero, so the filter has zero gain at the zero
    frequency and removes a unit root / deterministic trend.

    The first and last ``K`` observations are undefined (NaN) — the filter
    needs ``K`` leads and ``K`` lags.

    Parameters
    ----------
    series : pd.Series
        Input series (need not be stationary; the filter de-trends).
    low, high : int, optional
        Pass-band period bounds. Defaults by frequency: ``(6, 32)``
        quarterly, ``(18, 96)`` monthly, ``(2, 8)`` annual — the standard
        business-cycle band (1.5-8 years). Require ``2 <= low < high``.
    K : int, optional
        Number of leads/lags (filter half-length). Defaults: 12 quarterly,
        36 monthly, 3 annual (Baxter-King's recommendation).
    frequency : {"quarterly", "monthly", "annual"}
        Used to pick defaults for ``low`` / ``high`` / ``K`` when not given.

    Returns
    -------
    FilterResult with ``trend + cycle = series`` over the interior
    ``[K, n-K)`` indices and NaN in the first and last ``K``.

    References
    ----------
    Baxter, M. and King, R. G. (1999). "Measuring Business Cycles:
    Approximate Band-Pass Filters for Economic Time Series." Review of
    Economics and Statistics, 81(4), 575-593.
    """
    if frequency not in _BANDPASS_PERIOD_DEFAULTS:
        raise ValueError(
            f"frequency must be one of {list(_BANDPASS_PERIOD_DEFAULTS)}"
        )
    low_d, high_d = _BANDPASS_PERIOD_DEFAULTS[frequency]
    low = low_d if low is None else low
    high = high_d if high is None else high
    if K is None:
        K = _BK_TRUNCATION_DEFAULTS[frequency]
    if series.isna().any():
        raise ValueError("BK filter does not accept NaN — drop or interpolate first")
    if not (2 <= low < high):
        raise ValueError(f"require 2 <= low < high, got low={low}, high={high}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    y = series.astype(float).values
    n = len(y)
    if n <= 2 * K:
        raise ValueError(
            f"series too short ({n}) for K={K}; need at least {2 * K + 1}"
        )

    # Ideal band-pass weights for periods in [low, high]:
    #   passed frequencies omega in [2*pi/high, 2*pi/low].
    omega_low = 2.0 * np.pi / high   # lowest passed frequency (longest period)
    omega_high = 2.0 * np.pi / low   # highest passed frequency (shortest period)
    h = np.arange(1, K + 1)
    b_half = np.empty(K + 1)
    b_half[0] = (omega_high - omega_low) / np.pi
    b_half[1:] = (np.sin(omega_high * h) - np.sin(omega_low * h)) / (np.pi * h)
    # Full symmetric weight vector indexed -K..K: [b_K..b_1, b_0, b_1..b_K]
    weights = np.concatenate([b_half[:0:-1], b_half])
    # Baxter-King normalisation: subtract the mean so the weights sum to zero
    weights = weights - weights.sum() / (2 * K + 1)

    cycle_arr = np.full(n, np.nan)
    for t in range(K, n - K):
        # c_t = sum_{j=-K}^{K} weights[j+K] * y[t-j]  (weights symmetric)
        cycle_arr[t] = float(weights @ y[t - K : t + K + 1][::-1])
    trend_arr = y - cycle_arr

    return FilterResult(
        method="baxter_king",
        trend=pd.Series(trend_arr, index=series.index, name=f"{series.name}_trend"),
        cycle=pd.Series(cycle_arr, index=series.index, name=f"{series.name}_cycle"),
        metadata={
            "low": int(low),
            "high": int(high),
            "K": int(K),
            "frequency": frequency,
            "two_sided": True,
            "n_obs_dropped": int(2 * K),
        },
    )


def cf_filter(
    series: pd.Series,
    low: int | None = None,
    high: int | None = None,
    drift: bool = True,
    frequency: str = "quarterly",
) -> FilterResult:
    """Christiano-Fitzgerald (2003) band-pass filter (asymmetric, full sample).

    The random-walk approximation to the ideal band-pass filter. Unlike
    Baxter-King the weights are time-varying and use the entire sample, so
    no observations are lost — at the cost of asymmetry (the filter weights
    differ at the start, middle, and end of the sample). This is CF's
    recommended default for persistent macro series.

    Wraps ``statsmodels.tsa.filters.cf_filter.cffilter`` (just as
    :func:`hp_filter` wraps statsmodels' HP) and packages the output as a
    :class:`FilterResult`.

    Parameters
    ----------
    series : pd.Series
        Input series.
    low, high : int, optional
        Pass-band period bounds. Defaults by frequency: ``(6, 32)``
        quarterly, ``(18, 96)`` monthly, ``(2, 8)`` annual. Require
        ``2 <= low < high``.
    drift : bool, default True
        Remove a linear drift before filtering (CF's random-walk
        assumption). Set ``False`` for series without drift.
    frequency : {"quarterly", "monthly", "annual"}
        Used to pick defaults for ``low`` / ``high`` when not given.

    Returns
    -------
    FilterResult with ``trend + cycle = series`` at every index (no NaN —
    the CF filter uses the full sample).

    References
    ----------
    Christiano, L. J. and Fitzgerald, T. J. (2003). "The Band Pass Filter."
    International Economic Review, 44(2), 435-465.
    """
    if frequency not in _BANDPASS_PERIOD_DEFAULTS:
        raise ValueError(
            f"frequency must be one of {list(_BANDPASS_PERIOD_DEFAULTS)}"
        )
    low_d, high_d = _BANDPASS_PERIOD_DEFAULTS[frequency]
    low = low_d if low is None else low
    high = high_d if high is None else high
    if series.isna().any():
        raise ValueError("CF filter does not accept NaN — drop or interpolate first")
    if not (2 <= low < high):
        raise ValueError(f"require 2 <= low < high, got low={low}, high={high}")

    cycle_sm, _ = _sm_cffilter(
        series.astype(float), low=low, high=high, drift=drift
    )
    cycle = np.asarray(cycle_sm).ravel()
    # Trend is the additive complement (series - cycle). NB: statsmodels'
    # second return value omits the removed linear drift when drift=True, so
    # it is NOT the additive complement — recompute it here to honour the
    # FilterResult invariant ``trend + cycle == series``.
    trend = series.astype(float).values - cycle
    return FilterResult(
        method="christiano_fitzgerald",
        trend=pd.Series(trend, index=series.index,
                        name=f"{series.name}_trend"),
        cycle=pd.Series(cycle, index=series.index,
                        name=f"{series.name}_cycle"),
        metadata={
            "low": int(low),
            "high": int(high),
            "drift": bool(drift),
            "frequency": frequency,
            "two_sided": False,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_ar_p(y: np.ndarray, p: int) -> tuple[np.ndarray, float]:
    """OLS AR(p) on y. Returns (phi_array of length p, constant)."""
    T = len(y)
    rhs = []
    lhs = []
    for t in range(p, T):
        rhs.append([y[t - j - 1] for j in range(p)] + [1.0])
        lhs.append(y[t])
    X = np.asarray(rhs)
    y_lhs = np.asarray(lhs)
    coefs, *_ = np.linalg.lstsq(X, y_lhs, rcond=None)
    phi = coefs[:p]
    c = float(coefs[p])
    return phi, c


def _select_ar_order(y: np.ndarray, max_p: int, ic: str) -> int:
    """Select AR lag order by AIC or BIC. Uses np-only OLS (no statsmodels).

    All candidate orders are fitted and scored on the SAME dependent
    observations (starting at the largest admissible lag) with one common
    ``n`` in the criterion. Scoring each candidate on its own ``T - p``
    sample makes the selection depend on the series' measurement units:
    rescaling by ``c`` adds ``2 n log(c)`` to each score, and unequal ``n``
    then reorders the candidates. The selected order is refitted by the
    caller on its usual larger sample.
    """
    if ic not in ("bic", "aic"):
        raise ValueError(f"ic must be 'bic' or 'aic', got {ic!r}")
    T_full = len(y)
    # Largest admissible candidate: the common sample must keep positive
    # residual degrees of freedom for the biggest design.
    p_max = int(max_p)
    while p_max >= 1 and (T_full - p_max) < p_max + 2:
        p_max -= 1
    if p_max < 1:
        return 1
    start = p_max
    n = T_full - start
    y_lhs = y[start:]
    best_p = 1
    best_crit = np.inf
    for p in range(1, p_max + 1):
        X = np.column_stack(
            [y[start - j - 1:T_full - j - 1] for j in range(p)]
            + [np.ones(n)]
        )
        beta, *_ = np.linalg.lstsq(X, y_lhs, rcond=None)
        resid = y_lhs - X @ beta
        rss = float(resid @ resid)
        sigma2 = max(rss / n, 1e-300)
        k = p + 1   # AR coefs + constant
        if ic == "bic":
            crit = n * np.log(sigma2) + k * np.log(n)
        else:
            crit = n * np.log(sigma2) + 2 * k
        if crit < best_crit:
            best_crit = crit
            best_p = p
    return best_p


def _ar_residuals(y: np.ndarray, p: int, phi: np.ndarray, c: float) -> np.ndarray:
    """Residuals of dy_t = c + sum phi_j dy_{t-j} + e_t for t=p..T-1."""
    T = len(y)
    out = np.empty(T - p)
    for t in range(p, T):
        out[t - p] = y[t] - c - sum(phi[j] * y[t - j - 1] for j in range(p))
    return out
