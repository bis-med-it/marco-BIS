"""Out-of-sample forecasting and forecast evaluation.

Iterated point and density forecasts for ``VARFit`` and its subclasses
(``BVARFit``, ``MFVARFit``, ``TVPVARFit``; ``DFMFit`` is a separate class
and is not supported),
a rolling/expanding-window driver, and standard scoring utilities
(MSE/RMSE/MAE, density scores — CRPS / log score / PIT — plus
HLN-corrected Diebold-Mariano pairwise comparison).

All functions return either a :class:`ForecastContainer` (forecasts)
or a plain ``pd.DataFrame`` / scalar (scoring), so they slot into
research_log.json snapshots and the existing replication output
convention without ceremony.

Architecture:

* ``iterated_forecast(fit, horizon, ...)`` is the workhorse. It takes
  the *last p observations* of ``fit.data`` as the conditioning
  history, then iterates ``y_{T+h} = c + sum_j A_j y_{T+h-j}``
  forward, optionally adding innovation draws and/or parameter draws.

* ``rolling_forecasts(data, fit_fn, window_size, horizon, ...)``
  walks the sample, fitting ``fit_fn(data_up_to_t)`` at each origin
  and forecasting ``horizon`` steps ahead. Returns a long-format
  DataFrame paired with the realised values for easy scoring.

* ``forecast_score(forecasts, realised, metric)`` computes MSE / RMSE
  / MAE pointwise or by horizon. ``diebold_mariano(loss_a, loss_b,
  horizon)`` returns the standard t-statistic and p-value for
  comparing two forecast loss series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sp_stats

# Quantile half-widths for two-sided bands
_Z68_LO, _Z68_HI = 0.16, 0.84
_Z90_LO, _Z90_HI = 0.05, 0.95


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass
class ForecastContainer:
    """Output of :func:`iterated_forecast`.

    Attributes
    ----------
    point : np.ndarray, shape (horizon, n_vars)
        Point forecasts from the fit's point estimates (B_hat, Sigma_u).
        Always present; this is the no-uncertainty iterated forecast.
    paths : np.ndarray or None, shape (n_paths, horizon, n_vars)
        Simulated forecast paths if ``n_paths`` was specified in the call;
        otherwise ``None``. Each path samples innovations from
        ``N(0, Sigma_u)`` at each horizon (and parameters from the posterior
        if ``parameter_uncertainty=True``).
    median, lower_68, upper_68, lower_90, upper_90 : np.ndarray or None
        Quantile summaries across ``paths``, shape (horizon, n_vars). None
        when ``paths`` is None.
    horizons : np.ndarray, shape (horizon,)
        Forecast horizons 1, 2, ..., horizon.
    variable_names : list[str]
    forecast_origin : pd.Timestamp
        The last in-sample date — the conditioning origin for horizon 1.
    metadata : dict
        Free-form provenance (method, parameter_uncertainty flag, seed, etc.)
    """

    point: np.ndarray
    paths: np.ndarray | None
    median: np.ndarray | None
    lower_68: np.ndarray | None
    upper_68: np.ndarray | None
    lower_90: np.ndarray | None
    upper_90: np.ndarray | None
    horizons: np.ndarray
    variable_names: list[str]
    forecast_origin: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self, summary: str = "point") -> pd.DataFrame:
        """Return forecasts as a long-format DataFrame keyed by (horizon, variable).

        ``summary`` selects which quantity to extract: ``"point"`` (the
        no-uncertainty path), ``"median"``, ``"lower_68"``, ``"upper_68"``,
        ``"lower_90"``, or ``"upper_90"``. Raises if a quantile is requested
        but the container has no ``paths``.
        """
        arr = {
            "point": self.point, "median": self.median,
            "lower_68": self.lower_68, "upper_68": self.upper_68,
            "lower_90": self.lower_90, "upper_90": self.upper_90,
        }.get(summary)
        if arr is None:
            raise ValueError(
                f"summary={summary!r} not available — density forecasts "
                f"weren't computed for this container."
            )
        return pd.DataFrame(arr, index=self.horizons, columns=self.variable_names)


# ---------------------------------------------------------------------------
# Iterated forecast
# ---------------------------------------------------------------------------


def iterated_forecast(
    fit,
    horizon: int,
    n_paths: int | None = None,
    parameter_uncertainty: bool = False,
    seed: int | None = None,
) -> ForecastContainer:
    """Iterated VAR(p) forecast — point and (optionally) density.

    Parameters
    ----------
    fit : VARFit (or subclass — BVARFit, MFVARFit, TVPVARFit)
        Must expose ``lag_matrices()``, ``B_hat``, ``Sigma_u``,
        ``variable_names``, ``data``, ``p``, ``n_vars``, and
        ``fit_info`` (with a ``trend`` key). For density forecasts with
        parameter uncertainty, must also expose ``parameter_draws()``.
    horizon : int
        Forecast horizon (number of periods ahead). Must be ≥ 1.
    n_paths : int, optional
        If given, draw ``n_paths`` simulated forecast paths to produce a
        density forecast (median + 68% + 90% bands). If ``None``, only the
        point forecast is computed.
    parameter_uncertainty : bool, default False
        If True (and ``n_paths`` is given), draw one ``(B_draw, Sigma_draw)``
        from the posterior per simulated path, capturing joint parameter +
        innovation uncertainty. Ignored when ``n_paths is None`` (point
        forecast uses point estimates).
    seed : int, optional
        RNG seed for reproducible density forecasts.

    Returns
    -------
    ForecastContainer

    Raises
    ------
    ValueError
        If ``horizon < 1``, or if ``parameter_uncertainty=True`` is
        requested without ``n_paths``.
    NotImplementedError
        If ``fit.fit_info["trend"]`` is something other than ``"c"`` or
        ``"n"`` (``"ct"`` needs extra plumbing for the deterministic
        trend component; not yet supported), or if the fit is a VARX
        (``fit.has_exog=True``) — future exogenous paths would be
        required and the companion state-space cannot represent the
        exogenous block.
    RuntimeError
        If ``parameter_uncertainty=True`` and ``fit.parameter_draws()``
        returns no usable draws.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if parameter_uncertainty and n_paths is None:
        raise ValueError(
            "parameter_uncertainty=True requires n_paths; otherwise the "
            "forecast uses point estimates and there's no parameter "
            "distribution to draw from."
        )

    trend = fit.fit_info.get("trend", "c")
    if trend not in ("c", "n"):
        raise NotImplementedError(
            f"trend={trend!r} not yet supported by iterated_forecast. "
            f"Currently supports 'c' (constant) and 'n' (no trend); 'ct' "
            f"needs deterministic-time-trend handling."
        )
    # VARX guard: _extract_intercept reads B[-1], which is the LAST exog
    # coefficient (not the constant) when has_exog=True. Without future
    # exog paths a VARX forecast is undefined anyway — raise rather than
    # silently mis-compute.
    if getattr(fit, "has_exog", False):
        raise NotImplementedError(
            "iterated_forecast does not support VARX fits (fit.has_exog=True). "
            "VARX forecasting requires future paths for the exogenous block, "
            "and the current iterated_forecast implementation doesn't accept "
            "them. To add VARX forecasting, extend iterated_forecast to "
            "accept `future_exog` and update _extract_intercept to read "
            "the constant row by index (fit.p * fit.n_vars) rather than "
            "B[-1]."
        )

    p = int(fit.p)
    n = int(fit.n_vars)
    history = np.asarray(fit.data.values[-p:], dtype=float)  # (p, n)

    # Point forecast
    A_list = fit.lag_matrices()
    intercept = _extract_intercept(fit.B_hat, trend, n)
    point = _iterate(history, A_list, intercept, horizon)

    # Set up the forecast-origin timestamp and horizons
    origin = pd.Timestamp(fit.data.index[-1])
    horizons = np.arange(1, horizon + 1)

    if n_paths is None:
        return ForecastContainer(
            point=point, paths=None,
            median=None, lower_68=None, upper_68=None,
            lower_90=None, upper_90=None,
            horizons=horizons, variable_names=list(fit.variable_names),
            forecast_origin=origin,
            metadata={
                "method": "iterated_var",
                "horizon": int(horizon),
                "p": p, "n_vars": n, "trend": trend,
                "n_paths": None, "parameter_uncertainty": False,
                "seed": seed,
            },
        )

    rng = np.random.default_rng(seed)

    # If parameter uncertainty, draw (B, Sigma) per path; else share point estimates.
    if parameter_uncertainty:
        draws = fit.parameter_draws(n_draws=n_paths, seed=seed)
        # Stability filtering may return FEWER draws than requested. Size the
        # path array to what actually came back — quantiles over unwritten
        # np.empty slots would report uninitialised memory as forecasts.
        if len(draws) == 0:
            raise RuntimeError(
                "iterated_forecast: parameter_draws returned no draws "
                "(stability filtering rejected everything). Loosen the "
                "prior / increase max attempts, or run without "
                "parameter_uncertainty."
            )
        paths = np.empty((len(draws), horizon, n))
        # `draws` is a list of (B_arr, Sigma_arr_or_df); normalise Sigma to ndarray
        for k, (B_draw, Sigma_draw) in enumerate(draws):
            A_k = _lag_matrices_from_B(B_draw, n, p)
            intercept_k = _extract_intercept(B_draw, trend, n)
            Sigma_k = np.asarray(Sigma_draw)
            L_k = np.linalg.cholesky(_psd(Sigma_k))
            innovations = rng.standard_normal((horizon, n)) @ L_k.T
            paths[k] = _iterate(history, A_k, intercept_k, horizon,
                                innovations=innovations)
    else:
        paths = np.empty((n_paths, horizon, n))
        L = np.linalg.cholesky(_psd(np.asarray(fit.Sigma_u)))
        for k in range(n_paths):
            innovations = rng.standard_normal((horizon, n)) @ L.T
            paths[k] = _iterate(history, A_list, intercept, horizon,
                                innovations=innovations)

    median = np.median(paths, axis=0)
    lo_68 = np.quantile(paths, _Z68_LO, axis=0)
    hi_68 = np.quantile(paths, _Z68_HI, axis=0)
    lo_90 = np.quantile(paths, _Z90_LO, axis=0)
    hi_90 = np.quantile(paths, _Z90_HI, axis=0)

    return ForecastContainer(
        point=point, paths=paths,
        median=median, lower_68=lo_68, upper_68=hi_68,
        lower_90=lo_90, upper_90=hi_90,
        horizons=horizons, variable_names=list(fit.variable_names),
        forecast_origin=origin,
        metadata={
            "method": "iterated_var",
            "horizon": int(horizon),
            "p": p, "n_vars": n, "trend": trend,
            "n_paths": int(n_paths),
            "parameter_uncertainty": bool(parameter_uncertainty),
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Rolling / expanding window driver
# ---------------------------------------------------------------------------


def rolling_forecasts(
    data: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], Any],
    window_size: int,
    horizon: int,
    expanding: bool = False,
    start_at: int | None = None,
    step: int = 1,
) -> pd.DataFrame:
    """Walk the sample, fitting ``fit_fn`` at each origin and forecasting.

    For each forecast origin ``t``:
        * ``expanding=False`` (default): fit on ``data.iloc[t-window_size:t]``
        * ``expanding=True``: fit on ``data.iloc[:t]``
    Then forecast ``horizon`` steps ahead and align with ``data.iloc[t:t+horizon]``
    as the realised values.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series; columns are the forecast variables.
    fit_fn : callable
        ``fit_fn(data_subset)`` returns an object compatible with
        :func:`iterated_forecast` (a ``VARFit`` subclass).
    window_size : int
        Rolling-window length (number of observations in each fit).
    horizon : int
        Forecast horizon at each origin.
    expanding : bool, default False
        If True, use an expanding window starting at index 0 instead of a
        fixed rolling window. ``window_size`` then sets the *minimum*
        observations before the first forecast.
    start_at : int, optional
        First forecast origin (defaults to ``window_size``).
    step : int, default 1
        Step between consecutive origins. Use larger values to speed up
        the rolling exercise at the cost of fewer forecasts.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: ``origin``, ``horizon``, ``variable``,
        ``forecast``, ``realised``. One row per (origin, horizon, variable)
        triple. ``realised`` is NaN for horizons that extend past the end
        of ``data``. When ``fit_fn`` raises at an origin, that origin's
        rows carry NaN forecasts plus an ``error`` column with the
        exception text — score comparisons across models should use
        :func:`forecast_score_matched` so such failures cannot shift the
        comparison onto different samples.

    Notes
    -----
    Forecast columns are aligned to the input variables BY NAME (via the
    fitted model's ``variable_names``), never by position — a fit
    callback that reorders the model's variables cannot silently swap
    labels. A fitted model whose variable set does not match the input
    columns raises ``ValueError``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    T = len(data)
    if T < window_size + 1:
        raise ValueError(
            f"data has {T} rows; need at least window_size+1={window_size+1} "
            f"for at least one rolling forecast."
        )

    first = window_size if start_at is None else int(start_at)
    rows: list[dict] = []
    cols = list(data.columns)

    for t in range(first, T, step):
        if expanding:
            train = data.iloc[:t]
        else:
            train = data.iloc[t - window_size:t]
        try:
            fit = fit_fn(train)
            fc = iterated_forecast(fit, horizon=horizon)
        except Exception as exc:  # rolling robustness: log and skip
            for h in range(1, horizon + 1):
                for v in cols:
                    rows.append({
                        "origin": data.index[t - 1],
                        "horizon": h,
                        "variable": v,
                        "forecast": np.nan,
                        "realised": (data.iloc[t + h - 1][v]
                                     if t + h - 1 < T else np.nan),
                        "error": str(exc),
                    })
            continue
        # Map forecast columns to input variables BY NAME, never by
        # position: a fit callback that reorders the model's variables
        # would otherwise attach the wrong prediction to each realisation
        # without any error.
        fc_names = list(getattr(fc, "variable_names", cols))
        if sorted(fc_names) != sorted(cols) or len(set(fc_names)) != len(fc_names):
            raise ValueError(
                f"rolling_forecasts: the fitted model's variables "
                f"{fc_names!r} do not match the input columns {cols!r} — "
                f"cannot align forecasts to realisations."
            )
        col_pos = {v: fc_names.index(v) for v in cols}
        for h in range(1, horizon + 1):
            for v in cols:
                rows.append({
                    "origin": data.index[t - 1],   # last in-sample obs
                    "horizon": h,
                    "variable": v,
                    "forecast": float(fc.point[h - 1, col_pos[v]]),
                    "realised": (float(data.iloc[t + h - 1][v])
                                 if t + h - 1 < T else np.nan),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def forecast_score(
    rolling_df: pd.DataFrame,
    metric: str = "rmse",
    by: str | Sequence[str] = "horizon",
) -> pd.DataFrame:
    """Compute forecast accuracy from a :func:`rolling_forecasts` output.

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Output of :func:`rolling_forecasts` (long format with ``forecast``
        and ``realised`` columns).
    metric : {"mse", "rmse", "mae"}
        Loss metric. RMSE is the square root of the MSE.
    by : str or sequence of str, default "horizon"
        Group-by key(s). E.g. ``"horizon"`` gives one score per horizon;
        ``["horizon", "variable"]`` gives a (horizon, variable) panel;
        ``"variable"`` averages over horizons.

    Returns
    -------
    pd.DataFrame
        Score column named after the metric, indexed by the group-by key.
    """
    if metric not in ("mse", "rmse", "mae"):
        raise ValueError(f"unknown metric {metric!r}; use mse/rmse/mae")
    df = rolling_df.dropna(subset=["forecast", "realised"]).copy()
    if df.empty:
        raise ValueError("no rows with both forecast and realised — nothing to score")
    err = df["forecast"] - df["realised"]
    if metric == "mae":
        df["_loss"] = err.abs()
    else:
        df["_loss"] = err ** 2
    grouped = df.groupby(by)["_loss"].mean()
    if metric == "rmse":
        grouped = np.sqrt(grouped)
    out = grouped.reset_index().rename(columns={"_loss": metric})
    return out


def forecast_score_matched(
    rolling_dfs: dict[str, pd.DataFrame],
    metric: str = "rmse",
    by: str | Sequence[str] = "horizon",
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Score several models on their COMMON valid forecasts.

    Scoring each model separately and then comparing the aggregates can
    compare different origin sets whenever one model failed at some origin
    (its NaN rows are dropped independently) — a single failed forecast can
    reverse the apparent ranking. This helper inner-joins the models'
    :func:`rolling_forecasts` outputs on ``(origin, horizon, variable)``,
    keeps only rows where EVERY model's forecast and the shared realisation
    are finite, and scores each model on that matched sample.

    Parameters
    ----------
    rolling_dfs : dict of {model_name: rolling_forecasts output}
    metric, by : as in :func:`forecast_score`.

    Returns
    -------
    (scores, info)
        ``scores`` maps each model name to its :func:`forecast_score`
        result on the matched sample. ``info`` has ``n_common`` (matched
        rows scored) and ``n_excluded_<name>`` (that model's valid rows
        dropped because another model failed there).
    """
    if len(rolling_dfs) < 2:
        raise ValueError("forecast_score_matched needs at least two models")
    keys = ["origin", "horizon", "variable"]
    merged: pd.DataFrame | None = None
    for name, df in rolling_dfs.items():
        part = df[keys + ["forecast", "realised"]].rename(
            columns={"forecast": f"forecast_{name}", "realised": f"realised_{name}"})
        merged = part if merged is None else merged.merge(part, on=keys, how="inner")
    assert merged is not None
    names = list(rolling_dfs)
    # Realisations must agree across models (same data, same targets)
    base_real = merged[f"realised_{names[0]}"]
    for name in names[1:]:
        other = merged[f"realised_{name}"]
        both = base_real.notna() & other.notna()
        if both.any() and not np.allclose(base_real[both], other[both]):
            raise ValueError(
                f"realisations disagree between models {names[0]!r} and "
                f"{name!r} — the rolling exercises were not run on the same "
                f"data."
            )
    finite = base_real.notna()
    for name in names:
        finite &= merged[f"forecast_{name}"].notna()
    common = merged[finite]
    info: dict[str, int] = {"n_common": int(len(common))}
    for name in names:
        n_valid = int((rolling_dfs[name]["forecast"].notna()
                       & rolling_dfs[name]["realised"].notna()).sum())
        info[f"n_excluded_{name}"] = n_valid - int(len(common))
    if common.empty:
        raise ValueError("no common valid forecasts across the models")
    scores: dict[str, pd.DataFrame] = {}
    for name in names:
        long_df = common[keys].copy()
        long_df["forecast"] = common[f"forecast_{name}"].values
        long_df["realised"] = base_real[finite].values
        scores[name] = forecast_score(long_df, metric=metric, by=by)
    return scores, info


def density_scores(
    paths: np.ndarray,
    actuals: np.ndarray,
    variable_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Density-forecast evaluation: CRPS, log score, and PIT per cell.

    Scores a simulated predictive density (e.g. ``ForecastContainer.paths``
    or a conditional-forecast path ensemble) against realised values —
    the density-evaluation layer central-bank forecasting work expects
    alongside point RMSEs.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, H, n_vars)
        Simulated forecast paths (the predictive density as draws).
    actuals : np.ndarray, shape (H, n_vars)
        Realised values aligned with the forecast horizons. NaN cells are
        skipped (scores reported as NaN).
    variable_names : sequence of str, optional
        Column labels; defaults to ``var_0..var_{n-1}``.

    Returns
    -------
    pd.DataFrame (long format)
        Columns ``horizon`` (1-based), ``variable``, ``crps``,
        ``log_score``, ``pit``:

        * ``crps`` — sample CRPS (Gneiting-Raftery 2007, eq. 21):
          ``E|X - y| - 0.5 E|X - X'|``, the pairwise term computed via the
          O(m log m) sorted identity. Negatively oriented (smaller is
          better); proper.
        * ``log_score`` — log predictive density of the realisation under
          a GAUSSIAN approximation to the draw ensemble
          (``N(mean(paths), var(paths))``). Documented approximation: for
          strongly non-Gaussian ensembles prefer the CRPS, which uses the
          draws directly.
        * ``pit`` — probability integral transform, the fraction of draws
          at or below the realisation (Diebold-Gunther-Tay 1998).
          Uniform[0,1] across cells under correct calibration; a
          U-shaped histogram means the density is too narrow, a humped
          one too wide.
    """
    paths = np.asarray(paths, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if paths.ndim != 3:
        raise ValueError(f"paths must be 3-D (n_paths, H, n_vars); got {paths.shape}")
    m, H, n = paths.shape
    if m < 2:
        raise ValueError(f"need at least 2 paths for density scores, got {m}")
    if actuals.shape != (H, n):
        raise ValueError(
            f"actuals shape {actuals.shape} != (H, n_vars) = {(H, n)}"
        )
    if np.isnan(paths).any():
        raise ValueError("paths contain NaN")
    if variable_names is None:
        variable_names = [f"var_{j}" for j in range(n)]
    if len(variable_names) != n:
        raise ValueError(f"variable_names must have length {n}")

    rows = []
    idx_weights = 2.0 * np.arange(1, m + 1) - m - 1.0    # (2i - m - 1), i=1..m
    for h in range(H):
        for j in range(n):
            y = actuals[h, j]
            if np.isnan(y):
                rows.append((h + 1, variable_names[j],
                             np.nan, np.nan, np.nan))
                continue
            x = paths[:, h, j]
            # CRPS via the sorted identity: E|X-X'| = (2/m^2) sum (2i-m-1) x_(i)
            term1 = float(np.mean(np.abs(x - y)))
            xs = np.sort(x)
            term2 = float(2.0 / (m * m) * np.dot(idx_weights, xs))
            crps = term1 - 0.5 * term2
            # Gaussian-approximation log score
            mu = float(np.mean(x))
            sd = float(np.std(x, ddof=1))
            sd = max(sd, 1e-12)
            log_score = float(_sp_stats.norm.logpdf(y, loc=mu, scale=sd))
            # PIT
            pit = float(np.mean(x <= y))
            rows.append((h + 1, variable_names[j], crps, log_score, pit))

    return pd.DataFrame(
        rows, columns=["horizon", "variable", "crps", "log_score", "pit"]
    )


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    Tests H0: E[loss_a - loss_b] = 0 against H1: E[loss_a - loss_b] != 0.
    Uses a Newey-West HAC variance estimator with truncation ``horizon-1``
    (the standard choice for ``horizon``-step-ahead forecast losses),
    multiplies the statistic by the HLN (1997, IJF) small-sample factor

        sqrt((n + 1 - 2h + h(h-1)/n) / n)

    and evaluates the two-sided p-value against Student-t with ``n-1``
    degrees of freedom (not the normal). With the 20-40 forecast origins
    typical of quarterly evaluation, the uncorrected normal-based DM test
    over-rejects materially.

    The variance uses Bartlett weights ``1 - k/h`` on the autocovariances,
    which keeps it positive; the original test uses equal weights. At
    multi-step horizons the test rejects a true null somewhat too often, so
    read borderline results at long horizons with caution.

    Parameters
    ----------
    loss_a, loss_b : np.ndarray, shape (n,)
        Loss series for two competing forecasts, paired at the same
        origins / horizons. Typically squared errors or absolute errors.
    horizon : int, default 1
        Forecast horizon — used both for the HAC truncation
        (``q = max(horizon - 1, 0)``) and the HLN factor.

    Returns
    -------
    dict with keys ``"d_bar"`` (mean loss differential), ``"dm_stat"``
    (the HLN-corrected t-statistic), ``"p_value"`` (two-sided, t(n-1)),
    and ``"hln_factor"``.

    Raises
    ------
    ValueError
        If fewer than 3 paired losses remain; if ``n`` does not exceed
        the Newey-West truncation ``horizon - 1`` (with widely spaced
        forecast origins, pass the horizon in ORIGIN-SPACING units —
        the truncation and HLN factor assume consecutive origins); or
        if the loss differential is a NONZERO constant (one model
        dominates by a fixed amount — the test statistic is undefined;
        report the losses directly). Identical losses return
        ``dm_stat=0, p_value=1``.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape or loss_a.ndim != 1:
        raise ValueError("loss_a and loss_b must be 1-D arrays of the same length")
    mask = ~(np.isnan(loss_a) | np.isnan(loss_b))
    d = loss_a[mask] - loss_b[mask]
    n = len(d)
    if n < 3:
        raise ValueError(f"need at least 3 paired loss observations, got {n}")
    d_bar = float(np.mean(d))
    # Newey-West HAC variance of d_bar: globally centred Bartlett
    # cross-products with ONE denominator (1/n). Re-centring each truncated
    # slice separately (np.cov per lag) is NOT the Newey-West estimator and
    # can flip significance decisions; the Bartlett kernel with this
    # construction is guaranteed PSD, so no non-HAC fallback is needed.
    q = max(int(horizon) - 1, 0)
    if n <= q + 1:
        raise ValueError(
            f"need more than horizon paired observations for the "
            f"Newey-West truncation q={q}; got n={n}. With widely spaced "
            f"forecast origins, pass the horizon in ORIGIN-spacing units."
        )
    dc = d - d_bar
    gamma0 = float(dc @ dc) / n
    var_d_bar = gamma0
    for lag in range(1, q + 1):
        gamma = float(dc[lag:] @ dc[:-lag]) / n
        weight = 1.0 - lag / (q + 1.0)
        var_d_bar += 2.0 * weight * gamma
    var_d_bar = var_d_bar / n
    if var_d_bar <= 0:
        # Zero long-run variance: a CONSTANT loss differential. d_bar == 0
        # means literally identical losses — no evidence against H0. A
        # nonzero constant differential is a degenerate design (the test
        # statistic is undefined), not evidence of equal accuracy.
        if abs(d_bar) < 1e-300:
            return {"d_bar": d_bar, "dm_stat": 0.0, "p_value": 1.0,
                    "hln_factor": 1.0}
        raise ValueError(
            f"Diebold-Mariano test undefined: the loss differential is a "
            f"nonzero constant (d_bar={d_bar:.6g}, variance 0). One "
            f"forecast dominates by a fixed amount at every origin; report "
            f"the losses directly instead of a DM p-value."
        )
    # Harvey-Leybourne-Newbold (1997) small-sample correction + t(n-1)
    h = max(int(horizon), 1)
    hln = np.sqrt(max(n + 1.0 - 2.0 * h + h * (h - 1.0) / n, 0.0) / n)
    dm_stat = hln * d_bar / np.sqrt(var_d_bar)
    p_value = 2.0 * (1.0 - _sp_stats.t.cdf(abs(dm_stat), df=n - 1))
    return {"d_bar": d_bar, "dm_stat": float(dm_stat),
            "p_value": float(p_value), "hln_factor": float(hln)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_intercept(B: np.ndarray, trend: str, n: int) -> np.ndarray:
    """Pull the intercept row from B_hat (shape (k, n)) given trend.

    With ``trend='c'`` the intercept is the LAST row of B (matches
    `build_design_matrices` in lib.var). With ``trend='n'`` there is no
    intercept; return zeros.
    """
    if trend == "c":
        return np.asarray(B[-1], dtype=float)
    return np.zeros(n, dtype=float)


def _lag_matrices_from_B(B: np.ndarray, n: int, p: int) -> list[np.ndarray]:
    """Reconstruct [A_1, ..., A_p] from a B array (same shape as B_hat)."""
    return [np.asarray(B[lag * n:(lag + 1) * n, :]).T for lag in range(p)]


def _iterate(
    history: np.ndarray,
    A_list: list[np.ndarray],
    intercept: np.ndarray,
    horizon: int,
    innovations: np.ndarray | None = None,
) -> np.ndarray:
    """Iterate the VAR forward ``horizon`` steps from ``history``.

    Parameters
    ----------
    history : (p, n)
        Last p observations, with ``history[-1]`` being the most recent.
    A_list : list of (n, n), length p
        Lag-coefficient matrices ``A_1..A_p`` in order. ``A_j[i, k]`` is
        the effect of lag-j of variable k on variable i — same convention
        as ``lib.var.extract_lag_matrices``.
    intercept : (n,)
    horizon : int
    innovations : (horizon, n), optional
        Pre-drawn innovations. If None, deterministic point forecast.

    Returns
    -------
    out : (horizon, n)
    """
    p = len(A_list)
    n = history.shape[1]
    out = np.empty((horizon, n))
    # state[-1] is the most recent observation; state[-p] is the oldest used lag
    state = np.copy(history)
    for h in range(horizon):
        y_h = np.copy(intercept)
        for j, A in enumerate(A_list):
            y_h += A @ state[-(j + 1)]
        if innovations is not None:
            y_h += innovations[h]
        out[h] = y_h
        # Roll state forward
        state = np.roll(state, -1, axis=0)
        state[-1] = y_h
    return out


def _psd(Sigma: np.ndarray, rel_eps: float = 1e-12) -> np.ndarray:
    """Project ``Sigma`` to the nearest PSD matrix by clipping eigenvalues
    to a floor RELATIVE to the largest eigenvalue. The relative floor makes
    the projection unit-invariant: a valid positive-definite covariance
    passes through unchanged however small its absolute entries (an
    absolute floor would silently rewrite covariances expressed in small
    units and distort conditional distributions)."""
    Sigma = (Sigma + Sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    floor = max(float(eigvals.max()), 0.0) * rel_eps
    if floor == 0.0:
        # Degenerate (zero or negative-definite) input — keep a tiny
        # positive floor so downstream Cholesky factorisations don't abort.
        floor = np.finfo(float).tiny
    if eigvals.min() >= floor:
        return Sigma
    eigvals = np.clip(eigvals, floor, None)
    return (eigvecs * eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Conditional forecasts (Bańbura-Giannone-Lenza 2015)
# ---------------------------------------------------------------------------


@dataclass
class ConditionalForecastContainer:
    """Output of :func:`conditional_forecast`.

    Attributes
    ----------
    point : np.ndarray, shape (horizon, n_vars)
        Conditional mean forecast — ``E[y_{T+h} | y_{1:T}, conditioning]``
        at horizons 1..horizon. Conditioning entries pass through exactly.
    paths : np.ndarray or None, shape (n_paths, horizon, n_vars)
        Simulated draws from the conditional distribution. None when
        ``n_paths=0``.
    median, lower_68, upper_68, lower_90, upper_90 : np.ndarray or None
        Quantile summaries across ``paths`` (None when ``paths`` is None).
    horizons : np.ndarray, shape (horizon,)
    variable_names : list[str]
    forecast_origin : pd.Timestamp
    conditioning : dict[str, np.ndarray]
        The conditioning paths provided (variable name → array of length
        horizon, possibly with NaN entries indicating non-conditioning at
        that horizon).
    metadata : dict
    """

    point: np.ndarray
    paths: np.ndarray | None
    median: np.ndarray | None
    lower_68: np.ndarray | None
    upper_68: np.ndarray | None
    lower_90: np.ndarray | None
    upper_90: np.ndarray | None
    horizons: np.ndarray
    variable_names: list[str]
    forecast_origin: pd.Timestamp
    conditioning: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


def conditional_forecast(
    fit,
    horizon: int,
    conditioning: dict[str, np.ndarray] | None = None,
    n_paths: int = 0,
    parameter_uncertainty: bool = False,
    seed: int | None = None,
) -> ConditionalForecastContainer:
    """Conditional forecast — Bańbura, Giannone & Lenza (2015) Kalman approach.

    Forecasts ``y_{T+1}, ..., y_{T+horizon}`` conditional on pinned future
    values for a subset of variables (or none, recovering an unconditional
    forecast). Cast the VAR in companion-form state-space, treat the
    conditioning values as future "observations" (with NaN where there's
    no condition), and run the Kalman smoother. The smoothed mean of the
    state's first ``n_vars`` block gives the conditional mean forecast.

    Parameters
    ----------
    fit : VARFit (or subclass)
        Reduced-form VAR or BVAR fit.
    horizon : int
        Forecast horizon. Conditioning paths must have this length.
    conditioning : dict {variable_name: array(horizon)}, optional
        Per-variable conditioning path. NaN entries in the array mark
        "not conditioned at this horizon" — useful for mixed conditioning
        like "FFR pinned for next 4 quarters, free thereafter". Variables
        not in the dict are forecast endogenously. ``None`` recovers an
        unconditional forecast.
    n_paths : int, default 0
        Density forecast paths. ``0`` returns point forecast only.
    parameter_uncertainty : bool, default False
        If True, draw ``(B, Sigma)`` per path from
        ``fit.parameter_draws()``. Otherwise the conditional distribution
        is at point parameter estimates.
    seed : int, optional

    Returns
    -------
    ConditionalForecastContainer

    Notes
    -----
    * **Unconditional limit**: ``conditioning=None`` (or empty dict)
      produces the same point forecast as :func:`iterated_forecast`. The
      density paths may differ in numerics because this routine samples
      via Kalman state perturbation rather than direct innovation
      simulation, but the conditional distribution is identical in
      expectation.
    * **Hard conditions only**: the algorithm pins each conditioning
      value exactly. Soft conditioning (e.g., "GDP growth between 1%
      and 3%") via importance tilting is not implemented.
    * **Relation to Waggoner-Zha (1999)**: WZ implement conditional
      forecasts in SHOCK space (linear restrictions on future structural
      shocks via QR/SVD constraint solving); this routine is the
      Bańbura-Giannone-Lenza (2015) STATE-space formulation (Kalman
      smoother with pinned observations). The two deliver the same
      conditional mean asymptotically; the state-space form additionally
      yields full density paths via the simulation smoother. A
      methodology choice, not a divergence — but numbers compared
      against implementations of the WZ form may differ in finite
      samples.
    * **No measurement noise**: the state-space measurement equation is
      ``y_t = Z @ alpha_t`` with exactly zero error — VARs have observed
      states (the lag stack), and hard conditions enter as EXACT
      observations. The innovation covariance is only PSD-projected with
      a floor RELATIVE to its largest eigenvalue, so the conditional
      forecast is invariant to the data's units.

    Raises
    ------
    NotImplementedError
        If ``fit.fit_info["trend"]`` is not ``"c"`` or ``"n"``, or the
        fit is a VARX (``fit.has_exog=True``) — the companion
        state-space builder cannot represent those deterministic /
        exogenous blocks.

    Examples
    --------
    Counterfactual GDP given a hypothetical FFR path::

        fit = lib.fit_minnesota_bvar(data, lags=4)
        ffr_path = np.full(8, 5.0)   # FFR held at 5% for 8 quarters
        result = lib.conditional_forecast(
            fit, horizon=8, conditioning={"ffr": ffr_path}, n_paths=1000, seed=0,
        )
        gdp_response = result.point[:, fit.variable_names.index("rgdp")]
    """
    # Local imports to avoid circular deps at module load
    from . import state_space as ss

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_paths < 0:
        raise ValueError(f"n_paths must be >= 0, got {n_paths}")

    variable_names = list(fit.variable_names)
    n_vars = fit.n_vars
    p = fit.p

    # Validate + canonicalise conditioning
    cond_arr = np.full((horizon, n_vars), np.nan)
    canonical_cond: dict[str, np.ndarray] = {}
    for name, path in (conditioning or {}).items():
        if name not in variable_names:
            raise ValueError(
                f"conditioning variable {name!r} not in fit.variable_names "
                f"= {variable_names}"
            )
        arr = np.asarray(path, dtype=float).ravel()
        if arr.shape != (horizon,):
            raise ValueError(
                f"conditioning[{name!r}] must have shape ({horizon},); "
                f"got {arr.shape}"
            )
        col = variable_names.index(name)
        cond_arr[:, col] = arr
        canonical_cond[name] = arr

    # Same layout guards as iterated_forecast: the state-space builder
    # infers the constant from the trailing coefficient row, which is wrong
    # for trends and exogenous blocks — reject rather than silently build a
    # different model from the one fitted.
    _trend = fit.fit_info.get("trend", "c")
    if _trend not in ("c", "n"):
        raise NotImplementedError(
            f"trend={_trend!r} not supported by conditional_forecast "
            f"(supports 'c' and 'n'; a deterministic time trend needs "
            f"explicit handling in the state-space builder)."
        )
    if getattr(fit, "has_exog", False):
        raise NotImplementedError(
            "conditional_forecast does not support VARX fits "
            "(fit.has_exog=True): future exogenous paths are required and "
            "the state-space builder cannot represent the exogenous block."
        )

    # Cast VAR as companion-form state-space
    model = _build_var_state_space(fit)

    # Build the extended observation matrix:
    # Pre-forecast: actual data Y[lags:] (after pre-sample) — the state
    # filter consumes this to land at the right initial condition.
    # Forecast horizon: cond_arr rows (with NaN where not conditioned).
    actual_Y = fit.Y_act               # shape (T_eff, n_vars), aligns with state-space evolution
    Y_extended = np.vstack([actual_Y, cond_arr])

    # Point forecast: Kalman smooth, extract first n_vars of state
    res = ss.kalman_filter_smooth(model, Y_extended)
    smoothed = res.smoothed_state[:, :n_vars]
    forecast_block = smoothed[-horizon:]    # (horizon, n_vars)
    point = forecast_block.copy()
    # Override with conditioning values exactly where provided
    for col_idx in range(n_vars):
        mask = ~np.isnan(cond_arr[:, col_idx])
        point[mask, col_idx] = cond_arr[mask, col_idx]

    # Forecast origin
    fc_origin = fit.data.index[-1] if hasattr(fit, "data") else None

    # If no density forecast requested, return now
    if n_paths == 0:
        horizons_arr = np.arange(1, horizon + 1)
        return ConditionalForecastContainer(
            point=point, paths=None,
            median=None, lower_68=None, upper_68=None, lower_90=None, upper_90=None,
            horizons=horizons_arr,
            variable_names=variable_names,
            forecast_origin=fc_origin,
            conditioning=canonical_cond,
            metadata={
                "method": "BGL 2015 Kalman conditional smoothing",
                "n_paths": 0,
                "parameter_uncertainty": False,
                "seed": seed,
            },
        )

    # Density forecast: simulate via Kalman simulation smoother + parameter draws
    rng = np.random.default_rng(seed)
    if parameter_uncertainty and hasattr(fit, "parameter_draws"):
        param_draws = fit.parameter_draws(n_draws=n_paths, seed=seed)
    else:
        param_draws = None

    paths = np.empty((n_paths, horizon, n_vars))
    if param_draws is None:
        # Single model — batch the n_paths draws in one call to the simulation smoother
        sim_all = ss.simulation_smoother(
            model, Y_extended, n_draws=n_paths, seed=seed,
        )   # (n_paths, T_total, n_state)
        # Forecast block: last `horizon` time steps, first n_vars of state
        paths = sim_all[:, -horizon:, :n_vars]
    else:
        # One draw per parameter sample. Stability filtering may return
        # fewer draws than requested — iterate over what came back.
        if len(param_draws) == 0:
            raise RuntimeError(
                "conditional_forecast: parameter_draws returned no draws "
                "(stability filtering rejected everything)."
            )
        if len(param_draws) < n_paths:
            paths = paths[: len(param_draws)]
        for j in range(len(param_draws)):
            B_j, Sigma_j = param_draws[j]
            model_j = _build_var_state_space_from_BS(
                B_j, Sigma_j, fit.p, fit.k, fit.n_vars, fit.Y_act,
            )
            sim_j = ss.simulation_smoother(
                model_j, Y_extended, n_draws=1,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            paths[j] = sim_j[0, -horizon:, :n_vars]

    # Pin conditioning values across all paths
    for col_idx in range(n_vars):
        mask = ~np.isnan(cond_arr[:, col_idx])
        if mask.any():
            paths[:, mask, col_idx] = cond_arr[mask, col_idx]

    # Quantile summaries
    median = np.median(paths, axis=0)
    lower_68 = np.quantile(paths, 0.16, axis=0)
    upper_68 = np.quantile(paths, 0.84, axis=0)
    lower_90 = np.quantile(paths, 0.05, axis=0)
    upper_90 = np.quantile(paths, 0.95, axis=0)

    horizons_arr = np.arange(1, horizon + 1)
    return ConditionalForecastContainer(
        point=point, paths=paths,
        median=median, lower_68=lower_68, upper_68=upper_68,
        lower_90=lower_90, upper_90=upper_90,
        horizons=horizons_arr,
        variable_names=variable_names,
        forecast_origin=fc_origin,
        conditioning=canonical_cond,
        metadata={
            "method": "BGL 2015 Kalman conditional smoothing",
            "n_paths": int(n_paths),
            "parameter_uncertainty": bool(parameter_uncertainty),
            "seed": seed,
        },
    )


def _build_var_state_space(fit):
    """Build a companion-form state-space model from a VARFit.

    State: ``alpha_t = [y_t; y_{t-1}; ...; y_{t-p+1}]`` (n_vars * p,)
    Measurement: ``y_t = Z alpha_t`` (Z = [I_n | 0 | ... | 0], H = 0)
    Transition: ``alpha_{t+1} = T alpha_t + c + R eta_t``
    """
    n = fit.n_vars
    p = fit.p
    return _build_var_state_space_from_BS(
        np.asarray(fit.B_hat), np.asarray(fit.Sigma_u),
        p, fit.k, n, fit.Y_act,
    )


def _build_var_state_space_from_BS(B, Sigma, p, k, n, Y_act):
    """Build state-space from explicit (B, Sigma) — used for parameter-draw paths."""
    from . import state_space as ss
    lag_mats = _lag_matrices_from_B(B, n, p)
    intercept = _extract_intercept(B, "c" if k > n * p else "n", n)
    n_state = n * p
    T_mat = np.zeros((n_state, n_state))
    for l, A_l in enumerate(lag_mats):
        T_mat[:n, l * n:(l + 1) * n] = A_l
    if p > 1:
        T_mat[n:, :-n] = np.eye((p - 1) * n)
    R = np.zeros((n_state, n))
    R[:n] = np.eye(n)
    Q = _psd(Sigma)
    Z = np.zeros((n, n_state))
    Z[:, :n] = np.eye(n)
    # H = 0 exactly, as the docstring states: observed values and hard
    # conditions are EXACT. A tiny ridge here would turn hard conditions
    # into noisy observations and distort the conditional distribution of
    # the unconditioned variables whenever the data's units are small.
    H = np.zeros((n, n))
    c = np.zeros(n_state)
    c[:n] = intercept
    init_block = Y_act[:p].copy() if Y_act.shape[0] >= p else np.zeros((p, n))
    if init_block.shape[0] < p:
        init_block = np.vstack([np.zeros((p - init_block.shape[0], n)), init_block])
    initial_state = np.concatenate(
        [init_block[p - 1 - l] for l in range(p)]
    )
    return ss.StateSpaceModel(
        Z=Z, H=H, T=T_mat, R=R, Q=Q,
        state_intercept=c, initial_state_mean=initial_state,
    )
