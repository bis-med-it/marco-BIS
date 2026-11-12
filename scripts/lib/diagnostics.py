"""One-button diagnostics for a fitted VAR / BVAR / MF-VAR / TVP-VAR / DFM.

The classic VAR-diagnostic suite - stability, residual autocorrelation,
heteroskedasticity, normality, lag-selection comparison - bundled into a
single :func:`diagnostics` call that returns a structured
:class:`DiagnosticReport`. The report has a ``summary()`` method for a
human-readable console print and a ``to_log_dict()`` method for
``research_log.json`` provenance.

What's tested per residual series:

* **Mean** and **std** (sanity numbers, no test).
* **Ljung-Box** at the configured lag - serial correlation in residuals.
  H0: no autocorrelation up to lag ``lb_lags``. A small p-value says the
  model has left structure in the residuals - usually a sign of
  under-fitted lag order.
* **Jarque-Bera** - normality of residuals. H0: residuals normal. Small
  p-value flags fat tails / asymmetry; matters for inference.
* **ARCH-LM** at the configured lag - conditional heteroskedasticity.
  Small p-value suggests time-varying variance; consider an SV extension
  via [`lib.stochastic_volatility`](stochastic_volatility.py).

System-level:

* **Max companion-matrix eigenvalue modulus** (via :func:`max_eigenvalue_modulus`).
  Below 0.95 is "stationary"; 0.95–1.00 is "marginal (boundary)" - common
  for BVARs with random-walk priors; above 1.00 is "unstable". The
  unstable case typically indicates either a unit root in the data that
  the fit hasn't accommodated, or numerical drift in a constrained
  estimator.

* **Lag-selection comparison**: refit the model class at every
  ``p ∈ {1, ..., max_lag_search}`` and report what AIC, BIC, HQIC would
  each have picked. If the current ``p`` disagrees with what one or more
  ICs prefer, the report's warnings list flags it.

The dispatcher handles ``VARFit`` and ``BVARFit`` directly; subclasses
without their own lag-IC computation fall back to skipping that section
with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


# Thresholds for flagging concerns in the warnings list
_PVAL_ALPHA = 0.05
_STABILITY_MARGIN = 0.95


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    """Structured output of :func:`diagnostics`.

    Attributes
    ----------
    fit_class : str
        ``type(fit).__name__``.
    n_obs : int
        Effective sample size used in estimation (``fit.t_eff``).
    n_vars : int
    p : int
        Lag order of the fit.
    fit_info : dict
        Pass-through of ``fit.fit_info`` (AIC/BIC/HQIC/llf/etc).
    stability : dict
        ``max_eigenvalue_modulus``, ``stable`` (bool), ``status`` (str).
    residuals : dict
        Per-variable: mean, std, ``ljung_box_stat``/``_pvalue``,
        ``jarque_bera_stat``/``_pvalue``, ``arch_stat``/``_pvalue``.
        Keyed by variable name.
    lag_selection : dict
        Current ``p``, plus what AIC, BIC, HQIC each recommend across
        ``range(1, max_lag_search + 1)``. Falls back to ``None`` for fit
        classes whose lag-selection isn't implemented in this dispatch.
    warnings : list[str]
        Plain-text strings flagging concerns. Empty list = clean bill of
        health.
    """

    fit_class: str
    n_obs: int
    n_vars: int
    p: int
    fit_info: dict[str, Any]
    stability: dict[str, Any]
    residuals: dict[str, dict[str, float]]
    lag_selection: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)

    # -------- presentation --------

    def summary(self) -> str:
        """Human-readable text summary, suitable for console printing."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"  Diagnostics for {self.fit_class}")
        lines.append("=" * 70)

        # Fit summary
        ic_bits = []
        for k in ("aic", "bic", "hqic", "llf", "log_mdd"):
            if k in self.fit_info:
                v = self.fit_info[k]
                if isinstance(v, (int, float)):
                    ic_bits.append(f"{k.upper()}={v:.2f}")
        lines.append(f"Fit: n_obs={self.n_obs}, n_vars={self.n_vars}, p={self.p}"
                     + (f"  ({', '.join(ic_bits)})" if ic_bits else ""))

        # Stability
        st = self.stability
        lines.append(
            f"Stability: max |eig| = {st['max_eigenvalue_modulus']:.4f}  "
            f"[{st['status']}]"
        )

        # Residuals table
        lines.append("")
        lines.append("Residual diagnostics (per variable):")
        lines.append(
            f"  {'variable':<14}{'mean':>10}{'std':>10}"
            f"{'LB(p)':>10}{'JB(p)':>10}{'ARCH(p)':>10}"
        )
        for var, r in self.residuals.items():
            lb = r.get("ljung_box_pvalue", float("nan"))
            jb = r.get("jarque_bera_pvalue", float("nan"))
            ar = r.get("arch_pvalue", float("nan"))
            lines.append(
                f"  {var:<14}{r['mean']:>+10.4f}{r['std']:>10.4f}"
                f"{lb:>10.3f}{jb:>10.3f}{ar:>10.3f}"
            )

        # Lag selection
        if self.lag_selection is not None:
            ls = self.lag_selection
            lines.append("")
            lines.append(f"Lag selection (across p = 1..{ls['max_lag_search']}):")
            lines.append(f"  Current p = {ls['current_p']}")
            for ic in ("aic", "bic", "hqic"):
                rec = ls.get(f"{ic}_recommended")
                if rec is not None:
                    flag = "  <- differs" if rec != ls["current_p"] else ""
                    lines.append(f"  {ic.upper():<6} picks p = {rec}{flag}")

        # Warnings
        lines.append("")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("No warnings flagged.")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_log_dict(self) -> dict[str, Any]:
        """JSON-safe dict, suitable for research_log.json embedding."""
        return {
            "fit_class": self.fit_class,
            "n_obs": int(self.n_obs),
            "n_vars": int(self.n_vars),
            "p": int(self.p),
            "fit_info": _to_json_safe(self.fit_info),
            "stability": _to_json_safe(self.stability),
            "residuals": _to_json_safe(self.residuals),
            "lag_selection": _to_json_safe(self.lag_selection) if self.lag_selection else None,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# diagnostics()
# ---------------------------------------------------------------------------


def diagnostics(
    fit,
    max_lag_search: int = 12,
    lb_lags: "int | Sequence[int]" = 8,
    arch_lags: int = 4,
) -> DiagnosticReport:
    """One-button diagnostics for a fitted VAR / BVAR / etc.

    Parameters
    ----------
    fit : VARFit subclass
        Must expose ``residuals`` (ndarray), ``variable_names`` (sequence),
        ``p`` (int), ``n_vars`` (int), ``lag_matrices()`` (list of (n, n)
        matrices), ``fit_info`` (dict), and ``t_eff`` (int) - i.e., the
        standard ``VARFit`` interface.
    max_lag_search : int, default 12
        Max ``p`` to consider in the AIC/BIC/HQIC lag-selection comparison.
        Set to 0 or 1 to skip the lag-selection section.
    lb_lags : int or sequence of int, default 8
        Ljung-Box test lag(s), per variable. 8 is a standard choice for
        quarterly data; pass e.g. ``[6, 12, 24]`` for the monthly
        multi-lag convention. Each per-variable result carries
        ``"ljung_box"``: ``{lag: {"stat", "p_value"}}`` for every requested
        lag; the scalar ``"ljung_box_stat"`` / ``"ljung_box_pvalue"``
        fields keep reporting the FIRST lag for backward compatibility.
    arch_lags : int, default 4
        ARCH-LM test lag.

    Returns
    -------
    DiagnosticReport
    """
    # Local import to keep this module independent at load time
    import lib

    # --- Stability ---
    A_list = fit.lag_matrices()
    max_eig = float(lib.max_eigenvalue_modulus(A_list))
    if max_eig < _STABILITY_MARGIN:
        status = "stationary"
    elif max_eig < 1.0:
        status = "marginal (near unit root)"
    else:
        status = "non-stationary / unstable"
    stability = {
        "max_eigenvalue_modulus": max_eig,
        "stable": bool(max_eig < 1.0),
        "status": status,
    }

    # --- Residual tests (per variable) ---
    residuals_arr = np.asarray(fit.residuals)
    var_names = list(fit.variable_names)
    residuals: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    if max_eig >= 1.0:
        warnings.append(
            f"System has a root on or outside the unit circle "
            f"(max |eig| = {max_eig:.4f})."
        )
    elif max_eig >= _STABILITY_MARGIN:
        warnings.append(
            f"Max |eig| = {max_eig:.4f} sits near the unit circle - expected "
            f"for systems estimated in (log-)levels, OLS or BVAR with "
            f"random-walk priors alike. Not a failure by itself: check that "
            f"IRFs to transitory shocks flatten to a finite limit rather "
            f"than grow (see the diagnostics skill's levels-VAR arbitration)."
        )

    n_resid = residuals_arr.shape[0]
    lb_lag_list = [lb_lags] if isinstance(lb_lags, (int, np.integer)) else list(lb_lags)
    # Apply the standard T/5 cap per requested lag, dedupe preserving order
    lb_lags_eff: list[int] = []
    for L in lb_lag_list:
        L_eff = max(1, min(int(L), n_resid // 5))
        if L_eff not in lb_lags_eff:
            lb_lags_eff.append(L_eff)
    arch_lags_eff = max(1, min(arch_lags, n_resid // 10))
    for j, var in enumerate(var_names):
        r = residuals_arr[:, j]
        d: dict[str, Any] = {
            "mean": float(np.mean(r)),
            "std": float(np.std(r, ddof=1)),
        }
        # Ljung-Box (one test per requested lag; scalar fields = first lag)
        try:
            lb = acorr_ljungbox(r, lags=lb_lags_eff, return_df=True)
            d["ljung_box"] = {
                int(L): {
                    "stat": float(lb["lb_stat"].loc[L]),
                    "p_value": float(lb["lb_pvalue"].loc[L]),
                }
                for L in lb_lags_eff
            }
            d["ljung_box_stat"] = d["ljung_box"][lb_lags_eff[0]]["stat"]
            d["ljung_box_pvalue"] = d["ljung_box"][lb_lags_eff[0]]["p_value"]
            for L in lb_lags_eff:
                p_L = d["ljung_box"][L]["p_value"]
                if p_L < _PVAL_ALPHA:
                    warnings.append(
                        f"{var}: Ljung-Box at lag {L} rejects no-autocorrelation "
                        f"(p={p_L:.3f}) - residuals show serial dependence; "
                        f"consider a higher lag order."
                    )
        except Exception:
            d["ljung_box"] = {}
            d["ljung_box_stat"] = float("nan")
            d["ljung_box_pvalue"] = float("nan")
        # Jarque-Bera
        try:
            jb = jarque_bera(r)
            d["jarque_bera_stat"] = float(jb.statistic)
            d["jarque_bera_pvalue"] = float(jb.pvalue)
            if d["jarque_bera_pvalue"] < _PVAL_ALPHA:
                warnings.append(
                    f"{var}: Jarque-Bera rejects normality (p={d['jarque_bera_pvalue']:.3f}) - "
                    f"residuals show skew/kurtosis. Inference based on Gaussian innovations "
                    f"may be miscalibrated; consider bootstrap bands."
                )
        except Exception:
            d["jarque_bera_stat"] = float("nan")
            d["jarque_bera_pvalue"] = float("nan")
        # ARCH-LM
        try:
            arch = het_arch(r, nlags=arch_lags_eff)
            d["arch_stat"] = float(arch[0])
            d["arch_pvalue"] = float(arch[1])
            if d["arch_pvalue"] < _PVAL_ALPHA:
                warnings.append(
                    f"{var}: ARCH-LM at lag {arch_lags_eff} flags conditional heteroskedasticity "
                    f"(p={d['arch_pvalue']:.3f}) - consider a stochastic-volatility extension "
                    f"(`lib.stochastic_volatility`)."
                )
        except Exception:
            d["arch_stat"] = float("nan")
            d["arch_pvalue"] = float("nan")
        residuals[var] = d

    # --- Lag-selection comparison ---
    lag_selection = None
    if max_lag_search >= 2 and isinstance(fit.data, pd.DataFrame):
        try:
            lag_selection = _compare_lag_orders(
                fit.data, fit.p, max_lag_search=max_lag_search, trend=fit.fit_info.get("trend", "c"),
            )
            lag_selection["max_lag_search"] = int(max_lag_search)
            lag_selection["current_p"] = int(fit.p)
            for ic in ("aic", "bic", "hqic"):
                rec = lag_selection.get(f"{ic}_recommended")
                if rec is not None and rec != fit.p:
                    warnings.append(
                        f"Lag order: {ic.upper()} prefers p={rec}, current is p={fit.p}."
                    )
        except Exception as exc:
            lag_selection = {"max_lag_search": int(max_lag_search), "error": str(exc)}

    return DiagnosticReport(
        fit_class=type(fit).__name__,
        n_obs=int(getattr(fit, "t_eff", residuals_arr.shape[0])),
        n_vars=int(fit.n_vars),
        p=int(fit.p),
        fit_info=dict(fit.fit_info) if isinstance(fit.fit_info, dict) else {},
        stability=stability,
        residuals=residuals,
        lag_selection=lag_selection,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_lag_orders(
    data: pd.DataFrame, current_p: int, max_lag_search: int, trend: str,
) -> dict[str, Any]:
    """Refit an OLS VAR at each p ∈ {1, ..., max_lag_search} and record the
    AIC/BIC/HQIC values. Returns a dict with per-IC recommendations."""
    import lib

    table: list[dict[str, Any]] = []
    for p in range(1, max_lag_search + 1):
        if len(data) <= p + 2:
            break
        try:
            fit_p = lib.fit_var(data, lags=p, trend=trend)
            info = fit_p.fit_info
            table.append({
                "p": p,
                "aic": float(info.get("aic", np.nan)),
                "bic": float(info.get("bic", np.nan)),
                "hqic": float(info.get("hqic", np.nan)),
            })
        except Exception:
            continue

    out: dict[str, Any] = {"table": table}
    for ic in ("aic", "bic", "hqic"):
        vals = [row for row in table if not np.isnan(row[ic])]
        if vals:
            best = min(vals, key=lambda r: r[ic])
            out[f"{ic}_recommended"] = int(best["p"])
            out[f"{ic}_best_value"] = float(best[ic])
    return out


def _to_json_safe(obj):
    """Recursively normalise numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
