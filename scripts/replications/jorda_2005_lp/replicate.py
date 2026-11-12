"""Jordà (2005) Local Projections — demonstration on a CEE-style monetary VAR.

Estimates the same impulse response in two ways:

1. **VAR(p) + Cholesky** — closed-form structural IRF from a recursively
   identified VAR with FFR ordered last. Bands via the asymptotic MC method.
2. **Local Projections** — horizon-by-horizon OLS of future ``y_{t+h}`` on the
   structural monetary policy shock identified in step 1, with lagged ``y_t``
   as controls. Newey-West HAC bands at the LP-standard ``h+1`` truncation.

Jordà's claim: when the VAR lag specification is correct, the two methods
produce similar point estimates; LP bands are typically wider, reflecting
the cost of horizon-by-horizon estimation. LP avoids imposing the VAR's
recursive response dynamics, but consistency still requires valid shock
identification and an adequate projection/control specification. Here the
shock is recovered from the same fitted VAR.

Library usage: ``lib.fit_var`` (VAR estimation), ``lib.cholesky_irf``
(closed-form structural IRF), ``lib.lp_irf`` (local projections with HAC bands).
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_and_transform(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        start=data_cfg["sample_start"],
        end=data_cfg["sample_end"],
    )
    return lib.transform(raw[data_cfg["variable_order"]], data_cfg["transforms"])


def lp_rolling_forecasts(
    data: pd.DataFrame, lags: int, horizon: int,
    window_size: int, step: int = 1,
) -> pd.DataFrame:
    """LP-style direct multi-step rolling forecasts.

    Matches the semantics of ``lib.rolling_forecasts``: at each origin
    ``t-1`` (training window ends at index ``t``), forecast ``data[t..t+h-1]``
    by direct OLS regression of ``y_{tau+h}`` on ``[1, y_tau, y_{tau-1},
    ..., y_{tau-p+1}]`` over the training window, then predict from the
    origin's lag vector. Per-variable OLS (each equation independent).

    Returns the same long-format DataFrame shape as ``lib.rolling_forecasts``
    so the downstream RMSE + Diebold-Mariano machinery applies unchanged.
    """
    T, n_vars = data.shape
    cols = list(data.columns)
    rows: list[dict] = []
    for t in range(window_size, T, step):
        train = data.iloc[t - window_size:t]
        T_train = len(train)
        for h in range(1, horizon + 1):
            tau_min, tau_max = lags - 1, T_train - 1 - h
            if tau_max < tau_min:
                continue
            n_train = tau_max - tau_min + 1
            if n_train < lags * n_vars + 1 + 5:
                continue
            X = np.ones((n_train, lags * n_vars + 1))
            y_tgt = np.zeros((n_train, n_vars))
            for i, tau in enumerate(range(tau_min, tau_max + 1)):
                base = 1
                for lag in range(lags):
                    X[i, base:base + n_vars] = train.iloc[tau - lag].values
                    base += n_vars
                y_tgt[i] = train.iloc[tau + h].values
            beta, *_ = np.linalg.lstsq(X, y_tgt, rcond=None)
            x_origin = np.ones(lags * n_vars + 1)
            base = 1
            for lag in range(lags):
                x_origin[base:base + n_vars] = train.iloc[T_train - 1 - lag].values
                base += n_vars
            forecast = x_origin @ beta
            for var_idx, var in enumerate(cols):
                realised = (float(data.iloc[t + h - 1][var])
                            if t + h - 1 < T else np.nan)
                rows.append({
                    "origin": data.index[t - 1],
                    "horizon": h,
                    "variable": var,
                    "forecast": float(forecast[var_idx]),
                    "realised": realised,
                })
    return pd.DataFrame(rows)


def extract_cholesky_shock(fit, shock_var: str) -> pd.Series:
    """Recover the structural shock for ``shock_var`` from a fitted VAR.

    Under the Cholesky ordering of ``fit.variable_names``, the structural
    shocks are ``eps_t = P^{-1} u_t`` where ``P = chol(Sigma_u)``. The
    returned Series is indexed by the residual-period dates
    (``fit.data.index[fit.p:]``) so it aligns with ``data.loc[shock.index]``
    for downstream LP estimation.

    The shock has unit variance asymptotically; in finite samples the
    realised sample variance is ``(T-k)/T`` due to OLS df adjustment in
    ``Sigma_u``.
    """
    P = np.linalg.cholesky(fit.Sigma_u)
    P_inv = np.linalg.inv(P)
    structural = fit.residuals @ P_inv.T          # (T-p) x n
    idx = fit.data.index[fit.p:]
    var_idx = fit.variable_names.index(shock_var)
    return pd.Series(
        structural[:, var_idx], index=idx,
        name=f"{shock_var}_chol_shock",
    )


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print(f"  {cfg['replication']['subtitle']}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Variables (Cholesky order): {list(data.columns)}")

    print("\n[2/4] Fitting VAR(p) and identifying the monetary policy shock...")
    fit = lib.fit_var(data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"])
    shock = extract_cholesky_shock(fit, cfg["model"]["shock_var"])
    shock_idx = fit.variable_names.index(cfg["model"]["shock_var"])
    print(f"      VAR(p={fit.p}), T_eff={fit.t_eff}, stable={fit.stable()}")
    print(f"      Structural shock: {cfg['model']['shock_var']} equation residual,"
          f" mean={shock.mean():+.4f}, std={shock.std():.4f}")

    print("\n[3/4] VAR Cholesky IRF (closed-form, asymptotic MC bands)...")
    var_irf = lib.cholesky_irf(
        fit,
        horizon=cfg["model"]["horizon"],
        bands=cfg["inference"]["var_bands"],
        n_draws=cfg["inference"]["n_draws"],
        seed=seed,
    )

    print("\n[4/4] LP IRF (horizon-by-horizon, HAC bands)...")
    data_for_lp = data.loc[shock.index]
    lp_irf_result = lib.lp_irf(
        data=data_for_lp,
        shock=shock,
        horizon=cfg["model"]["horizon"],
        lags=cfg["model"]["lags"],
        shock_name="mp_struct",
        hac_lag_rule=cfg["inference"]["hac_lag_rule"],
        credible_intervals=tuple(cfg["inference"]["credible_intervals"]),
    )

    # ---- Comparison table ----
    print("\n--- IRF comparison at selected horizons (response to one-sd MP shock) ---")
    horizons_to_print = [h for h in (0, 1, 2, 4, 8, 12, 16, 20)
                        if h <= cfg["model"]["horizon"]]
    comparisons: list[dict] = []
    for var in fit.variable_names:
        v_resp = fit.variable_names.index(var)
        l_resp = lp_irf_result.variable_names.index(var)
        print(f"\n  Response of '{var}':")
        print(f"    {'h':>3}  {'VAR median':>11}  {'VAR 68% CI':>22}"
              f"  {'LP median':>10}  {'LP 68% CI':>22}")
        print(f"    {'-' * 3}  {'-' * 11}  {'-' * 22}  {'-' * 10}  {'-' * 22}")
        for h in horizons_to_print:
            v_m = var_irf.median[h, v_resp, shock_idx]
            v_lo = var_irf.lower_68[h, v_resp, shock_idx]
            v_hi = var_irf.upper_68[h, v_resp, shock_idx]
            l_m = lp_irf_result.median[h, l_resp, 0]
            l_lo = lp_irf_result.lower_68[h, l_resp, 0]
            l_hi = lp_irf_result.upper_68[h, l_resp, 0]
            print(f"    {h:>3}  {v_m:>+11.4f}  [{v_lo:>+7.4f},{v_hi:>+7.4f}]"
                  f"  {l_m:>+10.4f}  [{l_lo:>+7.4f},{l_hi:>+7.4f}]")
            comparisons.append({
                "variable": var, "h": h,
                "var_median": float(v_m),
                "var_lower_68": float(v_lo), "var_upper_68": float(v_hi),
                "lp_median": float(l_m),
                "lp_lower_68": float(l_lo), "lp_upper_68": float(l_hi),
            })

    # ---- Diagnostic: agreement metric ----
    sq_diff = np.array([
        (c["var_median"] - c["lp_median"]) ** 2 for c in comparisons
    ])
    rms_diff = float(np.sqrt(sq_diff.mean()))
    print(f"\n  RMS (VAR median - LP median), all variables and horizons: {rms_diff:.4f}")
    print(f"  Avg LP 68% half-width: "
          f"{np.mean([(c['lp_upper_68']-c['lp_lower_68'])/2 for c in comparisons]):.4f}")
    print(f"  Avg VAR 68% half-width: "
          f"{np.mean([(c['var_upper_68']-c['var_lower_68'])/2 for c in comparisons]):.4f}")

    # ---- OOS forecast comparison: iterated VAR vs direct LP at multiple horizons ----
    print("\n--- Rolling-window OOS forecast comparison (iterated VAR vs direct LP) ---")
    fc_cfg = cfg["forecast_evaluation"]
    lags, trend = cfg["model"]["lags"], cfg["model"]["trend"]

    def fit_var_fn(df: pd.DataFrame):
        return lib.fit_var(df, lags=lags, trend=trend)

    print(f"      Rolling window: {fc_cfg['window_size']} quarters, "
          f"step={fc_cfg['step']}, horizon=1..{fc_cfg['horizon']}")
    rolling_var = lib.rolling_forecasts(
        data, fit_var_fn,
        window_size=fc_cfg["window_size"], horizon=fc_cfg["horizon"],
        step=fc_cfg["step"],
    )
    rolling_lp = lp_rolling_forecasts(
        data, lags=lags, horizon=fc_cfg["horizon"],
        window_size=fc_cfg["window_size"], step=fc_cfg["step"],
    )
    n_origins = int(rolling_var["origin"].nunique())
    print(f"      Generated forecasts at {n_origins} origins.")

    # Matched scoring: both approaches evaluated on the SAME valid
    # (origin, horizon, variable) cells, so a failed fit in one of them
    # cannot shift the comparison onto different samples.
    _scores, _match = lib.forecast_score_matched(
        {"lp": rolling_lp, "var": rolling_var},
        metric="rmse", by=["horizon", "variable"],
    )
    rmse_v, rmse_l = _scores["var"], _scores["lp"]
    print(f"      Matched comparison cells: {_match['n_common']}")
    rmse_cmp = rmse_l.merge(rmse_v, on=["horizon", "variable"], suffixes=("_lp", "_var"))
    rmse_cmp["ratio_lp_over_var"] = rmse_cmp["rmse_lp"] / rmse_cmp["rmse_var"]

    horizons = fc_cfg["horizons_to_report"]
    print(f"\n      RMSE ratio (LP-direct / VAR-iterated), <1 = LP wins:")
    print(f"        {'variable':<12}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    oos_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            row = rmse_cmp[(rmse_cmp["horizon"] == h) & (rmse_cmp["variable"] == v)]
            if row.empty:
                cells.append("       n/a")
                continue
            r = float(row["ratio_lp_over_var"].iloc[0])
            cells.append(f"{r:>10.3f}")
            oos_rows.append({
                "variable": v, "horizon": int(h),
                "rmse_var": float(row["rmse_var"].iloc[0]),
                "rmse_lp": float(row["rmse_lp"].iloc[0]),
                "ratio_lp_over_var": r,
            })
        print(f"        {v:<12}" + "".join(cells))
    n_lp_wins = sum(1 for r in oos_rows if r["ratio_lp_over_var"] < 1.0)
    print(f"\n      LP wins (ratio < 1) in {n_lp_wins} of {len(oos_rows)} cells.")

    print(f"\n      Diebold-Mariano p-values (squared-error losses, LP vs VAR):")
    print(f"        {'variable':<12}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    dm_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            l_h = rolling_lp[(rolling_lp["horizon"] == h)
                              & (rolling_lp["variable"] == v)
                              & rolling_lp["realised"].notna()]
            v_h = rolling_var[(rolling_var["horizon"] == h)
                               & (rolling_var["variable"] == v)
                               & rolling_var["realised"].notna()]
            merged = l_h.merge(v_h, on=["origin", "variable"], suffixes=("_lp", "_var"))
            err_l = (merged["forecast_lp"] - merged["realised_lp"]).values
            err_v = (merged["forecast_var"] - merged["realised_var"]).values
            if len(err_l) < 5:
                cells.append("       n/a")
                continue
            dm = lib.diebold_mariano(err_l ** 2, err_v ** 2, horizon=h)
            cells.append(f"{dm['p_value']:>10.3f}")
            dm_rows.append({"variable": v, "horizon": int(h),
                             "dm_stat": dm["dm_stat"], "p_value": dm["p_value"]})
        print(f"        {v:<12}" + "".join(cells))
    n_sig = sum(1 for r in dm_rows if r["p_value"] < 0.10)
    print(f"\n      p < 0.10 (LP != VAR predictively) in {n_sig} of {len(dm_rows)} cells.")

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        h1_ratios = [r["ratio_lp_over_var"] for r in oos_rows if r["horizon"] == 1]
        h1_ratio_avg = (
            float(sum(h1_ratios) / len(h1_ratios)) if h1_ratios else float("nan")
        )
        validation_results = {
            "n_vars": float(data.shape[1]),
            "n_oos_cells": float(len(oos_rows)),
            "n_lp_wins": float(n_lp_wins),
            "n_dm_significant": float(n_sig),
            "h1_ratio_avg": h1_ratio_avg,
            "var_stable": 1.0 if fit.stable() else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
            "variables": list(data.columns),
        },
        "model": {
            "lags": int(cfg["model"]["lags"]),
            "horizon": int(cfg["model"]["horizon"]),
            "shock_var": cfg["model"]["shock_var"],
            "trend": cfg["model"]["trend"],
        },
        "var": {
            "stable": bool(fit.stable()),
            "aic": float(fit.fit_info.get("aic", float("nan"))),
            "bic": float(fit.fit_info.get("bic", float("nan"))),
            "t_eff": int(fit.t_eff),
        },
        "shock": {
            "mean": float(shock.mean()),
            "std": float(shock.std()),
            "n": int(len(shock)),
        },
        "comparison_table": comparisons,
        "rms_var_vs_lp_median": rms_diff,
        "forecast_evaluation": {
            "window_size": fc_cfg["window_size"],
            "horizon": fc_cfg["horizon"],
            "step": fc_cfg["step"],
            "n_origins": n_origins,
            "rmse_comparison": oos_rows,
            "diebold_mariano": dm_rows,
            "n_lp_wins": n_lp_wins,
            "n_cells": len(oos_rows),
        },
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + Local Projections at h=0..{cfg['model']['horizon']}",
        "Variables": ", ".join(data.columns),
        "Identification (VAR)": "Cholesky (recursive)",
        "Shock variable": cfg["model"]["shock_var"],
        "LP HAC bandwidth rule": "horizon_plus_one",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "VAR stable": "yes" if fit.stable() else "no",
        "OOS: LP vs VAR-iterated": f"LP wins {n_lp_wins}/{len(oos_rows)} cells "
                                    f"(DM sig in {n_sig})",
        "RMS(VAR median − LP median)": f"{rms_diff:.4f}",
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    # Slice the VAR Cholesky IRF to only the shock variable's column, so the
    # shape matches the single-shock LP IRF — both become (H+1, n_vars, 1).
    # Then add_standard_sections with the LP IRF as alt_irf produces the
    # LP-vs-VAR comparison in the robustness section.
    H_lp = lp_irf_result.median.shape[0]
    var_slice = np.zeros((H_lp, var_irf.median.shape[1], 1))
    var_slice[:, :, 0] = var_irf.median[:H_lp, :, shock_idx]
    var_lo68 = np.zeros_like(var_slice)
    var_lo68[:, :, 0] = var_irf.lower_68[:H_lp, :, shock_idx]
    var_hi68 = np.zeros_like(var_slice)
    var_hi68[:, :, 0] = var_irf.upper_68[:H_lp, :, shock_idx]
    var_lo90 = np.zeros_like(var_slice)
    var_lo90[:, :, 0] = var_irf.lower_90[:H_lp, :, shock_idx]
    var_hi90 = np.zeros_like(var_slice)
    var_hi90[:, :, 0] = var_irf.upper_90[:H_lp, :, shock_idx]

    # Construct a thin wrapper that looks like an IRFContainer for the helper
    from types import SimpleNamespace
    var_irf_sliced = SimpleNamespace(
        median=var_slice,
        lower_68=var_lo68, upper_68=var_hi68,
        lower_90=var_lo90, upper_90=var_hi90,
        horizons=lp_irf_result.horizons,
        shock_names=[f"{cfg['model']['shock_var']} shock (VAR-Cholesky)"],
        horizon=H_lp - 1,
        variable_names=list(data.columns),
    )

    report_html.add_standard_sections(
        fit=fit, irf=var_irf_sliced,
        spec=spec_dict, log=log,
        var_labels=list(data.columns),
        alt_irf=lp_irf_result,
        alt_label="Local Projections (Jordà 2005)",
        baseline_label="VAR-Cholesky",
        shock_idx_for_summary=0,
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    log_path = outputs_dir / "research_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"      -> {log_path.relative_to(PROJECT_ROOT)}")

    # Save LP IRF as the canonical numerical snapshot (the headline output)
    lp_irf_result.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
