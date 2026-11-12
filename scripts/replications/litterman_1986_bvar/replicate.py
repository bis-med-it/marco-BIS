"""Litterman (1986) — Bayesian VAR with the Minnesota prior.

Thin spec-layer driver. The fit, IRF, and data pipeline run through
``lib.Spec.from_replication("litterman_1986_bvar")`` + ``lib.run``; this
script only adds the replication-specific bits (OLS baseline, shrinkage
diagnostic, rolling-window OOS comparison, DM test, output assembly).

What this replication does:

* Fits a seven-variable quarterly BVAR with the asymmetric Litterman
  prior at the paper's hyperparameters (``lambda_1=0.2, lambda_2=0.2,
  lambda_3=1``, own-first-lag prior mean ``= 1`` for all variables).
* Reports a shrinkage diagnostic (own-lag-1 OLS vs BVAR vs prior).
* Computes an illustrative Cholesky response to a commercial-paper-rate
  innovation.
* Reproduces the paper's actual headline: rolling-window OOS forecast
  RMSE — BVAR vs unrestricted VAR(6), with Diebold-Mariano significance
  tests.
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> int:
    spec = lib.Spec.from_replication("litterman_1986_bvar")
    seed = int(spec.metadata.get("general", {}).get("random_seed", 42))
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    # ---- [1/4] Spec-layer run: data + BVAR fit + IRF ----
    print("\n[1/4] Fitting Litterman BVAR via lib.run...")
    result = lib.run(spec, seed=seed)
    bvar = result.fit
    data = result.data
    irf = result.irf
    p = spec.model["prior"]

    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Variables (Cholesky order): {list(data.columns)}")
    print(f"      Prior: lambda_1={p['lambda_1']}, lambda_2={p['lambda_2']}, "
          f"lambda_3={p['lambda_3']}, mu={p['mu']}, delta={p['delta']}, "
          f"asymmetric={p.get('asymmetric', False)}")
    print(f"      Random-walk priors (delta_i): {p['rw_deltas']}  "
          f"(0 = stationary I(0); 1 = non-stationary)")
    eig_bvar = lib.max_eigenvalue_modulus(bvar.lag_matrices())
    if p.get("asymmetric", False):
        print(f"      log MDD: N/A (asymmetric Litterman prior; "
              f"presample sigmas={[round(s, 4) for s in bvar.sigmas]})")
        mdd_val = None
    else:
        mdd_val = float(lib.log_marginal_data_density(bvar))
        print(f"      log MDD: {mdd_val:.4f}")
    print(f"      Max |eigenvalue|: {eig_bvar:.4f}  "
          f"(stable={bvar.stable()}; near-unity expected with delta_i=1)")

    # ---- [2/4] OLS baseline via spec.update(model.class=var) ----
    print("\n[2/4] OLS baseline VAR (lib.run with model.class='var')...")
    ols_result = lib.run(spec.update(model={"class": "var"}, irf=None), seed=seed)
    ols = ols_result.fit
    eig_ols = lib.max_eigenvalue_modulus(ols.lag_matrices())
    print(f"      OLS VAR: AIC={ols.fit_info['aic']:.2f}, "
          f"BIC={ols.fit_info['bic']:.2f}, max|eig|={eig_ols:.4f}, "
          f"stable={ols.stable()}")

    # Shrinkage diagnostic: BVAR own-lag-1 vs OLS vs prior
    A1_ols = ols.lag_matrices()[0]
    A1_bvar = bvar.lag_matrices()[0]
    print(f"\n      Shrinkage diagnostic - own-lag-1 coefficients:")
    print(f"        {'variable':<12}{'OLS':>9}{'BVAR mean':>13}"
          f"{'prior mean':>14}{'shrink %':>11}")
    shrink_rows = []
    for j, var in enumerate(bvar.variable_names):
        ols_own = float(A1_ols[j, j])
        bvar_own = float(A1_bvar[j, j])
        prior_own = float(p["rw_deltas"][j])
        if abs(ols_own - prior_own) > 0.1:
            toward = 100.0 * (ols_own - bvar_own) / (ols_own - prior_own)
            cell = f"{toward:>10.1f}%"
        else:
            cell = "       N/A"
        print(f"        {var:<12}{ols_own:>9.4f}{bvar_own:>13.4f}"
              f"{prior_own:>14.4f}{cell}")
        shrink_rows.append({"variable": var, "ols": ols_own,
                             "bvar": bvar_own, "prior": prior_own})

    # ---- [3/4] IRF table ----
    shock = spec.irf["shock_var"]
    print(f"\n[3/4] Cholesky IRFs to a one-sigma shock in '{shock}':")
    shock_idx = bvar.variable_names.index(shock)
    for var in bvar.variable_names:
        v_idx = bvar.variable_names.index(var)
        print(f"\n    Response of '{var}':")
        print(f"      {'h':>3}  {'median':>10}  {'68% CI':>22}")
        for h in (0, 1, 4, 8, 12, 20):
            if h > spec.irf["horizon"]:
                continue
            m, lo, hi = (irf.median[h, v_idx, shock_idx],
                          irf.lower_68[h, v_idx, shock_idx],
                          irf.upper_68[h, v_idx, shock_idx])
            print(f"      {h:>3}  {m:>+10.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]")

    # ---- [4/4] Rolling-window OOS evaluation ----
    print("\n[4/4] Rolling-window OOS evaluation (BVAR vs unrestricted VAR)...")
    fc = spec.forecast_evaluation
    lags, trend = spec.model["lags"], spec.model["trend"]
    print(f"      Rolling window: {fc['window_size']} quarters, "
          f"step={fc['step']}, horizon=1..{fc['horizon']}")
    rolling_bvar = lib.rolling_forecasts(
        data, lambda df: lib.fit_minnesota_bvar(df, lags=lags, prior=bvar.prior, trend=trend),
        window_size=fc["window_size"], horizon=fc["horizon"], step=fc["step"],
    )
    rolling_ols = lib.rolling_forecasts(
        data, lambda df: lib.fit_var(df, lags=lags, trend=trend),
        window_size=fc["window_size"], horizon=fc["horizon"], step=fc["step"],
    )
    n_origins = int(rolling_bvar["origin"].nunique())
    print(f"      Generated forecasts at {n_origins} origins.")

    # Matched scoring: both models evaluated on the SAME valid
    # (origin, horizon, variable) cells, so a failed fit in one model
    # cannot shift the comparison onto different samples.
    _scores, _match = lib.forecast_score_matched(
        {"bvar": rolling_bvar, "ols": rolling_ols},
        metric="rmse", by=["horizon", "variable"],
    )
    rmse_b, rmse_o = _scores["bvar"], _scores["ols"]
    print(f"      Matched comparison cells: {_match['n_common']}")
    rmse_cmp = rmse_b.merge(rmse_o, on=["horizon", "variable"], suffixes=("_bvar", "_ols"))
    rmse_cmp["ratio_bvar_over_ols"] = rmse_cmp["rmse_bvar"] / rmse_cmp["rmse_ols"]

    horizons = fc["horizons_to_report"]
    print(f"\n      RMSE ratio (BVAR / unrestricted VAR), values < 1 = BVAR wins:")
    print(f"        {'variable':<10}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    headline_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            row = rmse_cmp[(rmse_cmp["horizon"] == h) & (rmse_cmp["variable"] == v)]
            r = float(row["ratio_bvar_over_ols"].iloc[0])
            cells.append(f"{r:>10.3f}")
            headline_rows.append({
                "variable": v, "horizon": int(h),
                "rmse_bvar": float(row["rmse_bvar"].iloc[0]),
                "rmse_ols": float(row["rmse_ols"].iloc[0]),
                "ratio_bvar_over_ols": r,
            })
        print(f"        {v:<10}" + "".join(cells))
    n_wins = sum(1 for r in headline_rows if r["ratio_bvar_over_ols"] < 1.0)
    print(f"\n      BVAR wins (ratio < 1) in {n_wins} of {len(headline_rows)} cells.")

    print(f"\n      Diebold-Mariano p-values (squared-error losses):")
    print(f"        {'variable':<10}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    dm_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            bvar_h = rolling_bvar[(rolling_bvar["horizon"] == h)
                                   & (rolling_bvar["variable"] == v)
                                   & rolling_bvar["realised"].notna()]
            ols_h = rolling_ols[(rolling_ols["horizon"] == h)
                                 & (rolling_ols["variable"] == v)
                                 & rolling_ols["realised"].notna()]
            merged = bvar_h.merge(ols_h, on=["origin", "variable"],
                                   suffixes=("_bvar", "_ols"))
            err_b = (merged["forecast_bvar"] - merged["realised_bvar"]).values
            err_o = (merged["forecast_ols"] - merged["realised_ols"]).values
            if len(err_b) < 5:
                cells.append("       n/a")
                continue
            dm = lib.diebold_mariano(err_b ** 2, err_o ** 2, horizon=h)
            cells.append(f"{dm['p_value']:>10.3f}")
            dm_rows.append({"variable": v, "horizon": int(h),
                             "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                             "d_bar": dm["d_bar"]})
        print(f"        {v:<10}" + "".join(cells))
    n_sig = sum(1 for r in dm_rows if r["p_value"] < 0.10)
    print(f"\n      p < 0.10 (BVAR != VAR predictively) in {n_sig} of {len(dm_rows)} cells.")

    # ---- Paper-target validation ----
    targets_path = HERE / "paper_targets.yaml"
    if targets_path.exists():
        # Build the results dict matching keys in paper_targets.yaml
        def _ratio(var: str, h: int) -> float | None:
            for r in headline_rows:
                if r["variable"] == var and int(r["horizon"]) == h:
                    return float(r["ratio_bvar_over_ols"])
            return None
        validation_results = {
            "n_bvar_wins": float(n_wins),
            "n_dm_significant": float(n_sig),
            "max_eigenvalue_modulus": float(eig_bvar),
            "rmse_ratio_rgnp_h4": _ratio("rgnp", 4),
            "rmse_ratio_cprate_h4": _ratio("cprate", 4),
            "rmse_ratio_invest_h4": _ratio("invest", 4),
            "rmse_ratio_urate_h8": _ratio("urate", 8),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
    else:
        comp_report = None

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log = {
        **result.to_log_dict(),
        "ols_baseline": {
            "aic": float(ols.fit_info["aic"]),
            "bic": float(ols.fit_info["bic"]),
            "max_eigenvalue_modulus": float(eig_ols),
            "stable": bool(ols.stable()),
        },
        "shrinkage_diagnostic": shrink_rows,
        "log_mdd": mdd_val,
        "max_eigenvalue_modulus": float(eig_bvar),
        "forecast_evaluation": {
            "window_size": fc["window_size"], "horizon": fc["horizon"],
            "step": fc["step"], "n_origins": n_origins,
            "rmse_comparison": headline_rows, "diebold_mariano": dm_rows,
            "n_bvar_wins": n_wins, "n_cells": len(headline_rows),
        },
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (full eight-section structure) ----
    spec_dict = {
        "Model": f"BVAR({bvar.p}) with Minnesota prior",
        "Variables": ", ".join(bvar.variable_names),
        "Identification": "Cholesky (recursive)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(bvar.p),
        "Prior": "Minnesota (Litterman, per-equation)",
        "Lambda_1 (overall tightness)": f"{p['lambda_1']:.4f}",
        "Lambda_2 (cross-variable)": f"{p.get('lambda_2', 0.5):.4f}",
        "Sum-of-coefficients (mu)": f"{p.get('mu', 1.0):.4f}",
        "Dummy-initial-obs (delta)": f"{p.get('delta', 1.0):.4f}",
        "Max |eigenvalue|": f"{float(eig_bvar):.4f}",
        "OOS: BVAR vs OLS": f"BVAR wins {n_wins}/{len(headline_rows)} cells",
    }
    report_html = lib.VARReport(
        title=spec.metadata.get("title", "Litterman 1986 BVAR replication"),
        subtitle=spec.metadata.get("subtitle",
            "BVAR with Minnesota prior; Cholesky IRFs + rolling-window OOS"),
    )
    report_html.add_standard_sections(
        fit=bvar, irf=irf, spec=spec_dict, log=log,
        shock_idx_for_summary=shock_idx,
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2, default=str))
    irf.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
