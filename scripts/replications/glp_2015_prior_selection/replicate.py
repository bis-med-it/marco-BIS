"""Giannone-Lenza-Primiceri (2015) hyperparameter selection — spec-layer driver.

Compares:
  1. Default Minnesota prior (lambda_1=0.1, mu=delta=1)
  2. GLP-optimised hyperparameters (Nelder-Mead over the MDD)

on the CEE (1999) seven-variable monetary BVAR. Reports the MDD
improvement, the implied IRF differences, and a rolling-window OOS
comparison.

The data pipeline, both BVAR fits and both IRFs go through
``lib.Spec.from_replication`` and ``lib.run``. The optimisation
(``lib.optimize_minnesota_hyperparameters``, Nelder-Mead) and the rolling
out-of-sample comparison are done inline.

The config has ``model.initial_prior`` rather than ``model.prior`` (it
distinguishes "initial / default" from "optimised"); we lift it into
``model.prior`` via ``spec.update`` at lib.run-call time.
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402


def _prior_to_dict(prior) -> dict:
    return {
        "lambda_1": prior.lambda_1, "lambda_2": prior.lambda_2,
        "lambda_3": prior.lambda_3, "lambda_4": prior.lambda_4,
        "mu": prior.mu, "delta": prior.delta,
    }


def main() -> int:
    spec = lib.Spec.from_replication("glp_2015_prior_selection")
    seed = int(spec.metadata.get("general", {}).get("random_seed", 42))
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}")
    print("=" * 70)

    print("\n[1/5] Loading CEE data via lib.load_data...")
    data = lib.load_data(spec)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")

    lags, trend = spec.model["lags"], spec.model["trend"]
    initial_prior_dict = dict(spec.model["initial_prior"])
    irf_kwargs = {
        "method": "cholesky", "horizon": 40,
        "bands": "parameter_draws", "n_draws": 1000,
    }

    print("\n[2/5] Default Minnesota prior — fit via lib.run...")
    spec_default = spec.update(model={"prior": initial_prior_dict}, irf=irf_kwargs)
    result_default = lib.run(spec_default, seed=seed)
    bvar_init = result_default.fit
    irf_init = result_default.irf
    init_prior = bvar_init.prior   # MinnesotaPrior reconstructed by the dispatcher
    mdd_init = lib.log_marginal_data_density(bvar_init)
    print(f"      lambda_1={init_prior.lambda_1}, lambda_2={init_prior.lambda_2}, "
          f"mu={init_prior.mu}, delta={init_prior.delta}")
    print(f"      log MDD = {mdd_init:.4f}")

    print("\n[3/5] GLP 2015 optimisation...")
    glp_cfg = spec.metadata["glp"]
    print(f"      Optimising over: {glp_cfg['params_to_optimize']}")
    optimal_prior, diag = lib.optimize_minnesota_hyperparameters(
        data, lags=lags,
        params_to_optimize=glp_cfg["params_to_optimize"],
        initial=init_prior,
        method=glp_cfg["method"],
    )
    print(f"      Converged: {diag['converged']}  "
          f"(n_evaluations={diag['n_evaluations']})")
    print(f"      Optimised parameters:")
    for k, v in diag["optimized_params"].items():
        old = getattr(init_prior, k)
        print(f"        {k}: {old:.4f}  ->  {v:.4f}")
    print(f"      log MDD: {diag['initial_mdd']:.4f}  ->  {diag['best_mdd']:.4f}"
          f"  (change = {diag['improvement']:+.4f})")
    print(f"      Bayes factor vs initial: exp({diag['improvement']:.2f}) "
          f"~ {np.exp(min(diag['improvement'], 700)):.3e}")

    print("\n[4/5] Optimised BVAR — fit via lib.run, compare IRFs...")
    spec_opt = spec.update(model={"prior": _prior_to_dict(optimal_prior)},
                           irf=irf_kwargs)
    result_opt = lib.run(spec_opt, seed=seed)
    bvar_opt = result_opt.fit
    irf_opt = result_opt.irf
    ffr_idx = bvar_init.variable_names.index("ffr")
    print(f"      {'h':>3}  {'rgdp (init)':>14}  {'rgdp (GLP-opt)':>17}  "
          f"{'deflator (init)':>17}  {'deflator (GLP-opt)':>19}")
    for h in [0, 4, 8, 12, 20, 30, 40]:
        rgdp_init = irf_init.median[h, 0, ffr_idx]
        rgdp_opt = irf_opt.median[h, 0, ffr_idx]
        defl_init = irf_init.median[h, 1, ffr_idx]
        defl_opt = irf_opt.median[h, 1, ffr_idx]
        print(f"      {h:>3}  {rgdp_init:>+14.5f}  {rgdp_opt:>+17.5f}  "
              f"{defl_init:>+17.5f}  {defl_opt:>+19.5f}")

    # ---- [5/5] Rolling-window OOS comparison (replication-specific) ----
    print("\n[5/5] Rolling-window OOS forecast evaluation "
          "(GLP-optimal vs default prior)...")
    fc_cfg = spec.forecast_evaluation

    # Hyperparameters for the OOS exercise are tuned on the INITIAL training
    # window only, then frozen. The full-sample optimum above serves the
    # full-sample IRF exhibit; reusing it here would let the evaluation
    # observations inform the hyperparameters at every origin (look-ahead).
    train = data.iloc[:fc_cfg["window_size"]]
    optimal_prior_train, diag_train = lib.optimize_minnesota_hyperparameters(
        train, lags=lags,
        params_to_optimize=glp_cfg["params_to_optimize"],
        initial=init_prior,
        method=glp_cfg["method"],
    )
    print(f"      OOS tuning on the first {fc_cfg['window_size']} quarters "
          f"only (no look-ahead); full-sample optimum shown above is for "
          f"the in-sample IRF exhibit.")
    for k, v in diag_train["optimized_params"].items():
        full_v = diag["optimized_params"][k]
        print(f"        {k}: training-window {v:.4f}  "
              f"(full-sample {full_v:.4f})")

    def fit_default(df):
        return lib.fit_minnesota_bvar(df, lags=lags, prior=init_prior, trend=trend)

    def fit_glp(df):
        return lib.fit_minnesota_bvar(df, lags=lags, prior=optimal_prior_train,
                                      trend=trend)

    print(f"      Rolling window: {fc_cfg['window_size']} quarters, "
          f"step={fc_cfg['step']}, horizon=1..{fc_cfg['horizon']}")
    rolling_default = lib.rolling_forecasts(
        data, fit_default,
        window_size=fc_cfg["window_size"], horizon=fc_cfg["horizon"],
        step=fc_cfg["step"],
    )
    rolling_glp = lib.rolling_forecasts(
        data, fit_glp,
        window_size=fc_cfg["window_size"], horizon=fc_cfg["horizon"],
        step=fc_cfg["step"],
    )
    n_origins = int(rolling_default["origin"].nunique())
    print(f"      Generated forecasts at {n_origins} origins.")

    # Matched scoring: both priors evaluated on the SAME valid
    # (origin, horizon, variable) cells, so a failed fit in one model
    # cannot shift the comparison onto different samples.
    _scores, _match = lib.forecast_score_matched(
        {"glp": rolling_glp, "default": rolling_default},
        metric="rmse", by=["horizon", "variable"],
    )
    rmse_d, rmse_g = _scores["default"], _scores["glp"]
    print(f"      Matched comparison cells: {_match['n_common']}")
    rmse_cmp = rmse_g.merge(rmse_d, on=["horizon", "variable"], suffixes=("_glp", "_default"))
    rmse_cmp["ratio_glp_over_default"] = rmse_cmp["rmse_glp"] / rmse_cmp["rmse_default"]

    horizons = fc_cfg["horizons_to_report"]
    print(f"\n      RMSE ratio (GLP-optimal / default), values < 1 = GLP wins:")
    print(f"        {'variable':<14}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    headline_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            row = rmse_cmp[(rmse_cmp["horizon"] == h) & (rmse_cmp["variable"] == v)]
            r = float(row["ratio_glp_over_default"].iloc[0])
            cells.append(f"{r:>10.3f}")
            headline_rows.append({
                "variable": v, "horizon": int(h),
                "rmse_default": float(row["rmse_default"].iloc[0]),
                "rmse_glp": float(row["rmse_glp"].iloc[0]),
                "ratio_glp_over_default": r,
            })
        print(f"        {v:<14}" + "".join(cells))
    n_wins = sum(1 for r in headline_rows if r["ratio_glp_over_default"] < 1.0)
    print(f"\n      GLP wins (ratio < 1) in {n_wins} of {len(headline_rows)} cells.")

    print(f"\n      Diebold-Mariano p-values (squared-error losses, GLP vs default):")
    print(f"        {'variable':<14}" + "".join(f"{'h=' + str(h):>10}" for h in horizons))
    dm_rows = []
    for v in data.columns:
        cells = []
        for h in horizons:
            g_h = rolling_glp[(rolling_glp["horizon"] == h)
                               & (rolling_glp["variable"] == v)
                               & rolling_glp["realised"].notna()]
            d_h = rolling_default[(rolling_default["horizon"] == h)
                                    & (rolling_default["variable"] == v)
                                    & rolling_default["realised"].notna()]
            merged = g_h.merge(d_h, on=["origin", "variable"],
                                suffixes=("_glp", "_default"))
            err_g = (merged["forecast_glp"] - merged["realised_glp"]).values
            err_d = (merged["forecast_default"] - merged["realised_default"]).values
            if len(err_g) < 5:
                cells.append("       n/a")
                continue
            dm = lib.diebold_mariano(err_g ** 2, err_d ** 2, horizon=h)
            cells.append(f"{dm['p_value']:>10.3f}")
            dm_rows.append({"variable": v, "horizon": int(h),
                             "dm_stat": dm["dm_stat"], "p_value": dm["p_value"]})
        print(f"        {v:<14}" + "".join(cells))
    n_sig = sum(1 for r in dm_rows if r["p_value"] < 0.10)
    print(f"\n      p < 0.10 (GLP-optimal != default predictively) in {n_sig} "
          f"of {len(dm_rows)} cells.")

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log = {
        "replication_id": rep["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": spec.data["source"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
        },
        "initial_prior": _prior_to_dict(init_prior),
        "initial_mdd": float(mdd_init),
        "glp_optimisation": diag,
        "optimal_prior": _prior_to_dict(optimal_prior),
        "forecast_evaluation": {
            "window_size": fc_cfg["window_size"],
            "horizon": fc_cfg["horizon"],
            "step": fc_cfg["step"],
            "n_origins": n_origins,
            "rmse_comparison": headline_rows,
            "diebold_mariano": dm_rows,
            "n_glp_wins": n_wins,
            "n_cells": len(headline_rows),
        },
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(data.shape[1]),
            "glp_improves_mdd":
                1.0 if diag["best_mdd"] > diag["initial_mdd"] else 0.0,
            "log_mdd_change": float(diag["improvement"]),
            "optimal_lambda_1": float(optimal_prior.lambda_1),
            "n_oos_cells": float(len(headline_rows)),
            "n_glp_wins": float(n_wins),
            "n_dm_significant": float(n_sig),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (full eight-section structure) ----
    spec_dict = {
        "Model": f"BVAR({bvar_opt.p}) with GLP-optimised Minnesota prior",
        "Variables": ", ".join(bvar_opt.variable_names),
        "Identification": "Cholesky (recursive)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(bvar_opt.p),
        "Prior": "Minnesota, GLP (2015) marginal-likelihood maximised",
        "Initial lambda_1": f"{init_prior.lambda_1:.4f}",
        "Optimal lambda_1": f"{optimal_prior.lambda_1:.4f}",
        "Initial log MDD": f"{diag['initial_mdd']:.4f}",
        "Optimal log MDD": f"{diag['best_mdd']:.4f}",
        "log Bayes factor": f"{diag['improvement']:+.4f}",
        "OOS: GLP vs default": f"GLP wins {n_wins}/{len(headline_rows)} cells "
                                f"(DM sig in {n_sig})",
    }
    report_html = lib.VARReport(
        title=rep.get("title", "GLP 2015 prior selection replication"),
        subtitle=rep.get("subtitle",
            "Marginal-likelihood-based hyperparameter selection for Minnesota BVAR"),
    )
    report_html.add_standard_sections(
        fit=bvar_opt, irf=irf_opt, spec=spec_dict, log=log,
        shock_idx_for_summary=ffr_idx,
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"\n      ->{(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      ->{(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")
    irf_opt.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      ->{(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
