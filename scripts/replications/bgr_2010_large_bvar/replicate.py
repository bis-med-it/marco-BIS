"""Banbura-Giannone-Reichlin (2010) Large BVARs — spec-layer driver.

Demonstrates the BGR scaling rule: as the number of variables grows, the
Minnesota overall-tightness lambda_1 must SHRINK to keep the large BVAR's
in-sample fit on a fixed benchmark set roughly equal to the small VAR's.

Architecture note: BGR is a *multi-config exploration* (3 system sizes ×
default + calibrated priors + rolling-window OOS for 3 fitters), not a
single-canonical-run replication. The spec layer (``lib.run``) is designed
for the latter, so this replicate.py uses the spec only for the data step
via :func:`lib.load_data` — the rest (the brentq calibration, the multi-
system loop, the OOS comparison) stays inline since it doesn't fit the
"one model, one run" abstraction.

Pipeline:
  1. ``lib.Spec.from_replication("bgr_2010_large_bvar")`` + ``lib.load_data``
     for the FRED-MD load + transform.
  2. Fit OLS VAR on the SMALL (3-var) benchmark system; record per-variable
     in-sample MSE.
  3. For each larger system (medium=7, large=17):
       (a) Default-lambda_1 BVAR — record benchmark MSE.
       (b) BGR-calibrated lambda_1 via 1-D ``scipy.optimize.brentq`` such
           that benchmark-variable MSE matches the small-VAR baseline.
  4. Report the lambda_1* sequence and fit ratios.
  5. Rolling-window OOS evaluation: small VAR vs large BVAR (default) vs
     large BVAR (BGR-calibrated).

The headline: lambda_1* shrinks monotonically with system size. Without
calibration, the 17-variable BVAR overfits and forecasts much worse than
the small one; with calibration, it matches.
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
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402


def benchmark_mse(fit, benchmark_vars: list[str]) -> dict[str, float]:
    """Per-variable in-sample MSE for the benchmark variables.

    Works for both VARFit (OLS residuals) and BVARFit (residuals at the
    posterior mean of B).
    """
    resid = fit.residuals
    out = {}
    for v in benchmark_vars:
        i = fit.variable_names.index(v)
        out[v] = float(np.mean(resid[:, i] ** 2))
    return out


def avg_benchmark_log_mse(fit, benchmark_vars: list[str]) -> float:
    mses = benchmark_mse(fit, benchmark_vars)
    return float(np.mean(np.log(list(mses.values()))))


def calibrate_lambda_1(
    data: pd.DataFrame,
    benchmark_vars: list[str],
    target_log_mse: float,
    lags: int,
    trend: str,
    default_prior: dict,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """Find lambda_1 such that log-average benchmark MSE matches target.

    Returns (lambda_1_star, gap_at_optimum).
    """
    def gap(lam1: float) -> float:
        prior = lib.MinnesotaPrior(
            lambda_1=lam1,
            lambda_2=default_prior["lambda_2"],
            lambda_3=default_prior["lambda_3"],
            lambda_4=default_prior["lambda_4"],
            mu=default_prior["mu"],
            delta=default_prior["delta"],
        )
        fit = lib.fit_minnesota_bvar(data, lags=lags, prior=prior, trend=trend)
        return avg_benchmark_log_mse(fit, benchmark_vars) - target_log_mse

    lo, hi = bounds
    g_lo = gap(lo)
    g_hi = gap(hi)
    if g_lo * g_hi > 0:
        # Same sign at both endpoints; pick the closer one and report
        chosen = lo if abs(g_lo) < abs(g_hi) else hi
        return float(chosen), float(min(abs(g_lo), abs(g_hi)))
    lam_star = brentq(gap, lo, hi, xtol=1e-5)
    return float(lam_star), float(gap(lam_star))


def main() -> int:
    spec = lib.Spec.from_replication("bgr_2010_large_bvar")
    seed = int(spec.metadata.get("general", {}).get("random_seed", 42))
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/5] Loading FRED-MD via lib.load_data...")
    data = lib.load_data(spec)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} columns)")

    bench = spec.data["benchmark_vars"]
    p = spec.model["lags"]
    trend = spec.model["trend"]
    default_prior = spec.model["default_prior"]
    bounds = tuple(spec.model["calibration"]["lambda_1_search_bounds"])

    # --- Small system: OLS VAR baseline ---
    print(f"\n[2/5] Small benchmark VAR(p={p}, OLS) on {bench}...")
    small_data = data[bench]
    small_var = lib.fit_var(small_data, lags=p, trend=trend)
    small_mse = benchmark_mse(small_var, bench)
    target_log_mse = float(np.mean(np.log(list(small_mse.values()))))
    print(f"      Benchmark MSEs: " +
          ", ".join(f"{v}={small_mse[v]:.5f}" for v in bench))
    print(f"      log-average benchmark MSE (TARGET for large systems): "
          f"{target_log_mse:.4f}")

    # --- Larger systems: default vs calibrated lambda_1 ---
    print(f"\n[3/5] BVAR runs at each system size:")
    results = []
    for sys_name in ["small", "medium", "large"]:
        sys_cfg = spec.data["systems"][sys_name]
        sub = data[sys_cfg["vars"]]
        n_vars = sub.shape[1]
        print(f"\n  {sys_cfg['label']}")

        # Default-lambda_1 BVAR
        prior_default = lib.MinnesotaPrior(
            lambda_1=default_prior["lambda_1"],
            lambda_2=default_prior["lambda_2"],
            lambda_3=default_prior["lambda_3"],
            lambda_4=default_prior["lambda_4"],
            mu=default_prior["mu"],
            delta=default_prior["delta"],
        )
        bvar_default = lib.fit_minnesota_bvar(
            sub, lags=p, prior=prior_default, trend=trend
        )
        default_mse = benchmark_mse(bvar_default, bench)
        default_log_mse = float(np.mean(np.log(list(default_mse.values()))))
        print(f"    default lambda_1={default_prior['lambda_1']:.3f}: "
              f"log avg benchmark MSE = {default_log_mse:+.4f}  "
              f"(ratio to small VAR: {np.exp(default_log_mse - target_log_mse):.3f})")

        # BGR-calibrated lambda_1 (only meaningful for n_vars > 3)
        if n_vars > len(bench):
            lam_star, gap = calibrate_lambda_1(
                data=sub,
                benchmark_vars=bench,
                target_log_mse=target_log_mse,
                lags=p,
                trend=trend,
                default_prior=default_prior,
                bounds=bounds,
            )
            prior_cal = lib.MinnesotaPrior(
                lambda_1=lam_star,
                lambda_2=default_prior["lambda_2"],
                lambda_3=default_prior["lambda_3"],
                lambda_4=default_prior["lambda_4"],
                mu=default_prior["mu"],
                delta=default_prior["delta"],
            )
            bvar_cal = lib.fit_minnesota_bvar(
                sub, lags=p, prior=prior_cal, trend=trend
            )
            cal_mse = benchmark_mse(bvar_cal, bench)
            cal_log_mse = float(np.mean(np.log(list(cal_mse.values()))))
            print(f"    BGR-calibrated lambda_1*={lam_star:.4f}: "
                  f"log avg benchmark MSE = {cal_log_mse:+.4f}  "
                  f"(ratio to small VAR: {np.exp(cal_log_mse - target_log_mse):.3f}; "
                  f"residual gap: {gap:+.2e})")
        else:
            lam_star = default_prior["lambda_1"]
            cal_log_mse = default_log_mse
            cal_mse = default_mse
            print(f"    (no calibration: n_vars == |benchmark|, "
                  f"so the system IS the small VAR)")

        results.append({
            "system": sys_name,
            "label": sys_cfg["label"],
            "n_vars": int(n_vars),
            "default_lambda_1": default_prior["lambda_1"],
            "default_log_mse": default_log_mse,
            "default_mse_ratio": float(np.exp(default_log_mse - target_log_mse)),
            "calibrated_lambda_1": float(lam_star),
            "calibrated_log_mse": cal_log_mse,
            "calibrated_mse_ratio": float(np.exp(cal_log_mse - target_log_mse)),
            "benchmark_mse_calibrated": cal_mse,
        })

    # --- Headline: lambda_1* scaling ---
    print(f"\n[4/5] BGR scaling rule — lambda_1* by system size:")
    print(f"      {'system':<10}{'n_vars':>8}{'default lam1':>14}"
          f"{'default ratio':>15}{'calibrated lam1':>17}{'calibrated ratio':>18}")
    for r in results:
        print(f"      {r['system']:<10}{r['n_vars']:>8}"
              f"{r['default_lambda_1']:>14.4f}"
              f"{r['default_mse_ratio']:>15.3f}"
              f"{r['calibrated_lambda_1']:>17.4f}"
              f"{r['calibrated_mse_ratio']:>18.3f}")
    print(f"\n      The calibrated lambda_1 should DECREASE with system size.")
    print(f"      A smaller default in-sample MSE ratio can indicate tighter "
          f"in-sample fit; compare out-of-sample accuracy separately.")
    print(f"      The calibrated ratio should hover near 1.0.")

    # --- BGR's actual headline: OOS forecast comparison ---
    print(f"\n[5/5] Out-of-sample forecast evaluation (the OOS headline)...")
    fc_cfg = spec.forecast_evaluation
    fc_window = int(fc_cfg["window_size"])
    fc_horizon = int(fc_cfg["horizon"])
    fc_step = int(fc_cfg["step"])
    horizons_show = fc_cfg["horizons_to_report"]

    # OOS calibration uses ONLY the initial training window. Calibrating on
    # the full sample and reusing the result at earlier forecast origins
    # would leak future information into the "out-of-sample" comparison —
    # the calibrated-vs-default contest must be decided with information
    # available at the first origin. (The full-sample lambda_1* table in
    # [3/5] above is the paper's in-sample calibration exercise and is
    # reported as such.)
    large_vars_pre = spec.data["systems"]["large"]["vars"]
    train = data.iloc[:fc_window]
    small_var_train = lib.fit_var(train[bench], lags=p, trend=trend)
    train_target = float(
        np.mean(np.log(list(benchmark_mse(small_var_train, bench).values())))
    )
    lam_oos, gap_oos = calibrate_lambda_1(
        data=train[large_vars_pre],
        benchmark_vars=bench,
        target_log_mse=train_target,
        lags=p,
        trend=trend,
        default_prior=default_prior,
        bounds=bounds,
    )
    lambda_1_cal = float(lam_oos)
    print(f"      lambda_1* calibrated on the initial {fc_window}-month "
          f"training window only: {lambda_1_cal:.4f} "
          f"(full-sample value, NOT used here: "
          f"{next(r for r in results if r['system'] == 'large')['calibrated_lambda_1']:.4f})")

    # Three setups: small VAR, large BVAR (default), large BVAR (calibrated)
    large_vars = spec.data["systems"]["large"]["vars"]

    def fit_small_var(df):
        return lib.fit_var(df[bench], lags=p, trend=trend)

    def make_large_bvar_fitter(lam1):
        def _f(df):
            prior_local = lib.MinnesotaPrior(
                lambda_1=lam1,
                lambda_2=default_prior["lambda_2"],
                lambda_3=default_prior["lambda_3"],
                lambda_4=default_prior["lambda_4"],
                mu=default_prior["mu"], delta=default_prior["delta"],
            )
            return lib.fit_minnesota_bvar(
                df[large_vars], lags=p, prior=prior_local, trend=trend,
            )
        return _f

    print(f"      Rolling window: {fc_window} months, step={fc_step}, "
          f"horizon=1..{fc_horizon}")
    print(f"      Setup 1: small VAR(3) on {bench}")
    rolling_small = lib.rolling_forecasts(
        data[bench], fit_small_var,
        window_size=fc_window, horizon=fc_horizon, step=fc_step,
    )
    print(f"      Setup 2: large BVAR(17) at default lambda_1={default_prior['lambda_1']}")
    rolling_large_def = lib.rolling_forecasts(
        data[large_vars], make_large_bvar_fitter(default_prior["lambda_1"]),
        window_size=fc_window, horizon=fc_horizon, step=fc_step,
    )
    print(f"      Setup 3: large BVAR(17) at BGR-calibrated lambda_1*={lambda_1_cal:.4f}")
    rolling_large_cal = lib.rolling_forecasts(
        data[large_vars], make_large_bvar_fitter(lambda_1_cal),
        window_size=fc_window, horizon=fc_horizon, step=fc_step,
    )

    # Score on benchmark variables only — MATCHED across the three setups
    # (same valid origin/horizon/variable cells), so a failed fit in one
    # setup cannot shift the comparison onto different samples.
    _scores, _match = lib.forecast_score_matched(
        {
            "small": rolling_small[rolling_small["variable"].isin(bench)],
            "default": rolling_large_def[rolling_large_def["variable"].isin(bench)],
            "cal": rolling_large_cal[rolling_large_cal["variable"].isin(bench)],
        },
        metric="rmse", by=["horizon", "variable"],
    )
    rmse_s, rmse_d, rmse_c = _scores["small"], _scores["default"], _scores["cal"]
    print(f"      Matched comparison cells: {_match['n_common']}")

    # Print headline table: BVAR-default RMSE / small-VAR RMSE
    # vs BVAR-calibrated RMSE / small-VAR RMSE
    print(f"\n      RMSE ratio vs small VAR baseline (lower = better):")
    print(f"        {'variable':<10}{'h':>4}  "
          f"{'small VAR':>12}{'large BVAR default':>22}{'large BVAR cal*':>20}")
    forecast_rows = []
    for v in bench:
        for h in horizons_show:
            r_s = float(rmse_s[(rmse_s["horizon"] == h) & (rmse_s["variable"] == v)]["rmse"].iloc[0])
            r_d = float(rmse_d[(rmse_d["horizon"] == h) & (rmse_d["variable"] == v)]["rmse"].iloc[0])
            r_c = float(rmse_c[(rmse_c["horizon"] == h) & (rmse_c["variable"] == v)]["rmse"].iloc[0])
            print(f"        {v:<10}{h:>4}  "
                  f"{r_s:>12.5f}"
                  f"{r_d:>16.5f} ({r_d / r_s:.2f}x)"
                  f"{r_c:>14.5f} ({r_c / r_s:.2f}x)")
            forecast_rows.append({
                "variable": v, "horizon": h,
                "rmse_small_var": r_s,
                "rmse_large_default": r_d, "ratio_default": r_d / r_s,
                "rmse_large_calibrated": r_c, "ratio_calibrated": r_c / r_s,
            })

    avg_ratio_def = float(np.mean([r["ratio_default"] for r in forecast_rows]))
    avg_ratio_cal = float(np.mean([r["ratio_calibrated"] for r in forecast_rows]))
    print(f"\n      Avg RMSE ratio vs small VAR: "
          f"default-large={avg_ratio_def:.2f}, calibrated-large={avg_ratio_cal:.2f}")
    print(f"      BGR's claim: calibrated large should be competitive (~1.0);")
    print(f"      default large should be much worse (>>1.0).")

    # --- Paper-target validation ---
    targets_path = HERE / "paper_targets.yaml"
    if targets_path.exists():
        lam_by_n = {r["n_vars"]: float(r["calibrated_lambda_1"]) for r in results}
        lam_7 = lam_by_n.get(7)
        lam_17 = lam_by_n.get(17)
        validation_results = {
            "lambda_1_star_7var": lam_7,
            "lambda_1_star_17var": lam_17,
            "avg_rmse_ratio_calibrated_over_small": avg_ratio_cal,
            "avg_rmse_ratio_default_over_small": avg_ratio_def,
            "lambda_1_star_decrease_7_to_17": (
                lam_7 - lam_17 if (lam_7 is not None and lam_17 is not None) else None
            ),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        print("\n" + comp_report.table())
    else:
        comp_report = None

    # --- Save outputs ---
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
            "n_columns_loaded": int(data.shape[1]),
            "benchmark_vars": bench,
        },
        "small_var_baseline": {
            "vars": bench,
            "benchmark_mse": small_mse,
            "target_log_mse": target_log_mse,
        },
        "systems": results,
        "model": {
            "lags": p, "trend": trend,
            "default_prior": default_prior,
            "calibration": spec.model["calibration"],
        },
        "forecast_evaluation": {
            "window_size": fc_window,
            "horizon": fc_horizon,
            "step": fc_step,
            "lambda_1_calibrated": lambda_1_cal,
            "rows": forecast_rows,
            "avg_ratio_default_over_small": avg_ratio_def,
            "avg_ratio_calibrated_over_small": avg_ratio_cal,
        },
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — no IRFs computed in this replication) ----
    # BGR is a hyperparameter-calibration replication: the headline is lambda_1*
    # scaling with system size (Bańbura-Giannone-Reichlin 2010 Table 1). No
    # impulse-response analysis. The HTML report carries the specification +
    # research log; the system-size comparison and OOS RMSE numbers live in
    # the research log (rendered as collapsible JSON).
    spec_dict = {
        "Model": "BVAR with Minnesota prior, multiple system sizes",
        "Lag order": str(p),
        "Trend": str(trend),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "System sizes tested": ", ".join(str(r["n_vars"]) for r in results),
        "Benchmark variables": ", ".join(bench),
        "OOS window size": str(fc_window),
        "OOS horizon": f"1..{fc_horizon}",
        "lambda_1* by n_vars": " | ".join(
            f"{r['n_vars']}-var: {r['calibrated_lambda_1']:.4f}" for r in results
        ),
        "Calibration target": "Match small-VAR avg log MSE",
    }
    report_html = lib.VARReport(
        title=rep.get("title", "Bańbura-Giannone-Reichlin (2010) large BVAR"),
        subtitle=rep.get("subtitle",
            "Hyperparameter calibration by system size — IRF analysis not "
            "produced (see research log JSON for full results)"),
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
