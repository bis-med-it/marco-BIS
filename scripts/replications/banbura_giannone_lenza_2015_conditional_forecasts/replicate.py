"""Bańbura-Giannone-Lenza (2015) — Conditional forecasts via Kalman smoothing.

Methodological demonstration. Fits a CEE-style 7-variable quarterly BVAR
on 1965-2019 FRED-QD data, then runs three forecasts:

  1. Unconditional forecast (baseline)
  2. "Higher for longer" scenario: FFR held at 5% for 8 quarters
  3. "ZLB" counterfactual: FFR pinned at 0.25% for 8 quarters

Each scenario uses the ``lib.conditional_forecast`` function — the
Kalman-smoother conditional-forecast machinery from BGL 2015.

NB on signs: these are REDUCED-FORM conditional forecasts — the pinned
FFR path is explained by ALL shocks, so a higher-for-longer path can
raise GDP/inflation through reverse causality (the model reads high
rates as a response to a strong economy); see the README's results
discussion. Structural shock-identified scenarios (BGL Sec. 3.4) would
be needed for a "policy-shock-only" reading. The point of the BGL
framework is that the conditional distribution is computed via Kalman
smoothing rather than brute-force shock-path simulation.
"""

from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):       # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402


def main() -> int:
    spec = lib.Spec.from_replication("banbura_giannone_lenza_2015_conditional_forecasts")
    seed = int(spec.metadata["general"]["random_seed"])
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/4] Loading data via lib.load_data...")
    data = lib.load_data(spec)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Variables: {list(data.columns)}")

    print("\n[2/4] Fitting BVAR with Minnesota prior...")
    lags, trend = spec.model["lags"], spec.model["trend"]
    pd_ = spec.model["prior"]
    prior = lib.MinnesotaPrior(
        lambda_1=pd_["lambda_1"], lambda_2=pd_["lambda_2"],
        lambda_3=pd_["lambda_3"], lambda_4=pd_["lambda_4"],
        mu=pd_["mu"], delta=pd_["delta"],
    )
    bvar = lib.fit_minnesota_bvar(data, lags=lags, prior=prior, trend=trend)
    print(f"      Variables (canonical order): {bvar.variable_names}")
    print(f"      Max |eigenvalue|: {lib.max_eigenvalue_modulus(bvar.lag_matrices()):.4f}")

    fc_cfg = spec.forecast
    horizon = int(fc_cfg["horizon"])
    n_paths = int(fc_cfg["n_paths"])
    print(f"      Forecast horizon: {horizon} quarters; "
          f"density draws: {n_paths}")

    # ---- [3/4] Unconditional baseline ----
    print(f"\n[3/4] Unconditional forecast (no scenario)...")
    uncond = lib.conditional_forecast(
        bvar, horizon=horizon, conditioning=None,
        n_paths=n_paths, seed=seed,
    )
    print(f"      Headline path (point forecast):")
    print(f"        {'h':>3}  " + "  ".join(f"{v:>10}" for v in data.columns))
    for h in range(horizon):
        row = "  ".join(f"{uncond.point[h, i]:>+10.4f}"
                         for i in range(len(data.columns)))
        print(f"        {h + 1:>3}  {row}")

    # ---- [4/4] Conditional scenarios ----
    print(f"\n[4/4] Conditional scenarios:")
    scenarios = spec.metadata["scenarios"]
    scenario_results = []
    for sc in scenarios:
        print(f"\n  Scenario: {sc['name']!r}  ({sc['description']})")
        cond_dict = {k: np.asarray(v) for k, v in sc["conditioning"].items()}
        cf = lib.conditional_forecast(
            bvar, horizon=horizon, conditioning=cond_dict,
            n_paths=n_paths, seed=seed,
        )
        scenario_results.append({"scenario": sc, "result": cf})

        # Print shifts vs unconditional for the unconditioned variables
        print(f"        {'h':>3}  " + "  ".join(f"{v:>10}" for v in data.columns))
        for h in range(horizon):
            row = "  ".join(f"{cf.point[h, i]:>+10.4f}"
                             for i in range(len(data.columns)))
            print(f"        {h + 1:>3}  {row}")

        # Headline: rgdp and deflator shift at h=4 (1 year ahead)
        rgdp_idx = bvar.variable_names.index("rgdp")
        defl_idx = bvar.variable_names.index("deflator")
        h_mid = horizon // 2 - 1   # ~h=4 with horizon=8
        print(f"\n        At h={h_mid + 1} quarters (~ 1 year):")
        print(f"          rgdp shift vs unconditional: "
              f"{cf.point[h_mid, rgdp_idx] - uncond.point[h_mid, rgdp_idx]:+.4f}")
        print(f"          deflator shift vs unconditional: "
              f"{cf.point[h_mid, defl_idx] - uncond.point[h_mid, defl_idx]:+.4f}")
        if cf.paths is not None:
            print(f"          rgdp 68% band at h={h_mid + 1}: "
                  f"[{cf.lower_68[h_mid, rgdp_idx]:+.4f}, "
                  f"{cf.upper_68[h_mid, rgdp_idx]:+.4f}]")

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
            "variables": list(data.columns),
        },
        "model": {
            "class": "bvar", "lags": int(lags), "trend": trend,
            "prior": pd_,
        },
        "unconditional": {
            "point": uncond.point.tolist(),
            "median": uncond.median.tolist() if uncond.median is not None else None,
            "lower_68": uncond.lower_68.tolist() if uncond.lower_68 is not None else None,
            "upper_68": uncond.upper_68.tolist() if uncond.upper_68 is not None else None,
        },
        "scenarios": [
            {
                "name": s["scenario"]["name"],
                "description": s["scenario"]["description"],
                "conditioning": {k: list(v) for k, v in s["scenario"]["conditioning"].items()},
                "point": s["result"].point.tolist(),
                "median": (s["result"].median.tolist()
                            if s["result"].median is not None else None),
                "lower_68": (s["result"].lower_68.tolist()
                              if s["result"].lower_68 is not None else None),
                "upper_68": (s["result"].upper_68.tolist()
                              if s["result"].upper_68 is not None else None),
            }
            for s in scenario_results
        ],
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        ffr_col_idx = bvar.variable_names.index("ffr")
        hfl = next(
            (s for s in scenario_results
             if s["scenario"]["name"] == "higher_for_longer"), None,
        )
        zlb = next(
            (s for s in scenario_results
             if s["scenario"]["name"] == "zlb"), None,
        )
        hfl_ffr_avg = (
            float(hfl["result"].point[:, ffr_col_idx].mean()) if hfl else float("nan")
        )
        zlb_ffr_avg = (
            float(zlb["result"].point[:, ffr_col_idx].mean()) if zlb else float("nan")
        )
        # Non-degenerate bands: lower_68 < upper_68 everywhere in unconditional
        if uncond.lower_68 is not None and uncond.upper_68 is not None:
            bands_ok = bool(np.all(uncond.upper_68 - uncond.lower_68 > 0))
        else:
            bands_ok = False
        # Conditioning alters path: max |hfl.point - uncond.point| > 0.001
        if hfl is not None:
            max_shift = float(np.abs(hfl["result"].point - uncond.point).max())
            alters = max_shift > 0.001
        else:
            alters = False
        validation_results = {
            "n_obs": float(len(data)),
            "n_scenarios": float(len(scenario_results)),
            "hfl_ffr_avg": hfl_ffr_avg,
            "zlb_ffr_avg": zlb_ffr_avg,
            "bands_non_degenerate": 1.0 if bands_ok else 0.0,
            "conditioning_alters_path": 1.0 if alters else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — conditional forecasts return forecast paths,
    #      not IRFs; no standard add_irf section applies) ----
    spec_dict = {
        "Model": f"BVAR({lags}) with Minnesota prior + BGL 2015 conditional forecasts",
        "Variables": ", ".join(list(data.columns)),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Forecast horizon": str(uncond.point.shape[0]),
        "Scenarios": ", ".join(s["scenario"]["name"] for s in scenario_results),
        "Conditioning method": "Kalman smoother on companion-form state space (BGL 2015)",
        "Lag order": str(lags),
    }
    for s in scenario_results:
        sc = s["scenario"]
        spec_dict[f"Scenario: {sc['name']}"] = sc["description"]

    report_html = lib.VARReport(
        title=rep.get("title", "Bańbura-Giannone-Lenza 2015 conditional forecasts"),
        subtitle=rep.get("subtitle",
            "Conditional forecast scenarios via Kalman smoother — see research log "
            "JSON for full numerical paths"),
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
