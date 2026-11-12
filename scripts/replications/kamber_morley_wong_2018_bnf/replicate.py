"""Kamber-Morley-Wong (2018) BN filter on US real GDP.

Demonstrates the KMW headline: imposing a low signal-to-noise ratio on the
AR(p) for first differences produces a LARGER, more persistent,
NBER-aligned cycle than the unconstrained Beveridge-Nelson decomposition.

Pipeline:
  1. Load log real GDP (FRED-QD GDPC1) and scale to log-percentage points.
  2. Unconstrained BN baseline at AR(12).
  3. KMW BN at delta=0.25 (the paper's low signal-to-noise ratio delta = psi(1)^2).
  4. SNR sensitivity sweep across [0.10, 0.25, 0.50, 1.00].
  5. NBER-recession-period average cycle for BN vs KMW (sign + amplitude).

Library calls: ``lib.beveridge_nelson`` (closed-form BN),
``lib.kmw_bn_filter`` (constrained AR(p): sum(phi) = 1 - 1/sqrt(delta-bar)
imposed exactly via the Dickey-Fuller substitution; closed-form OLS)
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
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_series(cfg: dict) -> pd.Series:
    spec = cfg["data"]["series"]
    raw = lib.load_dataset(
        source=cfg["data"]["source"],
        series={spec["code"]: spec["code"]},
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
    )
    df = lib.transform(raw, {spec["code"]: spec["transform"]})
    return df[spec["code"]].dropna()


def _phillips_curve_oos(
    inflation: pd.Series,
    bn_cycle: pd.Series,
    kmw_cycle: pd.Series,
    window_size: int,
    horizons: list[int],
    step: int,
    ar_lags: int,
) -> dict:
    """Expanding-window OOS Phillips-curve test.

    For each origin t (the last in-sample observation) and each horizon h:
      Three forecasting models, all fit on data within the training window:
        (a) AR baseline:   pi_{t+h} = c + sum_{j=1..p} phi_j pi_{t-j+1} + e
        (b) AR + BN gap:   pi_{t+h} = c + sum phi_j pi_{t-j+1} + alpha bn_t + e
        (c) AR + KMW gap:  pi_{t+h} = c + sum phi_j pi_{t-j+1} + alpha kmw_t + e
      Forecast each model at the origin's lag vector, compare to the realised
      pi_{t+h} once available.

    Returns dict with `n_origins`, `rmse_by_horizon` list, `dm_kmw_vs_bn`
    Diebold-Mariano comparison list, and per-origin forecast errors.
    """
    # Align all three series on a common index
    df = pd.concat(
        [inflation.rename("pi"), bn_cycle.rename("bn"), kmw_cycle.rename("kmw")],
        axis=1,
    ).dropna()

    T = len(df)
    if T < window_size + max(horizons) + 1:
        raise ValueError(
            f"Phillips curve OOS needs T >= window_size + max_horizon + 1; "
            f"got T={T}, window_size={window_size}, max_horizon={max(horizons)}"
        )

    # Build per-horizon forecast errors
    errs_ar: dict[int, list[float]] = {h: [] for h in horizons}
    errs_bn: dict[int, list[float]] = {h: [] for h in horizons}
    errs_kmw: dict[int, list[float]] = {h: [] for h in horizons}
    n_origins = 0

    for t in range(window_size, T - 1, step):
        train = df.iloc[:t]
        for h in horizons:
            target_idx = t + h - 1
            if target_idx >= T:
                continue
            tau_min, tau_max = ar_lags - 1, len(train) - 1 - h
            if tau_max < tau_min + 5:
                continue
            n_tr = tau_max - tau_min + 1
            # Build design matrices for each model
            X_ar = np.ones((n_tr, 1 + ar_lags))
            X_bn = np.ones((n_tr, 1 + ar_lags + 1))
            X_kmw = np.ones((n_tr, 1 + ar_lags + 1))
            y = np.zeros(n_tr)
            pi_arr = train["pi"].values
            bn_arr = train["bn"].values
            kmw_arr = train["kmw"].values
            for i, tau in enumerate(range(tau_min, tau_max + 1)):
                for lag in range(ar_lags):
                    X_ar[i, 1 + lag] = pi_arr[tau - lag]
                X_bn[i, 1:1 + ar_lags] = X_ar[i, 1:1 + ar_lags]
                X_kmw[i, 1:1 + ar_lags] = X_ar[i, 1:1 + ar_lags]
                X_bn[i, -1] = bn_arr[tau]
                X_kmw[i, -1] = kmw_arr[tau]
                y[i] = pi_arr[tau + h]
            beta_ar, *_ = np.linalg.lstsq(X_ar, y, rcond=None)
            beta_bn, *_ = np.linalg.lstsq(X_bn, y, rcond=None)
            beta_kmw, *_ = np.linalg.lstsq(X_kmw, y, rcond=None)

            origin_t = len(train) - 1
            x_origin_ar = np.array(
                [1.0] + [pi_arr[origin_t - lag] for lag in range(ar_lags)]
            )
            x_origin_bn = np.concatenate([x_origin_ar, [bn_arr[origin_t]]])
            x_origin_kmw = np.concatenate([x_origin_ar, [kmw_arr[origin_t]]])
            fhat_ar = float(x_origin_ar @ beta_ar)
            fhat_bn = float(x_origin_bn @ beta_bn)
            fhat_kmw = float(x_origin_kmw @ beta_kmw)
            realised = float(df["pi"].iloc[target_idx])
            errs_ar[h].append(fhat_ar - realised)
            errs_bn[h].append(fhat_bn - realised)
            errs_kmw[h].append(fhat_kmw - realised)
        n_origins += 1

    rmse_by_horizon = []
    dm_kmw_vs_bn = []
    for h in horizons:
        e_ar = np.asarray(errs_ar[h])
        e_bn = np.asarray(errs_bn[h])
        e_kmw = np.asarray(errs_kmw[h])
        rmse_by_horizon.append({
            "horizon": h,
            "ar_only": float(np.sqrt((e_ar ** 2).mean())),
            "ar_plus_bn": float(np.sqrt((e_bn ** 2).mean())),
            "ar_plus_kmw": float(np.sqrt((e_kmw ** 2).mean())),
            "n_origins": int(len(e_ar)),
        })
        if len(e_kmw) >= 5:
            dm = lib.diebold_mariano(e_kmw ** 2, e_bn ** 2, horizon=h)
            dm_kmw_vs_bn.append({
                "horizon": h, "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
            })
    return {
        "n_origins": n_origins,
        "rmse_by_horizon": rmse_by_horizon,
        "dm_kmw_vs_bn": dm_kmw_vs_bn,
    }


def nber_average(cycle: pd.Series, recessions: list[dict]) -> list[dict]:
    """Mean cycle within each NBER recession window (peak month -> trough month)."""
    out = []
    for r in recessions:
        mask = (cycle.index >= pd.Timestamp(r["start"])) & (
            cycle.index <= pd.Timestamp(r["end"])
        )
        seg = cycle[mask].dropna()
        out.append({
            "name": r["name"],
            "start": r["start"],
            "end": r["end"],
            "n_quarters": int(len(seg)),
            "mean_cycle": float(seg.mean()) if len(seg) > 0 else None,
            "min_cycle": float(seg.min()) if len(seg) > 0 else None,
        })
    return out


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print(f"  {cfg['replication']['subtitle']}")
    print("=" * 70)

    print("\n[1/5] Loading log real GDP...")
    series = load_series(cfg)
    print(f"      {series.name} (log * 100): "
          f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m} "
          f"({len(series)} obs)")

    p = cfg["filter"]["ar_order"]
    delta = cfg["filter"]["delta"]

    print(f"\n[2/5] Unconstrained Beveridge-Nelson (AR({p}))...")
    bn = lib.beveridge_nelson(series, ar_order=p)
    bn_std = float(bn.cycle.dropna().std())
    print(f"      cycle std: {bn_std:.3f} log-pp")
    print(f"      AR phi sum: {sum(bn.metadata['phi']):+.4f}")
    print(f"      max |eig|: {bn.metadata['max_companion_eigenvalue']:.4f}")

    print(f"\n[3/5] KMW BN with delta={delta} (AR({p}))...")
    kmw = lib.kmw_bn_filter(series, delta=delta, ar_order=p)
    kmw_std = float(kmw.cycle.dropna().std())
    print(f"      cycle std: {kmw_std:.3f} log-pp  "
          f"(x{kmw_std / bn_std:.2f} vs BN)")
    print(f"      sum(phi) = {kmw.metadata['sum_phi']:+.4f} (= 1 - 1/sqrt(delta), exact)")
    print(f"      delta unconstrained: {kmw.metadata['delta_unconstrained']:.3f}  "
          f"(imposed delta-bar = {delta})")
    print(f"      AR phi sum: {sum(kmw.metadata['phi']):+.4f}  "
          f"(unconstrained: {sum(kmw.metadata['phi_unconstrained']):+.4f})")
    print(f"      max |eig|: {kmw.metadata['max_companion_eigenvalue']:.4f}")

    # ---- SNR sensitivity sweep ----
    print(f"\n      SNR sensitivity sweep:")
    print(f"        {'delta':>6}  {'cycle std':>10}  {'sum_phi':>8}  "
          f"{'psi(1)':>8}  {'max|eig|':>9}")
    sensitivity_results = []
    for delta_val in cfg["filter"]["delta_sensitivity"]:
        kmw_s = lib.kmw_bn_filter(series, delta=delta_val, ar_order=p)
        std_s = float(kmw_s.cycle.dropna().std())
        sensitivity_results.append({
            "delta": float(delta_val),
            "cycle_std": std_s,
            "sum_phi": float(kmw_s.metadata["sum_phi"]),
            "psi_1": float(kmw_s.metadata["psi_1"]),
            "max_companion_eigenvalue": float(
                kmw_s.metadata["max_companion_eigenvalue"]
            ),
        })
        print(f"        {delta_val:>6.2f}  {std_s:>10.3f}  "
              f"{kmw_s.metadata['sum_phi']:>+8.3f}  "
              f"{kmw_s.metadata['psi_1']:>8.3f}  "
              f"{kmw_s.metadata['max_companion_eigenvalue']:>9.4f}")

    # ---- Phillips-curve OOS test: does the gap predict inflation? ----
    print(f"\n[4/5] Phillips-curve OOS test (does the gap predict inflation?)...")
    pc_cfg = cfg["phillips_curve"]
    raw_pi = lib.load_dataset(
        source=pc_cfg["inflation_source"],
        series={pc_cfg["inflation_code"]: pc_cfg["inflation_code"]},
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
    )
    inflation = lib.transform(
        raw_pi, {pc_cfg["inflation_code"]: pc_cfg["inflation_transform"]},
    )[pc_cfg["inflation_code"]].dropna()
    inflation.name = "pi"
    print(f"      Inflation series ({pc_cfg['inflation_code']}, "
          f"{pc_cfg['inflation_transform']}): {len(inflation)} obs, "
          f"mean={inflation.mean():+.4f}, std={inflation.std():.4f}")

    pc_results = _phillips_curve_oos(
        inflation=inflation, bn_cycle=bn.cycle, kmw_cycle=kmw.cycle,
        window_size=pc_cfg["window_size"], horizons=pc_cfg["horizons"],
        step=pc_cfg["step"], ar_lags=pc_cfg["ar_lags"],
    )
    print(f"      Expanding window from {pc_cfg['window_size']} quarters, "
          f"step={pc_cfg['step']}, AR({pc_cfg['ar_lags']}) inflation base.")
    print(f"      Generated forecasts at {pc_results['n_origins']} origins.\n")
    print(f"      Forecast RMSE by model (lower is better):")
    print(f"        {'horizon':>8}  {'AR baseline':>12}  {'+ BN gap':>10}  {'+ KMW gap':>11}")
    for row in pc_results["rmse_by_horizon"]:
        print(f"        {row['horizon']:>8}  {row['ar_only']:>12.5f}  "
              f"{row['ar_plus_bn']:>10.5f}  {row['ar_plus_kmw']:>11.5f}")
    print(f"\n      RMSE ratios vs AR baseline (< 1 = the gap adds predictive info):")
    print(f"        {'horizon':>8}  {'BN/AR':>10}  {'KMW/AR':>10}  {'KMW/BN':>10}")
    for row in pc_results["rmse_by_horizon"]:
        bn_r = row["ar_plus_bn"] / row["ar_only"]
        kmw_r = row["ar_plus_kmw"] / row["ar_only"]
        kb_r = row["ar_plus_kmw"] / row["ar_plus_bn"]
        print(f"        {row['horizon']:>8}  {bn_r:>10.3f}  "
              f"{kmw_r:>10.3f}  {kb_r:>10.3f}")
    print(f"\n      Diebold-Mariano (KMW-gap vs BN-gap; squared-error losses):")
    print(f"        {'horizon':>8}  {'DM-stat':>10}  {'p-value':>10}")
    for row in pc_results["dm_kmw_vs_bn"]:
        print(f"        {row['horizon']:>8}  {row['dm_stat']:>+10.3f}  "
              f"{row['p_value']:>10.3f}")

    # ---- NBER recession averages ----
    print(f"\n[5/5] Cycle averages within NBER recession windows:")
    print(f"        {'recession':<22}  {'BN mean':>10}  {'KMW mean':>10}  "
          f"{'KMW min':>10}")
    bn_nber = nber_average(bn.cycle, cfg["nber_recessions"])
    kmw_nber = nber_average(kmw.cycle, cfg["nber_recessions"])
    for bn_e, kmw_e in zip(bn_nber, kmw_nber):
        bn_m = f"{bn_e['mean_cycle']:+.2f}" if bn_e["mean_cycle"] is not None else "  n/a"
        kmw_m = f"{kmw_e['mean_cycle']:+.2f}" if kmw_e["mean_cycle"] is not None else "  n/a"
        kmw_min = f"{kmw_e['min_cycle']:+.2f}" if kmw_e["min_cycle"] is not None else "  n/a"
        print(f"        {bn_e['name']:<22}  {bn_m:>10}  {kmw_m:>10}  {kmw_min:>10}")

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    cycles = pd.DataFrame({
        "bn_cycle": bn.cycle,
        f"kmw_cycle_delta_{delta}": kmw.cycle,
    })
    cycles.to_csv(outputs_dir / "cycles.csv")
    print(f"\n      -> {(outputs_dir / 'cycles.csv').relative_to(PROJECT_ROOT)}")

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "series": cfg["data"]["series"]["code"],
            "transform": cfg["data"]["series"]["transform"],
            "sample_start": str(series.index[0].date()),
            "sample_end": str(series.index[-1].date()),
            "n_obs": int(len(series)),
        },
        "bn": {
            "ar_order": bn.metadata["ar_order"],
            "phi": bn.metadata["phi"],
            "cycle_std": bn_std,
            "max_companion_eigenvalue": bn.metadata["max_companion_eigenvalue"],
        },
        "kmw": {
            "ar_order": kmw.metadata["ar_order"],
            "delta_target": kmw.metadata["delta_target"],
            "sum_phi": kmw.metadata["sum_phi"],
            "psi_1": kmw.metadata["psi_1"],
            "delta_unconstrained": kmw.metadata["delta_unconstrained"],
            "phi": kmw.metadata["phi"],
            "phi_unconstrained": kmw.metadata["phi_unconstrained"],
            "cycle_std": kmw_std,
            "max_companion_eigenvalue": kmw.metadata["max_companion_eigenvalue"],
            "amplitude_ratio_vs_bn": kmw_std / bn_std,
        },
        "delta_sensitivity": sensitivity_results,
        "nber_recessions": {
            "bn": bn_nber,
            "kmw": kmw_nber,
        },
        "phillips_curve_oos": pc_results,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        validation_results = {
            "n_obs": float(len(series)),
            "bn_cycle_std": float(bn_std),
            "kmw_cycle_std": float(kmw_std),
            "kmw_bn_amplitude_ratio": float(kmw_std / bn_std),
            "sum_phi": float(kmw.metadata["sum_phi"]),
            "delta_unconstrained": float(kmw.metadata["delta_unconstrained"]),
            "gap_2009q2_negative": 1.0 if float(
                kmw.cycle.loc["2009-04":"2009-06"].dropna().iloc[0]) < 0 else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — filter-only replication, no IRFs) ----
    spec_dict = {
        "Model": "Beveridge-Nelson + KMW BN filter (delta-bar restriction, constrained OLS)",
        "Series": cfg["data"]["series"]["code"],
        "Transform": cfg["data"]["series"]["transform"],
        "Sample": f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m} ({len(series)} obs)",
        "AR order (p)": str(bn.metadata["ar_order"]),
        "BN cycle std": f"{bn_std:.4f}",
        "KMW cycle std": f"{kmw_std:.4f}",
        "Amplitude ratio (KMW/BN)": f"{kmw_std/bn_std:.3f}",
        "delta-bar (imposed)": f"{kmw.metadata['delta_target']:.3f}",
        "sum(phi) = 1 - 1/sqrt(delta)": f"{kmw.metadata['sum_phi']:+.4f}",
        "delta unconstrained": f"{kmw.metadata['delta_unconstrained']:.3f}",
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
