"""
Diebold-Yilmaz (2012) connectedness on 4 asset-class volatility proxies.

Original paper: daily Parkinson (1980) high-low volatilities on US stocks, bonds, FX,
and commodities, 1999-2010. This replication uses MONTHLY absolute log
returns for stocks/FX/oil and absolute yield changes for bonds as
volatility proxies from FRED-MD (1973-2024).

Methodology is paper-faithful (Pesaran-Shin GVD, total/directional/net
spillovers); the data is a sensible monthly substitute. The total
spillover index should still be a crisis barometer.
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

import lib   # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_and_transform(cfg: dict) -> pd.DataFrame:
    """Load raw FRED-MD series, then apply per-series transforms."""
    raw = lib.load_dataset(
        source=cfg["data"]["source"],
        series={s["name"]: s["fred_code"] for s in cfg["data"]["series"]},
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
    )
    return lib.transform(
        raw, {s["name"]: s["transform"] for s in cfg["data"]["series"]}
    )


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading volatility proxies...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months)")
    print(f"      Variables: {list(data.columns)}")

    print("\n[2/4] Estimating VAR for full-sample connectedness...")
    fit = lib.fit_var(data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"])
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), max |eigenvalue|={eig:.4f}")

    print("\n[3/4] Diebold-Yilmaz (2012) connectedness measures...")
    result = lib.connectedness(fit, horizon=cfg["connectedness"]["horizon"])
    print(f"      Total spillover index: {result.total:.2f}%")
    labels = cfg["data"]["variable_labels"]
    print(f"\n      {'variable':<35} {'from':>8} {'to':>8} {'net':>8}")
    for i, v in enumerate(result.variable_names):
        print(f"      {labels.get(v, v):<35} "
              f"{result.directional_from[i]:>+8.2f} "
              f"{result.directional_to[i]:>+8.2f} "
              f"{result.net[i]:>+8.2f}")

    print("\n[4/4] Computing rolling-window connectedness...")
    window = cfg["connectedness"]["rolling_window"]
    rolling = lib.rolling_connectedness(
        data, window=window, horizon=cfg["connectedness"]["horizon"],
        lags=cfg["model"]["lags"], step=cfg["connectedness"]["rolling_step"],
    )
    print(f"      Computed {len(rolling)} windows ({window} months each)")
    print(f"      Total spillover range: {rolling['total'].min():.1f}% to "
          f"{rolling['total'].max():.1f}%")
    # Identify peaks (likely crisis windows)
    print(f"\n      Top 5 connectedness peaks:")
    top = rolling['total'].sort_values(ascending=False).head(5)
    for date, val in top.items():
        print(f"        {date:%Y-%m}: {val:.2f}%")

    print("\n      Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    rolling.to_csv(outputs_dir / "rolling_connectedness.csv")
    print(f"      ->{(outputs_dir / 'rolling_connectedness.csv').relative_to(PROJECT_ROOT)}")

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
            "substitution": "monthly absolute log returns (stocks/FX/oil) and absolute yield changes (bonds) as volatility proxies (original DY 2012 uses daily Parkinson high-low)",
        },
        "model": {
            "class": "var",
            "lags": fit.p,
            "max_eigenvalue_modulus": float(eig),
        },
        "full_sample_connectedness": {
            "total": float(result.total),
            "horizon": int(result.horizon),
            "fevd": result.fevd.tolist(),
            "directional_from": result.directional_from.tolist(),
            "directional_to": result.directional_to.tolist(),
            "net": result.net.tolist(),
        },
        "rolling": {
            "window": int(window),
            "n_windows": int(len(rolling)),
            "total_min": float(rolling['total'].min()),
            "total_max": float(rolling['total'].max()),
            "total_mean": float(rolling['total'].mean()),
        },
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        peak_date = rolling['total'].idxmax()
        peak_during_gfc = peak_date.year in (2008, 2009)
        validation_results = {
            "n_vars": float(fit.n_vars),
            "max_eig_modulus": float(eig),
            "total_spillover_index": float(result.total),
            "rolling_max_spillover": float(rolling['total'].max()),
            "rolling_min_spillover": float(rolling['total'].min()),
            "peak_during_gfc": 1.0 if peak_during_gfc else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — connectedness measures are spillover
    #      indices, not IRFs; rolling time series + headline stats summary) ----
    peak_date = rolling['total'].idxmax()
    spec_dict = {
        "Model": f"VAR({fit.p}) + Diebold-Yilmaz (2012) connectedness",
        "Variables": ", ".join(list(data.columns)),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Max |eigenvalue|": f"{float(eig):.4f}",
        "Connectedness horizon": str(int(result.horizon)),
        "Total spillover index": f"{float(result.total):.2f}%",
        "Rolling window": f"{int(window)} months",
        "Rolling: n windows": str(int(len(rolling))),
        "Rolling: min spillover": f"{float(rolling['total'].min()):.2f}%",
        "Rolling: max spillover": f"{float(rolling['total'].max()):.2f}%",
        "Rolling: peak date": peak_date.strftime("%Y-%m"),
        "Peak during GFC (2008-09)":
            "yes" if peak_date.year in (2008, 2009) else "no",
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"] + " — see rolling_connectedness.csv "
                 "for the full time series",
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"      ->{(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      ->{(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
