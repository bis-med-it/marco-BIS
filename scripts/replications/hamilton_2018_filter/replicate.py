"""
Hamilton (2018) filter demonstration on US real GDP.

Compares four trend-cycle decompositions:
  HP (lambda=1600), Hamilton (h=8, p=4), BN (AR auto-selected by BIC),
  KMW-BN (delta=0.25).

Hamilton's argument is that HP introduces spurious dynamics and that his
regression-based filter is cleaner. KMW (2018) propose the BN filter with
a low signal-to-noise ratio delta imposed as a model-based alternative. Together these define
the modern trend-cycle decomposition landscape.

Thin orchestration script using ``scripts/lib``.
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


def load_series(cfg: dict) -> pd.Series:
    """Load a single raw FRED-QD series, then apply the transform."""
    spec = cfg["data"]["series"]
    raw = lib.load_dataset(
        source=cfg["data"]["source"],
        series={spec["code"]: spec["code"]},
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
    )
    df = lib.transform(raw, {spec["code"]: spec["transform"]})
    return df[spec["code"]]


def apply_filter(series: pd.Series, spec: dict) -> lib.FilterResult:
    method = spec["method"]
    if method == "hp":
        return lib.hp_filter(series, lam=spec.get("lam"))
    if method == "hamilton":
        return lib.hamilton_filter(series, h=spec.get("h"), p=spec.get("p", 4))
    if method == "bn":
        return lib.beveridge_nelson(
            series, ar_order=spec.get("ar_order"),
            max_ar_order=spec.get("max_ar_order", 12),
            ic=spec.get("ic", "bic"),
        )
    if method == "kmw_bn":
        return lib.kmw_bn_filter(
            series,
            delta=spec.get("delta", 0.25),
            ar_order=spec.get("ar_order"),
            max_ar_order=spec.get("max_ar_order", 12),
            ic=spec.get("ic", "bic"),
        )
    raise ValueError(f"unknown filter method: {method}")


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    series = load_series(cfg)
    print(f"      {series.name} (log * 100): "
          f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m} "
          f"({len(series)} obs)")

    print("\n[2/3] Applying filters...")
    results = {}
    for spec in cfg["filters"]:
        result = apply_filter(series, spec)
        results[result.method] = result
        cycle = result.cycle.dropna()
        cycle_std = float(cycle.std())
        print(f"      {result.method:<14} cycle std = {cycle_std:.3f}%"
              f"   ({len(cycle)} obs)")

    print("\n      Sample of cycle values across methods (last 8 quarters):")
    header = "      " + f"{'date':<10}" + "".join(
        f"{m:>14}" for m in results.keys()
    )
    print(header)
    for date in series.index[-8:]:
        row = f"      {str(date.date()):<10}"
        for method, result in results.items():
            v = result.cycle.loc[date]
            row += f"{v:>14.3f}" if pd.notna(v) else f"{'NaN':>14}"
        print(row)

    print("\n[3/3] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Save side-by-side cycles for easy comparison
    cycles = pd.DataFrame({m: r.cycle for m, r in results.items()})
    trends = pd.DataFrame({m: r.trend for m, r in results.items()})
    cycles.to_csv(outputs_dir / "cycles.csv")
    trends.to_csv(outputs_dir / "trends.csv")
    print(f"      ->{(outputs_dir / 'cycles.csv').relative_to(PROJECT_ROOT)}")
    print(f"      ->{(outputs_dir / 'trends.csv').relative_to(PROJECT_ROOT)}")

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
        "filters": [
            {
                "method": r.method,
                "metadata": {
                    k: (v if not isinstance(v, np.ndarray) else v.tolist())
                    for k, v in r.metadata.items()
                },
                "cycle_std": float(r.cycle.dropna().std()),
                "n_obs_with_cycle": int(r.cycle.notna().sum()),
            }
            for r in results.values()
        ],
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    stds = {m: float(r.cycle.dropna().std()) for m, r in results.items()}
    if targets_path.exists():
        validation_results = {
            "n_methods": float(len(results)),
            "n_obs": float(len(series)),
            "hp_cycle_std": stds.get("hp", float("nan")),
            "hamilton_cycle_std": stds.get("hamilton", float("nan")),
            "hamilton_amplitude_exceeds_hp":
                1.0 if stds.get("hamilton", 0) > stds.get("hp", 0) else 0.0,
            "bn_cycle_std": stds.get("beveridge_nelson", float("nan")),
            "kmw_amplitude_exceeds_bn":
                1.0 if stds.get("kmw_bn_filter", 0) > stds.get("beveridge_nelson", 0) else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — trend-cycle decomposition, no IRFs) ----
    spec_dict = {
        "Model": "Trend-cycle decomposition — multi-filter comparison",
        "Series": cfg["data"]["series"]["code"],
        "Transform": cfg["data"]["series"]["transform"],
        "Sample": f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m} ({len(series)} obs)",
        "Filters compared": ", ".join(sorted(results.keys())),
    }
    # Add per-filter cycle standard deviations
    for m in sorted(stds.keys()):
        spec_dict[f"Cycle std ({m})"] = f"{stds[m]:.4f}"

    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
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
