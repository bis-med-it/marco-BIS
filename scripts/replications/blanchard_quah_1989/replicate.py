"""
Replication of Blanchard & Quah (1989) — demand vs supply shock
decomposition via long-run identification.

Bivariate VAR on (Δlog real GDP, unemployment rate) with the Blanchard-Quah
restriction: the demand shock has no long-run effect on the level of GDP.

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


def load_and_transform(cfg: dict) -> pd.DataFrame:
    """Load raw FRED-QD series, then apply per-series transforms."""
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
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading and transforming data...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Variables (BQ order): {list(data.columns)}")

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, "
          f"T_eff={fit.t_eff}, max |eigenvalue|={eig:.4f}")

    print("\n[3/4] Long-run (Blanchard-Quah) identification...")
    sign_norm = cfg["identification"]["sign_normalization"]
    irf = lib.long_run_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        shock_names=cfg["identification"]["shock_names"],
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
        sign_normalization=sign_norm,
    )
    print(f"      IRF horizon: 0..{irf.horizon}, "
          f"{irf.n_vars} variables x {irf.n_shocks} shocks, "
          f"{cfg['irf']['n_draws']} draws")

    print("\n      Cumulative response of dlog_gdp (long-run effect on log GDP level):")
    for j, shock in enumerate(irf.shock_names):
        cum = irf.median[:, 0, j].sum()    # variable 0 = dlog_gdp
        print(f"        {shock:<10} -> {cum:+.5f}  "
              f"({'permanent' if abs(cum) > 0.1 else 'transitory'})")

    print("\n      Cumulative response of unrate:")
    for j, shock in enumerate(irf.shock_names):
        cum = irf.median[:, 1, j].sum()
        print(f"        {shock:<10} -> {cum:+.4f}")

    print("\n      Peak responses to one-standard-deviation structural shocks:")
    labels = cfg["data"]["variable_labels"]
    for j, shock in enumerate(irf.shock_names):
        for i, name in enumerate(irf.variable_names):
            path = irf.median[:, i, j]
            peak_h = int(np.argmax(np.abs(path)))
            print(f"        {shock:<10} -> {labels.get(name, name):<22} "
                  f"peak={path[peak_h]:+.4f} at h={peak_h}")

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in irf.variable_names]

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        gdp_idx = irf.variable_names.index("dlog_gdp")
        sup_idx = irf.shock_names.index("supply")
        dem_idx = irf.shock_names.index("demand")
        cum_sup_gdp = float(irf.median[:, gdp_idx, sup_idx].sum())
        cum_dem_gdp = float(irf.median[:, gdp_idx, dem_idx].sum())
        validation_results = {
            "n_vars": float(irf.median.shape[1]),
            "max_eig_modulus": float(eig),
            "cum_supply_log_gdp": cum_sup_gdp,
            "cum_demand_log_gdp": cum_dem_gdp,
            "supply_h0_gdp_growth_positive":
                1.0 if irf.median[0, gdp_idx, sup_idx] > 0 else 0.0,
            "demand_h0_gdp_growth_positive":
                1.0 if irf.median[0, gdp_idx, dem_idx] > 0 else 0.0,
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
            "variables": fit.variable_names,
        },
        "model": {
            "class": cfg["model"]["class"],
            "lags": fit.p,
            "fit_info": fit.fit_info,
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": dict(cfg["identification"]),
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p})",
        "Variables": ", ".join(var_labels),
        "Identification": "Long-run restrictions (Blanchard-Quah 1989)",
        "Restriction": "Demand shock has zero long-run effect on log GDP",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_standard_sections(
        fit=fit, irf=irf,
        spec=spec_dict, log=log,
        var_labels=var_labels,
        shock_idx_for_summary=0,
    narrative="auto",
    )
    report.save(outputs_dir / "report.html")
    print(f"      ->{(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    irf.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      ->{(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      ->{(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
