"""
Replication of Christiano, Eichenbaum & Evans (1999) — seven-variable
quarterly OLS VAR with Cholesky identification of the monetary policy
shock.

CEE 1999 §4 uses unrestricted OLS (no Bayesian prior), and so does this
replication.

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
    """Three-stage pipeline: load raw -> preprocess (nominalise m1) -> transform."""
    data_cfg = cfg["data"]

    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        start=data_cfg["sample_start"],
        end=data_cfg["sample_end"],
    )

    for step in data_cfg.get("preprocess", []):
        op = step["operation"]
        if op == "nominalise":
            raw[step["output"]] = lib.preprocess.nominalise(
                raw[step["real"]], raw[step["deflator"]]
            )
        elif op == "deflate":
            raw[step["output"]] = lib.preprocess.deflate(
                raw[step["nominal"]], raw[step["deflator"]]
            )
        else:
            raise ValueError(f"unknown preprocess operation: {op}")

    var_cols = data_cfg["variable_order"]
    return lib.transform(raw[var_cols], data_cfg["transforms"])


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
    print(f"      Variables (Cholesky order): {list(data.columns)}")

    print("\n[2/4] Fitting OLS VAR...")
    fit = lib.fit_var(
        data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}) by OLS, T_eff={fit.t_eff}, k={fit.k}")
    print(f"      Max |eigenvalue|: {eig:.4f}")
    print(f"      AIC: {fit.fit_info['aic']:.3f}, "
          f"BIC: {fit.fit_info['bic']:.3f}")

    print("\n[3/4] Cholesky IRFs with asymptotic MC bands...")
    labels = cfg["data"]["variable_labels"]
    shock_names = [f"{labels.get(v, v)} shock" for v in fit.variable_names]
    irf = lib.cholesky_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        shock_names=shock_names,
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
    )
    print(f"      IRF horizon: 0..{irf.horizon}, "
          f"{irf.n_vars} variables x {irf.n_shocks} shocks, "
          f"{cfg['irf']['n_draws']} draws")

    # CEE's headline: response of rgdp, deflator, etc. to FFR shock
    ffr_idx = fit.variable_names.index(cfg["identification"]["shock_of_interest"])
    print(f"\n      Monetary policy shock (column {ffr_idx} = ffr) — selected responses:")
    print(f"        {'h':>3} {'rgdp':>10} {'deflator':>10} {'commodity':>10} {'m1':>10}")
    for h in [0, 2, 4, 8, 12, 16, 24, 32, 40]:
        if h > irf.horizon:
            continue
        rgdp = irf.median[h, fit.variable_names.index("rgdp"), ffr_idx]
        defl = irf.median[h, fit.variable_names.index("deflator"), ffr_idx]
        comm = irf.median[h, fit.variable_names.index("commodity"), ffr_idx]
        m1   = irf.median[h, fit.variable_names.index("m1"), ffr_idx]
        print(f"        {h:3d} {rgdp:+10.5f} {defl:+10.5f} {comm:+10.5f} {m1:+10.5f}")

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in fit.variable_names]
    ffr_idx = fit.variable_names.index("ffr")

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        rgdp_idx = fit.variable_names.index("rgdp")
        rgdp_path = irf.median[4:17, rgdp_idx, ffr_idx]
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "max_eig_modulus": float(eig),
            "ffr_own_impact_positive":
                1.0 if irf.median[0, ffr_idx, ffr_idx] > 0 else 0.0,
            "rgdp_to_ffr_h12": float(irf.median[12, rgdp_idx, ffr_idx]),
            "rgdp_contracts_in_window":
                1.0 if rgdp_path.min() < 0 else 0.0,
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
            "trend": fit.fit_info["trend"],
            "aic": fit.fit_info["aic"],
            "bic": fit.fit_info["bic"],
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": dict(cfg["identification"]),
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    spec = {
        "Model": f"VAR({fit.p}), OLS",
        "Variables": ", ".join(var_labels),
        "Identification": "Recursive (Cholesky) — FFR ordered last in slow block",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Trend": str(fit.fit_info.get("trend", "n/a")),
        "Max |eigenvalue|": f"{float(eig):.4f}",
        "AIC / BIC": f"{fit.fit_info['aic']:.4f} / {fit.fit_info['bic']:.4f}",
    }

    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_standard_sections(
        fit=fit,
        irf=irf,
        spec=spec,
        log=log,
        var_labels=var_labels,
        shock_idx_for_summary=ffr_idx,
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
