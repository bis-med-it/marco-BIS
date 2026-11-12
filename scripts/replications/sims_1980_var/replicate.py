"""
Replication of Sims (1980) "Macroeconomics and Reality" — six-variable VAR.

Thin orchestration script. All heavy lifting is in ``scripts/lib``.

Pipeline:
  config.yaml  →  data loading  →  lib.fit_var  →  lib.cholesky_irf  →
  IRFContainer  →  report.html  +  numerical_results.json  +  research_log.json

See README.md in this folder for the specification rationale.
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make scripts/ importable
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))   # so `lib` is importable

import lib   # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_and_transform(cfg: dict) -> pd.DataFrame:
    """Three-stage pipeline: load raw -> preprocess (nominalise m1) -> transform.

    Sims's spec calls for nominal M1, but FRED-QD only provides M1REAL (real).
    The preprocess stage uses ``lib.preprocess.nominalise`` to recover nominal
    M1 from real M1 and the CPI (CPIAUCSL; on quarterly averages this is an
    approximation); the transform stage then applies the
    per-column transforms (log / level) for the final VAR.
    """
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


def hash_inputs(cfg: dict, data: pd.DataFrame) -> str:
    # Full-data digest: every observation enters the hash (an endpoint-only
    # summary would miss interior changes to the data).
    data_digest = hashlib.sha256(
        np.ascontiguousarray(data.to_numpy(dtype="float64")).tobytes()
        + "|".join(map(str, data.index)).encode()
        + "|".join(map(str, data.columns)).encode()
    ).hexdigest()
    payload = {
        "config": cfg,
        "data_sha256": data_digest,
        "n_obs": len(data),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


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

    print("\n[2/4] Estimating VAR...")
    fit = lib.fit_var(
        data=data,
        lags=cfg["model"]["lags"],
        trend=cfg["model"]["trend"],
    )
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, "
          f"BIC={fit.fit_info['bic']:.4f}, T_eff={fit.t_eff}, "
          f"max |eigenvalue|={lib.max_eigenvalue_modulus(fit.lag_matrices()):.4f}")

    print("\n[3/4] Computing Cholesky-identified IRFs with bands...")
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

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in fit.variable_names]
    max_eig = float(lib.max_eigenvalue_modulus(fit.lag_matrices()))

    # 4a. Paper-target validation (if paper_targets.yaml is present)
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        m1_idx = fit.variable_names.index("m1")
        rgdp_idx = fit.variable_names.index("rgdp")
        defl_idx = fit.variable_names.index("deflator")
        validation_results = {
            "n_obs": float(len(data)),
            "t_eff": float(fit.t_eff),
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "max_eig_modulus": max_eig,
            "m1_own_impact_positive":
                1.0 if irf.median[0, m1_idx, m1_idx] > 0 else 0.0,
            "rgdp_to_m1_h4": float(irf.median[4, rgdp_idx, m1_idx]),
            "deflator_to_m1_h12": float(irf.median[12, defl_idx, m1_idx]),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)

    # 4b. Build the research log (will be embedded in HTML + saved as JSON)
    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hash": hash_inputs(cfg, data),
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
            "trend": cfg["model"]["trend"],
            "fit_info": fit.fit_info,
            "max_eigenvalue_modulus": max_eig,
        },
        "identification": dict(cfg["identification"]),
        "irf": {
            "horizon": cfg["irf"]["horizon"],
            "bands": cfg["irf"]["bands"],
            "n_draws": cfg["irf"]["n_draws"],
            "credible_intervals": cfg["irf"]["credible_intervals"],
        },
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # 4c. Build the specification summary
    spec = {
        "Model": f"VAR({fit.p})",
        "Variables": ", ".join(var_labels),
        "Identification": str(cfg["identification"].get("scheme", "Cholesky")),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Trend": str(cfg["model"]["trend"]),
        "Max |eigenvalue|": f"{max_eig:.4f}",
        "AIC / BIC": f"{fit.fit_info['aic']:.4f} / {fit.fit_info['bic']:.4f}",
        "Asymptotic Monte Carlo draws":str(cfg['irf']['n_draws']),
    }

    # 4d. HTML report — full eight-section structure via add_standard_sections
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    # Use the M1 shock (Sims's headline) for the summary section
    m1_shock_idx = fit.variable_names.index("m1")
    report.add_standard_sections(
        fit=fit,
        irf=irf,
        spec=spec,
        log=log,
        var_labels=var_labels,
        shock_idx_for_summary=m1_shock_idx,
    narrative="auto",
    )
    report_path = outputs_dir / "report.html"
    report.save(report_path)
    print(f"      ->{report_path.relative_to(PROJECT_ROOT)}")

    # 4e. Numerical snapshot for regression testing
    snapshot_path = outputs_dir / "numerical_results.json"
    irf.save_snapshot(snapshot_path)
    print(f"      ->{snapshot_path.relative_to(PROJECT_ROOT)}")

    # 4f. Research log JSON (same content as embedded in HTML)
    log_path = outputs_dir / "research_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"      ->{log_path.relative_to(PROJECT_ROOT)}")

    # 4g. Print paper-target comparison to console (last, so it's most visible)
    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
