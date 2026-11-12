"""
Bernanke-Boivin-Eliasz (2005) FAVAR with FRED-MD data.

A simplified two-step FAVAR using a hand-picked 19-variable information
panel from FRED-MD. The full BBE paper uses ~120 series — this replication
demonstrates the method on a smaller, FRED-MD-only panel for ease of
reproduction.

Slow/fast rotation: 9 slow variables (real activity, prices, employment,
wages) are used to construct the BBE rotation, which removes the factors'
direct contemporaneous dependence on the FFR (the rotated factors are not
orthogonal to it).
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


def load_data(cfg: dict):
    """Load policy variable + info panel from FRED-MD.

    Some info-panel codes may not exist in the user's FRED-MD vintage
    (the paper used a ~120-series panel; this config uses a hand-picked
    subset, some of which get renamed or discontinued across vintages).
    ``ignore_missing=True`` skips absent series with a warning. Policy +
    info panel are loaded jointly so date alignment is automatic.
    """
    p_cfg = cfg["data"]["policy"]
    info_specs = cfg["data"]["info_panel"]

    series_spec = {p_cfg["code"]: p_cfg["code"]}
    series_spec.update({s["code"]: s["code"] for s in info_specs})

    raw = lib.load_dataset(
        source=cfg["data"]["source"],
        series=series_spec,
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
        ignore_missing=True,
    )

    transforms = {p_cfg["code"]: p_cfg["transform"]}
    for s in info_specs:
        if s["code"] in raw.columns:
            transforms[s["code"]] = s["transform"]
    data = lib.transform(raw, transforms)

    policy = data[p_cfg["code"]].rename(p_cfg["name"])
    available_codes = [s["code"] for s in info_specs if s["code"] in data.columns]
    info_panel = data[available_codes]
    return info_panel, policy, available_codes


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading FRED-MD data...")
    info, policy, codes = load_data(cfg)
    print(f"      Sample: {info.index[0]:%Y-%m} to {info.index[-1]:%Y-%m}"
          f" ({len(info)} months)")
    print(f"      Information panel: {info.shape[1]} variables")
    print(f"      Policy: {policy.name}")

    # Slow block — intersect with available info-panel columns
    requested_slow = cfg["data"]["slow_block"]
    slow_cols = [c for c in requested_slow if c in info.columns]
    print(f"      Slow block: {len(slow_cols)} variables ({', '.join(slow_cols[:5])}...)")

    print("\n[2/4] Fitting FAVAR (two-step PC)...")
    fit = lib.fit_favar(
        info_panel=info,
        policy_series=policy,
        n_factors=cfg["model"]["n_factors"],
        lags=cfg["model"]["lags"],
        slow_columns=slow_cols,
    )
    eig = lib.max_eigenvalue_modulus(fit.var_fit.lag_matrices())
    print(f"      VAR({fit.var_fit.p}) on [F_1..F_{fit.n_factors}, {fit.policy_var}], "
          f"T_eff={fit.var_fit.t_eff}, max |eigenvalue|={eig:.4f}")
    print(f"      Rotation: {fit.metadata['rotation']}")

    print("\n[3/4] Computing FAVAR IRFs to a monetary shock...")
    irf = lib.favar_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=cfg["general"]["random_seed"],
    )
    print(f"      IRF horizon: 0..{irf.horizon}, {irf.n_vars} responses, 1 shock")
    print(f"\n      Selected variable responses to a +1 unit FFR Cholesky shock:")
    print(f"        {'variable':<15} {'h=0':>8} {'h=6':>8} {'h=12':>8} {'h=24':>8} {'h=36':>8}")
    selected = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "GS1", "GS10"]
    for v in selected:
        if v not in irf.variable_names:
            continue
        i = irf.variable_names.index(v)
        vals = [irf.median[h, i, 0] for h in [0, 6, 12, 24, 36]]
        print(f"        {v:<15} " + " ".join(f"{val:>+8.4f}" for val in vals))

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "sample_start": str(info.index[0].date()),
            "sample_end": str(info.index[-1].date()),
            "n_obs": int(len(info)),
            "n_info_vars": int(info.shape[1]),
            "policy": fit.policy_var,
            "slow_block_size": len(slow_cols),
        },
        "model": {
            "n_factors": fit.n_factors,
            "lags": fit.var_fit.p,
            "rotation": fit.metadata["rotation"],
            "max_eigenvalue_modulus": float(eig),
        },
        "irf_metadata": irf.metadata,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        irf_shape_ok = (
            irf.median.ndim == 3
            and irf.median.shape[1] >= fit.n_factors + 1
            and irf.median.shape[2] == 1
        )
        validation_results = {
            "n_obs": float(len(info)),
            "n_info_vars": float(info.shape[1]),
            "n_factors": float(fit.n_factors),
            "n_lags": float(fit.var_fit.p),
            "stable": 1.0 if eig < 1.0 else 0.0,
            "rotation_is_bbe":
                1.0 if fit.metadata["rotation"] == "slow_fast_BBE" else 0.0,
            "irf_shape_ok": 1.0 if irf_shape_ok else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report — FAVAR fit is a FAVARFit wrapping a VAR on the
    #      factor + policy state; the IRF spans many response variables
    #      (factors + observed). The fit doesn't have standard residuals
    #      compatible with lib.diagnostics, so the helper's diagnostics
    #      section will gracefully skip — Specification + IRF + Research
    #      Log core. ----
    spec_dict = {
        "Model": f"FAVAR with {fit.n_factors} factors + {fit.policy_var}",
        "Information set size": str(int(info.shape[1])),
        "Slow block size": str(len(slow_cols)),
        "Policy variable": fit.policy_var,
        "Sample": f"{info.index[0]:%Y-%m} to {info.index[-1]:%Y-%m} ({len(info)} obs)",
        "Lag order (factor VAR)": str(fit.var_fit.p),
        "Rotation": fit.metadata["rotation"],
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_specification(spec_dict)
    report.add_irf(
        irf_point=irf.median, ci_68_lo=irf.lower_68, ci_68_hi=irf.upper_68,
        ci_90_lo=irf.lower_90, ci_90_hi=irf.upper_90,
        horizons=irf.horizons, var_names=irf.variable_names,
        shock_labels=irf.shock_names,
        x_axis_label="Months",
    )
    report.add_research_log(log)
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
