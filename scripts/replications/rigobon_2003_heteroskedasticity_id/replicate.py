"""
Rigobon (2003) heteroskedasticity-based identification — methodological demo.

Original paper: emerging-market sovereign bond returns with crisis vs
tranquil regimes. This replication substitutes:
  - US industrial production growth, 10Y yield, S&P 500 returns
  - NBER recessions as the high-variance regime

Demonstrates the library's `heteroskedasticity_irf` and compares the
identified shocks against the Cholesky baseline.
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


def build_recession_indicator(cfg: dict, index: pd.DatetimeIndex) -> np.ndarray:
    """1 during NBER recessions, 0 otherwise — aligned to `index`."""
    indicator = np.zeros(len(index), dtype=int)
    for peak, trough in cfg["identification"]["nber_recessions"]:
        peak_ts = pd.Timestamp(peak)
        trough_ts = pd.Timestamp(trough)
        mask = (index >= peak_ts) & (index <= trough_ts)
        indicator[mask] = 1
    return indicator


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading and transforming data...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months, {data.shape[1]} variables)")

    recession = build_recession_indicator(cfg, data.index)
    n_rec = int(recession.sum())
    n_exp = int((1 - recession).sum())
    print(f"      Regimes: {n_rec} recession months, {n_exp} expansion months")

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"])
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), max |eigenvalue|={eig:.4f}, T_eff={fit.t_eff}")

    print("\n[3/4] Heteroskedasticity-based identification (Rigobon 2003)...")
    irf, diag = lib.heteroskedasticity_irf(
        fit=fit,
        regime=recession,
        horizon=cfg["irf"]["horizon"],
        shock_names=cfg["identification"]["shock_names"],
        sign_normalization=cfg["identification"]["sign_normalization"],
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=cfg["general"]["random_seed"],
    )
    eigvals = np.asarray(diag["eigenvalues_omega1_over_omega2"])
    print(f"      Regime variance ratios (Omega_exp / Omega_rec eigenvalues):")
    for k, lam in enumerate(eigvals):
        print(f"        shock {k+1}: {lam:.3f}")
    # Identification needs EVERY pair of variance ratios separated — a large
    # max/min spread cannot certify that (two shocks can share a ratio while
    # a third is far away, leaving their subspace unidentified). Report the
    # label-invariant minimum pairwise separation instead; it is a
    # heuristic, not a formal weak-identification test.
    min_sep = diag["min_pairwise_log_separation"]
    print(f"      Minimum pairwise separation of the variance ratios "
          f"(log scale): {min_sep:.3f}")
    if diag["weak_identification"]:
        print("      CAUTION: at least two shocks have nearly equal variance "
              "ratios — individually they are close to unidentified (only "
              "their joint subspace is pinned down).")
    else:
        print("      All pairwise ratios are separated at this heuristic "
              "threshold; identification strength is not certified beyond "
              "that.")

    print("\n      Sample IRFs to shock 2 (yield-positive sign convention) at selected horizons:")
    labels = cfg["data"]["variable_labels"]
    shock_idx = irf.shock_names.index("shock_2")
    print(f"        {'h':>3}  {'ip_growth':>10} {'yield':>10} {'stock_ret':>10}")
    for h in [0, 1, 3, 6, 12, 24]:
        if h > irf.horizon:
            continue
        vals = [irf.median[h, i, shock_idx] for i in range(3)]
        print(f"        {h:>3} " + "  ".join(f"{v:>+9.4f}" for v in vals))

    print("\n[4/4] Comparison: Cholesky-identified IRFs on the same data...")
    irf_chol = lib.cholesky_irf(
        fit=fit, horizon=cfg["irf"]["horizon"],
        bands="none",
    )
    print(f"      Cholesky decomp of full-sample Sigma_u — first shock impact:")
    print(f"        {'variable':<12} {'Cholesky (col 0)':>16} {'Het-ID (shock_1)':>18}")
    for i, name in enumerate(data.columns):
        chol_val = irf_chol.median[0, i, 0]
        het_val = irf.median[0, i, 0]
        print(f"        {name:<12} {chol_val:>+16.4f} {het_val:>+18.4f}")

    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in irf.variable_names]

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        n_rec = int(recession[fit.p:].sum())
        n_exp = int((1 - recession[fit.p:]).sum())
        validation_results = {
            "n_vars": float(fit.n_vars),
            "n_recession_months": float(n_rec),
            "n_expansion_months": float(n_exp),
            "max_eig_modulus": float(eig),
            "weak_identification_flagged":
                1.0 if diag.get("weak_identification") else 0.0,
            "eig_spread": float(diag.get("eig_spread", float("nan"))),
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
            "variables": list(data.columns),
            "substitution": "US recessions for crisis regime (orig: EM sovereign bond crises)",
        },
        "regime": {
            "n_recession_months": int(recession[fit.p:].sum()),
            "n_expansion_months": int((1 - recession[fit.p:]).sum()),
        },
        "model": {
            "class": "var",
            "lags": fit.p,
            "max_eigenvalue_modulus": float(eig),
        },
        "identification_diagnostics": diag,
        "seed": cfg["general"]["random_seed"],
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report — heteroskedasticity-based id (primary) with
    #      Cholesky (alternative identification) shown as robustness ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + Heteroskedasticity-based identification",
        "Variables": ", ".join(var_labels),
        "Identification": "Regime-based heteroskedasticity (Rigobon 2003)",
        "Regimes": "Recession vs expansion (US NBER as substitution for EM crises)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Recession months": str(int(recession[fit.p:].sum())),
        "Expansion months": str(int((1 - recession[fit.p:]).sum())),
        "Lag order": str(fit.p),
        "Max |eigenvalue|": f"{float(eig):.4f}",
        "Near-equal variance-ratio flag (heuristic)": "yes" if diag.get("weak_identification") else "no",
        "Eigenvalue spread": f"{diag.get('eig_spread', float('nan')):.4f}",
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_standard_sections(
        fit=fit, irf=irf,
        spec=spec_dict, log=log,
        var_labels=var_labels,
        alt_irf=irf_chol,
        alt_label="Cholesky identification (alternative)",
        baseline_label="Het-ID (Rigobon)",
        shock_idx_for_summary=0,
        x_axis_label="Months",
    narrative="auto",
    )
    report.save(outputs_dir / "report.html")
    print(f"\n      ->{(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")
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
