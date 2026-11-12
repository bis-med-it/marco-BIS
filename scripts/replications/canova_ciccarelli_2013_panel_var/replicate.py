"""Canova-Ciccarelli (2013) G4 country panel VAR replication.

Replicates the methodological taxonomy from Canova-Ciccarelli "Panel
Vector Autoregressive Models: A Survey" on a G4 country panel (US, UK,
JP, DE) × {ipi, cpi, ltr, str}, monthly 1980-2019.

The paper itself is a survey, not a single empirical headline. This
replication demonstrates the three main estimators from the survey's
taxonomy:

  1. Pooled VAR with unit fixed effects (shared lag coefficients).
  2. Pesaran-Smith mean-group estimator (per-unit fits, averaged).
  3. Stacked panel VAR with full cross-unit interdependence — the
     survey's "dynamic interdependencies" model (CC §3.2), estimated
     with large-scale-BVAR shrinkage (§6.1). GVAR-flavoured but not a
     Dees et al. (2007) GVAR (no a-priori weight matrices, cf. §6.4).
     Stacks all units' variables into a single VAR of dimension
     N = 4 × 4 = 16 with Minnesota shrinkage on cross-country blocks
     (lambda_cross_unit = 0.05).

What this script does:

  1. Load the staging CSV (built by prepare_data.py from FRED).
  2. Fit pooled / mean-group / global-VAR estimators.
  3. Cholesky-identify a "domestic monetary tightening" shock —
     last-ordered shock to the short rate (str), within each unit.
  4. Report IRFs at selected horizons; per-unit max-eigenvalue for
     mean-group; cross-unit spillover ratio for global VAR.
  5. Compare pooled vs mean-group lag matrices (key Canova-Ciccarelli
     tradeoff: pooling biases vs heterogeneity-respecting noisy estimates).
  6. Compare own-unit blocks across estimators; quantify cross-country
     spillovers in the global VAR.
  7. Validate against paper_targets.yaml.

NOTES:

  * Pooled / mean-group Cholesky uses pooled-coefficient identification.
    Per-country shocks are NOT separately identified; the IRFs trace the
    response of the "representative" panel unit to a Cholesky-
    orthogonalised monetary shock.
  * Global VAR Cholesky on the full 16-dim stacked system identifies
    16 distinct shocks (one per country-variable pair). The
    "US monetary shock" is just the str shock for the US country block;
    spillovers to other countries' variables are direct outputs.
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import lib  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_panel(cfg: dict) -> pd.DataFrame:
    """Load the staging CSV. Returns a long-format DataFrame."""
    path = PROJECT_ROOT / cfg["data"]["staging_path"]
    if not path.exists():
        sys.stderr.write(
            f"\n*** Staging CSV not found: {path}\n"
            f"Run prepare_data.py first.\n\n"
        )
        raise SystemExit(1)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)
    rep = cfg["replication"]

    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/6] Loading panel...")
    panel_long = load_panel(cfg)
    var_names = [v["name"] for v in cfg["data"]["variables"]]
    panel_long = panel_long.set_index("date")
    # Convert to dict-of-DataFrames keyed by unit
    panel = {
        u: g[var_names].copy()
        for u, g in panel_long.groupby("unit")
    }
    for u, df in panel.items():
        print(f"      {u}: T={len(df)}, range {df.index.min().date()} to {df.index.max().date()}")

    # ---- Pooled VAR ----
    print("\n[2/6] Pooled VAR(4) with unit fixed effects...")
    fit_pool = lib.fit_pooled_var(
        panel, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig_pool = lib.max_eigenvalue_modulus(fit_pool.lag_matrices())
    print(f"      Total observations: {fit_pool.fit_info['T_total']}")
    print(f"      Pooled max|eig|: {eig_pool:.4f} (stable={eig_pool < 1.0})")
    print(f"      Per-unit intercepts:")
    for u in fit_pool.units:
        ic = fit_pool.unit_intercepts[u]
        print(f"        {u}: {ic.round(4).tolist()}")

    # ---- Mean-group ----
    print("\n[3/6] Mean-group estimator (Pesaran-Smith 1995)...")
    fit_mg = lib.fit_mean_group(
        panel, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig_mg = lib.max_eigenvalue_modulus(fit_mg.lag_matrices())
    print(f"      Mean-group max|eig|: {eig_mg:.4f}")
    # Per-unit max|eig| as a heterogeneity indicator
    per_unit_eigs = {
        u: float(lib.max_eigenvalue_modulus(lags))
        for u, lags in fit_mg.unit_lag_matrices().items()
    }
    print(f"      Per-unit max|eig|:")
    for u, e in per_unit_eigs.items():
        flag = " (UNSTABLE)" if e >= 1.0 else ""
        print(f"        {u}: {e:.4f}{flag}")

    # ---- Global VAR (cross-unit interdependence) ----
    print("\n[4/6] Stacked cross-country VAR (cross-unit interdependence; not a Dees et al. GVAR)...")
    gv_cfg = cfg["model"]["global_var"]
    # Need to align indices: all units must share the same DatetimeIndex.
    # The raw FRED series may have unit-specific NaN at edges. Align via
    # intersection of indices.
    common_idx = None
    for u, df_u in panel.items():
        common_idx = df_u.index if common_idx is None else common_idx.intersection(df_u.index)
    panel_aligned = {u: df_u.loc[common_idx] for u, df_u in panel.items()}
    n_aligned = len(common_idx)
    print(f"      Aligned sample: T={n_aligned} months "
          f"({common_idx.min().date()} to {common_idx.max().date()})")
    fit_gv = lib.fit_global_var(
        panel_aligned,
        lags=cfg["model"]["lags"],
        trend=cfg["model"]["trend"],
        lambda_1=gv_cfg["lambda_1"],
        lambda_2=gv_cfg["lambda_2"],
        lambda_cross_unit=gv_cfg["lambda_cross_unit"],
        lambda_3=gv_cfg["lambda_3"],
        lambda_4=gv_cfg["lambda_4"],
    )
    eig_gv = lib.max_eigenvalue_modulus(fit_gv.lag_matrices())
    print(f"      Stacked cross-country VAR dimension: N = "
          f"{fit_gv.fit_info['n_units']} units x "
          f"{fit_gv.fit_info['n_vars_per_unit']} vars = {fit_gv.n_vars}")
    print(f"      Stacked cross-country VAR max|eig|: {eig_gv:.4f} (stable={eig_gv < 1.0})")
    print(f"      Total lag coefficients: {fit_gv.n_vars * fit_gv.n_vars * fit_gv.p}")

    # Per-unit own-block max eigenvalue (within-unit dynamics only)
    print(f"      Per-unit own-block max|eig| (within-country dynamics):")
    for u in fit_gv.units:
        own_lag_blocks = [fit_gv.cross_unit_lag_block(u, u, lag=l + 1)
                           for l in range(fit_gv.p)]
        e_own = float(lib.max_eigenvalue_modulus(own_lag_blocks))
        flag = " (UNSTABLE)" if e_own >= 1.0 else ""
        print(f"        {u}: {e_own:.4f}{flag}")

    # Average cross-unit spillover ratio per lag
    print(f"      Cross-unit spillover ratios (avg Frobenius "
          f"cross-block / own-block) per lag:")
    spillover_ratios = {}
    for l in range(1, fit_gv.p + 1):
        ratio = fit_gv.average_cross_unit_spillover(lag=l)
        spillover_ratios[l] = ratio
        print(f"        lag {l}: {ratio:.4f}")

    # ---- Cross-country IRFs: US monetary shock -> foreign output ----
    # Identify a contractionary US monetary shock as the Cholesky shock to
    # US_str (the str variable in the US block). Since the global VAR is
    # one stacked VAR, this is just the str-equation Cholesky shock.
    print(f"\n      Cross-country IRFs: US monetary tightening (+US_str shock)")
    print(f"      -> foreign output responses (peak |response| in 24m):")
    # Reorder columns so US comes first and str is last within each unit's block
    # (Cholesky orders within US first as the policy block).
    units_order = ["US", "UK", "JP", "DE"]
    var_order_per_unit = ["ipi", "cpi", "ltr", "str"]
    stacked_order = [f"{u}_{v}" for u in units_order for v in var_order_per_unit]
    # Rebuild GVAR with this stacked variable order
    panel_for_gv = {u: panel_aligned[u][var_order_per_unit].copy() for u in units_order}
    fit_gv_ordered = lib.fit_global_var(
        panel_for_gv,
        lags=cfg["model"]["lags"],
        trend=cfg["model"]["trend"],
        lambda_1=gv_cfg["lambda_1"],
        lambda_2=gv_cfg["lambda_2"],
        lambda_cross_unit=gv_cfg["lambda_cross_unit"],
        lambda_3=gv_cfg["lambda_3"],
        lambda_4=gv_cfg["lambda_4"],
    )
    irf_gv = lib.cholesky_irf(
        fit_gv_ordered, horizon=cfg["irf"]["horizon"], bands="none",
    )
    us_str_shock_idx = fit_gv_ordered.variable_names.index("US_str")
    print(f"        {'country':>10}  {'ipi peak |r|':>14}  "
          f"{'cpi peak |r|':>14}  {'str peak |r|':>14}")
    cross_country_responses: dict = {}
    for u in units_order:
        row = {"unit": u}
        for v in ("ipi", "cpi", "str"):
            v_idx = fit_gv_ordered.variable_names.index(f"{u}_{v}")
            path = irf_gv.median[:, v_idx, us_str_shock_idx]
            peak_abs = float(np.max(np.abs(path)))
            peak_h = int(np.argmax(np.abs(path)))
            row[f"{v}_peak_abs"] = peak_abs
            row[f"{v}_peak_h"] = peak_h
        cross_country_responses[u] = row
        print(f"        {u:>10}  "
              f"{row['ipi_peak_abs']:>+13.5f} (h={row['ipi_peak_h']:2d})  "
              f"{row['cpi_peak_abs']:>+13.5f} (h={row['cpi_peak_h']:2d})  "
              f"{row['str_peak_abs']:>+13.5f} (h={row['str_peak_h']:2d})")

    # ---- Pooled vs mean-group lag-1 coefficient comparison ----
    print("\n[5/6] Pooled vs Mean-Group lag-1 coefficient matrix (B_1)...")
    pool_B1 = fit_pool.lag_matrices()[0]
    mg_B1 = fit_mg.lag_matrices()[0]
    diff_B1 = np.abs(pool_B1 - mg_B1)
    print(f"      Max |Pooled[i,j] - MeanGroup[i,j]|: {diff_B1.max():.4f}")
    print(f"      Mean |Pooled[i,j] - MeanGroup[i,j]|: {diff_B1.mean():.4f}")
    print(f"      Frobenius distance: {np.linalg.norm(pool_B1 - mg_B1):.4f}")
    print(f"      (Small distance => panel is roughly homogeneous; large => substantial heterogeneity)")

    # ---- IRF: Cholesky on pooled fit ----
    print(f"\n[6/6] Cholesky IRF (pooled fit), ordering {cfg['identification']['ordering']}...")
    # Reorder data columns to match the Cholesky ordering
    ordering = cfg["identification"]["ordering"]
    if ordering != var_names:
        # Re-fit pooled VAR with reordered columns
        panel_reordered = {u: df[ordering] for u, df in panel.items()}
        fit_pool = lib.fit_pooled_var(
            panel_reordered, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
        )
        var_names = ordering

    irf = lib.cholesky_irf(
        fit_pool,
        horizon=cfg["irf"]["horizon"],
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
    )
    shock_of_interest = cfg["identification"]["shock_of_interest"]
    shock_idx = var_names.index(shock_of_interest)

    print(f"\n      Response to a +1 unit shock to {shock_of_interest.upper()}"
          f" (last-ordered = 'monetary tightening'):")
    print("      Point estimates only: single-VAR bands are not valid for "
          "pooled panel fits (no panel bootstrap implemented).")
    horizons_to_report = cfg["report"]["horizons_to_report"]
    print(f"        {'h':>3}  {'variable':<10}  {'point':>10}")
    for h in horizons_to_report:
        for v_idx, var in enumerate(var_names):
            m = irf.median[h, v_idx, shock_idx]
            print(f"        {h:>3}  {var:<10}  {m:>+10.5f}")

    # ---- Paper-target validation ----
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        str_h0 = float(irf.median[0, var_names.index("str"), shock_idx])
        ipi_h12 = float(irf.median[12, var_names.index("ipi"), shock_idx])
        validation_results = {
            "pooled_max_eig": float(eig_pool),
            "mg_max_eig": float(eig_mg),
            "n_units": float(fit_pool.n_units),
            "total_observations": float(fit_pool.fit_info["T_total"]),
            "pool_vs_mg_max_diff": float(diff_B1.max()),
            # Sign indicators for IRF directional checks
            "str_h0_response_positive": 1.0 if str_h0 > 0 else 0.0,
            "ipi_h12_sign": -1.0 if ipi_h12 < 0 else 1.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        print("\n" + comp_report.table())

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log = {
        "replication_id": rep["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "n_units": fit_pool.n_units,
            "T_per_unit": fit_pool.fit_info["T_per_unit"],
            "T_total": fit_pool.fit_info["T_total"],
            "variables": var_names,
        },
        "model": {
            "class": cfg["model"]["class"],
            "lags": fit_pool.p, "trend": cfg["model"]["trend"],
            "estimators": cfg["model"]["estimators"],
        },
        "pooled": {
            "max_eigenvalue_modulus": float(eig_pool),
            "unit_intercepts": {
                u: fit_pool.unit_intercepts[u].tolist()
                for u in fit_pool.units
            },
        },
        "mean_group": {
            "max_eigenvalue_modulus": float(eig_mg),
            "per_unit_max_eig": per_unit_eigs,
        },
        "global_var": {
            "max_eigenvalue_modulus": float(eig_gv),
            "n_stacked_variables": int(fit_gv.n_vars),
            "n_total_lag_coefficients": int(
                fit_gv.n_vars * fit_gv.n_vars * fit_gv.p
            ),
            "lambda_1": gv_cfg["lambda_1"],
            "lambda_2": gv_cfg["lambda_2"],
            "lambda_cross_unit": gv_cfg["lambda_cross_unit"],
            "lambda_3": gv_cfg["lambda_3"],
            "lambda_4": gv_cfg["lambda_4"],
            "per_unit_own_block_max_eig": {
                u: float(lib.max_eigenvalue_modulus([
                    fit_gv.cross_unit_lag_block(u, u, lag=l + 1)
                    for l in range(fit_gv.p)
                ]))
                for u in fit_gv.units
            },
            "cross_unit_spillover_ratio_by_lag": spillover_ratios,
            "us_monetary_shock_cross_country_responses": cross_country_responses,
        },
        "pool_vs_mg": {
            "max_abs_diff_B1": float(diff_B1.max()),
            "mean_abs_diff_B1": float(diff_B1.mean()),
            "frobenius_diff_B1": float(np.linalg.norm(pool_B1 - mg_B1)),
        },
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report — pooled VAR IRF as the canonical view, with
    #      mean-group / global-VAR comparison summarised in spec table ----
    spec_dict = {
        "Model": f"Panel VAR ({fit_pool.p} lags) — pool + mean-group + stacked cross-country VAR",
        "Units": ", ".join(fit_pool.units),
        "Variables (per unit)": ", ".join(var_names),
        "Sample observations (T per unit)": str(fit_pool.fit_info["T_per_unit"]),
        "Total observations (T × N)": str(fit_pool.fit_info["T_total"]),
        "Pooled max |eig|": f"{float(eig_pool):.4f}",
        "Mean-group max |eig|": f"{float(eig_mg):.4f}",
        "Stacked cross-country VAR max |eig|": f"{float(eig_gv):.4f}",
        "Pool vs MG: max |B1 diff|": f"{float(diff_B1.max()):.4f}",
        "Pool vs MG: Frobenius |B1 diff|": f"{float(np.linalg.norm(pool_B1 - mg_B1)):.4f}",
        "Stacked cross-country VAR dimension": str(int(fit_gv.n_vars)),
        "IRF horizon": str(cfg["irf"]["horizon"]),
        "Shock of interest": shock_of_interest,
    }
    report_html = lib.VARReport(
        title=rep.get("title", "Canova-Ciccarelli 2013 panel VAR"),
        subtitle=rep.get("subtitle",
            "Pooled VAR (this section), mean-group, and stacked cross-country VAR estimates "
            "compared in the research log"),
    )
    report_html.add_standard_sections(
        fit=fit_pool, irf=irf,
        spec=spec_dict, log=log,
        var_labels=var_names,
        shock_idx_for_summary=shock_idx,
        x_axis_label="Months",
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2, default=str))
    irf.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
