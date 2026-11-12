"""
Replication of Galí (1999) — do technology shocks explain aggregate
fluctuations?

Bivariate VAR on (Δlog productivity, Δlog hours) with the long-run
restriction that only technology shocks may have a permanent effect on the
LEVEL of labour productivity (AER 89(1), pp. 255-257). Productivity is
constructed as log GDP minus log hours, mirroring the paper (p. 258).

Reproduces the paper's three headline exhibits:
  1. Conditional correlations of productivity and hours growth (Table 1,
     eq. 24) — negative conditional on technology, positive conditional on
     non-technology shocks.
  2. Impulse responses (Figure 2): hours FALL after a positive technology
     shock; productivity rises ~0.6 percent on impact.
  3. Historical decomposition with HP-filtered components (Section IV,
     Figure 6): the technology components of GDP and hours do not comove;
     the non-technology components do.

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


def build_variables(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load GDP + hours levels and construct the paper's VAR variables.

    Galí constructs productivity himself: x = log GDP - log hours (p. 258).
    Returns (VAR data [dlog_prod, dlog_hours] in percent, log hours level
    in percent for the detrended-hours variant).
    """
    raw = lib.load_dataset(
        source=cfg["data"]["source"],
        series={s["name"]: s["fred_code"] for s in cfg["data"]["series"]},
        start=cfg["data"]["sample_start"],
        end=cfg["data"]["sample_end"],
    )
    log_gdp = 100.0 * np.log(raw["gdp"])
    log_hours = 100.0 * np.log(raw["hours"])
    data = pd.DataFrame(
        {
            "dlog_prod": (log_gdp - log_hours).diff(),
            "dlog_hours": log_hours.diff(),
        }
    ).dropna()
    return data, log_hours


def conditional_corr(irf_point: np.ndarray, j: int) -> float:
    """Paper eq. (24): model-implied correlation of variables 1 and 2
    conditional on shock j, from the structural MA coefficients
    (infinite sums truncated at the IRF horizon, cf. paper fn. 16)."""
    c1 = irf_point[:, 0, j]
    c2 = irf_point[:, 1, j]
    return float((c1 * c2).sum() / np.sqrt((c1 ** 2).sum() * (c2 ** 2).sum()))


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/6] Loading data and constructing variables...")
    data, log_hours = build_variables(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print("      Productivity constructed as log GDP - log hours (paper p. 258)")
    corr_uncond = float(data["dlog_prod"].corr(data["dlog_hours"]))
    print(f"      Unconditional corr(dx, dn) = {corr_uncond:+.3f}"
          f"  (paper Table 1: -0.26)")

    print("\n[2/6] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, "
          f"T_eff={fit.t_eff}, max |eigenvalue|={eig:.4f}")

    print("\n[3/6] Long-run identification (only technology moves productivity"
          " permanently)...")
    sign_norm = cfg["identification"]["sign_normalization"]
    shock_names = cfg["identification"]["shock_names"]
    irf = lib.long_run_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        shock_names=shock_names,
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
        sign_normalization=sign_norm,
    )
    print(f"      IRF horizon: 0..{irf.horizon}, "
          f"{irf.n_vars} variables x {irf.n_shocks} shocks, "
          f"{cfg['irf']['n_draws']} draws")

    # Point IRF at a long horizon for the eq.-(24) conditional correlations
    # and the historical-decomposition impact matrix.
    trunc = cfg["conditional_correlations"]["truncation_horizon"]
    irf_point = lib.long_run_irf(
        fit=fit, horizon=trunc, shock_names=shock_names,
        bands="none", sign_normalization=sign_norm,
    )
    prod_impact = float(irf_point.median[0, 0, 0])
    hours_impact = float(irf_point.median[0, 1, 0])
    print(f"      Technology shock impact (1 s.d.): productivity "
          f"{prod_impact:+.3f}%  (paper: ~+0.6%), hours {hours_impact:+.3f}%"
          f"  (paper: negative)")

    print("\n[4/6] Conditional correlations of dx and dn (paper eq. 24,"
          f" sums truncated at h={trunc})...")
    corr_tech = conditional_corr(irf_point.median, 0)
    corr_nontech = conditional_corr(irf_point.median, 1)
    print("      shock           ours    paper (Table 1, 48:1-94:4)")
    print(f"      technology     {corr_tech:+.3f}   -0.82 (s.e. 0.12)")
    print(f"      nontechnology  {corr_nontech:+.3f}   +0.26 (s.e. 0.12)")
    print(f"      unconditional  {corr_uncond:+.3f}   -0.26 (s.e. 0.08)")

    print("\n[5/6] Robustness: detrended-hours variant (Table 1 Panel B)...")
    det = cfg.get("detrended_variant", {}).get("enabled", False)
    corr_tech_det = corr_nontech_det = eig_det = None
    if det:
        t = np.arange(len(log_hours), dtype=float)
        slope, intercept = np.polyfit(t, log_hours.to_numpy(dtype=float), 1)
        hours_detrended = log_hours - (intercept + slope * t)
        data_det = pd.DataFrame(
            {
                "dlog_prod": data["dlog_prod"],
                "hours_detrended": hours_detrended,
            }
        ).dropna()
        fit_det = lib.fit_var(
            data=data_det, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
        )
        eig_det = float(lib.max_eigenvalue_modulus(fit_det.lag_matrices()))
        irf_det = lib.long_run_irf(
            fit=fit_det, horizon=trunc, shock_names=shock_names,
            bands="none",
            sign_normalization={
                "technology": {"dlog_prod": +1},
                "nontechnology": {"hours_detrended": +1},
            },
        )
        # Hours enter in (detrended) LEVELS here; eq. (24) correlates GROWTH
        # rates, so difference the hours-level responses (impact diff = impact).
        m = irf_det.median.copy()
        m[:, 1, :] = np.diff(m[:, 1, :], axis=0, prepend=0.0)
        corr_tech_det = conditional_corr(m, 0)
        corr_nontech_det = conditional_corr(m, 1)
        print(f"      VAR({fit_det.p}) on [dlog_prod, detrended log hours], "
              f"max |eigenvalue|={eig_det:.4f}")
        print(f"      technology     {corr_tech_det:+.3f}   -0.81 (s.e. 0.11)")
        print(f"      nontechnology  {corr_nontech_det:+.3f}   +0.35 (s.e. 0.10)")

    print("\n[6/6] Historical decomposition, HP-filtered components"
          " (Section IV, Figure 6)...")
    S = np.asarray(irf_point.median[0], dtype=float)        # structural impact
    hd = lib.historical_decomposition(fit, S, shock_names=shock_names)
    lam = cfg["historical_decomposition"]["hp_lambda"]
    hp_corr = {}
    for shock in shock_names:
        contrib = hd.shocks[shock]                          # growth-rate contributions
        # y = x + n  =>  technology component of GDP growth is the sum of the
        # productivity-growth and hours-growth contributions; cumulate to levels.
        y_comp = (contrib["dlog_prod"] + contrib["dlog_hours"]).cumsum()
        n_comp = contrib["dlog_hours"].cumsum()
        y_cycle = lib.hp_filter(y_comp, lam=lam).cycle
        n_cycle = lib.hp_filter(n_comp, lam=lam).cycle
        hp_corr[shock] = float(y_cycle.corr(n_cycle))
    recon_err = float(hd.reconstruction_error().max())
    # Actual-data comovement for reference: the paper's point is that the
    # nontechnology components account for the bulk of it (pp. 266-267).
    y_act = (data["dlog_prod"] + data["dlog_hours"]).cumsum()
    n_act = data["dlog_hours"].cumsum()
    hp_corr_actual = float(
        lib.hp_filter(y_act, lam=lam).cycle.corr(lib.hp_filter(n_act, lam=lam).cycle)
    )
    print(f"      corr(GDP component, hours component), HP-filtered (lambda={lam}):")
    print(f"      technology     {hp_corr['technology']:+.3f}   -0.02 (paper p. 266;"
          f" lag-sensitive here, see README)")
    print(f"      nontechnology  {hp_corr['nontechnology']:+.3f}   +0.97 (paper p. 266)")
    print(f"      actual data    {hp_corr_actual:+.3f}")
    print(f"      decomposition reconstruction error: {recon_err:.2e}")

    print("\n      Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    labels = cfg["data"]["variable_labels"]
    var_labels = [labels.get(v, v) for v in irf.variable_names]

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        validation_results = {
            "n_vars": float(irf.median.shape[1]),
            "max_eig_modulus": float(eig),
            "corr_tech_fd": corr_tech,
            "corr_nontech_fd": corr_nontech,
            "corr_uncond_fd": corr_uncond,
            "prod_impact_tech": prod_impact,
            "hours_impact_negative": 1.0 if hours_impact < 0 else 0.0,
            "hp_corr_nontech": hp_corr["nontechnology"],
            "hd_nontech_dominates_comovement":
                1.0 if (hp_corr["nontechnology"] - hp_corr["technology"]) > 0.25
                else 0.0,
        }
        if corr_tech_det is not None:
            validation_results["corr_tech_detrended"] = corr_tech_det
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
            "construction": "dlog_prod = 100*dlog(GDPC1/HOANBS); "
                            "dlog_hours = 100*dlog(HOANBS) — paper p. 258",
        },
        "model": {
            "class": cfg["model"]["class"],
            "lags": fit.p,
            "lag_order_note": "not stated in the published paper; "
                              "4 = replication-literature convention",
            "fit_info": fit.fit_info,
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": dict(cfg["identification"]),
        "conditional_correlations": {
            "truncation_horizon": trunc,
            "technology": corr_tech,
            "nontechnology": corr_nontech,
            "unconditional": corr_uncond,
        },
        "detrended_variant": (
            {
                "technology": corr_tech_det,
                "nontechnology": corr_nontech_det,
                "max_eigenvalue_modulus": eig_det,
            }
            if det else None
        ),
        "historical_decomposition": {
            "hp_lambda": lam,
            "corr_components_technology": hp_corr["technology"],
            "corr_components_nontechnology": hp_corr["nontechnology"],
            "corr_actual_data": hp_corr_actual,
            "reconstruction_error": recon_err,
        },
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p})",
        "Variables": ", ".join(var_labels),
        "Identification": "Long-run restriction (Galí 1999)",
        "Restriction": "Only technology shocks have a permanent effect on "
                       "the level of labour productivity",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": f"{fit.p} (not stated in the published paper)",
        "Max |eigenvalue|": f"{float(eig):.4f}",
        "Conditional corr (technology)": f"{corr_tech:+.3f} (paper: -0.82)",
        "Conditional corr (non-technology)": f"{corr_nontech:+.3f} (paper: +0.26)",
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
