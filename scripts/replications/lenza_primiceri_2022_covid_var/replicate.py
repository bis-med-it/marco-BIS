"""Lenza-Primiceri (2022) — BVAR estimation with COVID-period variance scaling.

Compares three estimation strategies on a BVAR that spans the 2020 pandemic:

  1. **Naïve**: fit on the full sample (1960Q1 - 2024Q4) without any
     special handling. The 2020Q2 outliers distort parameter estimates.
  2. **Drop COVID**: fit on the sample with 2020Q2-2020Q3 removed (the
     comparator unaffected by the pandemic quarters).
  3. **LP-scaled**: fit on the full sample with per-observation scale
     factors ``s_t`` that down-weight COVID-period innovations via GLS.
     All observations are kept; the fixed scale path does not by itself
     model future forecast uncertainty (see README).

The LP scaling parameterisation (per the dictionary entry):

    s_t = s_bar         for t in [2020Q2, 2020Q3, 2020Q4]
    s_t = 1 + (s_bar - 1) * rho^(t - 2020Q4)   for t > 2020Q4
    s_t = 1             for t pre-2020Q2

with ``(s_bar, rho)`` fixed at demonstration values (``s_bar = 5``,
``rho = 0.8``). The paper instead estimates a monthly scale path
(posterior peaks ~17/65/20 in March-May 2020); see the README
§"Caveats".
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

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402


def build_lp_scale(
    data_index: pd.DatetimeIndex,
    lags: int,
    s_bar: float,
    rho: float,
    start: str,
    plateau_end: str,
) -> np.ndarray:
    """Build the per-observation scale vector aligned with ``Y_act`` rows.

    ``Y_act`` has length ``T_eff = T - lags`` and corresponds to dates
    ``data_index[lags:]``. The scaling kicks in at ``start``, plateaus
    through ``plateau_end``, then decays geometrically.
    """
    eff_dates = data_index[lags:]
    start_ts = pd.Timestamp(start)
    plateau_end_ts = pd.Timestamp(plateau_end)
    s = np.ones(len(eff_dates))
    for i, d in enumerate(eff_dates):
        if d < start_ts:
            s[i] = 1.0
        elif d <= plateau_end_ts:
            s[i] = s_bar
        else:
            # Quarters since plateau_end
            k = (d.year - plateau_end_ts.year) * 4 + (d.month - plateau_end_ts.month) // 3
            s[i] = 1.0 + (s_bar - 1.0) * (rho ** k)
    return s


def main() -> int:
    spec = lib.Spec.from_replication("lenza_primiceri_2022_covid_var")
    seed = int(spec.metadata["general"]["random_seed"])
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    data = lib.load_data(spec)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Variables: {list(data.columns)}")

    lags = spec.model["lags"]
    trend = spec.model["trend"]
    prior_dict = spec.model["prior"]
    prior = lib.MinnesotaPrior(
        lambda_1=prior_dict["lambda_1"], lambda_2=prior_dict["lambda_2"],
        lambda_3=prior_dict["lambda_3"], lambda_4=prior_dict["lambda_4"],
        mu=prior_dict["mu"], delta=prior_dict["delta"],
    )

    # ---- Setup 1: Naïve BVAR on full sample ----
    print("\n[2/5] Setup 1: Naïve BVAR (no COVID handling)...")
    fit_naive = lib.fit_minnesota_bvar(data, lags=lags, prior=prior, trend=trend)
    mdd_naive = lib.log_marginal_data_density(fit_naive)
    eig_naive = lib.max_eigenvalue_modulus(fit_naive.lag_matrices())
    print(f"      log MDD = {mdd_naive:.4f}, max|eig| = {eig_naive:.4f}")
    # Headline diagnostic: residual std at COVID quarter
    resid_naive = fit_naive.residuals
    covid_idx = np.where(data.index[lags:] == pd.Timestamp("2020-04-01"))[0]
    if len(covid_idx) > 0:
        i = int(covid_idx[0])
        print(f"      2020Q2 residual (rgdp): {resid_naive[i, 0]:+.4f}")

    # ---- Setup 2: Drop COVID observations (2020Q2 - 2020Q3) ----
    print("\n[3/5] Setup 2: BVAR with COVID observations dropped...")
    drop_mask = (data.index < pd.Timestamp("2020-04-01")) | (data.index > pd.Timestamp("2020-10-01"))
    data_drop = data[drop_mask]
    n_dropped = int((~drop_mask).sum())
    print(f"      Dropped {n_dropped} observations (2020Q2 - 2020Q3).")
    fit_drop = lib.fit_minnesota_bvar(data_drop, lags=lags, prior=prior, trend=trend)
    mdd_drop = lib.log_marginal_data_density(fit_drop)
    eig_drop = lib.max_eigenvalue_modulus(fit_drop.lag_matrices())
    print(f"      log MDD = {mdd_drop:.4f}, max|eig| = {eig_drop:.4f}")

    # ---- Setup 3: LP-scaled BVAR ----
    cs_cfg = spec.metadata["covid_scaling"]
    print(f"\n[4/5] Setup 3: LP-scaled BVAR (s_bar={cs_cfg['s_bar']}, "
          f"rho={cs_cfg['rho']})...")
    obs_scale = build_lp_scale(
        data_index=data.index, lags=lags,
        s_bar=cs_cfg["s_bar"], rho=cs_cfg["rho"],
        start=cs_cfg["start"], plateau_end=cs_cfg["plateau_end"],
    )
    n_scaled = int(np.sum(obs_scale > 1.001))
    print(f"      Scaling kicks in for {n_scaled} observations starting "
          f"{cs_cfg['start']}.")
    print(f"      First 5 scaled values: "
          f"{obs_scale[obs_scale > 1.001][:5].round(3).tolist()}")
    fit_lp = lib.fit_minnesota_bvar(
        data, lags=lags, prior=prior, trend=trend, obs_scale=obs_scale,
    )
    # Note: log_mdd on scaled fit isn't directly comparable to unscaled
    # without the Jacobian correction. Compute both for completeness.
    mdd_lp_scaled = lib.log_marginal_data_density(fit_lp)
    n_vars = data.shape[1]
    jacobian = -n_vars * np.sum(np.log(obs_scale))
    mdd_lp_original_scale = mdd_lp_scaled + jacobian
    eig_lp = lib.max_eigenvalue_modulus(fit_lp.lag_matrices())
    print(f"      log MDD (on scaled data): {mdd_lp_scaled:.4f}")
    print(f"      log MDD (Jacobian-corrected, full-sample scale; "
          f"do not compare with drop-sample MDD): "
          f"{mdd_lp_original_scale:.4f}")
    print(f"      max|eig| = {eig_lp:.4f}")

    # ---- Compare own-lag-1 coefficients across setups ----
    print("\n[5/5] Parameter comparison — own-lag-1 coefficients:")
    print(f"      {'variable':<10}{'naive':>10}{'drop':>10}{'LP-scaled':>12}{'naive-drop gap':>18}")
    coef_rows = []
    A1_n = fit_naive.lag_matrices()[0]
    A1_d = fit_drop.lag_matrices()[0]
    A1_l = fit_lp.lag_matrices()[0]
    for j, var in enumerate(data.columns):
        n_val = float(A1_n[j, j])
        d_val = float(A1_d[j, j])
        l_val = float(A1_l[j, j])
        gap = abs(n_val - d_val)
        print(f"      {var:<10}{n_val:>10.4f}{d_val:>10.4f}{l_val:>12.4f}{gap:>18.4f}")
        coef_rows.append({"variable": var, "naive": n_val, "drop": d_val,
                          "lp_scaled": l_val, "naive_minus_drop": n_val - d_val})

    # Average parameter shift
    avg_shift_naive = float(np.mean([abs(r["naive_minus_drop"]) for r in coef_rows]))
    avg_shift_lp = float(np.mean(
        [abs(r["lp_scaled"] - r["drop"]) for r in coef_rows]
    ))
    print(f"\n      Avg |naive - drop|     = {avg_shift_naive:.4f}  (COVID distortion)")
    print(f"      Avg |LP-scaled - drop| = {avg_shift_lp:.4f}  "
          f"(LP recovers drop-style parameters)")

    # ---- Compare residual std-deviation ratios (forecast-uncertainty proxy) ----
    print(f"\n      Residual variance comparison "
          f"(higher => wider forecast bands going forward):")
    print(f"        {'variable':<10}{'naive':>10}{'drop':>10}{'LP-scaled':>12}")
    std_rows = []
    sig_n = np.sqrt(np.diag(fit_naive.Sigma_u))
    sig_d = np.sqrt(np.diag(fit_drop.Sigma_u))
    sig_l = np.sqrt(np.diag(fit_lp.Sigma_u))
    for j, var in enumerate(data.columns):
        print(f"        {var:<10}{sig_n[j]:>10.5f}{sig_d[j]:>10.5f}{sig_l[j]:>12.5f}")
        std_rows.append({"variable": var, "naive": float(sig_n[j]),
                          "drop": float(sig_d[j]), "lp_scaled": float(sig_l[j])})

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log = {
        "replication_id": rep["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": spec.data["source"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
            "variables": list(data.columns),
        },
        "model": {
            "class": "bvar", "lags": int(lags), "trend": trend,
            "prior": prior_dict,
        },
        "covid_scaling": dict(cs_cfg),
        "naive": {
            "log_mdd": float(mdd_naive),
            "max_eig": float(eig_naive),
            "sigma_u_diag": [float(s) for s in sig_n],
        },
        "drop_covid": {
            "n_dropped": n_dropped,
            "log_mdd": float(mdd_drop),
            "max_eig": float(eig_drop),
            "sigma_u_diag": [float(s) for s in sig_d],
        },
        "lp_scaled": {
            "n_scaled": n_scaled,
            "log_mdd_scaled_data": float(mdd_lp_scaled),
            "log_mdd_jacobian_corrected": float(mdd_lp_original_scale),
            "max_eig": float(eig_lp),
            "sigma_u_diag": [float(s) for s in sig_l],
        },
        "own_lag_1_coefficients": coef_rows,
        "residual_std_deviations": std_rows,
        "summary": {
            "avg_abs_shift_naive_vs_drop": avg_shift_naive,
            "avg_abs_shift_lp_vs_drop": avg_shift_lp,
        },
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # urate-specific own-lag-1 coefficients
        urate_row = next(
            (r for r in coef_rows if r["variable"] == "urate"), None,
        )
        if urate_row is not None:
            naive_u = float(urate_row["naive"])
            drop_u = float(urate_row["drop"])
            lp_u = float(urate_row["lp_scaled"])
        else:
            naive_u = drop_u = lp_u = float("nan")
        validation_results = {
            "naive_urate_lag1": naive_u,
            "drop_urate_lag1": drop_u,
            "lp_urate_lag1": lp_u,
            "lp_close_to_drop_compared_to_naive":
                1.0 if avg_shift_lp <= 0.25 * avg_shift_naive else 0.0,
            "n_dropped": float(n_dropped),
            "all_stable": 1.0 if (
                eig_naive < 1.0 and eig_drop < 1.0 and eig_lp < 1.0
            ) else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — three BVAR fits compared on own-lag coefs +
    #      residual std deviations; no IRFs computed in this replication) ----
    spec_dict = {
        "Model": f"BVAR({lags}) with COVID observation scaling (Lenza-Primiceri 2022)",
        "Variables": ", ".join(list(data.columns)),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(lags),
        "Comparison": "naïve / drop-COVID / LP-scaled",
        "Naïve log MDD": f"{float(mdd_naive):.4f}",
        "Drop-COVID log MDD": f"{float(mdd_drop):.4f}",
        "LP-scaled log MDD (Jacobian-corrected)": f"{float(mdd_lp_original_scale):.4f}",
        "Avg |shift| naïve vs drop": f"{avg_shift_naive:.4f}",
        "Avg |shift| LP-scaled vs drop": f"{avg_shift_lp:.4f}",
        "Ratio (LP-shift / naïve-shift)": (
            f"{avg_shift_lp / avg_shift_naive:.3f}" if avg_shift_naive > 0 else "n/a"
        ),
        "Naïve stable": "yes" if eig_naive < 1.0 else "no",
        "Drop-COVID stable": "yes" if eig_drop < 1.0 else "no",
        "LP-scaled stable": "yes" if eig_lp < 1.0 else "no",
        "COVID observations downweighted": str(n_scaled),
    }
    report_html = lib.VARReport(
        title=rep.get("title", "Lenza-Primiceri 2022 COVID-VAR"),
        subtitle=rep.get("subtitle",
            "Prior + obs_scale comparison (naïve / drop / LP-scaled); see "
            "research log JSON for full coefficient and residual tables"),
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
