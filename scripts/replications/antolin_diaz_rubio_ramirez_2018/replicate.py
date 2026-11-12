"""Antolín-Díaz & Rubio-Ramírez (2018) — Narrative Sign Restrictions for SVARs.

Demonstrates the ADRR method on a 4-variable US monthly monetary VAR:
identify a contractionary monetary policy shock via (i) standard sign
restrictions on the impact responses and (ii) narrative sign restrictions
at known historical tightening dates (Volcker October 1979, late-Greenspan
May 2000).

The qualitative claim from the paper: narrative restrictions substantially
sharpen identification — acceptance rate drops and confidence bands
narrow compared with pure sign restrictions.

This replication uses a simplified spec relative to the paper's oil-market
illustration (Section IV). It is a rejection-only narrative-restriction
adaptation: narrative checks reject QR rotations, and the paper's
importance weighting is omitted (see README).
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


def main() -> int:
    spec = lib.Spec.from_replication("antolin_diaz_rubio_ramirez_2018")
    seed = int(spec.metadata["general"]["random_seed"])
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/4] Loading data via lib.load_data...")
    data = lib.load_data(spec)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months, {data.shape[1]} variables)")
    print(f"      Variables: {list(data.columns)}")

    print("\n[2/4] Fitting reduced-form VAR...")
    fit = lib.fit_var(data, lags=spec.model["lags"], trend=spec.model["trend"])
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), T_eff={fit.t_eff}, max|eig|={eig:.4f}")

    ident = spec.metadata.get("identification") or spec.data.get("identification") \
            or {}
    # `identification` lives at top level in the YAML so it folds into metadata
    ident = spec.metadata["identification"]
    horizon = int(ident["horizon"])
    sign_rs = ident["sign_restrictions"]
    rh = int(ident["restriction_horizons"])
    narr = ident["narrative_restrictions"]
    target = int(ident["target_accepted"])
    max_rot = int(ident["max_rotations"])

    # ---- [3/4] Sign-only baseline ----
    print(f"\n[3/4] Sign-only identification (Uhlig-style baseline)...")
    irf_base = lib.sign_restricted_irf(
        fit=fit, horizon=horizon,
        sign_restrictions=sign_rs,
        restriction_horizons=rh,
        target_accepted=target, max_rotations=max_rot,
        seed=seed, parameter_uncertainty=True,
    )
    print(f"      n_accepted: {irf_base.metadata['n_accepted']}, "
          f"acceptance_rate: {irf_base.metadata['acceptance_rate']:.4f}")
    # Headline: 68% band half-width on IP and CPI on impact
    shock_idx = 0
    ip_idx = fit.variable_names.index("ip")
    cpi_idx = fit.variable_names.index("cpi")
    hw_ip_0_base = float((irf_base.upper_68[0, ip_idx, shock_idx]
                           - irf_base.lower_68[0, ip_idx, shock_idx]) / 2)
    hw_cpi_0_base = float((irf_base.upper_68[0, cpi_idx, shock_idx]
                            - irf_base.lower_68[0, cpi_idx, shock_idx]) / 2)
    print(f"      68% band half-width at h=0: ip={hw_ip_0_base:.4f}, "
          f"cpi={hw_cpi_0_base:.4f}")

    # ---- [4/4] Sign + narrative restrictions ----
    print(f"\n[4/4] Sign + narrative identification (ADRR-style)...")
    print(f"      Narrative restrictions:")
    for nr in narr:
        kind = nr.get("type", "sign")
        if kind == "sign":
            print(f"        {nr['date']}  Type 1: shock sign = {nr['sign']:+d}")
        else:
            print(f"        {nr['date']}  Type 2 ({kind}): variable = {nr['variable']}")
    irf_narr = lib.sign_restricted_irf(
        fit=fit, horizon=horizon,
        sign_restrictions=sign_rs,
        restriction_horizons=rh,
        narrative_restrictions=narr,
        target_accepted=target, max_rotations=max_rot,
        seed=seed, parameter_uncertainty=True,
    )
    print(f"\n      n_accepted: {irf_narr.metadata['n_accepted']}, "
          f"acceptance_rate: {irf_narr.metadata['acceptance_rate']:.4f}")
    print(f"      n_rejected_by_narrative: "
          f"{irf_narr.metadata['n_rejected_by_narrative']}")
    hw_ip_0_narr = float((irf_narr.upper_68[0, ip_idx, shock_idx]
                           - irf_narr.lower_68[0, ip_idx, shock_idx]) / 2)
    hw_cpi_0_narr = float((irf_narr.upper_68[0, cpi_idx, shock_idx]
                            - irf_narr.lower_68[0, cpi_idx, shock_idx]) / 2)
    print(f"      68% band half-width at h=0: ip={hw_ip_0_narr:.4f}, "
          f"cpi={hw_cpi_0_narr:.4f}")

    # ---- Comparison: band shrinkage from narrative restrictions ----
    print(f"\n      Band-shrinkage comparison (narrative vs sign-only):")
    print(f"        {'h':>3}  {'variable':<10}  {'sign-only':>12}  {'narrative':>12}  {'ratio':>8}")
    band_rows = []
    for h_show in (0, 3, 6, 12, 24):
        if h_show > horizon:
            continue
        for j, var in enumerate(fit.variable_names):
            hw_b = (irf_base.upper_68[h_show, j, shock_idx]
                     - irf_base.lower_68[h_show, j, shock_idx]) / 2
            hw_n = (irf_narr.upper_68[h_show, j, shock_idx]
                     - irf_narr.lower_68[h_show, j, shock_idx]) / 2
            ratio = float(hw_n / hw_b) if hw_b > 1e-12 else float("nan")
            print(f"        {h_show:>3}  {var:<10}  {hw_b:>12.4f}  "
                  f"{hw_n:>12.4f}  {ratio:>8.3f}")
            band_rows.append({
                "horizon": int(h_show), "variable": var,
                "half_width_sign_only": float(hw_b),
                "half_width_narrative": float(hw_n),
                "ratio": ratio,
            })

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
            "class": "var", "lags": int(spec.model["lags"]),
            "trend": spec.model["trend"],
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": {
            "method": "narrative_sign_restrictions",
            "sign_restrictions": sign_rs,
            "narrative_restrictions": [dict(nr) for nr in narr],
            "restriction_horizons": rh,
            "horizon": horizon,
        },
        "sign_only_baseline": {
            "n_accepted": irf_base.metadata["n_accepted"],
            "n_attempted": irf_base.metadata["n_attempted"],
            "acceptance_rate": irf_base.metadata["acceptance_rate"],
        },
        "narrative": {
            "n_accepted": irf_narr.metadata["n_accepted"],
            "n_attempted": irf_narr.metadata["n_attempted"],
            "n_rejected_by_narrative": irf_narr.metadata["n_rejected_by_narrative"],
            "acceptance_rate": irf_narr.metadata["acceptance_rate"],
        },
        "band_comparison": band_rows,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # Spread row at h=0 and h=3
        spread_h0 = next(
            (r for r in band_rows
             if r["horizon"] == 0 and r["variable"] == "spread"), None,
        )
        spread_h3 = next(
            (r for r in band_rows
             if r["horizon"] == 3 and r["variable"] == "spread"), None,
        )
        sign_only_ar = float(irf_base.metadata["acceptance_rate"])
        narr_ar = float(irf_narr.metadata["acceptance_rate"])
        validation_results = {
            "n_obs": float(len(data)),
            "n_lags": float(spec.model["lags"]),
            "max_eig_modulus": float(eig),
            "sign_only_acceptance_rate": sign_only_ar,
            "narrative_substantially_filters":
                1.0 if narr_ar < 0.1 * sign_only_ar else 0.0,
            "spread_h0_band_ratio":
                float(spread_h0["ratio"]) if spread_h0 else float("nan"),
            "spread_h3_band_ratio":
                float(spread_h3["ratio"]) if spread_h3 else float("nan"),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    # Uses narrative IRF as primary; sign-only baseline IRF as alt for the
    # robustness section (showing the band tightening from narrative
    # restrictions).
    spec_dict = {
        "Model": f"VAR({spec.model['lags']}) + sign + narrative restrictions",
        "Variables": ", ".join(list(data.columns)),
        "Identification": "Sign + narrative (Type 1 + Type 2); rejection-only adaptation of Antolín-Díaz / Rubio-Ramírez (2018), omits the paper's importance weighting",
        "Narrative restrictions": str(len(narr)),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(spec.model["lags"]),
        "Sign-only acceptance rate": f"{irf_base.metadata['acceptance_rate']:.4f}",
        "Narrative acceptance rate": f"{irf_narr.metadata['acceptance_rate']:.4f}",
        "Narrative tightening factor": (
            f"{irf_base.metadata['acceptance_rate'] / max(irf_narr.metadata['acceptance_rate'], 1e-9):.2f}x"
        ),
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report_html = lib.VARReport(
        title=rep.get("title", "ADRR 2018 narrative sign restrictions"),
        subtitle=rep.get("subtitle",
            "Sign + narrative restrictions; band-tightening shown in Robustness"),
    )
    report_html.add_standard_sections(
        fit=fit, irf=irf_narr,
        spec=spec_dict, log=log,
        var_labels=list(data.columns),
        alt_irf=irf_base,
        alt_label="Sign-only baseline",
        baseline_label="Sign + narrative",
        shock_idx_for_summary=0,
        x_axis_label="Months",
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")
    irf_narr.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
