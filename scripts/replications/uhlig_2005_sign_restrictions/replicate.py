"""
Replication of Uhlig (2005) "What are the effects of monetary policy on output?"

Sign-restricted monetary policy SVAR with output left UNRESTRICTED — the
paper's key 'agnostic identification' contribution.

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

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data,
        lags=cfg["model"]["lags"],
        trend=cfg["model"]["trend"],
    )
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, T_eff={fit.t_eff}, "
          f"max |eigenvalue|={lib.max_eigenvalue_modulus(fit.lag_matrices()):.4f}")

    print("\n[3/4] Sign-restricted identification (Uhlig 2005)...")
    print(f"      Restrictions: {cfg['identification']['restrictions']['monetary_policy']}")
    print(f"      Imposed at horizons 0..{cfg['identification']['restriction_horizons']-1}")
    irf = lib.sign_restricted_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        sign_restrictions=cfg["identification"]["restrictions"],
        restriction_horizons=cfg["identification"]["restriction_horizons"],
        target_accepted=cfg["irf"]["target_accepted"],
        max_rotations=cfg["irf"]["max_rotations"],
        seed=seed,
        parameter_uncertainty=True,
    )
    print(f"      Accepted: {irf.metadata['n_accepted']}/{cfg['irf']['target_accepted']} "
          f"({irf.metadata['acceptance_rate']*100:.2f}% rate, "
          f"{irf.metadata['n_attempted']} attempts)")

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    labels = cfg["data"]["variable_labels"]
    var_labels = [labels.get(v, v) for v in fit.variable_names]

    # Replicate paper's Section 5 finding: is the output response robustly signed?
    indpro_idx = fit.variable_names.index("indpro")
    output_med = irf.median[:, indpro_idx, 0]
    output_lo90 = irf.lower_90[:, indpro_idx, 0]
    output_hi90 = irf.upper_90[:, indpro_idx, 0]
    # Windows are inclusive: months 1-24 = indices 1..24, months 3-18 =
    # indices 3..18 (month 0 is the impact period, excluded by design).
    band_zero_mask = (output_lo90[1:25] < 0) & (output_hi90[1:25] > 0)
    band_includes_zero = bool(np.any(band_zero_mask))
    n_band_zero_months = int(np.sum(band_zero_mask))
    band_robust_neg = bool(np.all(output_hi90[3:19] < 0))
    # Largest-magnitude median response over months 1-24 (sign kept).
    largest_idx = int(np.argmax(np.abs(output_med[1:25])))
    largest_med = float(output_med[1:25][largest_idx])
    largest_month = largest_idx + 1
    print(f"\n      Output (indpro) response to monetary contraction:")
    print(f"        Median response of largest magnitude: {largest_med:+.4f} "
          f"at month {largest_month}")
    print(f"        90% band includes zero in at least one of months 1-24: "
          f"{'YES (consistent with Uhlig)' if band_includes_zero else 'NO'}")
    print(f"        Months (of 24) whose 90% band includes zero: "
          f"{n_band_zero_months}")
    print(f"        90% band entirely below zero across months 3-18: "
          f"{'YES' if band_robust_neg else 'NO (consistent with Uhlig)'}")

    max_eig = float(lib.max_eigenvalue_modulus(fit.lag_matrices()))

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        out_med_window = output_med[1:25]
        peak_h = int(np.argmax(np.abs(out_med_window)))
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "max_eig_modulus": max_eig,
            "acceptance_rate": float(irf.metadata.get("acceptance_rate", 0.0)),
            "n_accepted": float(irf.metadata.get("n_accepted", 0)),
            "output_band_includes_zero":
                1.0 if band_includes_zero else 0.0,
            "output_median_peak_abs": float(abs(out_med_window[peak_h])),
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
            "max_eigenvalue_modulus": max_eig,
        },
        "identification": dict(cfg["identification"]),
        "irf_metadata": irf.metadata,
        "key_findings": {
            "output_band_includes_zero_months_1_24": band_includes_zero,
            "output_robustly_negative_months_3_18": band_robust_neg,
            "output_median_peak": float(np.min(output_med[1:25])),
            "output_median_peak_month": int(np.argmin(output_med[1:25])) + 1,
            "output_band_zero_months_count_1_24": n_band_zero_months,
            "output_median_largest_abs_response": largest_med,
            "output_median_largest_abs_month": largest_month,
        },
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (full eight-section structure) ----
    spec_dict = {
        "Model": f"VAR({fit.p})",
        "Variables": ", ".join(var_labels),
        "Identification": "Sign restrictions (Uhlig 2005, pure-rejection / agnostic)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Acceptance rate": f"{irf.metadata.get('acceptance_rate', 0.0):.3f}",
        "Accepted rotations": str(irf.metadata.get('n_accepted', 0)),
        "Max |eigenvalue|": f"{max_eig:.4f}",
        "Headline finding": (
            "Output response 90% band includes zero in at least one of months "
            f"1-24 ({n_band_zero_months} of 24; consistent with Uhlig)"
            if band_includes_zero else
            "Output response 90% band excludes zero in every month 1-24"
        ),
    }
    # IRFContainer from sign_restricted_irf has shape (H+1, n_vars, 1) for
    # single-shock identification — pass as-is; add_standard_sections derives
    # the rest from fit + irf.
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_standard_sections(
        fit=fit, irf=irf, spec=spec_dict, log=log,
        var_labels=var_labels,
        shock_idx_for_summary=0,
        x_axis_label="Months",
        units="Percent",          # data is 100*log -> responses read in percent
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
