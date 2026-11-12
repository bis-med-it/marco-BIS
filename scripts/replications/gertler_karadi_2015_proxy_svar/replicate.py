"""
Gertler-Karadi (2015) proxy-SVAR with FRED-MD substitutes.

The GK paper uses:
  - GS1 (1-year Treasury) as the policy variable
  - INDPRO, CPI as macro responses
  - Excess Bond Premium (Gilchrist-Zakrajšek 2012) as the credit indicator
  - ff4 (3-month-ahead fed funds future surprise) as the external instrument

This replication uses a FRED-MD substitute for EBP and an externally
retrieved JK monetary-policy instrument:
  - BAA-AAA spread for EBP (conceptually similar credit indicator)
  - Jarocinski-Karadi MP_median for ff4 (similar HF monetary surprise)

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


def load_fred_data(cfg: dict) -> pd.DataFrame:
    """Load via lib.load_dataset, derive BAA-AAA spread, apply transforms."""
    data_cfg = cfg["data"]
    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        end=data_cfg.get("sample_end"),
    )
    # BAA-AAA spread (no general "subtract" op in lib.transform; per-column
    # univariate transforms stay in lib.transform, this cross-column derivation
    # stays here).
    raw["spread"] = raw["baa"] - raw["aaa"]
    return lib.transform(raw[data_cfg["variable_order"]], data_cfg["transforms"]).dropna()


def load_jk_instrument(cfg: dict) -> pd.DataFrame:
    """Fetch the JK shock series via lib.fetch_data (external_archive).

    First call hits the GitHub URL; subsequent calls (including from any
    other replication using the same dataset_id) serve from the cache at
    data/<dataset_id>/dataset.csv.
    """
    inst_cfg = cfg["instrument"]
    df = lib.fetch_data(
        api=inst_cfg["api"],
        series={inst_cfg["series_key"]: inst_cfg["spec"]},
        dataset_id=inst_cfg["dataset_id"],
        dropna=False,
    )
    # The JK file leaves specific months missing (unscheduled/emergency
    # announcements excluded from the surprise measurement). Only the
    # months DECLARED in config as zero_surprise_months are set to zero —
    # an unexplained missing month is not an observed zero and aborts the
    # run with its dates. This keeps the VAR sample contiguous without a
    # blanket fill.
    key = inst_cfg["series_key"]
    allowed = {pd.Period(m, freq="M").to_timestamp()
               for m in inst_cfg.get("zero_surprise_months", [])}
    missing = set(df.index[df[key].isna()])
    unexplained = sorted(missing - allowed)
    if unexplained:
        raise RuntimeError(
            f"JK instrument has UNEXPLAINED missing months "
            f"{[str(d.date()) for d in unexplained]} — not in the config's "
            f"zero_surprise_months list. Verify each against the source "
            f"before declaring it a zero-surprise month."
        )
    if missing:
        print(f"      JK instrument: {len(missing)} declared no-event "
              f"month(s) set to zero surprise "
              f"({sorted(str(d.date()) for d in missing)})")
        df[key] = df[key].fillna(0.0)
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading FRED-MD and instrument...")
    fred = load_fred_data(cfg)
    jk = load_jk_instrument(cfg)
    z_key = cfg["instrument"]["series_key"]
    merged = fred.join(jk[[z_key]], how="inner").dropna()
    instrument = merged[z_key].values
    data = merged.drop(columns=[z_key])
    print(f"      FRED-MD: {fred.shape[1]} vars, {fred.index[0]:%Y-%m} to {fred.index[-1]:%Y-%m}")
    print(f"      JK shocks: {jk[z_key].notna().sum()} obs, "
          f"{jk[z_key].first_valid_index():%Y-%m} to {jk[z_key].last_valid_index():%Y-%m}")
    print(f"      Merged sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months)")
    print(f"      Instrument std={instrument.std():.4f}")

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, "
          f"T_eff={fit.t_eff}, max |eigenvalue|={eig:.4f}")

    print("\n[3/4] Proxy-SVAR identification...")
    irf, fs = lib.proxy_svar_irf(
        fit=fit,
        instrument=instrument,
        policy_var=cfg["identification"]["policy_variable"],
        horizon=cfg["irf"]["horizon"],
        shock_name="Monetary policy (proxy-SVAR via JK MP_median)",
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
        normalization=cfg["identification"]["normalization"],
    )
    print(f"      First-stage F={fs['F']:.2f}, gamma={fs['gamma']:+.4f}, R^2={fs['R2']:.4f}"
          f" ({'STRONG' if fs['F'] >= 10 else 'WEAK'} instrument)")
    print(f"      IRF horizon: 0..{irf.horizon}, "
          f"{irf.n_vars} variables, 1 shock; wild bootstrap with {cfg['irf']['n_draws']} draws")

    print("\n      GK-style monetary contraction (unit shock to GS1):")
    labels = cfg["data"]["variable_labels"]
    for i, name in enumerate(irf.variable_names):
        path = irf.median[:, i, 0]
        # Find peak by absolute value for hump-shape diagnostics
        peak_h = int(np.argmax(np.abs(path[1:])) + 1)
        sig68 = (irf.lower_68[peak_h, i, 0] > 0) or (irf.upper_68[peak_h, i, 0] < 0)
        sig90 = (irf.lower_90[peak_h, i, 0] > 0) or (irf.upper_90[peak_h, i, 0] < 0)
        print(f"      {labels.get(name, name):<32} peak={path[peak_h]:+.5f} "
              f"at h={peak_h:<2}  sig68={'Y' if sig68 else 'N'} sig90={'Y' if sig90 else 'N'}")

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in irf.variable_names]

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    F = float(fs.get("F", float("nan")))
    if targets_path.exists():
        gs1_idx = irf.variable_names.index(
            cfg["identification"]["policy_variable"]
        )
        gs1_path = irf.median[:, gs1_idx, 0]
        gs1_peak = float(np.abs(gs1_path).max())
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "max_eig_modulus": float(eig),
            "first_stage_F": F,
            "weak_instrument_flagged": 1.0 if F < 10.0 else 0.0,
            "gs1_peak_abs": gs1_peak,
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
            "substitutions": {
                "EBP": "BAA - AAA spread (FRED-MD)",
                "ff4_instrument": "JK MP_median",
            },
        },
        "model": {
            "class": cfg["model"]["class"],
            "lags": fit.p,
            "fit_info": fit.fit_info,
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": dict(cfg["identification"]),
        "first_stage": fs,
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + Proxy SVAR identification",
        "Variables": ", ".join(var_labels),
        "Identification": f"External instrument ({cfg['identification']['policy_variable']} as policy variable)",
        "Instrument": "Jarociński-Karadi MP_median (high-frequency MP surprises)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "First-stage F": f"{F:.2f}",
        "Weak-instrument flag": "yes" if F < 10.0 else "no",
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_standard_sections(
        fit=fit, irf=irf, spec=spec_dict, log=log,
        var_labels=var_labels,
        shock_idx_for_summary=0,
        x_axis_label="Months",
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
