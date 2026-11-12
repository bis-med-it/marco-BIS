"""
LP-IV (Stock-Watson 2018) applied to price-level and output responses with
Jarocinski-Karadi (2020) high-frequency monetary policy shocks as the
external instrument.

Thin orchestration script using ``scripts/lib``.

The methodology citation is Stock & Watson (2018) "Identification and
Estimation of Dynamic Causal Effects in Macroeconomics Using External
Instruments", EJ. The instrument is from Jarocinski & Karadi (2020) "Deconstructing Monetary Policy Surprises", AEJ:Macro.
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
    """Load FRED-MD via lib.load_dataset, apply per-column transforms."""
    data_cfg = cfg["data"]
    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        end=data_cfg.get("sample_end"),
    )
    return lib.transform(raw[data_cfg["variable_order"]], data_cfg["transforms"]).dropna()


def load_instrument(cfg: dict) -> pd.DataFrame:
    """Fetch JK shocks via lib.fetch_data (external_archive).

    Shares dataset_id="jk_shocks" with gertler_karadi_2015_proxy_svar so the
    cache at data/jk_shocks/dataset.csv is populated incrementally — the
    first replication to need each column pays the API call; subsequent
    runs read from the cache.
    """
    inst_cfg = cfg["instrument"]
    df = lib.fetch_data(
        api=inst_cfg["api"],
        series=inst_cfg["series"],
        dataset_id=inst_cfg["dataset_id"],
        dropna=False,
    )
    # Only the months DECLARED in config as zero_surprise_months
    # (unscheduled/emergency announcements the JK file excludes) are set
    # to zero — an unexplained missing month aborts with its dates. Same
    # convention as gertler_karadi_2015.
    allowed = {pd.Period(m, freq="M").to_timestamp()
               for m in inst_cfg.get("zero_surprise_months", [])}
    missing = set(df.index[df.isna().any(axis=1)])
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
        df = df.fillna(0.0)
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading FRED-MD and instrument...")
    data_full = load_fred_data(cfg)
    jk = load_instrument(cfg)
    # JK provides both MP and CBI shocks. Join on both so the sample matches
    # papers that report both; the instrument we actually use is mp_shock.
    primary = cfg["instrument"]["primary_series"]
    shock_cols = list(cfg["instrument"]["series"].keys())
    merged = data_full.join(jk[shock_cols], how="inner").dropna()
    z = merged[primary]
    # Drop the non-primary shock(s) from the data passed to LP-IV; keep
    # the primary in the frame because lp_iv_irf indexes responses out of `data`.
    extras = [c for c in shock_cols if c != primary]
    data = merged.drop(columns=extras)
    print(f"      Merged sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months)")
    print(f"      Instrument: {primary} (= "
          f"{cfg['instrument']['series'][primary]['column']}), "
          f"mean={z.mean():.4f}, std={z.std():.4f}")

    print("\n[2/4] Estimating LP-IV...")
    irf, fs = lib.lp_iv_irf(
        data=data,
        instrument=z.values,
        policy_var=cfg["model"]["policy_variable"],
        horizon=cfg["irf"]["horizon"],
        lags=cfg["model"]["lags"],
        response_vars=cfg["model"]["response_variables"],
        shock_name="Monetary policy (LP-IV via JK MP_median)",
        hac_lag_rule=cfg["model"]["hac_lag_rule"],
        credible_intervals=tuple(cfg["irf"]["credible_intervals"]),
    )
    print(f"      First-stage F: {fs['F']:.2f}, gamma={fs['gamma_hat']:+.4f} "
          f"({'STRONG' if fs['F'] >= 10 else 'WEAK'} instrument)")
    print(f"      IRF horizon: 0..{irf.horizon}, {irf.n_vars} responses")

    print("\n[3/4] Summary — peak responses to a +1 unit FFR shock identified by JK:")
    labels = cfg["data"]["variable_labels"]
    for i, name in enumerate(irf.variable_names):
        path = irf.median[:, i, 0]
        peak_h = int(np.argmax(np.abs(path)))
        # Significance: does the band exclude zero at the peak?
        sig68 = (irf.lower_68[peak_h, i, 0] > 0) or (irf.upper_68[peak_h, i, 0] < 0)
        sig90 = (irf.lower_90[peak_h, i, 0] > 0) or (irf.upper_90[peak_h, i, 0] < 0)
        print(f"      {labels.get(name, name):<24} "
              f"peak={path[peak_h]:+.5f} at h={peak_h:<2}  "
              f"sig68={'Y' if sig68 else 'N'} sig90={'Y' if sig90 else 'N'}")

    # --- Paper-target validation ---
    targets_path = HERE / "paper_targets.yaml"
    if targets_path.exists():
        # Per-variable peak responses (variable name -> (peak_value, peak_h))
        var_peaks: dict[str, tuple[float, int]] = {}
        for i, name in enumerate(irf.variable_names):
            path = irf.median[:, i, 0]
            peak_h = int(np.argmax(np.abs(path)))
            var_peaks[name] = (float(path[peak_h]), peak_h)

        validation_results = {
            "first_stage_F": float(fs["F"]),
            "first_stage_gamma": float(fs["gamma_hat"]),
            "strong_instrument": 1.0 if fs["F"] >= 10.0 else 0.0,
            "ffr_peak": var_peaks.get("ffr", (np.nan, 0))[0],
            "ffr_peak_h": float(var_peaks.get("ffr", (np.nan, 0))[1]),
            "ip_peak": var_peaks.get("ipi", var_peaks.get("indpro", (np.nan, 0)))[0],
            "cpi_peak": var_peaks.get("cpi", (np.nan, 0))[0],
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        print("\n" + comp_report.table())
    else:
        comp_report = None

    print("\n[4/4] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    var_labels = [labels.get(v, v) for v in irf.variable_names]

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
        },
        "instrument": {
            "source": cfg["instrument"]["source"],
            "api": cfg["instrument"]["api"],
            "dataset_id": cfg["instrument"]["dataset_id"],
            "primary_series": primary,
            "column": cfg["instrument"]["series"][primary]["column"],
            "mean": float(z.mean()),
            "std": float(z.std()),
            "first_stage": fs,
        },
        "model": dict(cfg["model"]),
        "irf_metadata": irf.metadata,
        "seed": seed,
    }
    if comp_report is not None:
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (LP-IV: no VARFit, so no auto-diagnostics/FEVD;
    #      use Specification + IRF + Research Log manually) ----
    spec_dict = {
        "Model": "Local Projections with external instrument (LP-IV)",
        "Variables": ", ".join(var_labels),
        "Instrument": (
            f"{cfg['instrument']['series'][primary]['column']} "
            f"(from {cfg['instrument']['dataset_id']} archive)"
        ),
        "Policy variable": str(cfg["model"]["policy_variable"]),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "LP horizon": str(cfg["irf"]["horizon"]),
        "First-stage F (Stock-Watson)": f"{fs.get('F', float('nan')):.2f}",
        "First-stage F ≥ 10 heuristic check":"yes" if fs.get("F", 0) >= 10.0 else "no",
        "LP HAC bandwidth rule": str(cfg["model"].get("hac_lag_rule",
                                                       "horizon_plus_one")),
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report.add_specification(spec_dict)
    report.add_irf(
        irf_point=irf.median,
        ci_68_lo=irf.lower_68, ci_68_hi=irf.upper_68,
        ci_90_lo=irf.lower_90, ci_90_hi=irf.upper_90,
        horizons=irf.horizons,
        var_names=var_labels,
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

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
