"""Romer-Romer (2004) narrative monetary shocks in a proxy-SVAR.

Same architectural pattern as
[`gertler_karadi_2015_proxy_svar/`](../gertler_karadi_2015_proxy_svar/) —
FRED-MD macro data + an external monetary-policy instrument fed to
``lib.proxy_svar_irf``. The only meaningful difference is the
instrument: GK uses high-frequency FOMC-window surprises; RR uses
Greenbook-purged narrative intentions. Both identify a contractionary
monetary policy shock under different assumptions about what the Fed's
information set looks like.

The RR shock CSV is NOT on a stable public URL. The replicate.py loads
it via ``lib.fetch_data(api="csv_file", ...)`` from a user-staged path
under ``data/staging/``. If the file is missing, the script exits with
a clear message pointing to the README's source list.

Pipeline:
  lib.load_dataset (FRED-MD, 4 macro series)
    -> lib.transform (log/level)
      -> lib.fetch_data(api="csv_file") for the RR shock series
        -> inner-join macro + shock
          -> lib.fit_var (VAR(12), trend='c')
            -> lib.proxy_svar_irf (wild-bootstrap, 1000 draws)
              -> save IRF snapshot + research log
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

import lib  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_macro(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        start=data_cfg.get("sample_start"),
        end=data_cfg.get("sample_end"),
    )
    return lib.transform(raw[data_cfg["variable_order"]], data_cfg["transforms"]).dropna()


def load_rr_shock(cfg: dict) -> pd.DataFrame:
    """Fetch the user-staged RR shock CSV via lib.fetch_data (csv_file)."""
    inst_cfg = cfg["instrument"]
    csv_path_rel = inst_cfg["spec"].split(":", 1)[0]
    csv_path = PROJECT_ROOT / csv_path_rel
    if not csv_path.exists():
        sys.stderr.write(
            "\n*** Romer-Romer shock CSV not found ***\n\n"
            f"Expected file:   {csv_path}\n"
            f"Expected layout: a 'date' column (first), then an 'rr_shock' column.\n\n"
            "How to obtain the RR shock series:\n"
            "  1. Romer-Romer (2004) AER replication archive: openICPSR 116025\n"
            "     (https://www.openicpsr.org/openicpsr/project/116025) — original\n"
            "     1969M1-1996M12 monthly series.\n"
            "  2. Coibion (2012) AEJ:Macro replication archive — extension to 2002.\n"
            "  3. Wieland-Yang (2020) JMCB replication archive — extends into the late 2000s.\n"
            "  4. Aruoba-Drechsel (2024) — NLP-based extension beyond Greenbook horizon.\n\n"
            "Once you have the CSV, place it at the path above and re-run.\n"
            "Format example:\n"
            "    date,rr_shock\n"
            "    1969-01-01,0.005\n"
            "    1969-02-01,-0.012\n"
            "    ...\n\n"
        )
        raise SystemExit(1)
    return lib.fetch_data(
        api=inst_cfg["api"],
        series={inst_cfg["dataset_id"]: inst_cfg["spec"]},
        dataset_id=inst_cfg["dataset_id"],
    )


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print(f"  {cfg['replication']['subtitle']}")
    print("=" * 70)

    print("\n[1/4] Loading FRED-MD + RR shock series...")
    macro = load_macro(cfg)
    rr = load_rr_shock(cfg)
    shock_col = cfg["instrument"]["dataset_id"]
    merged = macro.join(rr[[shock_col]], how="inner").dropna()
    instrument = merged[shock_col].values
    data = merged.drop(columns=[shock_col])
    print(f"      Merged sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months)")
    print(f"      Instrument: RR_shock, "
          f"mean={instrument.mean():+.4f}, std={instrument.std():.4f}")

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), AIC={fit.fit_info['aic']:.4f}, "
          f"T_eff={fit.t_eff}, max|eig|={eig:.4f}")

    print("\n[3/4] Proxy-SVAR identification with RR instrument...")
    irf, fs = lib.proxy_svar_irf(
        fit=fit,
        instrument=instrument,
        policy_var=cfg["identification"]["policy_variable"],
        horizon=cfg["irf"]["horizon"],
        shock_name="Monetary policy (proxy-SVAR via RR narrative)",
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
        normalization=cfg["identification"]["normalization"],
    )
    print(f"      First-stage F={fs['F']:.2f}, gamma={fs['gamma']:+.4f}, "
          f"R^2={fs['R2']:.4f}  "
          f"({'STRONG' if fs['F'] >= 10 else 'WEAK'} instrument)")

    print("\n[4/4] Peak responses to a unit RR shock to FFR:")
    labels = cfg["data"]["variable_labels"]
    peaks: list[dict] = []
    for i, name in enumerate(irf.variable_names):
        path = irf.median[:, i, 0]
        h_peak = int(np.argmax(np.abs(path[1:])) + 1)
        sig68 = bool((irf.lower_68[h_peak, i, 0] > 0)
                     or (irf.upper_68[h_peak, i, 0] < 0))
        sig90 = bool((irf.lower_90[h_peak, i, 0] > 0)
                     or (irf.upper_90[h_peak, i, 0] < 0))
        peaks.append({
            "variable": name, "label": labels.get(name, name),
            "peak_horizon": h_peak, "peak_value": float(path[h_peak]),
            "sig68": sig68, "sig90": sig90,
        })
        print(f"      {labels.get(name, name):<28} "
              f"peak={path[h_peak]:+.5f} at h={h_peak:<2}  "
              f"sig68={'Y' if sig68 else 'N'} sig90={'Y' if sig90 else 'N'}")

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "macro_source": cfg["data"]["source"],
            "instrument_source": cfg["instrument"]["source"],
            "instrument_spec": cfg["instrument"]["spec"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
            "variables": list(data.columns),
        },
        "model": {
            "class": cfg["model"]["class"],
            "lags": fit.p,
            "trend": cfg["model"]["trend"],
            "fit_info": fit.fit_info,
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": dict(cfg["identification"]),
        "first_stage": fs,
        "instrument_summary": {
            "mean": float(instrument.mean()),
            "std": float(instrument.std()),
            "n_obs": int(len(instrument)),
        },
        "peak_responses": peaks,
        "irf_metadata": irf.metadata,
        "seed": seed,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        policy_var = cfg["identification"]["policy_variable"]
        pol_idx = irf.variable_names.index(policy_var)
        pol_peak = float(irf.median[:, pol_idx, 0].max())
        validation_results = {
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "stable": 1.0 if eig < 1.0 else 0.0,
            "first_stage_F": float(fs.get("F", float("nan"))),
            "policy_var_peak_positive": 1.0 if pol_peak > 0 else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + Proxy SVAR identification",
        "Variables": ", ".join(list(data.columns)),
        "Policy variable": cfg["identification"]["policy_variable"],
        "Instrument": "Romer-Romer narrative MP shock series",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "First-stage F": f"{fs.get('F', float('nan')):.2f}",
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report_html.add_standard_sections(
        fit=fit, irf=irf, spec=spec_dict, log=log,
        var_labels=list(data.columns),
        shock_idx_for_summary=0,
        x_axis_label="Months",
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    irf.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
