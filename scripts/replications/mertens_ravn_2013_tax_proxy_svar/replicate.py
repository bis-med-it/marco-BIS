"""Mertens & Ravn (2013) — Multi-instrument proxy SVAR for tax shocks.

Seven-variable quarterly SVAR with TWO simultaneous narrative-instrument
identifications: personal and corporate income tax shocks recovered
jointly via the Mertens-Ravn (2013) §II algebra, implemented in the
multi-instrument path of ``lib.proxy_svar_irf``, using the library's
identity policy-impact block rather than MR's triangular tax-shock
decomposition.

What this replication does:

* Loads APITR, ACITR, PITB, CITB (BEA NIPA constructions), real
  government spending per capita, real GDP per capita, and log real
  federal debt held by the public per capita from a user-staged CSV (the AER replication archive layout).
* Estimates a reduced-form VAR(4).
* Identifies two simultaneous structural shocks via the personal and
  corporate narrative tax instruments using the multi-instrument proxy
  SVAR (MR 2013 §II / Stock-Watson 2018 multi-instrument extension).
* Reports IRFs to (i) a 1pp personal income tax cut and (ii) a 1pp
  corporate income tax cut, with wild-bootstrap confidence bands.
* Reports first-stage diagnostics: per (instrument, policy) pair F-stat
  and the condition number of Σ_p.

What it deliberately substitutes / omits:

* Inference uses MR's original Rademacher wild bootstrap (the paper: 10,000
  draws, 95% bands, p. 1225; here: 1,000 draws, 68/90 bands — a documented
  divergence). Jentsch-Lunsford (2019, AER Comment) PROVED that scheme
  asymptotically invalid for proxy-SVARs; in their 2019 reply MR
  acknowledge this and show the headline's 68%-level significance survives
  under several valid methods (Delta, MOSW k=2 parametric bootstrap with
  NW HAC, MBB with uncentered proxies or Hall intervals) while pushing
  back on JL's exact double-centered MBB. lib implements the JL MBB for
  the SINGLE-instrument path (bands="moving_block_bootstrap"); not yet for
  the multi-instrument path used here. Treat the reported bands as the
  legacy convention, likely UNDERSTATING uncertainty.
* The replication archive's exact APITR/ACITR construction is honoured
  via the staging CSV; if your CSV uses a different construction,
  document the difference in your local notes.
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_staging_csv(cfg: dict) -> pd.DataFrame:
    """Load the pre-staged MR data CSV. Exits cleanly if missing."""
    staging_rel = cfg["data"]["staging_path"]
    csv_path = PROJECT_ROOT / staging_rel
    if not csv_path.exists():
        endo_cols = [s["name"] for s in cfg["data"]["series"]]
        inst_cols = cfg["instruments"]["columns"]
        sys.stderr.write(
            "\n*** Mertens-Ravn 2013 staging CSV not found ***\n\n"
            f"Expected file:   {csv_path}\n"
            f"Expected layout: a 'date' column (first), then the columns "
            f"below.\n\n"
            f"Endogenous variables ({len(endo_cols)}):\n"
        )
        for c in endo_cols:
            sys.stderr.write(f"  - {c}\n")
        sys.stderr.write(f"\nInstruments ({len(inst_cols)}):\n")
        for c in inst_cols:
            sys.stderr.write(f"  - {c}\n")
        sys.stderr.write(
            "\nHow to obtain the data:\n"
            "  1. Mertens-Ravn (2013) AER replication archive\n"
            "     https://www.aeaweb.org/articles?id=10.1257/aer.103.4.1212\n"
            "     (look under 'Additional Materials' / 'Replication\n"
            "     package' / 'data files'). The archive provides\n"
            "     constructed APITR, ACITR, PITB, CITB and the two\n"
            "     narrative shock series.\n"
            "  2. The macro controls (log real FEDERAL purchases per\n"
            "     capita, log real GDP per capita, log real federal debt\n"
            "     held by the public per capita) can be recomputed if\n"
            "     missing from the archive. FGCEC1 and GDPC1 are already\n"
            "     real: divide by population (16+) and log. Federal debt\n"
            "     held by the public (DEBTHP / FYGFDPUN) is nominal:\n"
            "     divide by the GDP deflator and population, then log.\n"
            "     See the README's data table.\n\n"
            "Once you have the CSV at the path above, re-run with no\n"
            "other changes.\n\n"
        )
        raise SystemExit(1)

    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    sample_start = pd.Timestamp(cfg["data"]["sample_start"])
    sample_end = pd.Timestamp(cfg["data"]["sample_end"])
    df = df.loc[sample_start:sample_end]
    return df


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    rep = cfg["replication"]
    print("=" * 70)
    print(f"  {rep['title']}\n  {rep['subtitle']}")
    print("=" * 70)

    print("\n[1/5] Loading staging CSV (BEA NIPA tax variables + narrative shocks)...")
    staged = load_staging_csv(cfg)

    endo_cols = cfg["data"]["variable_order"]
    inst_cols = cfg["instruments"]["columns"]
    missing = [c for c in endo_cols + inst_cols if c not in staged.columns]
    if missing:
        sys.stderr.write(
            f"\n*** Staging CSV is missing required columns: {missing}\n"
            f"Available columns: {list(staged.columns)}\n\n"
        )
        raise SystemExit(2)

    data = lib.transform(
        staged[endo_cols], cfg["data"]["transforms"],
    ).dropna()
    if data.empty:
        sys.stderr.write(
            "\n*** No data after sample-window filter + transform. Check the\n"
            "    staging CSV's date range against config.data.sample_start/end.\n\n"
        )
        raise SystemExit(3)

    # Align instruments with the post-transform sample
    instruments_df = staged.loc[data.index, inst_cols]
    if instruments_df.isna().any().any():
        # A NaN instrument cell is treated as "no narrative event -> zero
        # shock" ONLY when the config's staging contract asserts it
        # (instrument.nan_means_no_event: true). An unexplained missing
        # cell is NOT an observed zero and aborts with its dates.
        n_nan = int(instruments_df.isna().sum().sum())
        if not cfg["instruments"].get("nan_means_no_event", False):
            _dates = instruments_df.index[instruments_df.isna().any(axis=1)]
            raise RuntimeError(
                f"staged instrument file has {n_nan} missing cells "
                f"(e.g. {[str(d.date()) for d in _dates[:5]]}) and the "
                f"config does not assert nan_means_no_event. Verify the "
                f"staged file's convention before enabling the zero fill."
            )
        print(f"      Note: {n_nan} NaN instrument values set to 0 per the "
              f"config's staging contract (nan_means_no_event: narrative "
              f"shocks are zero outside documented events).")
        instruments_df = instruments_df.fillna(0.0)
    instruments = instruments_df.values

    def _qfmt(ts: pd.Timestamp) -> str:
        return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"

    print(f"      Sample: {_qfmt(data.index[0])} to {_qfmt(data.index[-1])}"
          f" ({len(data)} obs, {data.shape[1]} variables)")
    print(f"      Endogenous: {list(data.columns)}")
    print(f"      Instruments: {inst_cols}")
    print(f"        personal_shock  mean={instruments[:, 0].mean():+.4f} "
          f"std={instruments[:, 0].std():.4f} nonzero={int((instruments[:, 0] != 0).sum())}")
    print(f"        corporate_shock mean={instruments[:, 1].mean():+.4f} "
          f"std={instruments[:, 1].std():.4f} nonzero={int((instruments[:, 1] != 0).sum())}")

    print("\n[2/5] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}): AIC={fit.fit_info['aic']:.4f}, "
          f"T_eff={fit.t_eff}, max|eig|={eig:.4f} (stable={fit.stable()})")

    print("\n[3/5] Multi-instrument proxy-SVAR identification (MR 2013 §II)...")
    irf, fs = lib.proxy_svar_irf(
        fit=fit,
        instrument=instruments,
        policy_var=cfg["identification"]["policy_variables"],
        horizon=cfg["irf"]["horizon"],
        shock_name=cfg["irf"]["shock_names"],
        bands=cfg["irf"]["bands"],
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
        normalization=cfg["identification"]["normalization"],
    )
    print(f"      Sigma_p condition number: {fs['Sigma_p_cond']:.2f}  "
          f"(lower = more independent instruments)")
    print(f"      Per-pair first-stage diagnostics:")
    print(f"        {'instr':>6} {'policy':>10} {'gamma':>10} {'R^2':>8} {'F':>10}")
    for p in fs["per_pair"]:
        instr_name = inst_cols[p["instrument_idx"]]
        policy_name = data.columns[p["policy_idx"]]
        flag = "***" if p["instrument_idx"] == p["policy_position"] else "   "
        print(f"        {instr_name[:6]:>6} {policy_name[:10]:>10}"
              f" {p['gamma']:>+10.4f} {p['R2']:>8.4f} {p['F']:>10.2f} {flag}")
    print(f"      Min own-pair F: {fs['weak_instrument_min_own_F']:.2f} "
          f"({'WEAK' if fs['weak_instrument_F_lt_10'] else 'STRONG'})")

    print("\n[4/5] IRFs to a 1pp personal tax cut and a 1pp corporate tax cut...")
    # MR report responses to a TAX CUT — sign-flip the unit-policy IRFs
    # (unit_policy gives +1 on impact for the policy variable; a cut is -1).
    horizons_to_report = cfg["report"]["horizons_to_report"]
    for shock_idx, shock_label in enumerate(cfg["irf"]["shock_names"]):
        print(f"\n    Response to {shock_label} (sign-flipped to represent a 1pp CUT):")
        print(f"      {'h':>3}  {'variable':<14} {'median':>10}  {'68% CI':>22}  {'90% CI':>22}")
        for h in horizons_to_report:
            if h > cfg["irf"]["horizon"]:
                continue
            for v_idx, var in enumerate(data.columns):
                # Sign-flip: tax-cut response = -1 * unit-rise response
                m = -irf.median[h, v_idx, shock_idx]
                lo68 = -irf.upper_68[h, v_idx, shock_idx]
                hi68 = -irf.lower_68[h, v_idx, shock_idx]
                lo90 = -irf.upper_90[h, v_idx, shock_idx]
                hi90 = -irf.lower_90[h, v_idx, shock_idx]
                print(f"      {h:>3}  {var:<14} {m:>+10.4f}  "
                      f"[{lo68:>+7.4f}, {hi68:>+7.4f}]  "
                      f"[{lo90:>+7.4f}, {hi90:>+7.4f}]")

    print("\n[5/5] Saving outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "staging_path": cfg["data"]["staging_path"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
            "variables": list(data.columns),
            "instruments": inst_cols,
        },
        "model": {
            "class": "var",
            "lags": int(cfg["model"]["lags"]),
            "trend": cfg["model"]["trend"],
            "max_eigenvalue_modulus": float(eig),
            "aic": float(fit.fit_info["aic"]),
        },
        "identification": {
            "method": "external_instrument_multi",
            "policy_variables": cfg["identification"]["policy_variables"],
            "normalization": cfg["identification"]["normalization"],
            "first_stage": fs,
        },
        "irf": {
            "horizon": int(cfg["irf"]["horizon"]),
            "bands": cfg["irf"]["bands"],
            "n_draws": int(cfg["irf"]["n_draws"]),
            "shock_names": cfg["irf"]["shock_names"],
            "convention": "responses to a +1 unit shock (use sign-flip for a tax cut)",
        },
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # Personal-tax F: own pair of instrument 0 (personal) and policy 0 (APITR)
        sigma_p_cond, personal_F = first_stage_summary(fs)
        validation_results = {
            "n_vars": float(fit.n_vars),
            "n_instruments": float(len(inst_cols)),
            "n_policy_vars": float(len(cfg["identification"]["policy_variables"])),
            "n_lags": float(fit.p),
            "stable": 1.0 if eig < 1.0 else 0.0,
            "sigma_p_invertible": 1.0 if sigma_p_cond < 1e6 else 0.0,
            "personal_F": personal_F,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + Multi-instrument Proxy SVAR (Mertens-Ravn 2013)",
        "Variables": ", ".join(list(data.columns)),
        "Policy variables": ", ".join(cfg["identification"]["policy_variables"]),
        "Instruments": ", ".join(inst_cols),
        "Normalization": cfg["identification"]["normalization"],
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Max |eigenvalue|": f"{float(eig):.4f}",
        "Σ_p condition number": f"{fs['Sigma_p_cond']:.2e}",
        "Sign convention": "responses to a +1 unit shock (sign-flip for tax cut)",
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report_html.add_standard_sections(
        fit=fit, irf=irf, spec=spec_dict, log=log,
        var_labels=list(data.columns),
        shock_idx_for_summary=0,
    narrative="auto",
    )
    report_html.save(outputs_dir / "report.html")
    print(f"      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(
        json.dumps(log, indent=2, default=str),
    )
    irf.save_snapshot(outputs_dir / "numerical_results.json")
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")
    print(f"      -> {(outputs_dir / 'numerical_results.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


def first_stage_summary(fs):
    """Condition number of Sigma_p and the personal-tax own-pair F from the
    multi-instrument diagnostics of ``lib.proxy_svar_irf``.

    Reads the keys the library returns (``Sigma_p_cond``, ``per_pair``);
    a missing key is a contract error, not a failed identification.
    """
    try:
        cond = float(fs["Sigma_p_cond"])
        per_pair = fs["per_pair"]
    except KeyError as exc:
        raise KeyError(f"first-stage diagnostics missing {exc}; "
                       "the lib.proxy_svar_irf return contract changed") from exc
    own = [p["F"] for p in per_pair
           if p["instrument_idx"] == 0 and p["policy_position"] == 0]
    if not own:
        raise KeyError("first-stage diagnostics have no personal-tax own pair")
    return cond, float(own[0])


if __name__ == "__main__":
    raise SystemExit(main())
