"""
Schorfheide-Song (2015) mixed-frequency Bayesian VAR — methodological demo.

Original paper: 11-variable US MF-VAR with 8 monthly and 3 quarterly
indicators in log levels, real-time vintages, MDD-optimised
hyperparameters; constant-parameter linear Gaussian (no stochastic
volatility — that arrives in the authors' 2021 pandemic follow-up).

This replication is a 4-variable simplified demo:
  - 3 monthly: IP growth, CPI inflation, FFR (from FRED-MD)
  - 1 quarterly: real GDP growth (from FRED-QD)

Quarterly GDP growth is placed on the monthly index at end-of-quarter
months with NaN intra-quarter. The library's ``fit_mfvar_bayesian``
runs the Gibbs sampler with Mariano-Murasawa aggregation.

``lib.run`` does not dispatch ``mfvar`` and ``lib.load_data`` assumes
single-frequency data, so the two-frequency join and the end-of-quarter
NaN placement are done inline. The top-level ``prior:`` and ``gibbs:``
blocks of config.yaml are read from ``spec.metadata``.
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

import lib   # noqa: E402


def load_mfvar_data(spec_data: dict) -> pd.DataFrame:
    """Load monthly + quarterly via lib.load_dataset + lib.transform, then
    place each quarterly observation at the END month of its quarter on the
    monthly grid (NaN intra-quarter) — the data shape the MF-VAR Gibbs
    sampler expects.

    The end-of-quarter NaN-placement is specific to MF-VAR aggregation
    (Mariano-Murasawa five-month average over the quarterly state); it
    doesn't generalise to ``lib.load_data``'s flat-data assumption.
    """
    start = spec_data["sample_start"]
    end = spec_data["sample_end"]

    raw_m = lib.load_dataset(
        source=spec_data["monthly"]["source"],
        series={s["name"]: s["fred_code"] for s in spec_data["monthly"]["series"]},
        start=start, end=end,
    )
    monthly_df = lib.transform(
        raw_m[spec_data["monthly"]["variable_order"]],
        spec_data["monthly"]["transforms"],
    ).dropna()

    raw_q = lib.load_dataset(
        source=spec_data["quarterly"]["source"],
        series={s["name"]: s["fred_code"] for s in spec_data["quarterly"]["series"]},
        start=start, end=end,
    )
    quarterly_df = lib.transform(
        raw_q[spec_data["quarterly"]["variable_order"]],
        spec_data["quarterly"]["transforms"],
    ).dropna()

    # Align onto monthly index: place each quarterly value at the END of
    # quarter (Mar/Jun/Sep/Dec) as a first-of-month timestamp; NaN everywhere
    # else. The MF-VAR Gibbs sampler uses the NaN pattern to drive the
    # state-space augmentation.
    full_idx = monthly_df.index
    aligned_q = pd.DataFrame(np.nan, index=full_idx, columns=quarterly_df.columns)
    for qname in quarterly_df.columns:
        for q_date in quarterly_df.index:
            quarter_end_month = q_date.to_period("Q").end_time.to_period("M").start_time
            if quarter_end_month in full_idx:
                aligned_q.loc[quarter_end_month, qname] = quarterly_df.loc[q_date, qname]
    merged = pd.concat([monthly_df, aligned_q], axis=1)
    return merged.dropna(subset=list(monthly_df.columns))


def main() -> int:
    spec = lib.Spec.from_replication("schorfheide_song_2015_mfvar")
    seed = int(spec.metadata["general"]["random_seed"])
    np.random.seed(seed)

    rep = spec.metadata["replication"]
    print("=" * 70)
    print(f"  {rep['title']}")
    print("=" * 70)

    print("\n[1/4] Loading and aligning mixed-frequency data...")
    data = load_mfvar_data(spec.data)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} months)")
    n_q_obs = int(data["gdp_growth"].notna().sum())
    print(f"      Monthly variables: "
          f"{[s['name'] for s in spec.data['monthly']['series']]}")
    print(f"      Quarterly variables: "
          f"{[s['name'] for s in spec.data['quarterly']['series']]} "
          f"({n_q_obs} end-of-quarter observations)")

    print("\n[2/4] Setting up MF-VAR specification...")
    mfvar_spec = lib.MFVARSpec(
        monthly_vars=[s["name"] for s in spec.data["monthly"]["series"]],
        quarterly_vars=[s["name"] for s in spec.data["quarterly"]["series"]],
        lags=spec.model["lags"],
        state_lags=spec.model["state_lags"],
    )
    prior_dict = spec.metadata["prior"]
    prior = lib.MinnesotaPrior(
        lambda_1=prior_dict["lambda_1"], lambda_2=prior_dict["lambda_2"],
        lambda_3=prior_dict["lambda_3"], lambda_4=prior_dict["lambda_4"],
        mu=prior_dict["mu"], delta=prior_dict["delta"],
    )
    print(f"      Variables (canonical order): {mfvar_spec.variable_order}")
    print(f"      Lags: {mfvar_spec.lags}, state lags: {mfvar_spec.state_lags}")
    print(f"      Prior: lambda_1={prior.lambda_1}, lambda_2={prior.lambda_2}, "
          f"mu={prior.mu}, delta={prior.delta}")

    print("\n[3/4] Running Gibbs sampler...")
    gibbs_cfg = spec.metadata["gibbs"]
    fit = lib.fit_mfvar_bayesian(
        data=data, spec=mfvar_spec, prior=prior,
        n_iter=gibbs_cfg["n_iter"], burn_in=gibbs_cfg["burn_in"],
        thin=gibbs_cfg["thin"], seed=seed,
        trend=spec.model["trend"], verbose=True,
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"\n      Stored draws: {fit.gibbs_B_draws.shape[0]}")
    print(f"      Max |eigenvalue| (posterior mean): {eig:.4f}")

    print("\n[4/4] Imputed monthly GDP growth (latent state at quarter-ends):")
    state = fit.state_path_posterior_mean
    n_M = mfvar_spec.n_monthly
    gdp_state = state[:, n_M]   # latent monthly GDP growth
    # Compare observed quarterly obs vs aggregated state at quarter ends
    w = lib.mariano_murasawa_weights()
    T = len(state)
    print(f"        {'date':<10} {'observed':>10} {'agg(state)':>11}")
    obs = data["gdp_growth"].values
    n_shown = 0
    for t in range(4, T):
        if not np.isnan(obs[t]):
            agg = sum(w[lag] * gdp_state[t - lag] for lag in range(5))
            print(f"        {data.index[t]:%Y-%m}    {obs[t]:>+10.4f}    {agg:>+11.4f}")
            n_shown += 1
            if n_shown >= 8:
                print(f"        ...")
                break

    print("\n      Latent monthly GDP growth — recent values:")
    print(f"        {'date':<10} {'gdp_growth_state':>18}")
    for t in range(max(0, T - 12), T):
        print(f"        {data.index[t]:%Y-%m}      {gdp_state[t]:>+10.4f}")

    print("\n      Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    state_df = pd.DataFrame(
        fit.state_path_posterior_mean,
        columns=fit.variable_names,
        index=data.index,
    )
    state_df.to_csv(outputs_dir / "state_path_posterior_mean.csv")
    print(f"      ->{(outputs_dir / 'state_path_posterior_mean.csv').relative_to(PROJECT_ROOT)}")

    log = {
        "replication_id": rep["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "monthly_source": spec.data["monthly"]["source"],
            "quarterly_source": spec.data["quarterly"]["source"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_months": int(len(data)),
            "n_quarterly_observations": n_q_obs,
            "monthly_vars": mfvar_spec.monthly_vars,
            "quarterly_vars": mfvar_spec.quarterly_vars,
        },
        "model": {
            "class": "mfvar",
            "lags": fit.p,
            "state_lags": mfvar_spec.state_lags,
            "max_eigenvalue_modulus": float(eig),
            "prior": fit.fit_info["prior"],
        },
        "gibbs": {
            "n_iter": gibbs_cfg["n_iter"],
            "burn_in": gibbs_cfg["burn_in"],
            "thin": gibbs_cfg["thin"],
            "n_draws_stored": int(fit.gibbs_B_draws.shape[0]),
        },
        "seed": seed,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # MM-aggregated state at quarter-ends should match observed quarterly value
        w_mm = lib.mariano_murasawa_weights()
        n_M = mfvar_spec.n_monthly
        gdp_state_arr = fit.state_path_posterior_mean[:, n_M]
        obs_arr = data["gdp_growth"].values
        max_mismatch = 0.0
        n_T = len(obs_arr)
        for t in range(4, n_T):
            if not np.isnan(obs_arr[t]):
                agg = sum(w_mm[lag] * gdp_state_arr[t - lag] for lag in range(5))
                diff = abs(agg - obs_arr[t])
                if diff > max_mismatch:
                    max_mismatch = diff
        validation_results = {
            "n_months": float(len(data)),
            "n_quarterly_obs": float(n_q_obs),
            "n_vars": float(len(fit.variable_names)),
            "n_lags": float(fit.p),
            "stable": 1.0 if eig < 1.0 else 0.0,
            "n_draws_stored": float(fit.gibbs_B_draws.shape[0]),
            "state_matches_observed_at_qends":
                1.0 if max_mismatch < 0.05 else 0.0,
            "state_qend_max_mismatch": float(max_mismatch),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — MF-VAR replication produces a smoothed
    #      state path (CSV), not IRFs. The headline is the
    #      Mariano-Murasawa state aggregation matching observed quarterly
    #      values to machine precision) ----
    spec_dict = {
        "Model": f"Mixed-frequency BVAR({fit.p}) (Schorfheide-Song 2015)",
        "Monthly variables": ", ".join(mfvar_spec.monthly_vars),
        "Quarterly variables": ", ".join(mfvar_spec.quarterly_vars),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} months)",
        "Quarterly observations": str(n_q_obs),
        "Lag order": str(fit.p),
        "State-space lags": str(mfvar_spec.state_lags),
        "Prior": str(fit.fit_info["prior"]),
        "Gibbs iterations": str(gibbs_cfg["n_iter"]),
        "Burn-in": str(gibbs_cfg["burn_in"]),
        "Posterior draws stored": str(int(fit.gibbs_B_draws.shape[0])),
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report_html = lib.VARReport(
        title=rep.get("title", "Schorfheide-Song 2015 mixed-frequency BVAR"),
        subtitle=rep.get("subtitle",
            "MF-VAR state-space Gibbs sampler — see state_path_posterior_mean.csv "
            "for the full smoothed monthly state"),
    )
    report_html.add_specification(spec_dict)
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"      ->{(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      ->{(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
