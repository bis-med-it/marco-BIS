"""
Primiceri (2005) TVP-VAR — time-varying coefficients (no SV).

Three-variable quarterly VAR(2) on inflation / unemployment / 3-month
T-bill. Gibbs sampler over time-varying coefficients with constant Σ.

The full Primiceri (2005) model also includes stochastic volatility for
Σ_t (via the Kim-Shephard-Chib mixture); this replication implements
only the time-varying-coefficients portion. The full model with SV is
available as `lib.tvp.fit_tvp_var_sv_primiceri`.
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
    """Load raw FRED-QD series, then apply per-series transforms."""
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
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} quarters, {data.shape[1]} variables)")

    print("\n[2/4] Setting up TVP-VAR spec...")
    spec = lib.TVPVARSpec(
        lags=cfg["model"]["lags"],
        training_sample=cfg["model"]["training_sample"],
        Q_prior_scale=cfg["model"]["Q_prior_scale"],
        Q_prior_df_factor=cfg["model"]["Q_prior_df_factor"],
    )
    print(f"      Lags: {spec.lags}, training: {spec.training_sample} quarters")
    print(f"      Q prior scale (k_Q^2): {spec.Q_prior_scale}")

    print("\n[3/4] Running Gibbs sampler...")
    fit = lib.fit_tvp_var(
        data, spec,
        n_iter=cfg["gibbs"]["n_iter"], burn_in=cfg["gibbs"]["burn_in"],
        thin=cfg["gibbs"]["thin"], seed=cfg["general"]["random_seed"],
        trend="c", verbose=True,
    )
    eig_last = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"\n      Stored draws: {fit.beta_path_draws.shape[0]}")
    print(f"      Max |eigenvalue| (posterior mean at last period): {eig_last:.4f}")

    print("\n[4/4] Time-varying coefficient diagnostics...")
    beta_path = fit.beta_path_posterior_mean   # (T_eff, k, n)
    var_names = fit.variable_names

    print(f"\n      Posterior-mean own-lag-1 coefficient over time:")
    print(f"        {'date':<10}  " + "  ".join(f"{v:>10}" for v in var_names))
    T_eff = beta_path.shape[0]
    dates_eff = data.index[spec.lags:]
    n = len(var_names)
    own_diag = np.zeros((T_eff, n))
    for t in range(T_eff):
        own_diag[t] = np.diag(beta_path[t][:n, :])
    # Show snapshots
    for idx in [0, T_eff // 4, T_eff // 2, 3 * T_eff // 4, T_eff - 1]:
        d = dates_eff[idx]
        vals = own_diag[idx]
        print(f"        {d:%Y-%m}    " + "  ".join(f"{v:>+10.3f}" for v in vals))

    print(f"\n      Range of own-lag-1 over sample (min, max):")
    for j, v in enumerate(var_names):
        print(f"        {v:<14}  [{own_diag[:, j].min():+.3f}, {own_diag[:, j].max():+.3f}]")

    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Save the time-varying own-lag-1 coefficients
    own_lag_df = pd.DataFrame(own_diag, index=dates_eff, columns=var_names)
    own_lag_df.to_csv(outputs_dir / "own_lag1_path.csv")
    print(f"\n      ->{(outputs_dir / 'own_lag1_path.csv').relative_to(PROJECT_ROOT)}")

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
        "model": {
            "class": "tvp_var",
            "lags": spec.lags,
            "training_sample": spec.training_sample,
            "Q_prior_scale": spec.Q_prior_scale,
            "max_eigenvalue_modulus_last_period": float(eig_last),
        },
        "gibbs": {
            "n_iter": cfg["gibbs"]["n_iter"],
            "burn_in": cfg["gibbs"]["burn_in"],
            "n_stored": int(fit.beta_path_draws.shape[0]),
        },
        "own_lag1_summary": {
            v: {
                "min": float(own_diag[:, j].min()),
                "max": float(own_diag[:, j].max()),
                "mean": float(own_diag[:, j].mean()),
            } for j, v in enumerate(var_names)
        },
        "seed": cfg["general"]["random_seed"],
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # Any coefficient drift: max - min across own-lag-1 paths > 0
        max_range = max(
            float(own_diag[:, j].max() - own_diag[:, j].min())
            for j in range(len(var_names))
        )
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(len(var_names)),
            "n_lags": float(spec.lags),
            "stable_last_period": 1.0 if eig_last < 1.0 else 0.0,
            "max_eig_last": float(eig_last),
            "n_stored": float(fit.beta_path_draws.shape[0]),
            "tvp_coefs_drift": 1.0 if max_range > 0 else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — TVP-VAR coefficients evolve over time;
    #      static IRF tensor doesn't apply. The headline is the
    #      time-varying own-lag-1 path summarised in the specification
    #      table and saved as CSV) ----
    spec_dict = {
        "Model": f"TVP-VAR({spec.lags}), homoskedastic (Primiceri 2005 without SV)",
        "Variables": ", ".join(var_names),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(spec.lags),
        "Training sample": str(spec.training_sample),
        "Q-prior scale (k_Q²)": str(spec.Q_prior_scale),
        "Gibbs iterations": str(cfg["gibbs"]["n_iter"]),
        "Burn-in": str(cfg["gibbs"]["burn_in"]),
        "Posterior draws stored": str(int(fit.beta_path_draws.shape[0])),
        "Max |eigenvalue| (last period)": f"{float(eig_last):.4f}",
    }
    for j, v in enumerate(var_names):
        spec_dict[f"Own-lag-1 range ({v})"] = (
            f"[{own_diag[:, j].min():+.3f}, {own_diag[:, j].max():+.3f}]"
        )

    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"] + " — see own_lag1_path.csv "
                 "for the full time path",
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
