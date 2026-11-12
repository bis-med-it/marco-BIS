"""
Bańbura-Modugno (2014) DFM with missing data — Bayesian Gibbs demo.

(The paper estimates by frequentist ML via a modified EM algorithm and
defers Bayesian treatment to future research, p. 150; the Gibbs sampler
here is this repo's documented substitution.)

Original paper: EM algorithm for DFM with arbitrary missing-data patterns,
applied to euro-area nowcasting. This replication substitutes:
  - Gibbs sampling (lib.fit_dfm_bayesian) for EM
  - US FRED-MD subset for the euro-area panel
  - Simulated ragged-edge pattern to demonstrate NaN handling

The demonstration shares the paper's missing-data state-space structure
but uses Bayesian Gibbs estimation: missing observations are handled via
the Kalman simulation smoother.
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


def apply_ragged_edge(df: pd.DataFrame, ragged: dict) -> pd.DataFrame:
    """NaN out the last k months of each named series to simulate
    publication-lag ragged edge."""
    df_ragged = df.copy()
    for col, k in ragged.items():
        if col in df_ragged.columns and k > 0:
            df_ragged.iloc[-k:, df_ragged.columns.get_loc(col)] = np.nan
    return df_ragged


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    data_clean = load_and_transform(cfg)
    print(f"      Sample: {data_clean.index[0]:%Y-%m} to {data_clean.index[-1]:%Y-%m}"
          f" ({len(data_clean)} months, {data_clean.shape[1]} variables)")

    print("\n[2/4] Inducing ragged-edge missing pattern...")
    data = apply_ragged_edge(data_clean, cfg["ragged_edge"])
    n_missing = int(data.isna().sum().sum())
    print(f"      Variables with ragged edge: {list(cfg['ragged_edge'].keys())}")
    print(f"      Total NaN cells: {n_missing}")

    print("\n[3/4] Estimating Bayesian DFM...")
    spec = lib.DFMSpec(
        n_factors=cfg["model"]["n_factors"],
        standardize=cfg["model"]["standardize"],
    )
    fit = lib.fit_dfm_bayesian(
        data, spec,
        n_iter=cfg["gibbs"]["n_iter"], burn_in=cfg["gibbs"]["burn_in"],
        thin=cfg["gibbs"]["thin"], seed=cfg["general"]["random_seed"],
        verbose=True,
    )
    print(f"\n      Stored draws: {fit.factor_path_draws.shape[0]}")
    print(f"      Factor AR(1) coefficient matrix Phi (posterior mean):")
    Phi = fit.factor_AR_posterior_mean
    for i in range(Phi.shape[0]):
        print(f"        [{', '.join(f'{Phi[i, j]:+7.3f}' for j in range(Phi.shape[1]))}]")
    print(f"      Max |Phi eigenvalue|: {np.max(np.abs(np.linalg.eigvals(Phi))):.4f}")

    print("\n[4/4] Comparing factors with PC baseline...")
    # Compute PC factors on the same data (NaN-imputed for PC)
    Y_imputed = data.interpolate(method="linear", limit_direction="both").fillna(0).values
    means = np.nanmean(data.values, axis=0)
    stds = np.nanstd(data.values, axis=0, ddof=1)
    stds[stds < 1e-12] = 1.0
    Y_std = (Y_imputed - means) / stds
    from lib.factors import _pc_factors
    F_pc, _ = _pc_factors(Y_std, K=spec.n_factors)

    # Correlate each Bayesian factor with each PC factor (factors are
    # identified only up to rotation/sign, so we look at the best match
    # via absolute correlation).
    F_bayes = fit.factor_path_posterior_mean
    print(f"\n      Correlation matrix (Bayesian DFM rows × PC columns):")
    print(f"        {'factor':<10}  " + "  ".join(f"PC_{j+1:>2}" for j in range(spec.n_factors)))
    for i in range(spec.n_factors):
        cors = [np.corrcoef(F_bayes[:, i], F_pc[:, j])[0, 1]
                 for j in range(spec.n_factors)]
        print(f"        bayes_F_{i+1}:  " + "  ".join(f"{c:>+6.3f}" for c in cors))
    print(f"\n      (Factors identified up to rotation/sign — pick max |corr| per row.)")

    print("\n      Imputed values at the ragged edge:")
    print(f"        {'date':<10}  " + "  ".join(
        f"{c:<10}" for c in cfg["ragged_edge"].keys()
    ))
    # Show the last few rows where NaN values were imputed via factor model.
    # Use the posterior mean of the common component accumulated WITHIN each
    # draw: factors and loadings are identified only up to sign/rotation
    # across draws, so mean(F) @ mean(Λ)' is not a valid reconstruction
    # (equivalent draws (F, Λ) and (−F, −Λ) would cancel).
    y_hat_std = fit.common_component_posterior_mean       # standardised
    y_hat = y_hat_std * fit.stds + fit.means              # back to original scale
    for t_off in [3, 2, 1, 0]:
        idx = -t_off - 1 if t_off > 0 else -1
        if abs(idx) > len(data):
            continue
        d = data.index[idx]
        vals = []
        for c in cfg["ragged_edge"].keys():
            obs = data.iloc[idx][c]
            j = data.columns.get_loc(c)
            imputed = y_hat[idx, j]
            if pd.isna(obs):
                vals.append(f"({imputed:+6.3f})")
            else:
                vals.append(f" {obs:+6.3f} ")
        print(f"        {d:%Y-%m}    " + "  ".join(vals))
    print("        (Parentheses = imputed via factor model; bare numbers = observed)")

    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Save factor path and imputed series
    factor_df = pd.DataFrame(
        F_bayes,
        index=data.index,
        columns=[f"F{i+1}" for i in range(spec.n_factors)],
    )
    factor_df.to_csv(outputs_dir / "factor_path.csv")
    print(f"\n      ->{(outputs_dir / 'factor_path.csv').relative_to(PROJECT_ROOT)}")

    # Per-factor posterior-mean loadings are published WITH the caveat that
    # factors are identified only up to sign/rotation across draws — use the
    # draw-wise common component (above) for any reconstruction.
    loadings_df = pd.DataFrame(
        fit.loadings_posterior_mean,
        index=data.columns,
        columns=[f"F{i+1}_loading" for i in range(spec.n_factors)],
    )
    loadings_df.to_csv(outputs_dir / "loadings.csv")
    print(f"      ->{(outputs_dir / 'loadings.csv').relative_to(PROJECT_ROOT)}")

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_version": getattr(lib, "__version__", "unversioned"),
        "data": {
            "source": cfg["data"]["source"],
            "sample_start": str(data.index[0].date()),
            "sample_end": str(data.index[-1].date()),
            "n_obs": int(len(data)),
            "n_variables": int(data.shape[1]),
            "n_missing_cells": int(n_missing),
            "ragged_edge": dict(cfg["ragged_edge"]),
        },
        "model": {
            "class": "bayesian_dfm",
            "n_factors": spec.n_factors,
            "standardize": spec.standardize,
        },
        "gibbs": {
            "n_iter": cfg["gibbs"]["n_iter"],
            "burn_in": cfg["gibbs"]["burn_in"],
            "n_stored": int(fit.factor_path_draws.shape[0]),
        },
        "max_factor_AR_eigenvalue": float(np.max(np.abs(np.linalg.eigvals(Phi)))),
        "seed": cfg["general"]["random_seed"],
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        max_eig = float(np.max(np.abs(np.linalg.eigvals(Phi))))
        # Best |corr| of first Bayesian factor with any PC factor
        f1_cors = [abs(np.corrcoef(F_bayes[:, 0], F_pc[:, j])[0, 1])
                    for j in range(spec.n_factors)]
        validation_results = {
            "n_obs": float(len(data)),
            "n_variables": float(data.shape[1]),
            "n_factors": float(spec.n_factors),
            "phi_stable": 1.0 if max_eig < 1.0 else 0.0,
            "max_phi_eig": max_eig,
            "n_stored": float(fit.factor_path_draws.shape[0]),
            "bayes_f1_pc_corr": float(max(f1_cors)),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (slim — BDFM produces factor paths + loadings,
    #      not IRFs) ----
    max_phi = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    spec_dict = {
        "Model": f"Bayesian Dynamic Factor Model with {spec.n_factors} factors (Banbura-Modugno 2014)",
        "Information set size": str(int(data.shape[1])),
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Missing cells (ragged-edge)": str(int(n_missing)),
        "Factors": str(spec.n_factors),
        "Standardise": "yes" if spec.standardize else "no",
        "Gibbs iterations": str(cfg["gibbs"]["n_iter"]),
        "Burn-in": str(cfg["gibbs"]["burn_in"]),
        "Posterior draws stored": str(int(fit.factor_path_draws.shape[0])),
        "Max factor AR eigenvalue": f"{max_phi:.4f}",
        "Bayes F1 vs PC best |corr|": (
            f"{max([abs(np.corrcoef(F_bayes[:, 0], F_pc[:, j])[0, 1]) for j in range(spec.n_factors)]):.4f}"
        ),
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"] + " — see loadings.csv and "
                 "research log for factor diagnostics",
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
