"""
Replication of Lanne, Meitz, Saikkonen (2017) — non-Gaussian SVAR.

The paper is methodological: B in u_t = B ε_t is identified up to column
permutation and scaling (sign under unit-variance normalisation)
whenever the structural shocks are mutually
independent with at most one Gaussian component. This script demonstrates
the methodology on a standard small US monetary VAR (INDPRO / CPI /
FEDFUNDS), monthly 1960-2007 (pre-ZLB).

Three passes:
  1. Cholesky baseline (FFR ordered last).
  2. LMS / FastICA (Hyvärinen 1999 fixed-point ICA).
  3. LMS-style pml_t (parametric t-ML ~ LMS Sec. 4.5 step 2,
     additionally estimating per-shock degrees of freedom).

Outputs:
  outputs/report.html              — pml_t IRFs with bootstrap bands
  outputs/numerical_results.json   — IRF snapshot
  outputs/research_log.json        — config + diagnostics for all 3 passes
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))   # so `lib` is importable

import lib   # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def load_and_transform(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw = lib.load_dataset(
        source=data_cfg["source"],
        series={s["name"]: s["fred_code"] for s in data_cfg["series"]},
        start=data_cfg["sample_start"],
        end=data_cfg["sample_end"],
    )
    var_cols = data_cfg["variable_order"]
    return lib.transform(raw[var_cols], data_cfg["transforms"])


def hash_inputs(cfg: dict, data: pd.DataFrame) -> str:
    # Full-data digest: every observation enters the hash (an endpoint-only
    # summary would miss interior changes to the data).
    data_digest = hashlib.sha256(
        np.ascontiguousarray(data.to_numpy(dtype="float64")).tobytes()
        + "|".join(map(str, data.index)).encode()
        + "|".join(map(str, data.columns)).encode()
    ).hexdigest()
    payload = {
        "config": cfg,
        "data_sha256": data_digest,
        "n_obs": len(data),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    # ---------------------------------------------------------------
    print("\n[1/6] Loading and transforming data...")
    data = load_and_transform(cfg)
    print(
        f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} "
        f"({len(data)} obs, {data.shape[1]} variables)"
    )

    # ---------------------------------------------------------------
    print("\n[2/6] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data,
        lags=cfg["model"]["lags"],
        trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(
        f"      VAR({fit.p}), T_eff={fit.t_eff}, "
        f"max |eigenvalue|={eig:.4f}, AIC={fit.fit_info['aic']:.3f}"
    )

    # Per-equation residual normality check — informs whether LMS can identify
    print("\n      Residual non-Gaussianity (Jarque-Bera per equation):")
    from scipy import stats as _stats
    for i, name in enumerate(fit.variable_names):
        u = np.asarray(fit.residuals)[:, i]
        jb_stat, jb_p = _stats.jarque_bera(u)
        skew = _stats.skew(u)
        kurt = _stats.kurtosis(u)
        normality = "REJECT  Gaussian" if jb_p < 0.01 else (
            "ambiguous"
        )
        print(
            f"        {name:>8}: JB={jb_stat:>10.2f}  p={jb_p:.2e}  "
            f"skew={skew:+.3f}  excess-kurt={kurt:+.2f}  ({normality})"
        )

    # ---------------------------------------------------------------
    print("\n[3/6] Cholesky baseline identification...")
    chol_cfg = cfg["identification"]["cholesky"]
    irf_chol = lib.cholesky_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        shock_names=[f"{v}_shock" for v in fit.variable_names],
        bands="asymptotic_mc",
        n_draws=cfg["irf"]["n_draws"],
        seed=seed,
    )
    shock_of_interest = chol_cfg["shock_of_interest"]
    sh_idx_chol = irf_chol.shock_names.index(shock_of_interest)
    print(
        f"      Cholesky impulse responses to the {shock_of_interest} "
        f"(textbook contractionary MP shock, ffr ordered last):"
    )
    print(f"        {'h':>3}  " + "  ".join(
        f"{name:>10}" for name in fit.variable_names
    ))
    for h in [0, 1, 3, 6, 12, 24, 48]:
        if h > irf_chol.horizon:
            continue
        vals = [irf_chol.median[h, i, sh_idx_chol] for i in range(fit.n_vars)]
        print(f"        {h:>3}  " + "  ".join(f"{v:>+10.5f}" for v in vals))

    # ---------------------------------------------------------------
    ng_cfg = cfg["identification"]["nongaussian"]

    def _run_lms(method: str, **method_kwargs):
        return lib.nongaussian_svar_irf(
            fit=fit,
            horizon=cfg["irf"]["horizon"],
            method=method,
            shock_names=ng_cfg["shock_names"],
            label_by=ng_cfg["label_by"],
            sign_normalization=ng_cfg["sign_normalization"],
            bands=cfg["irf"]["bands"],
            n_draws=cfg["irf"]["n_draws"],
            seed=seed,
            **method_kwargs,
        )

    print("\n[4/6] LMS / FastICA non-Gaussian identification...")
    irf_fica, diag_fica = _run_lms(
        "fastica", nonlinearity=ng_cfg["fastica"]["nonlinearity"],
    )
    print(
        f"      FastICA converged: {diag_fica['fastica_converged']} "
        f"in {diag_fica['fastica_n_iter']} iterations "
        f"(final diff = {diag_fica['fastica_final_diff']:.2e})"
    )
    print(f"      Bootstrap bands: {cfg['irf']['n_draws']} draws")

    print("\n[5/6] pml_t parametric t-ML (~ LMS Sec. 4.5 step 2)...")
    irf_pml, diag_pml = _run_lms(
        "pml_t",
        nonlinearity=ng_cfg["fastica"]["nonlinearity"],   # used for warm-start
        pml_t_n_restarts=ng_cfg["pml_t"]["n_restarts"],
        pml_t_nu_init=ng_cfg["pml_t"]["nu_init"],
    )
    print(
        f"      pml_t converged: {diag_pml['pml_t_converged']} "
        f"(best restart: {diag_pml['pml_t_best_restart_index']} / "
        f"{diag_pml['pml_t_n_restarts_tried']})"
    )
    print(
        f"      pml_t log-likelihood: {diag_pml['pml_t_log_likelihood']:.2f}"
    )
    print(f"      pml_t recovered df per shock:")
    for sh, nu_val in zip(irf_pml.shock_names, diag_pml["pml_t_nu"]):
        print(f"        {sh:>16}: nu = {nu_val:>7.2f} "
              f"({'near-Gaussian' if nu_val > 30 else 'heavy-tailed' if nu_val < 5 else 'leptokurtic'})")

    # ---------------------------------------------------------------
    print("\n      Recovered B side-by-side (rows = variables; cols = shocks):")
    B_fica = np.array(diag_fica["B_point"])
    B_pml = np.array(diag_pml["B_point"])
    print(f"        {'variable':>10}  "
          + "  ".join(f"{sh+' (fica)':>22}" for sh in irf_fica.shock_names))
    for i, name in enumerate(fit.variable_names):
        print(f"        {name:>10}  "
              + "  ".join(
                  f"{B_fica[i, k]:>+10.4f}  /  {B_pml[i, k]:>+8.4f}"
                  for k in range(fit.n_vars)
              ))
    # B agreement: Frobenius norm of difference
    B_diff = float(np.linalg.norm(B_fica - B_pml, ord="fro"))
    B_rel = B_diff / max(float(np.linalg.norm(B_fica, ord="fro")), 1e-12)
    print(f"      ||B_fica - B_pml||_F / ||B_fica||_F = {B_rel:.4f}")
    if B_rel < 0.05:
        print(
            "      -> FastICA and pml_t produce nearly identical B "
            "(< 5% relative Frobenius difference)"
        )
    else:
        print(
            f"      -> FastICA and pml_t diverge by {B_rel * 100:.1f}% "
            "(relative Frobenius; investigate distributional assumption)"
        )

    print()
    print("      Diagnostics on recovered shocks (pml_t, post labelling/sign):")
    print(f"        {'shock':>16}  {'JB stat':>10}  {'p-value':>10}  "
          f"{'skew':>8}  {'kurt':>8}")
    for sh in irf_pml.shock_names:
        jb = diag_pml["jarque_bera"][sh]
        sk = diag_pml["skewness"][sh]
        ku = diag_pml["excess_kurtosis"][sh]
        print(
            f"        {sh:>16}  {jb['stat']:>10.2f}  {jb['pvalue']:>10.2e}  "
            f"{sk:>+8.3f}  {ku:>+8.2f}"
        )
    print(f"      weak_nongaussianity flag: {diag_pml['weak_nongaussianity']} "
          f"(shocks not rejecting normality: {diag_pml['n_gaussian_like']})")
    if diag_pml["weak_nongaussianity"]:
        print(
            "      CAUTION: two or more recovered shocks cannot reject "
            "Gaussianity. LMS identification permits at most one Gaussian "
            "component, so identification is weak/inconclusive here - "
            "interpret with care and seek corroborating evidence (this is "
            "not by itself proof of failure)."
        )

    print()
    print("      MP-shock IRF (medians) side-by-side at selected horizons:")
    print(f"        {'h':>3}  "
          + "  ".join(
              f"{name + ' (chol/fica/pml)':>32}" for name in fit.variable_names
          ))
    sh_idx_lms_fica = irf_fica.shock_names.index("monetary_policy")
    sh_idx_lms_pml = irf_pml.shock_names.index("monetary_policy")
    for h in [0, 1, 3, 6, 12, 24, 48]:
        if h > cfg["irf"]["horizon"]:
            continue
        cells = []
        for i in range(fit.n_vars):
            chol_v = irf_chol.median[h, i, sh_idx_chol]
            fica_v = irf_fica.median[h, i, sh_idx_lms_fica]
            pml_v = irf_pml.median[h, i, sh_idx_lms_pml]
            cells.append(f"{chol_v:>+8.4f} / {fica_v:>+8.4f} / {pml_v:>+8.4f}")
        print(f"        {h:>3}  " + "  ".join(f"{c:>32}" for c in cells))

    # ---------------------------------------------------------------
    print("\n[6/6] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    labels = cfg["data"]["variable_labels"]
    var_labels = [labels.get(v, v) for v in fit.variable_names]

    snapshot_path = outputs_dir / "numerical_results.json"
    irf_pml.save_snapshot(snapshot_path)
    print(f"      ->{snapshot_path.relative_to(PROJECT_ROOT)}")

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hash": hash_inputs(cfg, data),
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
            "trend": cfg["model"]["trend"],
            "max_eigenvalue_modulus": float(eig),
        },
        "identification": {
            "cholesky": {
                "ordering": chol_cfg["ordering"],
                "shock_of_interest": shock_of_interest,
            },
            "nongaussian_fastica": {
                "method": "fastica",
                "nonlinearity": ng_cfg["fastica"]["nonlinearity"],
                "shock_names": ng_cfg["shock_names"],
                "label_by": ng_cfg["label_by"],
                "sign_normalization": ng_cfg["sign_normalization"],
                "diagnostics": {
                    "jarque_bera": diag_fica["jarque_bera"],
                    "skewness": diag_fica["skewness"],
                    "excess_kurtosis": diag_fica["excess_kurtosis"],
                    "max_jb_pvalue": diag_fica["max_jb_pvalue"],
                    "weak_nongaussianity": diag_fica["weak_nongaussianity"],
                    "fastica_converged": diag_fica["fastica_converged"],
                    "fastica_n_iter": diag_fica["fastica_n_iter"],
                    "B_point": diag_fica["B_point"],
                },
            },
            "nongaussian_pml_t": {
                "method": "pml_t",
                "n_restarts": ng_cfg["pml_t"]["n_restarts"],
                "nu_init": ng_cfg["pml_t"]["nu_init"],
                "shock_names": ng_cfg["shock_names"],
                "label_by": ng_cfg["label_by"],
                "sign_normalization": ng_cfg["sign_normalization"],
                "diagnostics": {
                    "jarque_bera": diag_pml["jarque_bera"],
                    "skewness": diag_pml["skewness"],
                    "excess_kurtosis": diag_pml["excess_kurtosis"],
                    "max_jb_pvalue": diag_pml["max_jb_pvalue"],
                    "weak_nongaussianity": diag_pml["weak_nongaussianity"],
                    "pml_t_log_likelihood": diag_pml["pml_t_log_likelihood"],
                    "pml_t_nu": diag_pml["pml_t_nu"],
                    "pml_t_converged": diag_pml["pml_t_converged"],
                    "pml_t_n_restarts_tried": diag_pml["pml_t_n_restarts_tried"],
                    "pml_t_best_restart_index": diag_pml["pml_t_best_restart_index"],
                    "B_point": diag_pml["B_point"],
                },
            },
            "fastica_vs_pml_t_B_frobenius_relative": B_rel,
        },
        "irf": {
            "horizon": cfg["irf"]["horizon"],
            "bands": cfg["irf"]["bands"],
            "n_draws": cfg["irf"]["n_draws"],
            "credible_intervals": cfg["irf"]["credible_intervals"],
        },
        "seed": seed,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        nu_list = diag_pml["pml_t_nu"]
        nu_all_heavy = all(2.0 < v < 10.0 for v in nu_list)
        validation_results = {
            "n_obs": float(len(data)),
            "max_eig_modulus": float(eig),
            "fastica_converged":
                1.0 if diag_fica["fastica_converged"] else 0.0,
            "fastica_n_iter": float(diag_fica["fastica_n_iter"]),
            "weak_nongaussianity":
                1.0 if diag_fica["weak_nongaussianity"] else 0.0,
            "pml_t_converged":
                1.0 if diag_pml["pml_t_converged"] else 0.0,
            "pml_t_nu_all_heavy_tailed":
                1.0 if nu_all_heavy else 0.0,
            "fastica_vs_pml_t_frobenius": float(B_rel),
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report — pml_t as headline, FastICA as alt (Robustness) ----
    nu_str = ", ".join(f"{nu:.2f}" for nu in diag_pml["pml_t_nu"])
    spec_dict = {
        "Model": f"VAR({fit.p}) + Non-Gaussian SVAR (Lanne-Meitz-Saikkonen 2017)",
        "Variables": ", ".join(var_labels),
        "Identification": "Independent components — pml_t (parametric t-ML, ~ LMS Sec. 4.5 step 2)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "pml_t shock degrees-of-freedom ν̂": nu_str,
        "All shocks heavy-tailed (2 < ν < 10)": (
            "yes" if all(2.0 < v < 10.0 for v in diag_pml["pml_t_nu"]) else "no"
        ),
        "FastICA converged": "yes" if diag_fica["fastica_converged"] else "no",
        "FastICA iterations": str(diag_fica["fastica_n_iter"]),
        "Weak non-Gaussianity flag (FastICA)":
            "yes" if diag_fica["weak_nongaussianity"] else "no",
        "||B_fica − B_pml||_F / ||B_fica||_F": f"{B_rel:.4f}",
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report_ng = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"] + " — LMS / pml_t",
    )
    report_ng.add_standard_sections(
        fit=fit, irf=irf_pml,
        spec=spec_dict, log=log,
        var_labels=var_labels,
        alt_irf=irf_fica,
        alt_label="FastICA (method='fastica')",
        baseline_label="pml_t (method='pml_t')",
        shock_idx_for_summary=0,
        x_axis_label="Months",
    narrative="auto",
    )
    report_ng_path = outputs_dir / "report.html"
    report_ng.save(report_ng_path)
    print(f"      ->{report_ng_path.relative_to(PROJECT_ROOT)}")

    log_path = outputs_dir / "research_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"      ->{log_path.relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
