"""
Replication of Andrade, Ferroni, Melosi (2023) — higher-moment SVAR
restrictions.

Two-pass methodology comparison on a US monetary VAR (INDPRO / CPI / FFR /
BAA-FFR spread, monthly 1965-2007):

  PASS 1: multi-shock sign restrictions (RWZ 2010), three identified
          shocks — demand / supply / monetary
  PASS 2: same sign restrictions PLUS an own-shock kurtosis threshold
          (the AFM 2023 form; the paper's US application uses ROBUST
          excess kurtosis > 1.2):
            excess_kurtosis(monetary) >= 1.0

Headline question: do AFM's higher-moment restrictions tighten the
68% credible bands on the monetary-shock IRFs?

Outputs:
  outputs/report.html              — Pass 2 (sign + AFM) IRFs
  outputs/numerical_results.json   — Pass 2 IRF snapshot
  outputs/research_log.json        — both passes + comparison + provenance
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
sys.path.insert(0, str(HERE.parent.parent))

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


def avg_68_width(irf: lib.IRFContainer, shock_name: str,
                 vars_to_avg: list[str]) -> float:
    """Average 68% half-width across the named variables and all horizons."""
    sh_idx = irf.shock_names.index(shock_name)
    widths = []
    for v in vars_to_avg:
        v_idx = irf.variable_names.index(v)
        w = irf.upper_68[:, v_idx, sh_idx] - irf.lower_68[:, v_idx, sh_idx]
        widths.append(np.abs(w).mean())
    return float(np.mean(widths))


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    # ---------------------------------------------------------------
    print("\n[1/5] Loading and transforming data...")
    data = load_and_transform(cfg)
    print(
        f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} "
        f"({len(data)} obs, {data.shape[1]} variables)"
    )

    # ---------------------------------------------------------------
    print("\n[2/5] Estimating reduced-form VAR...")
    fit = lib.fit_var(
        data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(
        f"      VAR({fit.p}), T_eff={fit.t_eff}, "
        f"max |eigenvalue|={eig:.4f}, AIC={fit.fit_info['aic']:.3f}"
    )

    # ---------------------------------------------------------------
    print("\n[3/5] PASS 1: Multi-shock sign restrictions only (RWZ 2010)...")
    id_cfg = cfg["identification"]
    irf_sign = lib.sign_restricted_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        sign_restrictions=id_cfg["sign_restrictions"],
        restriction_horizons=id_cfg["restriction_horizons"],
        target_accepted=cfg["irf"]["target_accepted"],
        max_rotations=cfg["irf"]["max_rotations"],
        seed=seed,
        parameter_uncertainty=True,
    )
    print(
        f"      Accepted: {irf_sign.metadata['n_accepted']}/"
        f"{irf_sign.metadata['n_attempted']} rotations "
        f"(acceptance rate: {irf_sign.metadata['acceptance_rate']:.4f})"
    )
    print(
        f"      Rejection breakdown: sign={irf_sign.metadata['n_rejected_by_sign']}"
    )

    # ---------------------------------------------------------------
    print(
        "\n[4/5] PASS 2: Sign restrictions + higher-moment (AFM 2023)..."
    )
    irf_hm = lib.sign_restricted_irf(
        fit=fit,
        horizon=cfg["irf"]["horizon"],
        sign_restrictions=id_cfg["sign_restrictions"],
        restriction_horizons=id_cfg["restriction_horizons"],
        higher_moment_restrictions=id_cfg["higher_moment_restrictions"],
        target_accepted=cfg["irf"]["target_accepted"],
        max_rotations=cfg["irf"]["max_rotations"],
        seed=seed,
        parameter_uncertainty=True,
    )
    print(
        f"      Accepted: {irf_hm.metadata['n_accepted']}/"
        f"{irf_hm.metadata['n_attempted']} rotations "
        f"(acceptance rate: {irf_hm.metadata['acceptance_rate']:.4f})"
    )
    print(
        f"      Rejection breakdown: "
        f"sign={irf_hm.metadata['n_rejected_by_sign']}, "
        f"higher_moment={irf_hm.metadata['n_rejected_by_higher_moment']}"
    )

    # ---------------------------------------------------------------
    print(
        "\n      Pass 1 vs Pass 2: monetary_shock IRF medians at selected "
        "horizons:"
    )
    sh = "monetary_shock"
    var_names = list(fit.variable_names)
    sh_idx_1 = irf_sign.shock_names.index(sh)
    sh_idx_2 = irf_hm.shock_names.index(sh)
    horizons_to_print = cfg["report"]["horizons_to_report"]
    print(
        f"        {'h':>3}  "
        + "  ".join(f"{name:>20}" for name in var_names)
    )
    print(
        f"        {'':>3}  "
        + "  ".join(f"{'(sign / sign+HM)':>20}" for _ in var_names)
    )
    for h in horizons_to_print:
        if h > irf_sign.horizon:
            continue
        row = []
        for i in range(fit.n_vars):
            v1 = irf_sign.median[h, i, sh_idx_1]
            v2 = irf_hm.median[h, i, sh_idx_2]
            row.append(f"{v1:+.4f} / {v2:+.4f}")
        print(f"        {h:>3}  " + "  ".join(f"{c:>20}" for c in row))

    print()
    print("      Band-tightening from higher-moment restrictions:")
    print(
        f"        {'variable':>10}  "
        f"{'sign-only 68% width':>22}  "
        f"{'sign+HM 68% width':>22}  "
        f"{'ratio (HM/sign)':>18}"
    )
    band_summary: dict = {}
    for v in var_names:
        v_idx = var_names.index(v)
        w_sign = float(np.abs(
            irf_sign.upper_68[:, v_idx, sh_idx_1]
            - irf_sign.lower_68[:, v_idx, sh_idx_1]
        ).mean())
        w_hm = float(np.abs(
            irf_hm.upper_68[:, v_idx, sh_idx_2]
            - irf_hm.lower_68[:, v_idx, sh_idx_2]
        ).mean())
        ratio = w_hm / max(w_sign, 1e-12)
        band_summary[v] = {
            "sign_only_avg_68_width": w_sign,
            "hm_avg_68_width": w_hm,
            "ratio": ratio,
        }
        print(
            f"        {v:>10}  {w_sign:>22.5f}  {w_hm:>22.5f}  "
            f"{ratio:>+18.3f}"
        )
    avg_ratio = float(np.mean([s["ratio"] for s in band_summary.values()]))
    print(f"\n      Average 68% half-width ratio (HM/sign-only): {avg_ratio:.3f}")
    if avg_ratio < 1.0:
        print(
            "      -> AFM higher-moment restrictions TIGHTEN the bands "
            f"(by {(1 - avg_ratio) * 100:.1f}% on average)"
        )
    else:
        print(
            "      -> AFM higher-moment restrictions DO NOT TIGHTEN the bands "
            "(consider stronger restrictions or check sample)"
        )

    # ---------------------------------------------------------------
    print("\n[5/5] Writing outputs...")
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    labels = cfg["data"]["variable_labels"]
    var_labels = [labels.get(v, v) for v in fit.variable_names]

    snapshot_path = outputs_dir / "numerical_results.json"
    irf_hm.save_snapshot(snapshot_path)
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
        "identification": dict(cfg["identification"]),
        "pass_1_sign_only": {
            "n_accepted": irf_sign.metadata["n_accepted"],
            "n_attempted": irf_sign.metadata["n_attempted"],
            "acceptance_rate": irf_sign.metadata["acceptance_rate"],
            "n_rejected_by_sign": irf_sign.metadata["n_rejected_by_sign"],
        },
        "pass_2_sign_plus_higher_moment": {
            "n_accepted": irf_hm.metadata["n_accepted"],
            "n_attempted": irf_hm.metadata["n_attempted"],
            "acceptance_rate": irf_hm.metadata["acceptance_rate"],
            "n_rejected_by_sign": irf_hm.metadata["n_rejected_by_sign"],
            "n_rejected_by_higher_moment":
                irf_hm.metadata["n_rejected_by_higher_moment"],
        },
        "band_summary": band_summary,
        "avg_ratio_hm_over_sign": avg_ratio,
        "irf": {
            "horizon": cfg["irf"]["horizon"],
            "target_accepted": cfg["irf"]["target_accepted"],
            "max_rotations": cfg["irf"]["max_rotations"],
        },
        "seed": seed,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(fit.n_vars),
            "n_lags": float(fit.p),
            "max_eig_modulus": float(eig),
            "pass1_acceptance_rate": float(irf_sign.metadata["acceptance_rate"]),
            "hm_filter_active":
                1.0 if irf_hm.metadata["n_rejected_by_higher_moment"] > 1000 else 0.0,
            "avg_band_ratio_hm_vs_sign": float(avg_ratio),
            "both_passes_hit_target":
                1.0 if (irf_sign.metadata["n_accepted"] >= cfg["irf"]["target_accepted"]
                         and irf_hm.metadata["n_accepted"] >= cfg["irf"]["target_accepted"]) else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report — sign+AFM as headline; sign-only as Robustness alt
    #      (shows band tightening / widening from higher-moment filter) ----
    spec_dict = {
        "Model": f"VAR({fit.p}) + sign + higher-moment restrictions (AFM 2023)",
        "Variables": ", ".join(var_labels),
        "Identification": "Sign + sample excess-kurtosis threshold (>1 for the monetary shock; AFM-inspired)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Pass 1 acceptance rate (sign-only)": f"{irf_sign.metadata['acceptance_rate']:.4f}",
        "Pass 2 acceptance rate (sign+AFM)": f"{irf_hm.metadata['acceptance_rate']:.4f}",
        "AFM rejection count": str(irf_hm.metadata["n_rejected_by_higher_moment"]),
        "Avg band ratio (Pass2 / Pass1)": f"{avg_ratio:.4f}",
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"] + " — Pass 2 (sign + AFM)",
    )
    report.add_standard_sections(
        fit=fit, irf=irf_hm,
        spec=spec_dict, log=log,
        var_labels=var_labels,
        alt_irf=irf_sign,
        alt_label="Pass 1: sign-only baseline",
        baseline_label="Pass 2: sign + AFM",
        shock_idx_for_summary=0,
        x_axis_label="Months",
    narrative="auto",
    )
    report_path = outputs_dir / "report.html"
    report.save(report_path)
    print(f"      ->{report_path.relative_to(PROJECT_ROOT)}")

    log_path = outputs_dir / "research_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"      ->{log_path.relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
