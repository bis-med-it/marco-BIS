"""
Replication of Kilian (2009), AER 99(3) — "Not All Oil Price Shocks Are Alike".

Recursive (Cholesky) global oil-market SVAR: three structural shocks — oil
supply, aggregate demand, and oil-market-specific (precautionary) demand —
from a 3-variable monthly VAR(24), 1973m2-2007m12. Thin orchestration over
scripts/lib (lib.fit_var -> lib.cholesky_irf). Point-identified / deterministic.

Data: Kilian's deposited data.txt, fetched on first run from the public
JCR-Econ (2025) replication repo and cached under data/.
"""

from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import sys
import urllib.request
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


def load_data(cfg: dict) -> pd.DataFrame:
    cache = HERE / cfg["data"]["cache"]
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"      fetching {cfg['data']['source_url']}")
        urllib.request.urlretrieve(cfg["data"]["source_url"], cache)
    raw = pd.read_csv(cache, sep=r"\s+", header=None)
    raw.columns = cfg["data"]["variables"]
    raw.index = pd.period_range(cfg["data"]["start_period"], periods=len(raw), freq="M").to_timestamp()
    return raw


def main() -> int:
    cfg = load_config()
    np.random.seed(cfg["general"]["random_seed"])
    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print("=" * 70)

    print("\n[1/4] Loading Kilian's data.txt...")
    data = load_data(cfg)
    print(f"      {data.shape[0]} obs, {data.index[0]:%Y-%m}..{data.index[-1]:%Y-%m}, vars={list(data.columns)}")

    print("\n[2/4] Estimating reduced-form VAR...")
    fit = lib.fit_var(data=data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"])
    eig = lib.max_eigenvalue_modulus(fit.lag_matrices())
    print(f"      VAR({fit.p}), T_eff={fit.t_eff}, max|eigenvalue|={eig:.4f}")

    print("\n[3/4] Recursive (Cholesky) identification...")
    shock_names = cfg["identification"]["shock_names"]
    irf = lib.cholesky_irf(fit, horizon=cfg["irf"]["horizon"], shock_names=shock_names, bands="none")
    rpo_i = list(data.columns).index("rpo")
    oilspec = shock_names.index("oil_specific")
    rpo_path = irf.median[:, rpo_i, oilspec]
    impact = float(rpo_path[0])
    peak = float(rpo_path[np.argmax(np.abs(rpo_path))])
    print(f"      oil-specific demand -> real oil price: impact={impact:+.3f}%, "
          f"peak={peak:+.3f}% at h={int(np.argmax(np.abs(rpo_path)))}")

    print("\n[4/4] Validating against paper targets + writing outputs...")
    outputs = HERE / "outputs"
    outputs.mkdir(exist_ok=True)
    comp = None
    targets_path = HERE / "paper_targets.yaml"
    if targets_path.exists():
        vr = {"oil_specific_rpo_impact": impact, "oil_specific_rpo_peak": peak,
              "max_eig_modulus": float(eig), "n_vars": float(fit.n_vars), "n_lags": float(fit.p)}
        comp = lib.compare_to_paper(vr, targets_path)

    log = {
        "replication_id": cfg["replication"]["id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {"source": cfg["data"]["source_url"], "n_obs": int(len(data)),
                 "sample_start": str(data.index[0].date()), "sample_end": str(data.index[-1].date()),
                 "variables": list(data.columns)},
        "model": {"class": cfg["model"]["class"], "lags": fit.p,
                  "max_eigenvalue_modulus": float(eig), "fit_info": fit.fit_info},
        "identification": dict(cfg["identification"]),
        "seed": cfg["general"]["random_seed"],
    }
    if comp is not None:
        log["paper_target_comparison"] = comp.to_dict()

    irf.save_snapshot(outputs / "numerical_results.json")
    (outputs / "research_log.json").write_text(json.dumps(log, indent=2, default=str))

    var_labels = [cfg["data"]["variable_labels"][v] for v in data.columns]
    spec = {
        "Model": f"VAR({fit.p}) + recursive (Cholesky) identification",
        "Variables": ", ".join(var_labels),
        "Identification": "Recursive — Kilian (2009) ordering [oil production, real activity, real oil price]",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "Lag order": str(fit.p),
        "Max |eigenvalue|": f"{float(eig):.4f}",
    }
    report = lib.VARReport(title=cfg["replication"]["title"], subtitle=cfg["replication"]["subtitle"])
    report.add_standard_sections(fit=fit, irf=irf, spec=spec, log=log,
                                 var_labels=var_labels, shock_idx_for_summary=oilspec,
                                 x_axis_label="Months", narrative="auto")
    report.save(outputs / "report.html")
    print("      -> outputs/ (numerical_results.json, research_log.json, report.html)")

    if comp is not None:
        print("\n" + comp.table())
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
