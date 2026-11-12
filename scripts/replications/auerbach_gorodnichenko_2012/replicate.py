"""Auerbach-Gorodnichenko (2012) state-dependent LP — fiscal multipliers.

AG's headline result is that government-spending multipliers are
substantially LARGER in recessions than in expansions. On the FRED-QD
sample (1959Q1-2008Q4) this replication finds the reverse ordering;
see the README. Estimates a
state-interacted local projection where the regime weight is a smooth
logistic function of past output growth.

Identification: government spending is ordered first in a small VAR
(Blanchard-Perotti 2002 timing), giving the Cholesky-identified
spending shock at no contemporaneous cost from GDP/tax. That shock is
then fed to a state-interacted LP at horizons 0..20.

State-interacted LP at horizon h:

    y_{t+h} = alpha_E + alpha_R F_{t-1}
              + beta_E shock_t + beta_R [F_{t-1} . shock_t]
              + Gamma_E Z_{t-1} + Gamma_R [F_{t-1} . Z_{t-1}]
              + eps_{t+h}

where Z_{t-1} stacks lagged levels of (g, gdp, tax). The expansion
coefficient is beta_E (when F~=0); the recession coefficient is
beta_E + beta_R (when F~=1).

Bands: Newey-West HAC, lags = horizon + 1 (the LP-standard rule).

The state-interacted regression uses lib.state_dependent_lp_irf. The
Cholesky shock extraction matches the helper used by
`scripts/replications/jorda_2005_lp/`.
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
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent.parent))

import lib  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


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
    return lib.transform(
        raw[data_cfg["variable_order"]], data_cfg["transforms"],
    ).dropna()


def extract_cholesky_shock(fit, shock_var: str) -> pd.Series:
    """Cholesky-identified shock for ``shock_var``. Same helper as jorda_2005_lp."""
    P = np.linalg.cholesky(fit.Sigma_u)
    P_inv = np.linalg.inv(P)
    struct = fit.residuals @ P_inv.T          # (T-p, n)
    idx = fit.data.index[fit.p:]
    var_idx = fit.variable_names.index(shock_var)
    return pd.Series(struct[:, var_idx], index=idx, name=f"{shock_var}_shock")


def compute_state(
    log_gdp: pd.Series, window: int, gamma: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (z, F) where z is the normalised 7Q-MA of dlog_gdp and F is
    the smooth-transition weight on the RECESSION regime, lagged one period.

    AG 2012 (eqs. 1-4) define the recession weight as
    ``F(z) = exp(-gamma * z) / (1 + exp(-gamma * z))``: F multiplies the
    recession-regime block and (1 - F) the expansion-regime block. F goes
    to 1 in recessions (low z) and to 0 in expansions (high z). The code
    computes it as 1 minus the standard logistic
    ``(1 + exp(-gamma * z))^{-1}``, which is the same function, and uses
    it as F_rec so that the interaction term in the regression
    ``[shock_t + F_rec_{t-1} . shock_t]`` decomposes into
    ``beta_E (expansion) + beta_R_minus_E (recession - expansion)``.

    F_rec is close to 1 when z is very negative (recession) and close to 0
    when z is very positive (expansion).
    """
    dlog = log_gdp.diff()
    z = dlog.rolling(window=window, min_periods=window).mean()
    z = (z - z.mean()) / z.std()
    # Standard logistic — F_exp=1 in expansion (high z)
    F_exp = 1.0 / (1.0 + np.exp(-gamma * z.shift(1)))
    # Recession weight = 1 - F_exp = AG's F(z)
    F_rec = 1.0 - F_exp
    F_rec.name = "F_rec"
    return z, F_rec


def state_lp_irf(
    data: pd.DataFrame,
    shock: pd.Series,
    state_F: pd.Series,
    horizon: int,
    lags: int,
    hac_lag_rule: str = "horizon_plus_one",
    response_vars: list[str] | None = None,
) -> dict:
    """Thin adapter over :func:`lib.state_dependent_lp_irf` that repacks the
    returned :class:`IRFContainer` into the per-variable dict of lists used
    below (cumulative multipliers, research log).
    """
    irf = lib.state_dependent_lp_irf(
        data=data,
        shock=shock,
        state=state_F,
        horizon=horizon,
        lags=lags,
        hac_lag_rule=hac_lag_rule,
        response_vars=response_vars,
        shock_name="g_shock",
        state_names=("expansion", "recession"),
    )
    if response_vars is None:
        response_vars = list(data.columns)

    out: dict[str, dict] = {}
    for var in response_vars:
        v_idx = irf.variable_names.index(var)
        beta_E = irf.median[:, v_idx, 0]
        beta_R = irf.median[:, v_idx, 1]
        lo68_E = irf.lower_68[:, v_idx, 0]
        hi68_E = irf.upper_68[:, v_idx, 0]
        lo68_R = irf.lower_68[:, v_idx, 1]
        hi68_R = irf.upper_68[:, v_idx, 1]
        lo90_E = irf.lower_90[:, v_idx, 0]
        hi90_E = irf.upper_90[:, v_idx, 0]
        lo90_R = irf.lower_90[:, v_idx, 1]
        hi90_R = irf.upper_90[:, v_idx, 1]
        out[var] = {
            "beta_expansion": beta_E.tolist(),
            "beta_recession": beta_R.tolist(),
            "ci68_E_lo": lo68_E.tolist(),
            "ci68_E_hi": hi68_E.tolist(),
            "ci68_R_lo": lo68_R.tolist(),
            "ci68_R_hi": hi68_R.tolist(),
            "ci90_E_lo": lo90_E.tolist(),
            "ci90_E_hi": hi90_E.tolist(),
            "ci90_R_lo": lo90_R.tolist(),
            "ci90_R_hi": hi90_R.tolist(),
        }
    return out


def cumulative_multiplier(
    gdp_irf: list[float], g_irf: list[float], horizon: int, y_over_g: float,
) -> float:
    """Cumulative spending multiplier at horizon h.

    M(h) = sum_{j=0..h} dY(j) / sum_{j=0..h} dG(j) * (Y/G)

    The Y/G factor converts log-units to dollar-for-dollar. Uses the
    sample-mean Y/G ratio.
    """
    num = float(np.nansum(gdp_irf[:horizon + 1]))
    den = float(np.nansum(g_irf[:horizon + 1]))
    if abs(den) < 1e-10:
        return float("nan")
    return num / den * y_over_g


def main() -> int:
    cfg = load_config()
    seed = cfg["general"]["random_seed"]
    np.random.seed(seed)

    print("=" * 70)
    print(f"  {cfg['replication']['title']}")
    print(f"  {cfg['replication']['subtitle']}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    data = load_and_transform(cfg)
    print(f"      Sample: {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}"
          f" ({len(data)} obs, {data.shape[1]} variables)")

    # Y/G ratio from sample mean (converts log multiplier to dollar multiplier)
    y_over_g = float(
        np.exp(data["gdp"]).mean() / np.exp(data["g"]).mean()
    )
    print(f"      Sample mean Y/G ratio: {y_over_g:.3f} "
          f"(used to scale log-multipliers to dollar units)")

    print(f"\n[2/4] Fitting VAR + extracting Cholesky g shock...")
    fit = lib.fit_var(
        data, lags=cfg["model"]["lags"], trend=cfg["model"]["trend"],
    )
    g_shock = extract_cholesky_shock(fit, cfg["model"]["shock_var"])
    print(f"      VAR(p={fit.p}), T_eff={fit.t_eff}, stable={fit.stable()}")
    print(f"      g_shock: mean={g_shock.mean():+.4f}, std={g_shock.std():.4f}")

    print(f"\n[3/4] Computing state variable and running state-LP...")
    z, F = compute_state(
        data["gdp"], window=cfg["state"]["window"], gamma=cfg["state"]["gamma"],
    )
    print(f"      State variable z (7Q-MA dlog_gdp, normalised): "
          f"min={z.min():.2f}, max={z.max():.2f}")
    print(f"      F(z_{{t-1}}) range: {F.min():.3f} to {F.max():.3f}  "
          f"(F~=1 = recession, F~=0 = expansion)")
    # Frequency of recession quarters (F > 0.5)
    rec_frac = float((F.dropna() > 0.5).mean())
    print(f"      F > 0.5 share: {rec_frac:.1%}  "
          f"(approximate 'recession-state' fraction of the sample)")

    horizon = cfg["model"]["horizon"]
    irfs = state_lp_irf(
        data=data,
        shock=g_shock,
        state_F=F,
        horizon=horizon,
        lags=cfg["model"]["lags"],
        hac_lag_rule=cfg["inference"]["hac_lag_rule"],
        response_vars=list(data.columns),
    )

    print(f"\n[4/4] Cumulative multipliers (cum dY/cum dG * Y/G):")
    print(f"      {'horizon':>8}{'expansion mult':>17}{'recession mult':>17}{'ratio R/E':>12}")
    multipliers = []
    g_irf_E = irfs["g"]["beta_expansion"]
    g_irf_R = irfs["g"]["beta_recession"]
    gdp_irf_E = irfs["gdp"]["beta_expansion"]
    gdp_irf_R = irfs["gdp"]["beta_recession"]
    for h in cfg["multipliers"]["horizons"]:
        if h > horizon:
            continue
        mE = cumulative_multiplier(gdp_irf_E, g_irf_E, h, y_over_g)
        mR = cumulative_multiplier(gdp_irf_R, g_irf_R, h, y_over_g)
        ratio = mR / mE if abs(mE) > 1e-8 else float("nan")
        multipliers.append({
            "horizon": h, "expansion": mE, "recession": mR,
            "ratio_R_over_E": ratio,
        })
        print(f"      {h:>8}{mE:>17.3f}{mR:>17.3f}{ratio:>12.3f}")

    print(f"\n      Headline (per AG 2012): recession multiplier > expansion multiplier")
    print(f"                                expansion multiplier ~ 0-0.5")
    print(f"                                recession multiplier ~ 1.5-2.5")

    # Brief LP summary at selected horizons
    print(f"\n      GDP response to a unit g-shock at selected horizons:")
    print(f"      {'h':>3}{'beta_E':>10}{'68% CI':>22}{'beta_R':>10}{'68% CI':>22}")
    horizons_show = [0, 1, 2, 4, 8, 12, 16, 20]
    horizons_show = [h for h in horizons_show if h <= horizon]
    gdp = irfs["gdp"]
    for h in horizons_show:
        bE = gdp["beta_expansion"][h]
        bR = gdp["beta_recession"][h]
        lo_E = gdp["ci68_E_lo"][h]; hi_E = gdp["ci68_E_hi"][h]
        lo_R = gdp["ci68_R_lo"][h]; hi_R = gdp["ci68_R_hi"][h]
        print(f"      {h:>3}{bE:>+10.4f}  [{lo_E:>+7.4f}, {hi_E:>+7.4f}]"
              f"  {bR:>+10.4f}  [{lo_R:>+7.4f}, {hi_R:>+7.4f}]")

    # ---- Save outputs ----
    outputs_dir = HERE / "outputs"
    outputs_dir.mkdir(exist_ok=True)

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
            "y_over_g_ratio": y_over_g,
        },
        "model": {
            "lags": int(cfg["model"]["lags"]),
            "horizon": int(horizon),
            "shock_var": cfg["model"]["shock_var"],
            "state_window": cfg["state"]["window"],
            "state_gamma": cfg["state"]["gamma"],
            "recession_share": rec_frac,
        },
        "irfs": irfs,
        "multipliers": multipliers,
        "shock_summary": {
            "mean": float(g_shock.mean()),
            "std": float(g_shock.std()),
            "n_obs": int(len(g_shock)),
        },
        "seed": seed,
    }

    # Paper-target validation
    targets_path = HERE / "paper_targets.yaml"
    comp_report = None
    if targets_path.exists():
        # multipliers list: each dict has horizon, expansion, recession, ratio_R_over_E
        h8 = next((m for m in multipliers if m["horizon"] == 8), None)
        h4 = next((m for m in multipliers if m["horizon"] == 4), None)
        validation_results = {
            "n_obs": float(len(data)),
            "n_vars": float(data.shape[1]),
            "n_lags": float(cfg["model"]["lags"]),
            "recession_share": float(rec_frac),
            "expansion_h4_positive":
                1.0 if (h4 and h4["expansion"] > 0) else 0.0,
            "ratio_R_over_E_h8":
                float(h8["ratio_R_over_E"]) if h8 else float("nan"),
            "lp_outputs_populated":
                1.0 if all(
                    k in irfs for k in ("gdp", "g", "tax")
                ) else 0.0,
        }
        comp_report = lib.compare_to_paper(validation_results, targets_path)
        log["paper_target_comparison"] = comp_report.to_dict()

    # ---- HTML report (custom — state-LP doesn't fit standard IRFContainer
    #      shape because the third axis is regime, not shock). Uses
    #      add_specification + add_robustness (expansion vs recession) +
    #      add_research_log. ----
    response_vars = list(data.columns)
    H = cfg["model"]["horizon"] + 1
    exp_arr = np.zeros((H, len(response_vars), 1))
    rec_arr = np.zeros((H, len(response_vars), 1))
    for j, v in enumerate(response_vars):
        exp_arr[:, j, 0] = np.asarray(irfs[v]["beta_expansion"])
        rec_arr[:, j, 0] = np.asarray(irfs[v]["beta_recession"])

    spec_dict = {
        "Model": "State-dependent Local Projections (AG 2012 / RZ 2018)",
        "Variables": ", ".join(response_vars),
        "Shock": cfg["model"]["shock_var"] + " (Cholesky residual)",
        "Sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({len(data)} obs)",
        "VAR lag order": str(fit.p),
        "LP horizon": str(cfg["model"]["horizon"]),
        "State variable": "7Q moving avg of dlog(gdp), normalised",
        "State transition gamma": str(cfg["state"]["gamma"]),
        "Recession-state share (F > 0.5)": f"{rec_frac:.1%}",
        "Multiplier ratio R/E at h=8": (
            f"{h8['ratio_R_over_E']:.3f}" if h8 else "n/a"
        ),
    }
    report_html = lib.VARReport(
        title=cfg["replication"]["title"],
        subtitle=cfg["replication"]["subtitle"],
    )
    report_html.add_specification(spec_dict)
    # Expansion vs recession comparison
    report_html.add_robustness(
        baseline_irf=exp_arr,
        alt_irf=rec_arr,
        labels=("Expansion (F=0)", "Recession (F=1)"),
        var_names=response_vars,
        shock_idx=0,
    )
    report_html.add_research_log(log)
    report_html.save(outputs_dir / "report.html")
    print(f"\n      -> {(outputs_dir / 'report.html').relative_to(PROJECT_ROOT)}")

    (outputs_dir / "research_log.json").write_text(json.dumps(log, indent=2))
    print(f"      -> {(outputs_dir / 'research_log.json').relative_to(PROJECT_ROOT)}")

    if comp_report is not None:
        print("\n" + comp_report.table())

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
