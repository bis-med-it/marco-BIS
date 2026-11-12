"""
Local projections (Jordà 2005), LP-IV (Stock-Watson 2018), and
state-dependent LP (Ramey-Zubairy 2018; smooth state weights as in the
Auerbach-Gorodnichenko 2012 smooth-transition VAR).

A local projection is, for each horizon ``h`` and each response variable ``j``,
the regression of ``y_{j, t+h}`` on the shock at time ``t`` plus a set of
controls. The coefficient on the shock is the impulse response at horizon
``h``. HAC standard errors give frequentist confidence bands.

LP-IV (Stock-Watson 2018) uses an external instrument ``Z`` for the shock.
We implement the "scaled reduced-form" variant: regress ``y_{t+h}`` on ``Z``
(plus controls) to get the reduced-form coefficient, then divide by the
first-stage coefficient of the policy variable on ``Z``. Numerically
identical to just-identified 2SLS at ``h=0`` (same sample); at ``h>0`` the
first stage uses the full sample while the reduced form loses ``h``
end-rows, so the ratio is a fixed-first-stage variant — asymptotically
equivalent to 2SLS (same plim), not numerically identical.

State-dependent LP allows the shock effect to differ between two regimes
(e.g., expansion vs recession). The state can be binary (Harding-Pagan
recession dummy, NBER dummy) or continuous (smooth-transition weight).

Public API:

* :func:`lp_irf` — single-shock LP with HAC bands. Returns an ``IRFContainer``.
* :func:`lp_iv_irf` — LP-IV with an external instrument.
* :func:`state_dependent_lp_irf` — state-interacted LP with two regimes
  (expansion / recession). Returns ``IRFContainer`` shape
  ``(H+1, n_response, 2)`` where the shock dim is regime-indexed.
* :func:`build_lp_controls` — utility for constructing the control matrix.

Bands are returned via the standard ``IRFContainer`` interface even though
LP uses normal-distribution CIs rather than empirical quantiles. The container
stores the median (= point estimate) and the implied CI endpoints from
``point ± z * se`` where ``z`` is the standard-normal quantile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .irf import IRFContainer


# ---------------------------------------------------------------------------
# Control matrix construction
# ---------------------------------------------------------------------------


def build_lp_controls(
    data: pd.DataFrame,
    lags: int,
    extra_controls: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the LP control matrix and aligned time index.

    Controls for time ``t`` are the lagged values ``Y_{t-1}, ..., Y_{t-lags}``
    of every column in ``data``, plus optional extras (used at time ``t``
    without lagging).

    Returns
    -------
    controls : ndarray of shape ``(T - lags, k_controls)``
    valid_index : ndarray of int
        Original-row indices in ``data`` corresponding to each control row.
        ``data.iloc[valid_index[i]]`` is the row at which ``controls[i]``
        applies (i.e., where the shock at time ``t`` lives).
    """
    Y = data.values
    T = Y.shape[0]
    rows = []
    valid = []
    for t in range(lags, T):
        row = []
        for lag in range(1, lags + 1):
            row.extend(Y[t - lag])
        if extra_controls is not None:
            row.extend(extra_controls.values[t])
        rows.append(row)
        valid.append(t)
    return np.asarray(rows), np.asarray(valid)


# ---------------------------------------------------------------------------
# Single-shock LP
# ---------------------------------------------------------------------------


def lp_irf(
    data: pd.DataFrame,
    shock: np.ndarray | pd.Series,
    horizon: int,
    lags: int,
    response_vars: Sequence[str] | None = None,
    shock_name: str = "shock",
    hac_lag_rule: str = "horizon_plus_one",
    extra_controls: pd.DataFrame | None = None,
    min_obs: int = 20,
    credible_intervals: tuple[float, float] = (0.68, 0.90),
    lag_augmented: bool = False,
) -> IRFContainer:
    """Local projections (Jordà 2005) with HAC-robust confidence bands.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series of response variables. Columns become
        candidate response names; lagged values become the controls.
    shock : array-like
        Shock series. Must be aligned with ``data.index`` (same length).
        Pre-identified shocks: Romer-Romer narrative, Cholesky residuals from
        an estimated VAR, etc.
    horizon : int
        Maximum forecast horizon ``H``. Returned IRFs cover ``0..H``.
    lags : int
        Lag length for the control set.
    response_vars : sequence of str, optional
        Subset of ``data.columns`` to use as response variables. Defaults
        to all columns.
    shock_name : str
        Human-readable name for the shock; goes into ``IRFContainer.shock_names``.
    hac_lag_rule : {"horizon_plus_one", "auto", "fixed:K"}
        Newey-West HAC truncation:
          * "horizon_plus_one" — ``max(h+1, 1)`` lags (standard for LP).
          * "auto" — Andrews-style automatic selection (not implemented;
            falls back to "horizon_plus_one").
          * "fixed:K" — fixed ``K`` lags at every horizon.
    extra_controls : pd.DataFrame, optional
        Extra contemporaneous controls aligned with ``data.index``.
    min_obs : int
        Minimum sample size at each ``h``; skip horizons with fewer obs.
    credible_intervals : tuple[float, float]
        Nominal coverage levels for the two reported bands.
    lag_augmented : bool, default False
        Montiel Olea & Plagborg-Møller (2021, ECMA) lag-augmented LP:
        include one EXTRA lag of all variables (the shock series included
        in the lagged information set) and use plain Eicker-Huber-White
        (HC1) standard errors instead of HAC. Valid uniformly over
        persistence and over horizons growing with the sample —
        **recommended for persistent (log-levels) data and long
        horizons**, where the default NW bands under-cover. The default
        stays ``False`` for backward compatibility with existing
        replications.

    Returns
    -------
    IRFContainer with shape ``(H+1, n_response, 1)``.

    Notes
    -----
    **Inference caveats — the default (non-augmented) LP bands.** The
    default bands are Newey-West HAC with bandwidth ``h+1`` and NORMAL
    critical values:

    * *Persistence / long horizons:* non-augmented LP + HAC under-covers
      for persistent data at long horizons — the typical macro log-levels
      use case. Pass ``lag_augmented=True`` (Montiel Olea &
      Plagborg-Møller 2021) there.
    * *Fixed-b / EWC critical values are not implemented.* With the
      bandwidth growing as ``h+1``, the bandwidth-to-sample ratio at long
      horizons is large enough that normal critical values over-reject
      (Lazarus-Lewis-Stock-Watson 2018, JBES). Lag augmentation avoids
      the issue entirely by using non-HAC errors.
    """
    shock_arr = np.asarray(shock).ravel()
    if len(shock_arr) != len(data):
        raise ValueError(
            f"shock length ({len(shock_arr)}) != data length ({len(data)})"
        )

    if response_vars is None:
        response_vars = list(data.columns)
    n_response = len(response_vars)
    response_idx = [data.columns.get_loc(c) for c in response_vars]

    # Build controls aligned with time t in [lags, T). Lag-augmented LP
    # (MOP 2021): include ONE extra lag of everything, with the shock
    # series in the lagged information set, and switch to EHW errors.
    if lag_augmented:
        data_aug = data.copy()
        data_aug["__lp_shock__"] = shock_arr
        controls, valid = build_lp_controls(
            data_aug, lags + 1, extra_controls=extra_controls)
    else:
        controls, valid = build_lp_controls(
            data, lags, extra_controls=extra_controls)
    shock_valid = shock_arr[valid]
    Y_full = data.values
    T_eff = len(valid)

    coefs = np.full((horizon + 1, n_response), np.nan)
    ses = np.full((horizon + 1, n_response), np.nan)

    for h in range(horizon + 1):
        max_t = T_eff - h
        if max_t < min_obs:
            continue
        if lag_augmented:
            cov_type, cov_kwds = "HC1", {}
        else:
            nw_lags = _hac_lags(h, hac_lag_rule)
            cov_type, cov_kwds = "HAC", {"maxlags": nw_lags}
        # has_constant="add": with the default "skip", a constant-valued
        # extra control makes statsmodels silently NOT prepend the
        # intercept, shifting params[1] onto a control instead of the
        # shock (pinned in tests/test_lp.py).
        X = add_constant(np.column_stack([shock_valid[:max_t], controls[:max_t]]),
                         has_constant="add")
        for col_i, j in enumerate(response_idx):
            y_fwd = Y_full[np.asarray(valid[:max_t]) + h, j]
            try:
                res = OLS(y_fwd, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
                coefs[h, col_i] = res.params[1]   # shock coefficient
                ses[h, col_i] = res.bse[1]
            except Exception:
                pass

    return _build_irf_container(
        coefs=coefs, ses=ses,
        variable_names=list(response_vars),
        shock_name=shock_name,
        credible_intervals=credible_intervals,
        metadata={
            "method": "local_projections",
            "identification": "external_or_pre_identified_shock",
            "n_lags": lags,
            "n_obs_effective": int(T_eff),
            "lag_augmented": bool(lag_augmented),
            "cov_type": (
                "HC1 (EHW; lag-augmented, MOP 2021)" if lag_augmented
                else "HAC (Newey-West)"
            ),
            "hac_lag_rule": None if lag_augmented else hac_lag_rule,
            # "auto" falls back to h+1 (Andrews selection unimplemented) —
            # record what actually ran so the research log is honest.
            "hac_lag_rule_effective": (
                None if lag_augmented
                else ("horizon_plus_one" if hac_lag_rule == "auto"
                      else hac_lag_rule)
            ),
        },
    )


# ---------------------------------------------------------------------------
# LP-IV (external instrument)
# ---------------------------------------------------------------------------


def lp_iv_irf(
    data: pd.DataFrame,
    instrument: np.ndarray | pd.Series,
    policy_var: str,
    horizon: int,
    lags: int,
    response_vars: Sequence[str] | None = None,
    shock_name: str | None = None,
    hac_lag_rule: str = "horizon_plus_one",
    extra_controls: pd.DataFrame | None = None,
    min_obs: int = 20,
    credible_intervals: tuple[float, float] = (0.68, 0.90),
) -> tuple[IRFContainer, dict]:
    """LP-IV (Stock-Watson 2018): identify the structural shock via an
    external instrument.

    Implementation: scaled-reduced-form (exactly just-identified 2SLS at
    h=0; asymptotically equivalent at h>0, where the first stage keeps the
    h end-rows the reduced form loses — more numerically stable than
    chaining residuals).

    For each horizon ``h``:

        β_h^RF = OLS coef of y_{j, t+h} on Z_t with controls
        γ      = OLS coef of policy_var_t on Z_t with controls
        IRF_h  = β_h^RF / γ
        SE     = SE(β_h^RF) / |γ|    (delta method; ignores SE(γ))

    **Inference caveats.**

    * The delta-method SEs DROP first-stage uncertainty entirely
      (``SE = SE(β^RF)/|γ̂|`` with ``Var(γ̂)`` ignored), so the bands are
      anti-conservative exactly when the instrument is marginal — check
      ``first_stage["F"]`` and treat results near the weak-instrument
      flag (F < 10; see also Montiel Olea-Pflueger 2013 on robust F)
      with caution.
    * Weak-instrument-robust confidence sets (Anderson-Rubin) are NOT
      implemented; under a weak instrument the reported bands have no
      validity guarantee.
    * The :func:`lp_irf` caveats (no lag-augmentation per Montiel Olea &
      Plagborg-Møller 2021; NW(h+1) with normal critical values) apply
      to the reduced-form stage here too.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series. Must include ``policy_var``.
    instrument : array-like
        External instrument ``Z_t``, length = ``len(data)``.
    policy_var : str
        Name of the policy variable (e.g., ``"ffr"``) in ``data``.
    horizon, lags, response_vars, hac_lag_rule, extra_controls, min_obs,
    credible_intervals : as in :func:`lp_irf`.

    Returns
    -------
    IRFContainer
        Standardised container with the scaled IRFs.
    dict
        First-stage diagnostics: ``F``, ``gamma_hat``, ``gamma_se``,
        ``n_obs``, ``weak_instrument_F_lt_10``.

    Raises
    ------
    RuntimeError
        Identification failure: the instrument has no variation left
        after projecting out the intercept and controls (constant, or
        exactly spanned by the controls) — checked with a relative,
        scale-invariant tolerance on the first-stage sample and on each
        horizon's truncated sample. A pseudoinverse "coefficient" from
        such a design is not a first stage. Also raised when the
        first-stage coefficient is essentially zero.
    """
    if policy_var not in data.columns:
        raise ValueError(f"policy_var {policy_var!r} not in data.columns")
    z_arr = np.asarray(instrument).ravel()
    if len(z_arr) != len(data):
        raise ValueError(
            f"instrument length ({len(z_arr)}) != data length ({len(data)})"
        )

    if response_vars is None:
        response_vars = list(data.columns)
    n_response = len(response_vars)
    response_idx = [data.columns.get_loc(c) for c in response_vars]
    policy_idx = data.columns.get_loc(policy_var)

    controls, valid = build_lp_controls(data, lags, extra_controls=extra_controls)
    z_valid = z_arr[valid]
    Y_full = data.values
    T_eff = len(valid)

    # ---- Identification precondition: the instrument must retain variation
    # after projecting out the intercept and controls. Without this check a
    # rank-deficient first-stage design (constant instrument, or instrument
    # spanned by the controls) still returns a finite "coefficient" through
    # statsmodels' default pseudoinverse — and that number is not a first
    # stage. Relative tolerance so rescaling the instrument changes nothing.
    def _resid_variation(z_sub: np.ndarray, ctrl_sub: np.ndarray) -> float:
        proj_X = np.column_stack([np.ones(len(z_sub)), ctrl_sub])
        beta_p, *_ = np.linalg.lstsq(proj_X, z_sub, rcond=None)
        z_resid = z_sub - proj_X @ beta_p
        denom = float(z_sub @ z_sub)
        return float(z_resid @ z_resid) / denom if denom > 0 else 0.0

    _REL_TOL = 1e-10
    if _resid_variation(z_valid, controls) <= _REL_TOL:
        raise RuntimeError(
            "LP-IV identification failure: the instrument has no variation "
            "after projecting out the intercept and controls (it is constant "
            "or exactly spanned by the controls). The first-stage "
            "coefficient would be a pseudoinverse artefact, not an IV first "
            "stage."
        )

    # ---- First stage: policy_t = alpha + gamma * Z_t + controls + v_t
    # has_constant="add" guards against constant-valued controls silently
    # suppressing the intercept and shifting params[1] (see lp_irf).
    fs_X = add_constant(np.column_stack([z_valid, controls]),
                        has_constant="add")
    fs_y = Y_full[np.asarray(valid), policy_idx]
    fs_res = OLS(fs_y, fs_X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    gamma_hat = float(fs_res.params[1])
    gamma_se = float(fs_res.bse[1])
    f_stat = float(fs_res.tvalues[1] ** 2)

    fs_diag = {
        "F": f_stat,
        "gamma_hat": gamma_hat,
        "gamma_se": gamma_se,
        "n_obs": int(T_eff),
        "weak_instrument_F_lt_10": bool(f_stat < 10),
    }

    if abs(gamma_hat) < 1e-12:
        raise RuntimeError(
            f"First-stage coefficient is essentially zero ({gamma_hat:.2e}); "
            f"instrument has no power on {policy_var!r}."
        )

    # ---- Reduced-form LPs: y_{t+h} on Z_t + controls
    rf_coefs = np.full((horizon + 1, n_response), np.nan)
    rf_ses = np.full((horizon + 1, n_response), np.nan)
    for h in range(horizon + 1):
        max_t = T_eff - h
        if max_t < min_obs:
            continue
        nw_lags = _hac_lags(h, hac_lag_rule)
        if _resid_variation(z_valid[:max_t], controls[:max_t]) <= _REL_TOL:
            raise RuntimeError(
                f"LP-IV identification failure at horizon {h}: the "
                f"instrument has no variation after projecting out the "
                f"controls on this horizon's truncated sample."
            )
        X = add_constant(np.column_stack([z_valid[:max_t], controls[:max_t]]),
                         has_constant="add")
        for col_i, j in enumerate(response_idx):
            y_fwd = Y_full[np.asarray(valid[:max_t]) + h, j]
            try:
                res = OLS(y_fwd, X).fit(
                    cov_type="HAC", cov_kwds={"maxlags": nw_lags}
                )
                rf_coefs[h, col_i] = res.params[1]
                rf_ses[h, col_i] = res.bse[1]
            except Exception:
                pass

    # Scale by gamma (delta-method SE ignoring first-stage uncertainty)
    coefs = rf_coefs / gamma_hat
    ses = rf_ses / abs(gamma_hat)

    if shock_name is None:
        shock_name = f"{policy_var} (LP-IV)"

    irf = _build_irf_container(
        coefs=coefs, ses=ses,
        variable_names=list(response_vars),
        shock_name=shock_name,
        credible_intervals=credible_intervals,
        metadata={
            "method": "lp_iv",
            "identification": "external_instrument",
            "policy_variable": policy_var,
            "n_lags": lags,
            "n_obs_effective": int(T_eff),
            "hac_lag_rule": hac_lag_rule,
            "first_stage": fs_diag,
        },
    )
    return irf, fs_diag


# ---------------------------------------------------------------------------
# Bayesian Local Projections (Ferreira-Miranda-Agrippino-Ricco)
# ---------------------------------------------------------------------------


# =============================================================================
# EXPERIMENTAL
# The function below is less established than the core VAR code.
# It rests on the library's own tests on simulated data. Treat results with
# caution and confirm important ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
def blp_irf(
    data: pd.DataFrame,
    shock: np.ndarray | pd.Series,
    horizon: int,
    lags: int,
    shrinkage: float = 0.0,
    var_lags: int | None = None,
    var_trend: str = "c",
    response_vars: Sequence[str] | None = None,
    shock_name: str = "BLP shock",
    hac_lag_rule: str = "horizon_plus_one",
    extra_controls: pd.DataFrame | None = None,
    min_obs: int = 20,
    credible_intervals: tuple[float, float] = (0.68, 0.90),
) -> IRFContainer:
    """Bayesian Local Projections (Ferreira, Miranda-Agrippino & Ricco 2025).

    Shrinks each horizon's LP coefficient toward the iterated-VAR IRF
    at that horizon, with the strength controlled by ``shrinkage``.
    Reduces small-sample noise in LP at the cost of slight bias toward
    the (possibly misspecified) VAR baseline. With ``shrinkage = 0``
    the BLP collapses to standard frequentist LP; with very large
    ``shrinkage`` it converges to the iterated VAR's IRF.

    **Algorithm** (simplified convex-combination form of MAR 2021):

    For each horizon ``h`` and response variable ``j``:

    1. Run standard OLS LP: ``y_{j,t+h} = α + β^OLS_h shock_t + γ' Z_{t-1:t-p} + u``
       to get the unrestricted coefficient ``β^OLS_h`` and its HAC SE.
    2. Fit a baseline VAR(``var_lags``) on the same data; compute the
       MA representation ``Φ_h`` (shape ``(H+1, n, n)``).
    3. Estimate the shock's impact loading on each VAR variable via the
       LP at ``h=0``: ``ℓ_k = β^OLS_{0, k}`` for variable ``k``.
    4. Compute the VAR-implied IRF as the prior mean:
       ``β^VAR_{h, j} = Σ_k Φ_h[j, k] · ℓ_k``.
    5. BLP point estimate:
       ``β^BLP_{h, j} = (β^OLS_{h, j} + λ · β^VAR_{h, j}) / (1 + λ)``
    6. BLP credible-interval SE: ``SE^BLP_{h, j} = SE^OLS_{h, j} / √(1 + λ)``
       (Bayesian posterior variance under the natural-conjugate
       λ-precision-matches-data scaling).

    With this scaling, ``shrinkage = λ`` is interpretable as "prior
    weight relative to data": λ=0 → pure LP; λ=1 → equal weight; λ→∞
    → pure VAR.

    Parameters
    ----------
    data, shock, horizon, lags, response_vars, shock_name,
    hac_lag_rule, extra_controls, min_obs, credible_intervals : as in
        :func:`lp_irf`.
    shrinkage : float, default 0.0
        Prior weight λ. Larger = tighter shrinkage to the VAR baseline.
        Must be ≥ 0. At the default ``shrinkage = 0`` the output equals
        that of :func:`lp_irf`; shrinkage is an explicit user choice.
    var_lags : int, optional
        Lag order of the baseline VAR used to construct the prior mean.
        Defaults to ``lags``.
    var_trend : str, default "c"
        Trend in the baseline VAR ("c" / "n" / "ct").

    Returns
    -------
    IRFContainer
        ``metadata["method"] == "bayesian_local_projections"``;
        ``metadata["identification"] == "bayesian_lp"``;
        ``metadata["shrinkage"]`` records λ; ``metadata["var_lags"]``
        records the baseline VAR lag order.

    Raises
    ------
    ValueError
        If ``shrinkage < 0``.

    References
    ----------
    Ferreira, L. N., Miranda-Agrippino, S., & Ricco, G. (2025). "Bayesian
    Local Projections." Review of Economics and Statistics, 107(5),
    1424-1438. (This implementation follows the design of the 2021
    Miranda-Agrippino & Ricco working-paper version.)

    See Also
    --------
    :func:`lp_irf` — frequentist LP (= ``blp_irf`` with ``shrinkage=0``).
    :func:`lp_iv_irf` — LP with external instrument.

    Notes
    -----
    **Scope and simplifications vs the MAR (2021 WP) design** —
    three deliberate deviations a user must know before quoting bands:

    1. *Shrinkage is a horizon-CONSTANT scalar convex combination* on the
       shock coefficient only. MAR's estimator places an NIW prior on
       the full LP coefficient vector (incl. controls, not implemented here), and
       its effective shrinkage RISES with horizon as LP data information
       falls (BLP ≈ LP at short h, ≈ VAR at long h). The constant-λ form
       does not reproduce that profile.
    2. *λ is NOT tuned by marginal likelihood* (MAR's headline device);
       the caller chooses it. Defaults to **0** (no shrinkage — output
       identical to :func:`lp_irf`): shrinking toward a possibly
       misspecified VAR, for a method whose rationale is robustness to
       VAR misspecification, must be an explicit choice.
    3. *Bands are mechanical*: SE^BLP = SE^OLS/sqrt(1+λ), i.e. CONDITIONAL
       on the prior and ignoring that the VAR prior mean is estimated
       from the same sample. At λ=1 this halves the variance by fiat —
       label such bands as conditional-on-prior, not frequentist.

    **Prior mean construction**: the VAR-iterated IRF is computed
    treating the externally-provided shock series as the impact vector,
    not as a structural identification. The impact loading is recovered
    from the LP at h=0 (i.e., the contemporaneous response). For a
    PRE-IDENTIFIED structural shock (e.g., narrative MP), this is the
    standard impulse-response interpretation. For an instrument /
    proxy, the impact loading would need to be normalized by the
    first-stage policy coefficient, for a unit policy response; this
    interface does not expose that first stage.
    """
    from .var import fit_var
    if shrinkage < 0:
        raise ValueError(f"shrinkage must be >= 0, got {shrinkage}")

    shock_arr = np.asarray(shock).ravel()
    if len(shock_arr) != len(data):
        raise ValueError(
            f"shock length ({len(shock_arr)}) != data length ({len(data)})"
        )

    if response_vars is None:
        response_vars = list(data.columns)
    response_vars = list(response_vars)
    n_response = len(response_vars)
    response_idx = [data.columns.get_loc(c) for c in response_vars]

    var_lags_eff = lags if var_lags is None else var_lags

    # --- 1. Standard OLS LP for the point estimate + HAC bands ---
    ols_irf = lp_irf(
        data=data, shock=shock_arr, horizon=horizon, lags=lags,
        response_vars=response_vars, shock_name=shock_name,
        hac_lag_rule=hac_lag_rule, extra_controls=extra_controls,
        min_obs=min_obs, credible_intervals=credible_intervals,
    )
    # Recover the point + SE from the bands
    # 68% half-width = z68 * SE. Use this to back out SE.
    c68 = credible_intervals[0]
    z68 = sp_stats.norm.ppf(0.5 + c68 / 2)
    ols_coef = ols_irf.median[:, :, 0]                                     # (H+1, n_resp)
    ols_se = (ols_irf.upper_68[:, :, 0] - ols_irf.median[:, :, 0]) / z68   # (H+1, n_resp)

    # If shrinkage=0, return OLS LP unchanged (with bayesian_lp metadata tag).
    if shrinkage == 0.0:
        metadata = dict(ols_irf.metadata)
        metadata.update({
            "method": "bayesian_local_projections",
            "identification": "bayesian_lp",
            "shrinkage": 0.0,
            "var_lags": int(var_lags_eff),
        })
        return IRFContainer(
            median=ols_irf.median,
            lower_68=ols_irf.lower_68, upper_68=ols_irf.upper_68,
            lower_90=ols_irf.lower_90, upper_90=ols_irf.upper_90,
            variable_names=list(response_vars),
            shock_names=[shock_name],
            metadata=metadata,
        )

    # --- 2. Baseline VAR for the prior mean ---
    var_fit = fit_var(data, lags=var_lags_eff, trend=var_trend)
    Phi_var = var_fit.ma_rep(horizon)            # (H+1, n_vars, n_vars)
    n_vars = data.shape[1]

    # --- 3. Impact loading: shock's contemporaneous effect on ALL VAR variables.
    # Re-run a one-horizon LP regression on every column of `data` (not just
    # response_vars) to populate the impact loading vector.
    impact_loading = _blp_impact_loading(
        data=data, shock=shock_arr, lags=lags, extra_controls=extra_controls,
    )  # (n_vars,)

    # --- 4. Prior IRF at each horizon ---
    # prior_irf[h, j] = Σ_k Phi_var[h, j, k] · impact_loading[k]
    prior_irf_full = np.einsum("hjk,k->hj", Phi_var, impact_loading)  # (H+1, n_vars)
    prior_irf = prior_irf_full[:, response_idx]                       # (H+1, n_resp)

    # --- 5. BLP point estimate + 6. BLP SE ---
    blp_coef = (ols_coef + shrinkage * prior_irf) / (1.0 + shrinkage)
    blp_se = ols_se / np.sqrt(1.0 + shrinkage)

    metadata = {
        "method": "bayesian_local_projections",
        "identification": "bayesian_lp",
        "n_lags": int(lags),
        "var_lags": int(var_lags_eff),
        "shrinkage": float(shrinkage),
        "n_obs_effective": ols_irf.metadata.get("n_obs_effective"),
        "hac_lag_rule": hac_lag_rule,
    }
    return _build_irf_container(
        coefs=blp_coef, ses=blp_se,
        variable_names=list(response_vars),
        shock_name=shock_name,
        credible_intervals=credible_intervals,
        metadata=metadata,
    )


def _blp_impact_loading(
    data: pd.DataFrame,
    shock: np.ndarray,
    lags: int,
    extra_controls: pd.DataFrame | None,
) -> np.ndarray:
    """OLS regression of every column's contemporaneous value on the shock
    (plus lagged controls + constant). Returns the impact-coefficient
    vector of length ``data.shape[1]``."""
    controls, valid = build_lp_controls(data, lags, extra_controls=extra_controls)
    shock_valid = shock[valid]
    Y_full = data.values
    X = add_constant(np.column_stack([shock_valid, controls]),
                     has_constant="add")
    loadings = np.empty(data.shape[1])
    for j in range(data.shape[1]):
        y_j = Y_full[np.asarray(valid), j]
        beta, *_ = np.linalg.lstsq(X, y_j, rcond=None)
        loadings[j] = beta[1]    # after add_constant: [const, shock, controls...]
    return loadings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hac_lags(h: int, rule: str) -> int:
    if rule == "horizon_plus_one":
        return max(h + 1, 1)
    if rule == "auto":
        # Andrews-style automatic bandwidth selection is NOT implemented.
        # Warn instead of silently aliasing, so a research log that says
        # "auto" cannot be mistaken for Andrews selection having run.
        import warnings
        warnings.warn(
            "hac_lag_rule='auto' (Andrews bandwidth selection) is not "
            "implemented; falling back to 'horizon_plus_one' (h+1 lags). "
            "The effective rule is recorded in metadata.",
            UserWarning, stacklevel=3,
        )
        return max(h + 1, 1)
    if rule.startswith("fixed:"):
        return max(int(rule.split(":", 1)[1]), 1)
    raise ValueError(f"unknown hac_lag_rule: {rule!r}")


def _build_irf_container(
    coefs: np.ndarray,
    ses: np.ndarray,
    variable_names: list[str],
    shock_name: str,
    credible_intervals: tuple[float, float],
    metadata: dict,
) -> IRFContainer:
    c68, c90 = credible_intervals
    z68 = sp_stats.norm.ppf(0.5 + c68 / 2)
    z90 = sp_stats.norm.ppf(0.5 + c90 / 2)

    # (H+1, n_response, 1) shape — LP identifies one shock
    median = coefs[:, :, None]
    lo68 = (coefs - z68 * ses)[:, :, None]
    hi68 = (coefs + z68 * ses)[:, :, None]
    lo90 = (coefs - z90 * ses)[:, :, None]
    hi90 = (coefs + z90 * ses)[:, :, None]

    return IRFContainer(
        median=median,
        lower_68=lo68, upper_68=hi68,
        lower_90=lo90, upper_90=hi90,
        variable_names=list(variable_names),
        shock_names=[shock_name],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# State-dependent / state-interacted local projections
# ---------------------------------------------------------------------------


def state_dependent_lp_irf(
    data: pd.DataFrame,
    shock: np.ndarray | pd.Series,
    state: np.ndarray | pd.Series,
    horizon: int,
    lags: int,
    state_names: tuple[str, str] = ("expansion", "recession"),
    response_vars: Sequence[str] | None = None,
    shock_name: str = "shock",
    hac_lag_rule: str = "horizon_plus_one",
    extra_controls: pd.DataFrame | None = None,
    min_obs: int = 20,
    credible_intervals: tuple[float, float] = (0.68, 0.90),
    interact_controls: bool = True,
) -> IRFContainer:
    """State-interacted local projections (Ramey-Zubairy 2018; the state
    weighting follows the Auerbach-Gorodnichenko 2012 smooth-transition VAR).

    Two regimes — interpreted as expansion (state ≈ 0) and recession
    (state ≈ 1), or any other binary partition — with the shock effect
    allowed to differ across regimes. The estimating equation per
    horizon ``h`` and response variable ``y_j``:

    .. math::

       y_{j, t+h} = \\alpha^E_h + (\\alpha^R_h - \\alpha^E_h)\\, F_t
                  + \\beta^E_{jh}\\, s_t
                  + (\\beta^R_{jh} - \\beta^E_{jh})\\, F_t\\, s_t
                  + \\gamma_h^\\top Z_{t-1:t-p}
                  + \\delta_h^\\top F_t\\, Z_{t-1:t-p}
                  + u_{j, t+h}

    where ``s_t`` is the shock, ``F_t \\in [0, 1]`` is the state
    weight (1 = recession), and ``Z_{t-1:t-p}`` are the lagged
    controls. ``\\beta^E_{jh}`` is the expansion impulse response;
    ``\\beta^R_{jh}`` the recession one. The intercept is
    regime-dependent: ``\\alpha^E_h`` in expansion (``F=0``) and
    ``\\alpha^R_h`` in recession (``F=1``), implemented by including
    ``F_t`` as a separate regressor.

    The interaction term ``F_t Z_{t-1:t-p}`` is included only when
    ``interact_controls = True`` (the Ramey-Zubairy 2018 "saturated"
    specification). Setting it to False omits regime-dependent
    dynamics in the controls; the shock coefficient is still
    regime-interacted.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format time series of response variables. Lagged values
        become the controls.
    shock : array-like or pd.Series
        Shock series. Aligned to ``data.index`` (same length when
        array-like; intersected when Series).
    state : array-like or pd.Series
        State weight aligned to ``data.index``. ``state_t = 0`` is pure
        expansion; ``state_t = 1`` is pure recession. Binary ({0, 1})
        for hard-regime LP; continuous for smooth-transition LP.
    horizon : int
        Maximum forecast horizon ``H``. Returned IRFs cover ``0..H``.
    lags : int
        Lag length for the control set.
    state_names : tuple of str
        Names for the two regimes. Default ``("expansion", "recession")``.
        Resulting ``IRFContainer.shock_names`` are
        ``[f"{shock_name}_{state_names[0]}", f"{shock_name}_{state_names[1]}"]``.
    response_vars : sequence of str, optional
        Subset of ``data.columns`` to use as response variables.
        Defaults to all columns.
    shock_name : str
        Human-readable base name for the shock.
    hac_lag_rule : {"horizon_plus_one", "fixed:K"}
        Newey-West HAC truncation rule. Same as ``lp_irf``.
    extra_controls : pd.DataFrame, optional
        Extra contemporaneous controls aligned with ``data.index``.
    min_obs : int
        Minimum sample size at each ``h``; skip horizons with fewer obs.
    credible_intervals : tuple of float
        Nominal coverage levels for the two reported bands.
    interact_controls : bool
        If True (default), also interact the lagged controls with the
        state (Ramey-Zubairy 2018 saturated specification). If False,
        only the shock is interacted (more parsimonious; assumes
        regime-invariant dynamics in the controls).

    Returns
    -------
    IRFContainer with shape ``(H+1, n_response, 2)``:

    * ``median[:, :, 0]`` = expansion impulse response ``β^E``
    * ``median[:, :, 1]`` = recession impulse response ``β^R``
    * Lower / upper bands constructed from HAC standard errors
      (recession SE accounts for the covariance between the
      shock and shock×state coefficients in the joint regression).

    The metadata field ``identification`` is ``"state_dependent_lp"``.

    Notes
    -----
    The recession variance is
    ``Var(β^R) = Var(β^E) + Var(β^R - β^E) + 2 Cov(β^E, β^R - β^E)``,
    where the covariance comes from the joint HAC covariance matrix
    of the regression coefficients. Tiny negative variances from
    finite-sample HAC are replaced with NaN in the bands.

    **Alignment.** The shock/state/data intersection is sorted
    CHRONOLOGICALLY before lags and forward outcomes are built — a
    Series' storage order never determines time order — and duplicate
    dates in the aligned index raise a ``ValueError``.

    **Estimability guard.** A regime-specific response is published
    only when the ``shock`` and ``state × shock`` columns retain
    independent variation after projecting out every other regressor.
    When the shock occurs in one regime only, the two columns coincide
    (only their sum is identified): the cell is left NaN, counted in
    ``metadata["n_unidentified_cells"]``, and a warning fires. When the
    state itself has no variation, the single identified response is
    reported as the expansion column and the recession column is NaN.
    If NO cell is identified, the function raises ``RuntimeError``
    rather than publishing a pseudoinverse allocation with a tight
    band.

    References
    ----------
    Auerbach, A. and Gorodnichenko, Y. (2012). "Measuring the output
    responses to fiscal policy." AEJ: Economic Policy, 4(2), 1-27.
    Ramey, V. and Zubairy, S. (2018). "Government spending multipliers
    in good times and in bad: Evidence from US historical data."
    Journal of Political Economy, 126(2), 850-901.
    """
    if response_vars is None:
        response_vars = list(data.columns)
    response_vars = list(response_vars)

    # Align shock + state + data on the intersection of valid indices
    shock_ser = (
        shock if isinstance(shock, pd.Series)
        else pd.Series(np.asarray(shock).ravel(), index=data.index)
    )
    state_ser = (
        state if isinstance(state, pd.Series)
        else pd.Series(np.asarray(state).ravel(), index=data.index)
    )
    if len(shock_ser) != len(data) and not isinstance(shock, pd.Series):
        raise ValueError(
            f"shock length ({len(shock_ser)}) != data length ({len(data)})"
        )
    if len(state_ser) != len(data) and not isinstance(state, pd.Series):
        raise ValueError(
            f"state length ({len(state_ser)}) != data length ({len(data)})"
        )

    common_idx = (
        shock_ser.dropna().index
        .intersection(state_ser.dropna().index)
        .intersection(data.dropna().index)
    )
    # Index.intersection preserves the CALLING index's storage order — a
    # reverse-stored shock Series would silently reverse time before the
    # lags and forward outcomes are built. Time order comes from the
    # calendar, never from storage order.
    if common_idx.has_duplicates:
        raise ValueError(
            "state_dependent_lp: duplicate dates in the aligned index — "
            "cannot establish a unique chronological order."
        )
    common_idx = common_idx.sort_values()
    if len(common_idx) < lags + min_obs:
        raise ValueError(
            f"insufficient overlap between shock, state, and data: "
            f"|common| = {len(common_idx)}; need >= lags + min_obs = "
            f"{lags + min_obs}"
        )
    s_full = shock_ser.loc[common_idx].values
    F_full = state_ser.loc[common_idx].values
    d_full = data.loc[common_idx]

    if extra_controls is not None:
        extra_aligned = extra_controls.reindex(common_idx)
    else:
        extra_aligned = None

    # Build lagged controls: Z_{t-1..t-p} stacked for all data columns
    T = len(d_full)
    n_data_vars = d_full.shape[1]
    Z_lags = np.zeros((T, n_data_vars * lags))
    for j in range(lags):
        Z_lags[j + 1:, j * n_data_vars:(j + 1) * n_data_vars] = (
            d_full.shift(j + 1).values[j + 1:]
        )
    if extra_aligned is not None:
        Z_lags = np.hstack([Z_lags, extra_aligned.values])

    # Drop the first `lags` rows where the lag block is NaN-padded
    valid_slice = slice(lags, T)
    s = s_full[valid_slice]
    F = F_full[valid_slice]
    Z_lags = Z_lags[valid_slice]
    T_eff = len(s)

    if T_eff < min_obs:
        raise ValueError(
            f"effective sample T_eff = {T_eff} after lag dropping is below "
            f"min_obs = {min_obs}"
        )

    # Build the state-interacted regressor block.
    # Column layout (before add_constant):
    #   col 0: F                                              (intercept shift)
    #   col 1: shock                                          (beta_E)
    #   col 2: F * shock                                      (beta_R - beta_E)
    #   cols 3..3+n_z-1: Z_lags                               (gamma_E)
    #   cols 3+n_z..3+2*n_z-1: F * Z_lags                     (delta = gamma_R - gamma_E)
    pieces = [
        F.reshape(-1, 1),
        s.reshape(-1, 1),
        (F * s).reshape(-1, 1),
        Z_lags,
    ]
    if interact_controls:
        pieces.append(F.reshape(-1, 1) * Z_lags)
    X_base = np.concatenate(pieces, axis=1)

    n_response = len(response_vars)
    response_idx = [data.columns.get_loc(c) for c in response_vars]
    Y_full = d_full.values   # aligned to common_idx

    # Pre-allocate output arrays: (H+1, n_response) for each of the 4 quantities
    beta_E = np.full((horizon + 1, n_response), np.nan)
    beta_R = np.full((horizon + 1, n_response), np.nan)
    se_E = np.full((horizon + 1, n_response), np.nan)
    se_R = np.full((horizon + 1, n_response), np.nan)
    n_unidentified_cells = 0

    # Index of the shock coefficient and the F*shock coefficient in X_ok
    # (after add_constant prepends an intercept):
    #   const(0) | F(1) | shock(2) | F*shock(3) | Z_lags... | F*Z_lags...
    SHOCK_IDX = 2
    INTERACT_IDX = 3

    for h in range(horizon + 1):
        nw_lags = _hac_lags(h, hac_lag_rule)
        for col_i, j_idx in enumerate(response_idx):
            # y_{t+h} for the j-th response variable, in the valid range
            y_full = pd.Series(Y_full[:, j_idx], index=common_idx).shift(-h)
            y_h = y_full.values[valid_slice]
            ok = (
                ~np.isnan(y_h)
                & ~np.any(np.isnan(X_base), axis=1)
            )
            if ok.sum() < (X_base.shape[1] + 2):
                continue
            if ok.sum() < min_obs:
                continue
            y_ok = y_h[ok]
            X_ok = add_constant(X_base[ok], has_constant="add")
            # Estimability guard: the shock and state×shock coefficients are
            # separately identified only if BOTH columns retain independent
            # variation after projecting out every other regressor. When the
            # shock occurs in one regime only, the two columns coincide —
            # their SUM is identified but the pseudoinverse would publish an
            # arbitrary split with a (meaningless) tight band. Leave such
            # cells NaN and count them.
            ctrl_cols = np.delete(X_ok, [SHOCK_IDX, INTERACT_IDX], axis=1)
            shock_cols = X_ok[:, [SHOCK_IDX, INTERACT_IDX]]
            proj, *_ = np.linalg.lstsq(ctrl_cols, shock_cols, rcond=None)
            resid_cols = shock_cols - ctrl_cols @ proj
            sv = np.linalg.svd(resid_cols, compute_uv=False)
            if sv.size < 2 or sv[1] <= max(sv[0], 1e-300) * 1e-8:
                n_shock = float(np.linalg.norm(resid_cols[:, 0]))
                n_inter = float(np.linalg.norm(resid_cols[:, 1]))
                # The expansion-only fallback needs the SHOCK itself to have
                # usable residual variation — an all-zero (or control-spanned)
                # shock must not produce a spuriously precise zero response.
                shock_scale = float(np.linalg.norm(shock_cols[:, 0]))
                shock_ok = (n_shock > 1e-12
                            and n_shock > 1e-8 * max(shock_scale, 1e-300))
                if shock_ok and n_inter <= 1e-8 * max(n_shock, 1e-300):
                    # The interaction carries no variation at all (e.g.
                    # state ≡ 0): the EXPANSION response is still identified
                    # from the plain shock column; only the recession one
                    # is not. Estimate it without the dead column.
                    try:
                        X_e = np.delete(X_ok, INTERACT_IDX, axis=1)
                        fit_e = OLS(y_ok, X_e).fit(
                            cov_type="HAC", cov_kwds={"maxlags": nw_lags},
                        )
                        beta_E[h, col_i] = fit_e.params[SHOCK_IDX]
                        se_E[h, col_i] = fit_e.bse[SHOCK_IDX]
                    except Exception:
                        pass
                # Either way the regime SPLIT is unidentified in this cell.
                n_unidentified_cells += 1
                continue
            try:
                fit_h = OLS(y_ok, X_ok).fit(
                    cov_type="HAC", cov_kwds={"maxlags": nw_lags},
                )
                b_E = fit_h.params[SHOCK_IDX]
                b_diff = fit_h.params[INTERACT_IDX]
                se_E_h = fit_h.bse[SHOCK_IDX]
                se_diff_h = fit_h.bse[INTERACT_IDX]
                cov_obj = fit_h.cov_params()
                cov_E_diff = (
                    cov_obj.iloc[SHOCK_IDX, INTERACT_IDX]
                    if hasattr(cov_obj, "iloc")
                    else cov_obj[SHOCK_IDX, INTERACT_IDX]
                )
                beta_E[h, col_i] = b_E
                beta_R[h, col_i] = b_E + b_diff
                se_E[h, col_i] = se_E_h
                var_R = se_E_h ** 2 + se_diff_h ** 2 + 2 * cov_E_diff
                # Finite-sample HAC can yield tiny negatives — replace with NaN
                se_R[h, col_i] = np.sqrt(var_R) if var_R > 0 else np.nan
            except Exception:
                continue

    # Build the IRFContainer with shape (H+1, n_response, 2)
    c68, c90 = credible_intervals
    z68 = sp_stats.norm.ppf(0.5 + c68 / 2)
    z90 = sp_stats.norm.ppf(0.5 + c90 / 2)

    # Stack expansion and recession along the last axis
    median = np.stack([beta_E, beta_R], axis=-1)            # (H+1, n_resp, 2)
    se_stack = np.stack([se_E, se_R], axis=-1)
    lo68 = median - z68 * se_stack
    hi68 = median + z68 * se_stack
    lo90 = median - z90 * se_stack
    hi90 = median + z90 * se_stack

    state_mean = float(np.nanmean(F))
    metadata = {
        "method": "state-dependent local projections (AG 2012 / RZ 2018)",
        "identification": "state_dependent_lp",
        "n_lags": lags,
        "n_obs_effective": int(T_eff),
        "hac_lag_rule": hac_lag_rule,
        "state_mean": state_mean,
        "state_names": list(state_names),
        "interact_controls": bool(interact_controls),
        "n_unidentified_cells": int(n_unidentified_cells),
    }
    if n_unidentified_cells:
        import warnings
        warnings.warn(
            f"state_dependent_lp: {n_unidentified_cells} (horizon, variable) "
            f"cell(s) left NaN — the regime-specific shock coefficients are "
            f"not separately identified there (the shock has no independent "
            f"variation in one regime after controls).", UserWarning,
            stacklevel=2)
    if np.all(np.isnan(beta_E)) and np.all(np.isnan(beta_R)):
        raise RuntimeError(
            "state_dependent_lp: no (horizon, variable) cell is identified — "
            "the shock varies in only one regime, so regime-specific "
            "responses cannot be separated. Estimate a single-regime LP "
            "instead."
        )
    shock_names = [f"{shock_name}_{state_names[0]}",
                   f"{shock_name}_{state_names[1]}"]
    return IRFContainer(
        median=median,
        lower_68=lo68, upper_68=hi68,
        lower_90=lo90, upper_90=hi90,
        variable_names=list(response_vars),
        shock_names=shock_names,
        metadata=metadata,
    )
