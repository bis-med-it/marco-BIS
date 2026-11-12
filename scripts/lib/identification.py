"""
Structural identification for reduced-form VARs.

Each routine takes a ``VARFit`` plus identification-specific arguments and
returns an ``IRFContainer``. Confidence bands are computed internally via the
``inference`` module (Monte Carlo from asymptotic coefficient distribution,
residual / wild bootstrap, or posterior draws for BVAR — handled by the
caller for BVAR posterior draws via ``IRFContainer.from_draws``).

Identification methods implemented:

* :func:`cholesky_irf` — standard recursive ordering.
* :func:`sign_restricted_irf` — QR rotation acceptance with sign constraints.
* :func:`proxy_svar_irf` — external-instrument identification of a single
  structural shock (Stock-Watson 2018; Mertens-Ravn 2013), with wild-
  bootstrap bands (MR/GK-legacy Rademacher wild bootstrap — NB Jentsch-
  Lunsford 2019 prove it invalid for proxy-SVARs; see _proxy_wild_bootstrap).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _stats

from . import inference
from .irf import IRFContainer
from .var import (
    VARFit, extract_lag_matrices, ma_representation, max_eigenvalue_modulus,
    _build_x_row,
)


# ---------------------------------------------------------------------------
# Cholesky
# ---------------------------------------------------------------------------


def cholesky_impact(Sigma_u: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of the residual covariance.

    Columns are the structural impact vectors under recursive identification
    in the given variable order.
    """
    return np.linalg.cholesky(Sigma_u)


def cholesky_irf(
    fit: VARFit,
    horizon: int,
    shock_names: Sequence[str] | None = None,
    bands: str = "asymptotic_mc",
    n_draws: int = 2000,
    seed: int | None = None,
) -> IRFContainer:
    """Cholesky-identified IRFs with confidence bands.

    Parameters
    ----------
    fit : VARFit
    horizon : int
        IRF horizon ``H``. The container will hold horizons 0..H.
    shock_names : sequence of str, optional
        Names for the structural shocks (length = ``n_vars``). Defaults to
        ``[f"{var}_shock" for var in fit.variable_names]``.
    bands : {"asymptotic_mc", "residual_bootstrap", "none"}
        How to compute confidence bands. ``asymptotic_mc`` draws from the
        asymptotic coefficient distribution; ``residual_bootstrap`` uses the
        residual bootstrap; ``none`` returns point-only bands.
    n_draws : int
        Number of draws for the chosen ``bands`` method.
    seed : int, optional
        RNG seed.

    Returns
    -------
    IRFContainer
    """
    n = fit.n_vars
    if shock_names is None:
        shock_names = [f"{v}_shock" for v in fit.variable_names]
    if len(shock_names) != n:
        raise ValueError(f"shock_names has {len(shock_names)} entries, expected {n}")

    P = cholesky_impact(fit.Sigma_u)
    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,jk->hik", Phi, P)  # (H+1, n, n)

    metadata: dict[str, Any] = {
        "method": "VAR / Cholesky",
        "identification": "cholesky",
        "ordering": list(fit.variable_names),
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
    }

    if bands == "none":
        return IRFContainer.from_point(
            point=irf_point,
            variable_names=list(fit.variable_names),
            shock_names=list(shock_names),
            metadata=metadata,
        )

    # Defer to inference module
    if bands == "residual_bootstrap":
        draws = inference.residual_bootstrap_draws(
            fit=fit, horizon=horizon, n_draws=n_draws, seed=seed,
            structural_impact=lambda Sigma: np.linalg.cholesky(Sigma),
        )
    elif bands in ("parameter_draws", "asymptotic_mc", "posterior"):
        # Generic path: ask the fit for parameter draws (asymptotic for VAR,
        # NIW posterior for BVAR), then map to IRFs.
        params = fit.parameter_draws(n_draws=n_draws, seed=seed)
        draws_list = []
        for B_d, Sigma_d in params:
            try:
                P_d = np.linalg.cholesky(Sigma_d)
            except np.linalg.LinAlgError:
                continue
            lag_mats = extract_lag_matrices(B_d, fit.n_vars, fit.p)
            Phi = ma_representation(lag_mats, horizon)
            draws_list.append(np.einsum("hij,jk->hik", Phi, P_d))
        if not draws_list:
            raise RuntimeError("no usable parameter draws — check stability or prior")
        draws = np.stack(draws_list, axis=0)
        metadata["bands"] = bands  # echo back what the user asked for
    else:
        raise ValueError(f"unknown bands method: {bands!r}")

    return IRFContainer.from_draws(
        draws=draws,
        variable_names=list(fit.variable_names),
        shock_names=list(shock_names),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Sign restrictions (Uhlig 2005-style QR acceptance)
# ---------------------------------------------------------------------------


_VALID_MOMENTS = {"skewness", "excess_kurtosis"}


def _validate_higher_moment_restriction(
    hm: dict, where: str, multi_shock: bool,
) -> dict:
    """Validate a single higher-moment restriction dict and return a
    normalised copy with explicit ``lower`` / ``upper`` / ``abs`` fields.

    Parameters
    ----------
    hm : dict
        User-supplied restriction.
    where : str
        Free-form label used in error messages.
    multi_shock : bool
        If True, require a ``"shock"`` key (and validate ``"vs"`` for
        cross-shock comparisons).
    """
    if not isinstance(hm, dict):
        raise ValueError(
            f"{where}: higher_moment_restrictions entry must be a dict; "
            f"got {type(hm).__name__}"
        )
    if "moment" not in hm:
        raise ValueError(
            f"{where}: higher_moment_restrictions entry must have a "
            f"'moment' key; got {list(hm.keys())}"
        )
    moment = hm["moment"]
    if moment not in _VALID_MOMENTS:
        raise ValueError(
            f"{where}: unknown moment {moment!r}; expected one of "
            f"{sorted(_VALID_MOMENTS)}"
        )
    lower = float(hm.get("lower", float("-inf")))
    upper = float(hm.get("upper", float("inf")))
    if lower > upper:
        raise ValueError(
            f"{where}: higher_moment_restrictions has lower > upper; "
            f"got lower={lower}, upper={upper}"
        )
    abs_flag = bool(hm.get("abs", False))
    # Comparison fields (multi-shock only)
    vs = hm.get("vs")
    comparison = hm.get("comparison")
    margin = float(hm.get("margin", 0.0))
    if vs is not None:
        if not multi_shock:
            raise ValueError(
                f"{where}: cross-shock 'vs' comparisons require multi-shock "
                f"mode (more than one shock in sign_restrictions)"
            )
        if comparison not in ("greater", "less"):
            raise ValueError(
                f"{where}: 'vs' restrictions must include "
                f"'comparison' in {{'greater', 'less'}}; got "
                f"comparison={comparison!r}"
            )
    return {
        "moment": moment, "lower": lower, "upper": upper, "abs": abs_flag,
        "vs": vs, "comparison": comparison, "margin": margin,
    }


def _check_moment_threshold(value: float, lower: float, upper: float,
                             abs_flag: bool) -> bool:
    """Apply (possibly abs) threshold bounds to a moment value."""
    v = abs(value) if abs_flag else value
    return lower <= v <= upper


def _type2_dominates(contribs: np.ndarray, col: int, direction: str,
                     criterion: str = "most_important",
                     tol: float = 1e-12) -> bool:
    """ADRR (2018) Type-2 historical-contribution dominance check.

    ``contribs`` are the raw (signed) contributions of every candidate
    column to the named variable at the named date, computed under the
    +Q column orientation for BOTH the IRF and the shock series — the
    product is orientation-invariant (flipping a column flips both
    factors), so callers must pass contributions UNMULTIPLIED by any
    acceptance-orientation sign.

    ``criterion`` selects the dominance threshold (ADRR p. 2806-2807):

    * ``"most_important"`` (default, ADRR Type A): the named shock's
      |contribution| is the LARGEST single one — ``|c_self| >= |c_j|`` for
      all j. Ties allowed. (Self-inclusion in the max is harmless.)
    * ``"overwhelming"`` (ADRR Type B): the named shock's |contribution|
      exceeds the SUM of all the others — ``|c_self| > sum_{j!=col}|c_j|``.
      ADRR's monetary-policy application uses this stronger form. Strict.

    ``direction`` additionally restricts the sign of the contribution:

    * ``"absolute"``: magnitude only.
    * ``"positive"``: dominates AND ``c_self > 0``.
    * ``"negative"``: dominates AND ``c_self < 0``.

    Used by both the single-shock and multi-shock narrative paths.
    """
    c_self = contribs[col]
    abs_all = np.abs(contribs)
    if criterion == "most_important":
        bound = float(abs_all.max())          # largest |c| over all columns
        if direction == "absolute":
            return bool(abs(c_self) >= bound - tol)
        if direction == "positive":
            return bool(c_self >= bound - tol)
        if direction == "negative":
            return bool(c_self <= -bound + tol)
    elif criterion == "overwhelming":
        sum_others = float(abs_all.sum() - abs(c_self))   # sum of all OTHERS
        if direction == "absolute":
            return bool(abs(c_self) > sum_others + tol)
        if direction == "positive":
            return bool(c_self > sum_others + tol)
        if direction == "negative":
            return bool(c_self < -sum_others - tol)
    else:
        raise ValueError(
            f"unknown type-2 criterion {criterion!r}; expected "
            f"'most_important' or 'overwhelming'"
        )
    raise ValueError(f"unknown type-2 direction {direction!r}")


def _parse_type2_extras(nr: dict) -> tuple[str, str]:
    """Parse the optional ADRR Type-2 modifiers from a restriction dict.

    Returns ``(criterion, contribution)``:

    * ``criterion`` — dominance threshold, ``"most_important"`` (default,
      ADRR Type A: larger than every other shock) or ``"overwhelming"``
      (ADRR Type B: larger than the SUM of all others; the monetary
      application of ADRR 2018 uses this).
    * ``contribution`` — which object the contribution is measured on:
      ``"cumulative"`` (default) = the shock's contribution to the
      historical-decomposition LEVEL of the variable at the date (the
      Phi-convolution of the whole shock history up to it); ``"unexpected"``
      = its contribution to the one-period UNEXPECTED movement (forecast
      error) at the date, ``B[var, :] * eps[date, :]``. ADRR's canonical
      examples (e.g. the Oct-1979 monetary shock as the overwhelming driver
      of the unexpected change in the funds rate) restrict the latter.
    """
    criterion = nr.get("criterion", "most_important")
    if criterion not in ("most_important", "overwhelming"):
        raise ValueError(
            f"Type 2 narrative restriction 'criterion' must be 'most_important' "
            f"or 'overwhelming'; got {criterion!r}"
        )
    contribution = nr.get("contribution", "cumulative")
    if contribution not in ("cumulative", "unexpected"):
        raise ValueError(
            f"Type 2 narrative restriction 'contribution' must be 'cumulative' "
            f"or 'unexpected'; got {contribution!r}"
        )
    return criterion, contribution


def _type2_contributions(irf_all_ext, eps_all, B_impact, var_idx: int,
                         t_idx: int, contribution: str) -> np.ndarray:
    """Per-column contributions to ``var_idx`` at ``t_idx`` for the chosen object.

    ``"unexpected"`` → one-period forecast-error contribution ``B[var,:]*eps[t,:]``
    (needs only the impact matrix ``B_impact = P @ Q``). ``"cumulative"`` →
    full historical-decomposition convolution
    ``sum_{s=0..t} Phi_{t-s}[var,:] * eps[s,:]`` (needs ``irf_all_ext``).
    """
    if contribution == "unexpected":
        return B_impact[var_idx, :] * eps_all[t_idx, :]
    irf_slice = irf_all_ext[:t_idx + 1, var_idx, :]
    eps_slice = eps_all[:t_idx + 1, :]
    return np.sum(irf_slice[::-1, :] * eps_slice, axis=0)


def _resolve_magnitude_restriction(mr: dict, variable_names: list[str],
                                   horizon: int) -> list[tuple[int, int, int, float]]:
    """Validate one relative-magnitude restriction and expand it over its horizons.

    A restriction dict has ``stronger`` and ``weaker`` variable names, ``horizons``
    (int or list of ints in ``[0, horizon]``; ``horizon`` is also accepted as an
    alias key) and an optional ``ratio`` (> 0, default 1.0). The caller enforces
    ``|IRF[h, stronger, shock]| >= ratio * |IRF[h, weaker, shock]|`` at each listed
    horizon — the response of ``stronger`` to the (assigned) shock dominates that
    of ``weaker`` in magnitude. With ``ratio = 1`` this is the plain
    relative-magnitude restriction; ``ratio > 1`` is an elasticity-style
    extension (Kilian-Murphy 2012). The comparison is in absolute magnitudes,
    so it also binds in the rare case where the two responses have opposite
    signs.

    Returns a list of ``(stronger_idx, weaker_idx, h, ratio)`` tuples.
    """
    if not isinstance(mr, dict) or "stronger" not in mr or "weaker" not in mr:
        raise ValueError(
            f"magnitude_restrictions entry must be a dict with 'stronger' and "
            f"'weaker' variable names; got {mr!r}"
        )
    for key in ("stronger", "weaker"):
        if mr[key] not in variable_names:
            raise ValueError(
                f"unknown variable {mr[key]!r} in magnitude restriction"
            )
    s_idx = variable_names.index(mr["stronger"])
    w_idx = variable_names.index(mr["weaker"])
    if s_idx == w_idx:
        raise ValueError(
            "magnitude restriction 'stronger' and 'weaker' must be different variables"
        )
    raw_h = mr.get("horizons", mr.get("horizon", 0))
    hs = [raw_h] if isinstance(raw_h, (int, np.integer)) else list(raw_h)
    ratio = float(mr.get("ratio", 1.0))
    if ratio <= 0:
        raise ValueError(f"magnitude restriction 'ratio' must be > 0; got {ratio}")
    out: list[tuple[int, int, int, float]] = []
    for h in hs:
        h = int(h)
        if not (0 <= h <= horizon):
            raise ValueError(
                f"magnitude restriction horizon {h} outside [0, {horizon}]"
            )
        out.append((s_idx, w_idx, h, ratio))
    return out


def _compute_shock_moment(eps_series: np.ndarray, moment: str) -> float:
    """Compute the named moment of a 1D shock series.

    ``"skewness"``: ``scipy.stats.skew(bias=False)`` — sample, bias-corrected.
    ``"excess_kurtosis"``: ``scipy.stats.kurtosis(fisher=True, bias=False)``
        — Fisher-style (excess), bias-corrected.
    """
    if moment == "skewness":
        return float(_stats.skew(eps_series, bias=False))
    if moment == "excess_kurtosis":
        return float(_stats.kurtosis(eps_series, fisher=True, bias=False))
    raise ValueError(f"unknown moment {moment!r}")   # defensive: validated upstream


def _draw_residuals(fit, B_d) -> np.ndarray:
    """Residuals consistent with coefficient draw ``B_d``.

    Narrative / historical-contribution / higher-moment restrictions are
    statements about the STRUCTURAL SHOCKS of the model being accepted.
    Under parameter uncertainty each draw is its own reduced form, so its
    shocks must be reconstructed from its own residuals
    ``u_d = Y_act - X_act @ B_d`` — checking the point estimate's residuals
    against a draw's rotation tests a different model's shocks.
    Falls back to ``fit.residuals`` when the draw IS the point estimate.
    """
    if B_d is fit.B_hat:
        return np.asarray(fit.residuals)
    Y_act = getattr(fit, "Y_act", None)
    X_act = getattr(fit, "X_act", None)
    if Y_act is None or X_act is None:
        raise AttributeError(
            "fit lacks Y_act/X_act; cannot reconstruct draw-consistent "
            "residuals for narrative/higher-moment checks under parameter "
            "uncertainty."
        )
    return np.asarray(Y_act) - np.asarray(X_act) @ np.asarray(B_d)


# =============================================================================
# EXPERIMENTAL OPTIONS
# The narrative and higher-moment restrictions of the function below are less
# established than its plain sign restrictions. They rest on the library's own
# tests on simulated data. Treat results with caution and confirm important
# ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
def sign_restricted_irf(
    fit: VARFit,
    horizon: int,
    sign_restrictions: dict[str, dict[str, int]],
    restriction_horizons: int = 1,
    target_accepted: int = 1000,
    max_rotations: int = 100_000,
    seed: int | None = None,
    parameter_uncertainty: bool = True,
    n_param_draws: int | None = None,
    narrative_restrictions: list[dict] | None = None,
    fev_share_restrictions: list[dict] | None = None,
    magnitude_restrictions: list[dict] | None = None,
    higher_moment_restrictions: list[dict] | None = None,
    column_scan: bool = False,
) -> IRFContainer:
    """Sign-restricted IRFs via QR rotations, optionally with narrative restrictions.

    For each rotation draw, check whether the candidate impact vector
    satisfies the sign restrictions on the NAMED shock at horizons
    ``0..restriction_horizons-1`` (a per-candidate sign flip is allowed).
    By default the candidate is COLUMN 0 of the Haar rotation only — the
    canonical Uhlig (2005) / RWZ (2010) accept-reject, which samples the
    identified set under the uniform-conditional measure. With
    ``column_scan=True`` (legacy), all n columns are scanned and the first
    match accepted: more rotation-efficient, but the induced measure over
    the identified set is not the canonical one (Baumeister-Hamilton 2015:
    the draw measure is part of the reported object) — see CONVENTIONS.md.
    Multi-shock mode always uses the fixed assignment shock j ↔ column j
    (exchangeability of Haar columns makes this the canonical measure and
    keeps acceptance independent of the order shocks are listed in). If
    ``narrative_restrictions`` is provided, also reject candidates that
    don't satisfy the narrative criteria at specific historical dates
    (Antolín-Díaz & Rubio-Ramírez 2018). If ``higher_moment_restrictions``
    is provided, also reject candidates whose implied structural shock
    series fails specified skewness / excess-kurtosis bounds
    (Andrade-Ferroni-Melosi 2023).

    Parameters
    ----------
    fit : VARFit
    horizon : int
        IRF horizon for the output container.
    sign_restrictions : dict
        ``{shock_name: {variable_name: sign}}`` where ``sign`` is ``+1``, ``-1``,
        or ``0`` (unrestricted). Variable names must match ``fit.variable_names``.
        Each top-level key is one identified shock.
    restriction_horizons : int
        Number of horizons (``0..restriction_horizons-1``) at which the sign
        restrictions must hold.
    target_accepted : int
        Stop after this many accepted rotations.
    max_rotations : int
        Hard cap on rotation attempts.
    seed : int, optional
        RNG seed.
    parameter_uncertainty : bool
        If True, draw reduced-form parameters from their asymptotic distribution
        for each rotation attempt (joint shape + identification uncertainty).
        If False, use point estimates of ``B_hat`` and ``Sigma_u``.
    n_param_draws : int, optional
        If ``parameter_uncertainty`` is True, generate this many fresh parameter
        draws (one rotation attempt each). Defaults to ``max_rotations``.
    narrative_restrictions : list of dict, optional
        Antolín-Díaz & Rubio-Ramírez (2018) narrative sign restrictions.
        Each restriction is a dict with keys:

        * ``"date"`` (str or pd.Timestamp): historical date in ``fit.data.index``.
          The date must fall within the post-lag effective sample
          (``fit.data.index[fit.p:]``).
        * ``"sign"`` (int): ``+1`` or ``-1``. The structural shock at this
          date must have the specified sign.

        Implementation: the structural shock at date ``t`` from rotation
        column ``Q_col`` is ``eps_t = u_t' @ (P^{-T} @ Q_col)`` where
        ``P = chol(Sigma_d)`` and ``u_t`` is the reduced-form residual
        OF THE CURRENT PARAMETER DRAW (``Y - X B_d``; the point fit's
        residuals when ``parameter_uncertainty=False``).
        Rotations that fail the sign check are rejected just like
        sign-restriction failures.

        Both **sign-of-shock** restrictions (Type 1) and
        **historical-contribution dominance** restrictions (Type 2 —
        "this shock was the largest contributor to variable v on date X")
        are supported, in single- and multi-shock mode. A Type-2 dict uses
        ``type`` in {``"largest_contributor"``, ``"largest_positive_contributor"``,
        ``"largest_negative_contributor"``} plus ``variable`` and ``date``, and two
        optional ADRR (2018) modifiers:

        * ``criterion``: ``"most_important"`` (default; |contribution| at least
          every other shock's — ADRR Type A) or ``"overwhelming"`` (greater than
          the SUM of all the others — ADRR Type B; the ADRR monetary application
          uses this).
        * ``contribution``: ``"cumulative"`` (default; contribution to the
          historical-decomposition LEVEL at the date) or ``"unexpected"``
          (contribution to the one-period unexpected movement / forecast error
          at the date, ``B[var,:]·eps[date,:]`` — the object ADRR's canonical
          examples restrict).

        A restriction refers to a single date. A window of several
        periods, as in some of the paper's examples, cannot be expressed:
        keys other than those listed here are ignored, so list each date
        separately. The paper's "least important" and "negligible"
        variants are not implemented.

        See the Type-2 blocks below.
    fev_share_restrictions : list of dict, optional
        Quantity / variance-decomposition restrictions on the identified
        shock's share of forecast-error variance. Each restriction is a
        dict with keys:

        * ``"variable"`` (str): variable name in ``fit.variable_names``.
        * ``"horizon"`` (int): forecast horizon ``H`` at which to evaluate
          the share, in the SAME H-term convention as
          ``lib.structural_fevd`` (the H-step-ahead forecast-error variance
          sums ``H`` MA terms, ``s = 0..H-1``; ``H=1`` is the impact
          decomposition). Must satisfy ``1 <= H <= horizon + 1``.
        * ``"lower"`` (float, optional, default 0.0): minimum acceptable
          FEV share (inclusive). In ``[0, 1]``.
        * ``"upper"`` (float, optional, default 1.0): maximum acceptable
          FEV share (inclusive). In ``[0, 1]``.

        Implementation: for a candidate column ``b_struct = P Q_col``,
        the share of variable ``k``'s ``H``-step forecast-error variance
        attributable to the identified shock is
        ``sum_{s=0..H-1} (Phi_s b_struct)_k^2 / sum_{s=0..H-1} (Phi_s Sigma Phi_s')_{kk}``
        — numerically identical to ``lib.structural_fevd(fit, b, horizon=H)``.
        Rotations failing any ``[lower, upper]`` bound are rejected.
    magnitude_restrictions : list of dict, optional
        Relative-magnitude restrictions (Kilian-Murphy 2012
        elasticity bounds). Each dict has ``stronger`` and ``weaker`` variable
        names, ``horizons`` (int or list of ints in ``[0, horizon]``) and an
        optional ``ratio`` (> 0, default 1.0). Reject a rotation unless
        ``|IRF[h, stronger, shock]| >= ratio * |IRF[h, weaker, shock]|`` at every
        listed horizon — i.e. ``stronger``'s response to the (assigned) shock
        dominates ``weaker``'s in magnitude (``ratio = 1`` is plain dominance;
        ``ratio > 1`` is an elasticity-style bound). In multi-shock mode each
        dict also needs a ``shock`` key.
    higher_moment_restrictions : list of dict, optional
        Andrade-Ferroni-Melosi (2023) higher-moment restrictions on the
        implied structural shock series. Each restriction is a dict
        with keys:

        * ``"moment"`` (str): ``"skewness"`` or ``"excess_kurtosis"``.
        * ``"lower"`` (float, optional): minimum acceptable moment value.
          Default ``-inf``.
        * ``"upper"`` (float, optional): maximum acceptable moment value.
          Default ``+inf``.
        * ``"abs"`` (bool, optional): if True, apply bounds to
          ``|moment|`` instead of the signed moment. Default False.

        ``"excess_kurtosis"`` is the Fisher (excess) form
        (``scipy.stats.kurtosis``); zero means Gaussian-equivalent
        kurtosis, positive means leptokurtic.

        Implementation: for a candidate column ``Q[:, c]``, the implied
        structural shock series is
        ``ε_t = u_t^⊤ (P^{-⊤} Q[:, c])``. Skew / excess-kurtosis are
        computed via ``scipy.stats.skew`` / ``scipy.stats.kurtosis``
        (bias=False) on this series and checked against the bounds.

        Single-shock mode: only threshold restrictions
        (no ``"vs"`` key) are supported. Multi-shock mode additionally
        supports cross-shock comparisons with a ``"vs": "other_shock"``
        key — see :func:`_sign_restricted_irf_multi_shock`.

    Returns
    -------
    IRFContainer with shocks named per the keys of ``sign_restrictions``.

    Notes
    -----
    **Dispatch.** With one key in ``sign_restrictions`` the single-shock
    path checks column 0 of each rotation (±) by default; the legacy
    ``column_scan=True`` option scans all columns and accepts the first
    match. With multiple keys, the multi-shock path
    (Rubio-Ramírez-Waggoner-Zha 2010) checks shock j against column j of
    the same rotation (±).

    **Inference caveats** (documented limitations of set-identification
    inference, not implementation bugs — see CONVENTIONS.md §4 "Inference
    caveats"):

    * *Fry-Pagan (2011)*: the reported bands are pointwise cross-draw
      quantiles; the pointwise median is generally not the IRF of any
      single accepted rotation, and connecting pointwise medians across
      horizons/variables mixes rotations. Read the bands as summaries of
      the admissible set.
    * *Baumeister-Hamilton (2015)*: the uniform (Haar) draw over rotations
      is itself an informative prior over the identified set — "agnostic"
      restrictions are not prior-free, and bands do not shrink to the
      identified set as the sample grows.
    * *Narrative restrictions*: implemented by rejection (violating draws
      discarded, survivors equally weighted). ADRR (2018, p. 2812) note
      this is not their Algorithm 1, which importance-reweights each
      surviving draw by the inverse probability that the narrative holds;
      the resulting posterior differs from ADRR's.
    """
    rng = np.random.default_rng(seed)
    n = fit.n_vars
    shock_names = list(sign_restrictions.keys())

    if len(shock_names) > 1:
        # Multi-shock simultaneous identification via Rubio-Ramirez-Waggoner-
        # Zha (2010): each rotation must assign each named shock to a
        # DISTINCT column of the orthogonal rotation matrix.
        return _sign_restricted_irf_multi_shock(
            fit=fit, horizon=horizon,
            sign_restrictions=sign_restrictions,
            restriction_horizons=restriction_horizons,
            target_accepted=target_accepted,
            max_rotations=max_rotations,
            seed=seed,
            parameter_uncertainty=parameter_uncertainty,
            n_param_draws=n_param_draws,
            narrative_restrictions=narrative_restrictions,
            fev_share_restrictions=fev_share_restrictions,
            magnitude_restrictions=magnitude_restrictions,
            higher_moment_restrictions=higher_moment_restrictions,
        )
    shock_name = shock_names[0]
    var_signs = sign_restrictions[shock_name]
    for v in var_signs:
        if v not in fit.variable_names:
            raise ValueError(f"unknown variable in sign restriction: {v!r}")
    sign_array = np.zeros(n, dtype=int)
    for v, s in var_signs.items():
        sign_array[fit.variable_names.index(v)] = int(np.sign(s))

    # Resolve and validate FEV-share restrictions upfront
    fev_resolved: list[tuple[int, int, float, float]] = []
    if fev_share_restrictions:
        for fr in fev_share_restrictions:
            if not isinstance(fr, dict) or "variable" not in fr or "horizon" not in fr:
                raise ValueError(
                    f"fev_share_restrictions entry must be a dict with "
                    f"'variable' and 'horizon'; got {fr!r}"
                )
            v_name = fr["variable"]
            if v_name not in fit.variable_names:
                raise ValueError(f"unknown variable in FEV restriction: {v_name!r}")
            h = int(fr["horizon"])
            if h < 1 or h > horizon + 1:
                raise ValueError(
                    f"FEV restriction horizon {h} outside [1, {horizon + 1}] "
                    f"(H-term convention: horizon H sums s = 0..H-1, matching "
                    f"lib.structural_fevd)"
                )
            lo = float(fr.get("lower", 0.0))
            hi = float(fr.get("upper", 1.0))
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"FEV restriction bounds must satisfy 0 <= lower <= upper <= 1; "
                    f"got lower={lo}, upper={hi}"
                )
            # h-1: the cumsum arrays index s = 0..idx, so the H-term share
            # lives at idx = H-1 (same convention as lib.structural_fevd).
            fev_resolved.append((fit.variable_names.index(v_name), h - 1, lo, hi))

    # Resolve and validate relative-magnitude restrictions upfront.
    mag_resolved: list[tuple[int, int, int, float]] = []
    if magnitude_restrictions:
        for mr in magnitude_restrictions:
            mag_resolved += _resolve_magnitude_restriction(
                mr, list(fit.variable_names), horizon)

    # Resolve and validate narrative restrictions upfront (Type 1 = sign-of-shock,
    # Type 2 = historical-contribution per ADRR 2018; type defaults to "sign").
    narrative_resolved: list[tuple[int, int]] = []
    type2_resolved: list[dict] = []
    valid_type2_kinds = {
        "largest_contributor", "largest_positive_contributor",
        "largest_negative_contributor",
    }
    if narrative_restrictions:
        eff_index = fit.data.index[fit.p:]
        for nr in narrative_restrictions:
            if not isinstance(nr, dict) or "date" not in nr:
                raise ValueError(
                    f"narrative_restrictions entry must be a dict with a "
                    f"'date' key; got {nr!r}"
                )
            kind = nr.get("type", "sign")
            ts = pd.Timestamp(nr["date"])
            if ts not in eff_index:
                raise ValueError(
                    f"narrative restriction date {ts} not in effective sample "
                    f"[{eff_index[0]} ... {eff_index[-1]}]"
                )
            t_idx = list(eff_index).index(ts)

            if kind == "sign":
                if "sign" not in nr:
                    raise ValueError(
                        f"Type 1 (sign) narrative restriction must have a 'sign' key; "
                        f"got {nr!r}"
                    )
                sgn = int(np.sign(nr["sign"]))
                if sgn not in (-1, 1):
                    raise ValueError(
                        f"narrative restriction sign must be +1 or -1; got {nr['sign']!r}"
                    )
                narrative_resolved.append((t_idx, sgn))
            elif kind in valid_type2_kinds:
                if "variable" not in nr:
                    raise ValueError(
                        f"Type 2 ({kind}) narrative restriction must have a "
                        f"'variable' key; got {nr!r}"
                    )
                v_name = nr["variable"]
                if v_name not in fit.variable_names:
                    raise ValueError(
                        f"unknown variable {v_name!r} in Type 2 narrative restriction"
                    )
                v_idx = fit.variable_names.index(v_name)
                if kind == "largest_contributor":
                    direction = "absolute"
                elif kind == "largest_positive_contributor":
                    direction = "positive"
                else:
                    direction = "negative"
                criterion, contribution = _parse_type2_extras(nr)
                type2_resolved.append({
                    "t_idx": t_idx, "var_idx": v_idx,
                    "direction": direction, "kind": kind,
                    "criterion": criterion, "contribution": contribution,
                })
            else:
                raise ValueError(
                    f"unknown narrative restriction type {kind!r}; expected "
                    f"'sign' or one of {sorted(valid_type2_kinds)}"
                )

    # Resolve and validate higher-moment restrictions upfront (AFM 2023).
    hm_resolved: list[dict] = []
    if higher_moment_restrictions:
        for i, hm in enumerate(higher_moment_restrictions):
            res = _validate_higher_moment_restriction(
                hm, where=f"shock {shock_name!r}, restriction #{i}",
                multi_shock=False,
            )
            hm_resolved.append(res)
    any_hm = bool(hm_resolved)

    if parameter_uncertainty:
        n_param_draws = n_param_draws or max_rotations
        # Dispatches to NIW posterior for BVARFit, asymptotic for plain VARFit
        param_draws = fit.parameter_draws(n_draws=n_param_draws, seed=seed)
    else:
        param_draws = None

    accepted: list[np.ndarray] = []
    n_attempted = 0
    n_rejected_narrative = 0
    n_rejected_fev_share = 0
    n_rejected_magnitude = 0
    n_rejected_type2 = 0
    n_rejected_hm = 0
    # 'cumulative' Type-2 contributions need the MA representation extended to
    # the latest such restriction date; 'unexpected' ones need only the impact
    # matrix, so they do not drive this (default -1 => skip the MA extension).
    max_t_idx_type2_single = max(
        (r["t_idx"] for r in type2_resolved if r["contribution"] == "cumulative"),
        default=-1,
    )
    any_type2_single = bool(type2_resolved)
    while len(accepted) < target_accepted and n_attempted < max_rotations:
        # Reduced-form draw
        if param_draws is not None:
            B_d, Sigma_d = param_draws[n_attempted % len(param_draws)]
        else:
            B_d, Sigma_d = fit.B_hat, fit.Sigma_u

        try:
            P = np.linalg.cholesky(Sigma_d)
        except np.linalg.LinAlgError:
            n_attempted += 1
            continue

        # Random orthogonal rotation via QR with diagonal-sign normalisation
        X = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(X)
        Q = Q @ np.diag(np.sign(np.diag(R)))

        lag_mats = extract_lag_matrices(B_d, n, fit.p)
        Phi = ma_representation(lag_mats, horizon)

        # Pre-compute total FEV at each horizon (depends only on Sigma_d + Phi,
        # not on the rotation column). diag of sum_{s<=h} Phi_s Sigma Phi_s'.
        if fev_resolved:
            phi_sigma_phi = np.einsum(
                "hij,jk,hlk->hil", Phi, Sigma_d, Phi,
            )  # (H+1, n, n)
            total_fev = np.cumsum(
                np.array([np.diag(phi_sigma_phi[h]) for h in range(horizon + 1)]),
                axis=0,
            )  # (H+1, n)
        else:
            total_fev = None

        # Pre-compute the structural-shock projection for narrative / higher-
        # moment checks. For column Q[:, col], the structural shock at time t
        # is eps_t = u_t' @ (P^{-T} @ Q[:, col]).
        if narrative_resolved or any_hm:
            try:
                Pinv_T = np.linalg.inv(P).T
            except np.linalg.LinAlgError:
                n_attempted += 1
                continue
            # residuals shape: (T_eff, n). Project onto P^{-T} @ Q (whole basis).
            # We'll select the column inside the loop.
            eps_basis = _draw_residuals(fit, B_d) @ Pinv_T  # (T_eff, n)
        else:
            eps_basis = None

        # Candidate columns. DEFAULT (canonical Uhlig/RWZ measure): only
        # column 0 — the first column of a Haar Q is uniform on the unit
        # sphere, so accept/reject on it samples the identified set under
        # the uniform-conditional measure. ``column_scan=True`` restores
        # the legacy behaviour (accept the FIRST of the n columns that
        # matches): more rotation-efficient, but the induced measure over
        # the identified set is NOT the canonical one (later columns are
        # accepted conditional on earlier ones failing — KS-distinguishable,
        # band edges shift by several percent). See CONVENTIONS.md §4.
        cols_to_check = range(n) if column_scan else range(1)
        for col in cols_to_check:
            b_struct = P @ Q[:, col]
            irf_cand = np.einsum("hij,j->hi", Phi, b_struct)  # (H+1, n)
            for irf_signed, q_col in [(irf_cand, Q[:, col]), (-irf_cand, -Q[:, col])]:
                if not _matches_signs(irf_signed, sign_array, restriction_horizons):
                    continue
                # Sign restrictions pass — now check narrative
                if narrative_resolved:
                    eps_series = eps_basis @ q_col   # (T_eff,)
                    ok = all(
                        int(np.sign(eps_series[t_idx])) == sgn
                        for t_idx, sgn in narrative_resolved
                    )
                    if not ok:
                        n_rejected_narrative += 1
                        continue
                # FEV-share restrictions
                if fev_resolved:
                    # Cumulative sum of (Phi_s b_struct)_k^2 over s
                    irf_sq_cum = np.cumsum(irf_signed**2, axis=0)  # (H+1, n)
                    ok = True
                    for v_idx, h_check, lo, hi in fev_resolved:
                        denom = total_fev[h_check, v_idx]
                        if denom <= 0:
                            ok = False
                            break
                        share = irf_sq_cum[h_check, v_idx] / denom
                        if not (lo <= share <= hi):
                            ok = False
                            break
                    if not ok:
                        n_rejected_fev_share += 1
                        continue
                # Relative-magnitude restrictions: the response of
                # 'stronger' to this shock dominates 'weaker' in magnitude.
                if mag_resolved:
                    ok = True
                    for s_idx, w_idx, h_check, ratio in mag_resolved:
                        if (abs(irf_signed[h_check, s_idx]) + 1e-12
                                < ratio * abs(irf_signed[h_check, w_idx])):
                            ok = False
                            break
                    if not ok:
                        n_rejected_magnitude += 1
                        continue
                # Type 2 historical-contribution restrictions (ADRR 2018):
                # the assigned column's contribution to the named variable
                # at the named date must dominate ALL OTHER columns' contributions.
                if any_type2_single:
                    if eps_basis is None:
                        try:
                            Pinv_T = np.linalg.inv(P).T
                        except np.linalg.LinAlgError:
                            continue
                        eps_basis_local = _draw_residuals(fit, B_d) @ Pinv_T
                    else:
                        eps_basis_local = eps_basis
                    eps_all = eps_basis_local @ Q   # (T_eff, n)
                    B_impact = P @ Q                # structural impact (n, n)
                    # 'cumulative' restrictions need Phi convolved to their date;
                    # 'unexpected' ones need only B_impact, so skip the (possibly
                    # large) MA extension when no cumulative restriction is present.
                    if max_t_idx_type2_single >= 0:
                        Phi_ext = ma_representation(lag_mats, max_t_idx_type2_single)
                        irf_all_ext = np.einsum("hij,jk->hik", Phi_ext, B_impact)
                    else:
                        irf_all_ext = None
                    # Contributions use the +Q[:, col] orientation for BOTH the
                    # IRF and the shock series; the product is orientation-
                    # invariant (flipping the column flips both factors), so the
                    # acceptance orientation (q_col vs -q_col) must NOT be
                    # multiplied in. See _type2_dominates / _type2_contributions.
                    type2_ok = True
                    for r in type2_resolved:
                        contribs = _type2_contributions(
                            irf_all_ext, eps_all, B_impact,
                            r["var_idx"], r["t_idx"], r["contribution"],
                        )
                        if not _type2_dominates(
                            contribs, col, r["direction"], criterion=r["criterion"],
                        ):
                            type2_ok = False
                            break
                    if not type2_ok:
                        n_rejected_type2 += 1
                        continue
                # Higher-moment restrictions (AFM 2023). Single-shock mode
                # supports only threshold restrictions (no cross-shock 'vs').
                if any_hm:
                    eps_series = eps_basis @ q_col   # (T_eff,)
                    hm_ok = True
                    for hm in hm_resolved:
                        m_val = _compute_shock_moment(eps_series, hm["moment"])
                        if not _check_moment_threshold(
                            m_val, hm["lower"], hm["upper"], hm["abs"],
                        ):
                            hm_ok = False
                            break
                    if not hm_ok:
                        n_rejected_hm += 1
                        continue
                accepted.append(irf_signed[:, :, None])
                break
            else:
                continue
            break
        n_attempted += 1

    if not accepted:
        raise RuntimeError(
            f"no rotations satisfied the sign restrictions after "
            f"{n_attempted} attempts — check the restriction set or "
            f"increase max_rotations"
        )

    draws = np.stack(accepted, axis=0)  # (n_accepted, H+1, n, 1)
    # Preserve existing identification tags for backward compatibility, then
    # append "+fev_share" when FEV-share restrictions are also active.
    # Build identification tag from the four possible restriction kinds.
    bits = ["sign"]
    if narrative_resolved:
        bits.insert(0, "narrative")
    if any_type2_single:
        bits.append("type2_contribution")
    if fev_resolved:
        bits.append("fev_share")
    if mag_resolved:
        bits.append("magnitude")
    if any_hm:
        bits.append("higher_moment")
    if len(bits) == 1:
        id_str = "sign_restrictions"
        method_str = "VAR / sign restrictions"
    else:
        id_str = "_".join(bits) + "_restrictions"
        # Build human-readable method string
        method_parts = ["VAR / sign restrictions"]
        if narrative_resolved:
            method_parts[0] = "VAR / narrative sign restrictions (ADRR 2018)"
        if any_type2_single:
            method_parts.append("Type 2 contribution (ADRR 2018)")
        if fev_resolved:
            method_parts.append("FEV-share")
        if mag_resolved:
            method_parts.append("relative magnitude")
        if any_hm:
            method_parts.append("higher-moment (AFM 2023)")
        method_str = " + ".join(method_parts)

    metadata = {
        "method": method_str,
        "identification": id_str,
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "target_accepted": target_accepted,
        "n_accepted": len(accepted),
        "n_attempted": n_attempted,
        "acceptance_rate": len(accepted) / max(n_attempted, 1),
        "median_target": _median_target_meta(draws),
        "restriction_horizons": restriction_horizons,
        "restrictions": {shock_name: dict(var_signs)},
        "narrative_restrictions": (
            [dict(nr) for nr in narrative_restrictions]
            if narrative_restrictions else None
        ),
        "n_rejected_by_narrative": int(n_rejected_narrative),
        "fev_share_restrictions": (
            [dict(fr) for fr in fev_share_restrictions]
            if fev_share_restrictions else None
        ),
        "n_rejected_by_fev_share": int(n_rejected_fev_share),
        "magnitude_restrictions": (
            [dict(mr) for mr in magnitude_restrictions]
            if magnitude_restrictions else None
        ),
        "n_rejected_by_magnitude": int(n_rejected_magnitude),
        "n_rejected_by_type2": int(n_rejected_type2),
        "higher_moment_restrictions": (
            [dict(hm) for hm in higher_moment_restrictions]
            if higher_moment_restrictions else None
        ),
        "n_rejected_by_higher_moment": int(n_rejected_hm),
        "parameter_uncertainty": parameter_uncertainty,
        "seed": seed,
    }
    return IRFContainer.from_draws(
        draws=draws,
        variable_names=list(fit.variable_names),
        shock_names=[shock_name],
        metadata=metadata,
    )


def fry_pagan_median_target(draws) -> dict:
    """Fry-Pagan (2011) median-target summary of sign-restricted IRF draws.

    The pointwise-median IRF reported by sign-restriction routines mixes
    responses from different rotations and is therefore not itself the
    impulse response of any single model (Fry and Pagan 2011). The median
    target (MT) instead selects the SINGLE accepted draw whose responses
    are jointly closest to the pointwise median, after standardising each
    (horizon, variable, shock) cell by its cross-draw standard deviation:

        MT = argmin_d  sum_{h,i,j} ((irf_d - median) / sd)^2 [h, i, j]

    The MT draw is a coherent one-rotation summary: it satisfies every
    restriction the routine imposed (it is one of the accepted draws) and
    its responses across variables come from the same structural model.

    Parameters
    ----------
    draws : ndarray, shape (n_draws, H+1, n_vars, n_shocks)
        Accepted IRF draws (the array stacked internally by
        ``sign_restricted_irf`` before quantile summarisation).

    Returns
    -------
    dict
        ``irf``: the MT draw, shape ``(H+1, n_vars, n_shocks)``;
        ``index``: position of the MT draw in ``draws``;
        ``distance``: the criterion value at the optimum;
        ``distances``: per-draw criterion values, shape ``(n_draws,)``.

    References
    ----------
    Fry, R. and A. Pagan (2011): "Sign restrictions in structural vector
    autoregressions: a critical review", Journal of Economic Literature,
    49(4), 938-960.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 4:
        raise ValueError(
            f"draws must be 4-D (n_draws, H+1, n_vars, n_shocks); "
            f"got shape {draws.shape}"
        )
    if draws.shape[0] < 2:
        raise ValueError("median target needs at least 2 accepted draws")
    med = np.median(draws, axis=0)
    sd = draws.std(axis=0, ddof=1)
    # Cells with no cross-draw variation contribute exactly 0 to every
    # draw's distance (draws == median there); avoid 0/0.
    sd = np.where(sd > 0, sd, 1.0)
    z = (draws - med) / sd
    distances = np.einsum("dhij,dhij->d", z, z)
    idx = int(np.argmin(distances))
    return {
        "irf": draws[idx].copy(),
        "index": idx,
        "distance": float(distances[idx]),
        "distances": distances,
    }


def _median_target_meta(draws) -> dict | None:
    """JSON-safe median-target block for IRFContainer metadata."""
    if draws.shape[0] < 2:
        return None
    mt = fry_pagan_median_target(draws)
    return {
        "irf": mt["irf"].tolist(),
        "index": mt["index"],
        "distance": mt["distance"],
        "note": (
            "Fry-Pagan (2011) median target: the single accepted rotation "
            "closest to the pointwise median (standardised cells); a "
            "coherent one-model summary of the identified set"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-shock simultaneous sign identification (Rubio-Ramirez-Waggoner-Zha 2010)
# ---------------------------------------------------------------------------

def _resolve_per_shock_restrictions(
    restrictions: list[dict] | None,
    shock_names: list[str],
    kind: str,
) -> dict[str, list[dict]]:
    """Group restrictions by the ``shock`` key (required for multi-shock mode).

    Returns ``{shock_name: [restrictions_for_that_shock]}``. Raises a clear
    error if any restriction lacks the ``shock`` key.
    """
    out: dict[str, list[dict]] = {name: [] for name in shock_names}
    if not restrictions:
        return out
    for i, r in enumerate(restrictions):
        if not isinstance(r, dict):
            raise ValueError(
                f"{kind} entry #{i} must be a dict; got {type(r).__name__}"
            )
        if "shock" not in r:
            raise ValueError(
                f"In multi-shock mode, each {kind} entry must include a "
                f"'shock' key naming which shock it applies to. Entry #{i} "
                f"has keys {list(r.keys())}; expected 'shock' from "
                f"{shock_names}."
            )
        s = r["shock"]
        if s not in out:
            raise ValueError(
                f"{kind} entry #{i} references shock {s!r} not in "
                f"sign_restrictions keys {shock_names}"
            )
        out[s].append(r)
    return out


def _sign_restricted_irf_multi_shock(
    fit: "VARFit",
    horizon: int,
    sign_restrictions: dict[str, dict[str, int]],
    restriction_horizons: int,
    target_accepted: int,
    max_rotations: int,
    seed: int | None,
    parameter_uncertainty: bool,
    n_param_draws: int | None,
    narrative_restrictions: list[dict] | None,
    fev_share_restrictions: list[dict] | None,
    magnitude_restrictions: list[dict] | None = None,
    higher_moment_restrictions: list[dict] | None = None,
) -> IRFContainer:
    """Multi-shock simultaneous sign identification (RWZ 2010).

    For each candidate rotation Q, check named shock j against column j
    of (P @ Q): the corresponding structural IRF must satisfy that
    shock's sign pattern, with either sign of the column allowed. The
    column assignment is fixed; there is no search over columns.

    If all m shocks can be simultaneously assigned, accept the rotation
    and produce an IRF tensor of shape ``(H+1, n, m)`` where column j
    corresponds to ``shock_names[j]``. Otherwise reject.

    Narrative and FEV-share restrictions: in multi-shock mode each
    restriction must specify a ``shock`` key naming which identified
    shock the restriction applies to. Restrictions are checked AFTER a
    candidate column-and-sign assignment is made for that shock.
    """
    rng = np.random.default_rng(seed)
    n = fit.n_vars
    shock_names = list(sign_restrictions.keys())
    m = len(shock_names)

    # Build per-shock sign arrays
    sign_arrays: list[np.ndarray] = []
    for name in shock_names:
        var_signs = sign_restrictions[name]
        for v in var_signs:
            if v not in fit.variable_names:
                raise ValueError(
                    f"unknown variable {v!r} in sign restriction for shock "
                    f"{name!r}"
                )
        sa = np.zeros(n, dtype=int)
        for v, s in var_signs.items():
            sa[fit.variable_names.index(v)] = int(np.sign(s))
        sign_arrays.append(sa)

    # Resolve per-shock narrative + FEV restrictions
    narr_by_shock_raw = _resolve_per_shock_restrictions(
        narrative_restrictions, shock_names, "narrative_restrictions",
    )
    fev_by_shock_raw = _resolve_per_shock_restrictions(
        fev_share_restrictions, shock_names, "fev_share_restrictions",
    )
    mag_by_shock_raw = _resolve_per_shock_restrictions(
        magnitude_restrictions, shock_names, "magnitude_restrictions",
    )

    # Convert narrative-by-shock raw dicts into resolved entries.
    # Two restriction kinds (ADRR 2018):
    #   * Type 1 ("sign", default) — (t_idx, sign): structural shock at date t must have sign s
    #   * Type 2 ("largest_contributor" / "largest_positive_contributor" /
    #            "largest_negative_contributor") — (t_idx, var_idx, direction):
    #            this shock must dominate all other shocks in contribution to
    #            variable var at date t.
    narrative_by_shock: list[list[tuple[int, int]]] = []
    type2_by_shock: list[list[dict]] = []
    eff_index = fit.data.index[fit.p:]
    valid_type2 = {
        "largest_contributor", "largest_positive_contributor",
        "largest_negative_contributor",
    }
    for name in shock_names:
        type1_resolved: list[tuple[int, int]] = []
        type2_resolved: list[dict] = []
        for nr in narr_by_shock_raw[name]:
            kind = nr.get("type", "sign")
            if "date" not in nr:
                raise ValueError(
                    f"narrative restriction for shock {name!r} must have a "
                    f"'date' key; got {list(nr.keys())}"
                )
            ts = pd.Timestamp(nr["date"])
            if ts not in eff_index:
                raise ValueError(
                    f"narrative date {ts} for shock {name!r} not in "
                    f"effective sample [{eff_index[0]} ... {eff_index[-1]}]"
                )
            t_idx = list(eff_index).index(ts)

            if kind == "sign":
                # Type 1 (ADRR 2018) — sign of the structural shock at date t.
                if "sign" not in nr:
                    raise ValueError(
                        f"Type 1 (sign) narrative restriction for shock {name!r} "
                        f"must have a 'sign' key; got {list(nr.keys())}"
                    )
                sgn = int(np.sign(nr["sign"]))
                if sgn not in (-1, 1):
                    raise ValueError(
                        f"narrative sign for shock {name!r} must be +1 or -1; "
                        f"got {nr['sign']!r}"
                    )
                type1_resolved.append((t_idx, sgn))
            elif kind in valid_type2:
                # Type 2 (ADRR 2018) — historical-contribution restriction.
                if "variable" not in nr:
                    raise ValueError(
                        f"Type 2 ({kind}) narrative restriction for shock {name!r} "
                        f"must have a 'variable' key; got {list(nr.keys())}"
                    )
                v_name = nr["variable"]
                if v_name not in fit.variable_names:
                    raise ValueError(
                        f"unknown variable {v_name!r} in Type 2 narrative restriction "
                        f"for shock {name!r}"
                    )
                v_idx = fit.variable_names.index(v_name)
                # Direction: which form of dominance?
                if kind == "largest_contributor":
                    direction = "absolute"
                elif kind == "largest_positive_contributor":
                    direction = "positive"
                else:
                    direction = "negative"
                criterion, contribution = _parse_type2_extras(nr)
                type2_resolved.append({
                    "t_idx": t_idx,
                    "var_idx": v_idx,
                    "direction": direction,
                    "kind": kind,
                    "criterion": criterion,
                    "contribution": contribution,
                })
            else:
                raise ValueError(
                    f"unknown narrative restriction type {kind!r} for shock "
                    f"{name!r}; expected 'sign' or one of {sorted(valid_type2)}"
                )
        narrative_by_shock.append(type1_resolved)
        type2_by_shock.append(type2_resolved)

    # Resolve FEV-share by shock
    fev_by_shock: list[list[tuple[int, int, float, float]]] = []
    for name in shock_names:
        resolved_f: list[tuple[int, int, float, float]] = []
        for fr in fev_by_shock_raw[name]:
            if "variable" not in fr or "horizon" not in fr:
                raise ValueError(
                    f"FEV-share restriction for shock {name!r} must have "
                    f"'variable' and 'horizon' keys; got {list(fr.keys())}"
                )
            v_name = fr["variable"]
            if v_name not in fit.variable_names:
                raise ValueError(
                    f"unknown variable {v_name!r} in FEV restriction for "
                    f"shock {name!r}"
                )
            h = int(fr["horizon"])
            if h < 1 or h > horizon + 1:
                raise ValueError(
                    f"FEV horizon {h} for shock {name!r} outside "
                    f"[1, {horizon + 1}] (H-term convention: horizon H sums "
                    f"s = 0..H-1, matching lib.structural_fevd)"
                )
            lo = float(fr.get("lower", 0.0))
            hi = float(fr.get("upper", 1.0))
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"FEV bounds must satisfy 0 <= lower <= upper <= 1; "
                    f"got lower={lo}, upper={hi}"
                )
            # h-1: cumsum index for the H-term share (see single-shock path)
            resolved_f.append((fit.variable_names.index(v_name), h - 1, lo, hi))
        fev_by_shock.append(resolved_f)

    # Resolve relative-magnitude by shock.
    mag_by_shock: list[list[tuple[int, int, int, float]]] = []
    for name in shock_names:
        resolved_m: list[tuple[int, int, int, float]] = []
        for mr in mag_by_shock_raw[name]:
            resolved_m += _resolve_magnitude_restriction(
                mr, list(fit.variable_names), horizon)
        mag_by_shock.append(resolved_m)

    # Resolve higher-moment by shock (AFM 2023).
    # Per-shock thresholds (no 'vs' key) checked during the assignment loop.
    # Cross-shock comparisons (has 'vs' key) checked AFTER all m shocks are
    # assigned (since they depend on the assignment of the comparison shock).
    hm_by_shock_raw = _resolve_per_shock_restrictions(
        higher_moment_restrictions, shock_names, "higher_moment_restrictions",
    )
    hm_threshold_by_shock: list[list[dict]] = []
    hm_comparisons: list[dict] = []
    for name in shock_names:
        thresh_list: list[dict] = []
        for i, hm in enumerate(hm_by_shock_raw[name]):
            res = _validate_higher_moment_restriction(
                hm, where=f"shock {name!r}, restriction #{i}", multi_shock=True,
            )
            if res["vs"] is None:
                thresh_list.append(res)
            else:
                # Cross-shock comparison — defer to post-assignment phase
                if res["vs"] not in shock_names:
                    raise ValueError(
                        f"higher_moment_restrictions: cross-shock 'vs' "
                        f"references shock {res['vs']!r} not in "
                        f"sign_restrictions keys {shock_names}"
                    )
                hm_comparisons.append({
                    "shock": name, "vs": res["vs"],
                    "moment": res["moment"], "comparison": res["comparison"],
                    "margin": res["margin"], "abs": res["abs"],
                })
        hm_threshold_by_shock.append(thresh_list)

    any_narrative = any(narrative_by_shock)
    any_fev = any(fev_by_shock)
    any_mag = any(mag_by_shock)
    any_type2 = any(type2_by_shock)
    any_hm_threshold = any(hm_threshold_by_shock)
    any_hm_comparison = bool(hm_comparisons)
    any_hm = any_hm_threshold or any_hm_comparison

    # For Type 2 restrictions, pre-compute the max historical date we
    # need so the MA representation is extended accordingly.
    # Only 'cumulative' restrictions drive the MA-extension horizon; 'unexpected'
    # ones need just the impact matrix (default -1 => skip the MA extension).
    max_t_idx_type2 = -1
    for shock_type2 in type2_by_shock:
        for r in shock_type2:
            if r["contribution"] == "cumulative" and r["t_idx"] > max_t_idx_type2:
                max_t_idx_type2 = r["t_idx"]

    if parameter_uncertainty:
        n_param_draws = n_param_draws or max_rotations
        param_draws = fit.parameter_draws(n_draws=n_param_draws, seed=seed)
    else:
        param_draws = None

    accepted: list[np.ndarray] = []
    n_attempted = 0
    n_rejected_sign = 0          # rejected at the sign-assignment stage
    n_rejected_narrative = 0     # rejected at narrative check
    n_rejected_fev_share = 0     # rejected at FEV check
    n_rejected_magnitude = 0     # rejected at relative-magnitude check
    n_rejected_type2 = 0          # rejected at Type 2 historical-contribution check
    n_rejected_hm = 0             # rejected at higher-moment check (AFM 2023)

    while len(accepted) < target_accepted and n_attempted < max_rotations:
        if param_draws is not None:
            B_d, Sigma_d = param_draws[n_attempted % len(param_draws)]
        else:
            B_d, Sigma_d = fit.B_hat, fit.Sigma_u

        try:
            P = np.linalg.cholesky(Sigma_d)
        except np.linalg.LinAlgError:
            n_attempted += 1
            continue

        X = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(X)
        Q = Q @ np.diag(np.sign(np.diag(R)))

        lag_mats = extract_lag_matrices(B_d, n, fit.p)
        Phi = ma_representation(lag_mats, horizon)

        # Pre-compute total FEV (depends only on Sigma_d + Phi)
        if any_fev:
            total_fev = np.cumsum(
                np.array([
                    np.diag(Phi[h] @ Sigma_d @ Phi[h].T)
                    for h in range(horizon + 1)
                ]),
                axis=0,
            )  # (H+1, n)
        else:
            total_fev = None

        if any_narrative or any_hm or any_type2:
            try:
                Pinv_T = np.linalg.inv(P).T
            except np.linalg.LinAlgError:
                n_attempted += 1
                continue
            eps_basis = _draw_residuals(fit, B_d) @ Pinv_T  # (T_eff, n)
        else:
            eps_basis = None

        # Pre-compute all candidate IRFs: irf_columns[c] = IRF of column c
        # of Q (n-vector b_struct = P @ Q[:, c], IRF = Phi @ b_struct).
        b_struct_all = P @ Q                               # (n, n)
        irf_all = np.einsum("hij,jk->hik", Phi, b_struct_all)  # (H+1, n, n)

        # Fixed column assignment (canonical RWZ measure): shock j is
        # checked against column j of the Haar rotation, with a per-column
        # sign flip. Haar columns are exchangeable, so fixing the
        # assignment loses no generality in distribution, and acceptance
        # does not depend on the order in which the shocks are listed.
        assignments: list[tuple[int, int, np.ndarray]] = []   # (col, sign, signed_irf)
        rejection_reason: str | None = None
        for shock_idx, sign_array in enumerate(sign_arrays):
            found = False
            col = shock_idx
            for sgn in (+1, -1):
                signed_irf = sgn * irf_all[:, :, col]
                if not _matches_signs(signed_irf, sign_array, restriction_horizons):
                    continue
                # Sign restrictions for this shock pass.
                # Narrative check (per-shock)
                if narrative_by_shock[shock_idx]:
                    q_col = sgn * Q[:, col]
                    eps_series = eps_basis @ q_col
                    narrative_ok = all(
                        int(np.sign(eps_series[t_idx])) == sgn_n
                        for t_idx, sgn_n in narrative_by_shock[shock_idx]
                    )
                    if not narrative_ok:
                        rejection_reason = "narrative"
                        continue
                # FEV-share check (per-shock)
                if fev_by_shock[shock_idx]:
                    irf_sq_cum = np.cumsum(signed_irf ** 2, axis=0)
                    fev_ok = True
                    for v_idx, h_check, lo, hi in fev_by_shock[shock_idx]:
                        denom = total_fev[h_check, v_idx]
                        if denom <= 0:
                            fev_ok = False
                            break
                        share = irf_sq_cum[h_check, v_idx] / denom
                        if not (lo <= share <= hi):
                            fev_ok = False
                            break
                    if not fev_ok:
                        rejection_reason = "fev_share"
                        continue
                # Relative-magnitude check (per-shock)
                if mag_by_shock[shock_idx]:
                    mag_ok = True
                    for s_idx, w_idx, h_check, ratio in mag_by_shock[shock_idx]:
                        if (abs(signed_irf[h_check, s_idx]) + 1e-12
                                < ratio * abs(signed_irf[h_check, w_idx])):
                            mag_ok = False
                            break
                    if not mag_ok:
                        rejection_reason = "magnitude"
                        continue
                # Higher-moment threshold check (AFM 2023, per-shock)
                if hm_threshold_by_shock[shock_idx]:
                    q_col = sgn * Q[:, col]
                    eps_series = eps_basis @ q_col
                    hm_ok = True
                    for hm in hm_threshold_by_shock[shock_idx]:
                        m_val = _compute_shock_moment(
                            eps_series, hm["moment"],
                        )
                        if not _check_moment_threshold(
                            m_val, hm["lower"], hm["upper"], hm["abs"],
                        ):
                            hm_ok = False
                            break
                    if not hm_ok:
                        rejection_reason = "higher_moment"
                        continue
                # All checks pass — this column satisfies this shock
                assignments.append((col, sgn, signed_irf))
                found = True
                break
            if not found:
                if rejection_reason == "narrative":
                    n_rejected_narrative += 1
                elif rejection_reason == "fev_share":
                    n_rejected_fev_share += 1
                elif rejection_reason == "magnitude":
                    n_rejected_magnitude += 1
                elif rejection_reason == "higher_moment":
                    n_rejected_hm += 1
                else:
                    n_rejected_sign += 1
                break

        if len(assignments) == m:
            # Type 2 historical-contribution restrictions (ADRR 2018):
            # for each restriction, the assigned shock's column must
            # dominate all other shock columns in absolute (or signed)
            # contribution to the specified variable at the specified date.
            type2_ok = True
            if any_type2:
                # Pre-compute structural shocks for ALL columns of Q at
                # all dates: eps_all[t, c] = u_t' @ (P^{-T} @ Q[:, c]).
                # eps_basis was already computed when any_narrative; reuse
                # if available, else compute now.
                if not any_narrative:
                    try:
                        Pinv_T = np.linalg.inv(P).T
                    except np.linalg.LinAlgError:
                        n_attempted += 1
                        continue
                    eps_basis_t2 = _draw_residuals(fit, B_d) @ Pinv_T
                else:
                    eps_basis_t2 = eps_basis
                # eps_basis_t2 @ Q : structural shocks for ALL n columns
                eps_all = eps_basis_t2 @ Q          # (T_eff, n)
                # 'cumulative' restrictions need Phi convolved to their date;
                # 'unexpected' ones need only the impact matrix B = P @ Q.
                B_impact = P @ Q                      # (n, n)
                if max_t_idx_type2 >= 0:
                    Phi_ext = ma_representation(lag_mats, max_t_idx_type2)
                    irf_all_ext = np.einsum("hij,jk->hik", Phi_ext, B_impact)
                else:
                    irf_all_ext = None
                # Check each Type 2 restriction
                for shock_idx, restrictions in enumerate(type2_by_shock):
                    if not restrictions:
                        continue
                    assigned_col, assigned_sgn, _ = assignments[shock_idx]
                    for r in restrictions:
                        contribs = _type2_contributions(
                            irf_all_ext, eps_all, B_impact,
                            r["var_idx"], r["t_idx"], r["contribution"],
                        )
                        # The assigned column's contribution must dominate — ADRR
                        # dominance, unified with the single-shock path via
                        # _type2_dominates (criterion = most_important / overwhelming).
                        if not _type2_dominates(
                            contribs, assigned_col, r["direction"],
                            criterion=r["criterion"],
                        ):
                            type2_ok = False
                            break
                    if not type2_ok:
                        break
                if not type2_ok:
                    n_rejected_type2 += 1
                    n_attempted += 1
                    continue
            # Cross-shock higher-moment comparisons (AFM 2023): each entry
            # requires the moment of one assigned shock to dominate (or be
            # dominated by) the moment of another assigned shock. Computed
            # after the per-shock assignment phase since the comparison
            # depends on the assigned column / sign of BOTH shocks.
            if any_hm_comparison:
                # Recover per-shock structural-shock series for each
                # assignment. eps_per_shock[k] has shape (T_eff,).
                eps_per_shock = []
                for sh_idx in range(m):
                    col_k, sgn_k, _ = assignments[sh_idx]
                    eps_per_shock.append(eps_basis @ (sgn_k * Q[:, col_k]))
                cmp_ok = True
                for cmp in hm_comparisons:
                    a_idx = shock_names.index(cmp["shock"])
                    b_idx = shock_names.index(cmp["vs"])
                    a_val = _compute_shock_moment(
                        eps_per_shock[a_idx], cmp["moment"],
                    )
                    b_val = _compute_shock_moment(
                        eps_per_shock[b_idx], cmp["moment"],
                    )
                    if cmp["abs"]:
                        a_val = abs(a_val)
                        b_val = abs(b_val)
                    if cmp["comparison"] == "greater":
                        if not (a_val >= b_val + cmp["margin"] - 1e-12):
                            cmp_ok = False
                            break
                    else:    # "less"
                        if not (a_val <= b_val - cmp["margin"] + 1e-12):
                            cmp_ok = False
                            break
                if not cmp_ok:
                    n_rejected_hm += 1
                    n_attempted += 1
                    continue
            # Accept
            irf_tensor = np.stack([a[2] for a in assignments], axis=-1)
            accepted.append(irf_tensor)
        n_attempted += 1

    if not accepted:
        raise RuntimeError(
            f"no rotations satisfied the multi-shock sign restrictions after "
            f"{n_attempted} attempts — check the restriction set or "
            f"increase max_rotations. Rejection counts: sign={n_rejected_sign}, "
            f"narrative={n_rejected_narrative}, fev_share={n_rejected_fev_share}, "
            f"type2={n_rejected_type2}, higher_moment={n_rejected_hm}"
        )

    draws = np.stack(accepted, axis=0)   # (n_kept, H+1, n, m)

    # Build identification tag
    bits = ["multi_shock_sign"]
    if any_narrative:
        bits.append("narrative")
    if any_type2:
        bits.append("type2_contribution")
    if any_fev:
        bits.append("fev_share")
    if any_mag:
        bits.append("magnitude")
    if any_hm:
        bits.append("higher_moment")
    id_str = "_".join(bits) + "_restrictions"
    method_str = (
        "VAR / multi-shock sign restrictions (RWZ 2010)"
        + (" + Type 1 narrative" if any_narrative else "")
        + (" + Type 2 contribution (ADRR 2018)" if any_type2 else "")
        + (" + FEV-share" if any_fev else "")
        + (" + relative magnitude" if any_mag else "")
        + (" + higher-moment (AFM 2023)" if any_hm else "")
    )

    metadata = {
        "method": method_str,
        "identification": id_str,
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "n_shocks_identified": m,
        "shock_names": list(shock_names),
        "target_accepted": target_accepted,
        "n_accepted": len(accepted),
        "n_attempted": n_attempted,
        "acceptance_rate": len(accepted) / max(n_attempted, 1),
        "median_target": _median_target_meta(draws),
        "restriction_horizons": restriction_horizons,
        "restrictions": {name: dict(sign_restrictions[name]) for name in shock_names},
        "narrative_restrictions": (
            [dict(nr) for nr in narrative_restrictions]
            if narrative_restrictions else None
        ),
        "fev_share_restrictions": (
            [dict(fr) for fr in fev_share_restrictions]
            if fev_share_restrictions else None
        ),
        "higher_moment_restrictions": (
            [dict(hm) for hm in higher_moment_restrictions]
            if higher_moment_restrictions else None
        ),
        "magnitude_restrictions": (
            [dict(mr) for mr in magnitude_restrictions]
            if magnitude_restrictions else None
        ),
        "n_rejected_by_sign": int(n_rejected_sign),
        "n_rejected_by_narrative": int(n_rejected_narrative),
        "n_rejected_by_fev_share": int(n_rejected_fev_share),
        "n_rejected_by_magnitude": int(n_rejected_magnitude),
        "n_rejected_by_type2": int(n_rejected_type2),
        "n_rejected_by_higher_moment": int(n_rejected_hm),
        "parameter_uncertainty": parameter_uncertainty,
        "seed": seed,
    }
    return IRFContainer.from_draws(
        draws=draws,
        variable_names=list(fit.variable_names),
        shock_names=list(shock_names),
        metadata=metadata,
    )


def _matches_signs(irf: np.ndarray, sign_array: np.ndarray,
                   restriction_horizons: int) -> bool:
    """Check ``irf[0:H, i]`` agrees with ``sign_array[i]`` for every ``i`` with
    non-zero sign."""
    for h in range(restriction_horizons):
        for i, s in enumerate(sign_array):
            if s == 0:
                continue
            if s > 0 and irf[h, i] < 0:
                return False
            if s < 0 and irf[h, i] > 0:
                return False
    return True


# ---------------------------------------------------------------------------
# Proxy-SVAR (external-instrument identification)
# ---------------------------------------------------------------------------


def proxy_svar_irf(
    fit: VARFit,
    instrument: np.ndarray,
    policy_var: str | int | list,
    horizon: int,
    shock_name: str | list[str] | None = None,
    bands: str = "wild_bootstrap",
    n_draws: int = 2000,
    seed: int | None = None,
    normalization: str = "unit_policy",
) -> tuple[IRFContainer, dict]:
    """External-instrument SVAR identification (Stock-Watson 2018 / Mertens-Ravn 2013).

    Two modes, dispatched by instrument shape:

    **Single-instrument** (1D instrument, ``policy_var`` str/int).
    Given an external instrument ``Z_t`` correlated with one structural shock
    and uncorrelated with all others, recover the corresponding column of the
    structural impact matrix:

        Σ_{uZ} = E[u_t Z_t] = H_1 * α

    **Multi-instrument** (2D instrument with shape ``(T, m)``, ``policy_var``
    list of length ``m``). Mertens-Ravn (2013) §II: with ``m`` external
    instruments and ``m`` simultaneous policy variables, recover the
    ``m``-column block of the structural impact matrix via

        Σ_uZ = H_1 * Φ                    (n × m)

    where Φ is the ``m × m`` instrument-shock loading. Partitioning into
    policy and other blocks:

        Σ_p = E[u^p_t Z_t']               (m × m)
        Σ_q = E[u^q_t Z_t']               ((n-m) × m)

    and imposing the unit-policy normalisation ``H_1^p = I_m`` yields

        H_1^q = Σ_q · Σ_p^{-1}

    The ``m × m`` block ``Σ_p`` must be invertible — instruments must
    move distinct policy variables. A NUMERICALLY rank-deficient ``Σ_p``
    (e.g. duplicated or linearly dependent instruments) is rejected with
    a ``RuntimeError`` via a scale-aware SVD rank check before
    inversion; strong-looking per-pair F statistics cannot substitute
    for this rank requirement.

    Normalisations:

    * ``"unit_policy"`` — H_1^p = I_m (unit response on impact, zero
      contemporaneous spillover between policy variables).
    * ``"one_sd"`` — scale each shock column by ``std(u_{policy_i})``.
    * ``"reliability"`` — (single-instrument only) scale by the Mertens-Ravn
      (2013) / Gertler-Karadi (2015, fn. 4) reliability-adjusted policy-shock
      std ``s_p``, correcting for the proxy's measurement error. Same IRF shape
      as ``"unit_policy"``; only the level differs.

    Parameters
    ----------
    fit : VARFit
        Reduced-form VAR fit. Residuals have shape ``(T_eff, n)``.
    instrument : array-like
        Single mode: 1D, length ``len(fit.data)``.
        Multi mode: 2D, shape ``(len(fit.data), m)``.
        Entries may be NaN to mark periods where the instrument is NOT
        OBSERVED (e.g. a proxy that starts mid-sample, like GK's FF4
        before 1991). NaNs are zero-filled for the moment computations —
        the identified impact ratios are invariant to the fill — but
        EXCLUDED from the first-stage diagnostics, which zero-padding
        would otherwise dilute toward zero (a spurious weak-instrument
        flag). True zeros are observed zero-surprise periods (FOMC-free
        months) and DO enter the diagnostics. In multi mode the same
        semantics apply per column.
    policy_var : str, int, or list
        Single mode: str / int. Multi mode: list of length ``m``, each a
        str or int. Must be distinct.
    horizon : int
        IRF horizon ``H``. The container holds ``0..H``.
    shock_name : str, list of str, or None
        Single mode: name string (defaults to ``"<policy_var> shock"``).
        Multi mode: list of length ``m``, or a base string used as a prefix
        (``"<base>_0", "<base>_1", ...``), or None (defaults to per-policy
        variable names).
    bands : {"none", "moving_block_bootstrap", "wild_bootstrap"}
        Inference method. The default, ``"wild_bootstrap"``, is kept to
        compare with earlier studies and is not valid for inference in a
        proxy SVAR (Jentsch and Lunsford 2019: its bands are too narrow,
        most of all at impact); use ``"moving_block_bootstrap"`` for a
        single instrument.
    n_draws : int
        Bootstrap draws.
    seed : int, optional
    normalization : {"unit_policy", "one_sd", "reliability"}
        ("reliability" is single-instrument only.)

    Returns
    -------
    IRFContainer of shape ``(H+1, n_vars, m)`` (m=1 in single mode).
    dict
        First-stage diagnostics. Single mode: ``F``, ``gamma``, ``R2``,
        ``n_obs``, ``weak_instrument_F_lt_10``. Multi mode: ``per_pair``
        list of ``{instrument_idx, policy_idx, gamma, R2, F}`` plus
        ``weak_instrument_min_own_F``, ``weak_instrument_F_lt_10``, and
        ``Sigma_p_cond`` (condition number of the policy-block matrix).
    """
    z = np.asarray(instrument)
    if z.ndim == 2 or isinstance(policy_var, (list, tuple)):
        # Multi-instrument path
        return _proxy_svar_irf_multi(
            fit=fit, instrument=z, policy_vars=policy_var, horizon=horizon,
            shock_name=shock_name, bands=bands, n_draws=n_draws, seed=seed,
            normalization=normalization,
        )

    # Single-instrument path (legacy)
    z = z.ravel().astype(float)
    if z.shape[0] != len(fit.data):
        raise ValueError(
            f"instrument length ({z.shape[0]}) != data length ({len(fit.data)})"
        )
    # NaN = "not observed" (see docstring): zero-fill for the moment
    # computations, but keep the mask so the first-stage diagnostics run
    # on the observed subsample only.
    z_observed = np.isfinite(z)
    if not z_observed.any():
        raise ValueError("instrument has no observed (non-NaN) entries")
    z = np.where(z_observed, z, 0.0)
    if isinstance(policy_var, str):
        if policy_var not in fit.variable_names:
            raise ValueError(f"policy_var {policy_var!r} not in variable_names")
        policy_idx = fit.variable_names.index(policy_var)
    else:
        policy_idx = int(policy_var)
        if not 0 <= policy_idx < fit.n_vars:
            raise ValueError(f"policy_idx {policy_idx} out of range")
    if normalization not in ("unit_policy", "one_sd", "reliability"):
        raise ValueError(
            f"normalization must be 'unit_policy', 'one_sd' or 'reliability'; "
            f"got {normalization!r}"
        )

    # Align instrument with residuals (drop the first p observations)
    z_aligned = z[fit.p:]
    if len(z_aligned) != fit.t_eff:
        raise RuntimeError("internal alignment error")

    # Point estimate
    impact = _proxy_impact_column(
        fit.residuals, z_aligned, policy_idx, normalization,
        n_regressors=fit.k,
    )
    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,j->hi", Phi, impact)   # (H+1, n)
    irf_point = irf_point[:, :, None]                  # add shock axis

    # First-stage diagnostics — on the instrument's OBSERVED subsample only
    # (zero-padded unobserved periods would dilute F toward zero)
    obs_aligned = z_observed[fit.p:]
    fs_diag = _proxy_first_stage(
        np.asarray(fit.residuals)[obs_aligned, policy_idx],
        z_aligned[obs_aligned],
    )
    if not obs_aligned.all():
        fs_diag["n_unobserved_in_sample"] = int((~obs_aligned).sum())

    metadata = {
        "method": "VAR / proxy SVAR (external instrument)",
        "identification": "external_instrument",
        "policy_variable": fit.variable_names[policy_idx],
        "normalization": normalization,
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
        "first_stage": fs_diag,
    }
    if shock_name is None:
        shock_name = f"{fit.variable_names[policy_idx]} (proxy-SVAR)"

    if bands == "none":
        return (
            IRFContainer.from_point(
                point=irf_point,
                variable_names=list(fit.variable_names),
                shock_names=[shock_name],
                metadata=metadata,
            ),
            fs_diag,
        )

    if bands == "moving_block_bootstrap":
        draws, block_len = _proxy_moving_block_bootstrap(
            fit=fit, z_aligned=z_aligned, policy_idx=policy_idx,
            horizon=horizon, n_draws=n_draws, seed=seed,
            normalization=normalization,
        )
        metadata["block_length"] = int(block_len)
    elif bands == "wild_bootstrap":
        draws = _proxy_wild_bootstrap(
            fit=fit, z_aligned=z_aligned, policy_idx=policy_idx,
            horizon=horizon, n_draws=n_draws, seed=seed,
            normalization=normalization,
        )
    else:
        raise ValueError(
            f"unknown bands method: {bands!r}; expected 'none', "
            f"'moving_block_bootstrap' (Jentsch-Lunsford 2019, recommended) "
            f"or 'wild_bootstrap' (MR/GK legacy — JL prove it invalid)"
        )

    return (
        IRFContainer.from_draws(
            draws=draws,
            variable_names=list(fit.variable_names),
            shock_names=[shock_name],
            metadata=metadata,
        ),
        fs_diag,
    )


def _proxy_reliability_sp(
    residuals: np.ndarray, ratios: np.ndarray, policy_idx: int, n_regressors: int,
) -> float:
    """Mertens-Ravn (2013) / Gertler-Karadi (2015, fn. 4) reliability-adjusted
    policy-shock standard deviation ``s_p``.

    Given the IV-identified impact ratios (``ratios[policy_idx] == 1`` and
    ``ratios[k] == s_k / s_p``) and the df-adjusted reduced-form residual
    covariance ``Sigma`` partitioned into policy (1) vs other (2) variables as
    ``[[S11, S21'], [S21, S22]]``, recover ``s_p`` — the std of the policy
    structural shock net of the proxy's measurement error — by the closed form

        s_p^2 = S11 - (S21 - r S11)' Q^{-1} (S21 - r S11),
        Q     = r S11 r' - (S21 r' + r S21') + S22,        r = s_k / s_p.

    The covariance is mean-centred and uses the degrees-of-freedom divisor
    ``T - n_regressors``. Any variable can be the policy variable
    (``policy_idx``); it need not be ordered first.
    """
    T, n = residuals.shape
    others = [k for k in range(n) if k != policy_idx]
    rc = residuals - residuals.mean(axis=0)
    Sigma = (rc.T @ rc) / (T - n_regressors)
    S11 = float(Sigma[policy_idx, policy_idx])
    if not others:                    # no other variables -> no measurement-error
        return float(np.sqrt(S11))    # correction possible; s_p = std(u_policy)
    S21 = Sigma[np.ix_(others, [policy_idx])]      # (n-1, 1)
    S22 = Sigma[np.ix_(others, others)]            # (n-1, n-1)
    r = np.asarray(ratios)[others].reshape(-1, 1)  # (n-1, 1)
    Q = S11 * (r @ r.T) - (S21 @ r.T + r @ S21.T) + S22
    diff = S21 - S11 * r
    val = S11 - float((diff.T @ np.linalg.solve(Q, diff)).item())
    if val <= 0:
        raise RuntimeError(
            f"reliability normalization: non-positive shock-variance estimate "
            f"(s_p^2 = {val:.3e}); the proxy may be too weak or Sigma "
            f"ill-conditioned. Use 'unit_policy' or 'one_sd' instead."
        )
    return float(np.sqrt(val))


def _proxy_impact_column(
    residuals: np.ndarray, instrument: np.ndarray, policy_idx: int,
    normalization: str, n_regressors: int | None = None,
) -> np.ndarray:
    """Recover the (single) structural impact column from residuals + instrument.

    EXACT FORMULA:

      1. Cross-covariance: sigma_uZ_k = (1/T_eff) * sum_t u_{t,k} * z_t
         for each variable k. By the proxy-SVAR identifying assumptions
         (z relevant for the policy shock, exogenous to all others),
         sigma_uZ is proportional to the structural impact column s:
            sigma_uZ = s * E[eps_t * z_t]
         so s_k / s_{policy} = sigma_uZ_k / sigma_uZ_{policy}.

      2. Normalisation: depends on `normalization`:

         * "unit_policy": s_{policy} = 1, so s_k = sigma_uZ_k / sigma_uZ_{policy}.
           The policy variable jumps by 1 unit on impact.

         * "one_sd": s_{policy} = np.std(u_{policy}, ddof=1)
           — the sample standard deviation of the policy variable's
           reduced-form residual, with Bessel correction (T_eff - 1
           divisor). Each column entry is scaled accordingly:
              s_k = (sigma_uZ_k / sigma_uZ_{policy}) * np.std(u_{policy}, ddof=1).
           The shock has variance equal to the policy variable's
           reduced-form-innovation variance (under ddof=1), NOT unit
           variance. This is the standard "one standard deviation"
           scaling used in IRF plots (Mertens-Ravn 2013 §III).

           PRECISION NOTE: this differs from sqrt(Sigma_u[policy, policy])
           by a factor of sqrt(T_eff / (T_eff - 1)). Sigma_u is computed
           by statsmodels with a degrees-of-freedom divisor matching the
           VAR fit; np.std with ddof=1 uses T_eff - 1. On a T_eff = 200
           sample the relative difference is ~0.25%; on T_eff = 1000 it's
           ~0.05%. The library's choice is ddof=1.

         * "reliability": s_{policy} = the Mertens-Ravn (2013) / Gertler-Karadi
           (2015, fn. 4) reliability-adjusted policy-shock std s_p, recovered
           from the df-adjusted reduced-form residual covariance and the impact
           ratios (see _proxy_reliability_sp); requires n_regressors. The policy
           variable jumps by s_p (not 1) on impact, correcting for the proxy's
           measurement error. The IRF SHAPE is identical to "unit_policy"; only
           the level differs (by the scalar s_p).

    Raises RuntimeError if the instrument has near-zero covariance with
    the policy variable's reduced-form residual (weak instrument).
    """
    sigma_uZ = (residuals * instrument[:, None]).mean(axis=0)   # (n,)
    base = sigma_uZ[policy_idx]
    if abs(base) < 1e-12:
        raise RuntimeError(
            f"Instrument has essentially zero covariance with the policy "
            f"variable's reduced-form residual (base = {base:.2e}). "
            f"Identification fails — check instrument relevance."
        )
    impact = sigma_uZ / base
    if normalization == "one_sd":
        impact = impact * np.std(residuals[:, policy_idx], ddof=1)
    elif normalization == "reliability":
        if n_regressors is None:
            raise ValueError(
                "reliability normalization requires n_regressors (the VAR's "
                "per-equation regressor count) to df-adjust the residual covariance"
            )
        impact = impact * _proxy_reliability_sp(
            residuals, impact, policy_idx, n_regressors,
        )
    return impact


def _proxy_first_stage(u_policy: np.ndarray, instrument: np.ndarray) -> dict:
    """First-stage diagnostics: F-statistic, gamma, R²."""
    n = len(instrument)
    Z = np.column_stack([np.ones(n), instrument])
    beta, *_ = np.linalg.lstsq(Z, u_policy, rcond=None)
    fitted = Z @ beta
    resid = u_policy - fitted
    sse = float((resid ** 2).sum())
    sst = float(((u_policy - u_policy.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    f_stat = (n - 2) * r2 / max(1.0 - r2, 1e-12)
    return {
        "F": float(f_stat),
        "gamma": float(beta[1]),
        "R2": float(r2),
        "n_obs": int(n),
        "weak_instrument_F_lt_10": bool(f_stat < 10),
    }


def _proxy_moving_block_bootstrap(
    fit: VARFit, z_aligned: np.ndarray, policy_idx: int, horizon: int,
    n_draws: int, seed: int | None, normalization: str,
    block_length: int | None = None,
) -> tuple[np.ndarray, int]:
    """Residual moving-block bootstrap for proxy-SVAR (Jentsch-Lunsford 2019).

    JL's replacement for the (invalid) Rademacher wild bootstrap: resample
    overlapping blocks of the JOINT vector ``(u_t, z_t)``, preserving the
    higher cross-moments of residuals and instrument that the impact
    column depends on (exactly what the Rademacher weights destroy).

    Per replication:

    1. Draw ``ceil(T/l)`` overlapping-block start indices uniformly with
       replacement (block length ``l ~ 5.03 T^{1/4}``, JL's rule);
       concatenate the joint ``(u, z)`` blocks and truncate to ``T``.
    2. RECENTER per within-block offset ``j``: subtract the mean over all
       admissible starts of ``u_{i+j}`` (and ``z_{i+j}``) — JL eq. (5) —
       so the bootstrap residuals are mean-zero by construction.
    3. Rebuild ``y*`` from the original VAR coefficients + ``u*``,
       re-estimate ``B``, identify the impact column from the
       re-estimated residuals and ``z*``, compute the IRF.

    Returns ``(draws, block_length)`` with draws shaped
    ``(n_kept, H+1, n_vars, 1)``.
    """
    rng = np.random.default_rng(seed)
    T_eff = fit.t_eff
    n = fit.n_vars
    p = fit.p
    k = fit.k
    has_exog = fit.has_exog
    exog_arr = fit.exog.values if has_exog else None
    exog_lags = fit.exog_lags if has_exog else 0
    n_exog_cols = (fit.n_exog * (exog_lags + 1)) if has_exog else 0
    include_const = (k - n * p - n_exog_cols) >= 1
    include_trend = (k - n * p - n_exog_cols) >= 2

    if block_length is None:
        block_length = max(int(round(5.03 * T_eff ** 0.25)), 2)
    ell = min(int(block_length), T_eff)
    n_starts = T_eff - ell + 1
    n_blocks = int(np.ceil(T_eff / ell))

    Y_full = fit.data.values
    initial = Y_full[:p].copy()
    resid = fit.residuals                          # (T_eff, n)
    B_hat = fit.B_hat

    # Per-offset recentering means (JL): mu_u[j] = mean_i u_{i+j},
    # i over admissible block starts 0..n_starts-1, offset j = 0..ell-1.
    offsets = np.arange(ell)
    starts_all = np.arange(n_starts)
    mu_u = np.stack([resid[starts_all + j].mean(axis=0) for j in offsets])
    mu_z = np.array([z_aligned[starts_all + j].mean() for j in offsets])

    draws: list[np.ndarray] = []
    for _ in range(n_draws):
        starts = rng.integers(0, n_starts, size=n_blocks)
        u_star = np.empty((n_blocks * ell, n))
        z_star = np.empty(n_blocks * ell)
        for b, s in enumerate(starts):
            sl = slice(b * ell, (b + 1) * ell)
            u_star[sl] = resid[s:s + ell] - mu_u
            z_star[sl] = z_aligned[s:s + ell] - mu_z
        u_star = u_star[:T_eff]
        z_star = z_star[:T_eff]

        # Rebuild y* recursively from the point-estimate VAR + u*
        Y_star = np.zeros_like(Y_full, dtype=float)  # float even for integer input — otherwise every simulated row is truncated
        Y_star[:p] = initial
        for t in range(p, p + T_eff):
            x_arr = np.asarray(_build_x_row(
                Y_star, t, p, include_const, include_trend, exog_arr, exog_lags,
            ))
            Y_star[t] = x_arr @ B_hat + u_star[t - p]

        Y_act_b = Y_star[p:]
        X_rows = [
            _build_x_row(Y_star, t, p, include_const, include_trend, exog_arr, exog_lags)
            for t in range(p, p + T_eff)
        ]
        X_act_b = np.asarray(X_rows)
        try:
            B_b = np.linalg.solve(X_act_b.T @ X_act_b, X_act_b.T @ Y_act_b)
        except np.linalg.LinAlgError:
            continue
        resid_b = Y_act_b - X_act_b @ B_b

        try:
            impact_b = _proxy_impact_column(
                resid_b, z_star, policy_idx, normalization,
                n_regressors=k,
            )
        except RuntimeError:
            continue

        lag_mats = extract_lag_matrices(B_b, n, p)
        if max_eigenvalue_modulus(lag_mats) >= 1.01:
            continue
        Phi_b = ma_representation(lag_mats, horizon)
        irf_b = np.einsum("hij,j->hi", Phi_b, impact_b)
        draws.append(irf_b[:, :, None])

    if not draws:
        raise RuntimeError("no successful moving-block-bootstrap replications")
    return np.stack(draws, axis=0), ell


def _proxy_wild_bootstrap(
    fit: VARFit, z_aligned: np.ndarray, policy_idx: int, horizon: int,
    n_draws: int, seed: int | None, normalization: str,
) -> np.ndarray:
    """Rademacher wild bootstrap for proxy-SVAR (Mertens-Ravn 2013 /
    Gertler-Karadi 2015 legacy inference).

    .. warning::
       Jentsch & Lunsford (2019, AER Comment) PROVED this scheme
       asymptotically INVALID for proxy-SVARs: the Rademacher weights
       destroy the higher cross-moments of ``(u_t, z_t)`` that the impact
       column depends on, producing severe undercoverage. They propose the
       residual moving-block bootstrap instead — available here as
       ``bands="moving_block_bootstrap"`` (single-instrument path), which
       should be preferred. These wild bands are retained only as the
       legacy choice of the original applications (MR 2013; GK 2015) —
       interpret them as such, not as asymptotically justified sets.

    For each bootstrap replication:

    1. Draw Rademacher weights ``omega_t in {-1, +1}`` for ``t = 1..T_eff``.
    2. Form bootstrap residuals ``u*_t = omega_t u_t`` and bootstrap
       instrument ``z*_t = omega_t z_t``.
    3. Reconstruct ``y*`` from the original VAR coefficients and ``u*``,
       then re-estimate ``B`` by OLS on the bootstrap sample.
    4. Identify a new impact column from ``u*`` and ``z*``.
    5. Compute the IRF.

    Returns ndarray of shape ``(n_kept, H+1, n_vars, 1)``.
    """
    rng = np.random.default_rng(seed)
    T_eff = fit.t_eff
    n = fit.n_vars
    p = fit.p
    k = fit.k
    has_exog = fit.has_exog
    exog_arr = fit.exog.values if has_exog else None
    exog_lags = fit.exog_lags if has_exog else 0
    n_exog_cols = (fit.n_exog * (exog_lags + 1)) if has_exog else 0
    # Layout: [lag1..lagp, (const), (trend), exog]. Detect const/trend from
    # remaining width after lag and exog blocks.
    include_const = (k - n * p - n_exog_cols) >= 1
    include_trend = (k - n * p - n_exog_cols) >= 2

    Y_full = fit.data.values
    initial = Y_full[:p].copy()
    resid = fit.residuals
    B_hat = fit.B_hat

    draws: list[np.ndarray] = []
    for _ in range(n_draws):
        omega = rng.choice([-1.0, 1.0], size=T_eff)
        u_star = resid * omega[:, None]
        z_star = z_aligned * omega

        # Reconstruct bootstrap y from VAR + bootstrap residuals (exog fixed)
        Y_star = np.zeros_like(Y_full, dtype=float)  # float even for integer input — otherwise every simulated row is truncated
        Y_star[:p] = initial
        for t in range(p, p + T_eff):
            x_arr = np.asarray(_build_x_row(
                Y_star, t, p, include_const, include_trend, exog_arr, exog_lags,
            ))
            Y_star[t] = x_arr @ B_hat + u_star[t - p]

        # Re-estimate on bootstrap sample
        Y_act_b = Y_star[p:]
        X_rows = [
            _build_x_row(Y_star, t, p, include_const, include_trend, exog_arr, exog_lags)
            for t in range(p, p + T_eff)
        ]
        X_act_b = np.asarray(X_rows)
        try:
            B_b = np.linalg.solve(X_act_b.T @ X_act_b, X_act_b.T @ Y_act_b)
        except np.linalg.LinAlgError:
            continue
        resid_b = Y_act_b - X_act_b @ B_b

        # Impact column from bootstrap residuals + bootstrap instrument
        try:
            impact_b = _proxy_impact_column(
                resid_b, z_star, policy_idx, normalization,
                n_regressors=k,
            )
        except RuntimeError:
            continue

        lag_mats = extract_lag_matrices(B_b, n, p)
        if max_eigenvalue_modulus(lag_mats) >= 1.01:
            continue
        Phi_b = ma_representation(lag_mats, horizon)
        irf_b = np.einsum("hij,j->hi", Phi_b, impact_b)
        draws.append(irf_b[:, :, None])

    if not draws:
        raise RuntimeError("no successful wild-bootstrap replications")
    return np.stack(draws, axis=0)


# ---------------------------------------------------------------------------
# Multi-instrument proxy SVAR (Mertens-Ravn 2013 §II)
# ---------------------------------------------------------------------------


def _proxy_svar_irf_multi(
    fit: VARFit,
    instrument: np.ndarray,
    policy_vars: list | tuple,
    horizon: int,
    shock_name: str | list[str] | None,
    bands: str,
    n_draws: int,
    seed: int | None,
    normalization: str,
) -> tuple[IRFContainer, dict]:
    """Multi-instrument proxy SVAR — Mertens-Ravn (2013) §II."""
    z = np.atleast_2d(instrument)
    if z.ndim != 2:
        raise ValueError(f"multi-instrument expects 2D instrument; got shape {z.shape}")
    if z.shape[0] != len(fit.data):
        raise ValueError(
            f"instrument length ({z.shape[0]}) != data length ({len(fit.data)})"
        )
    T_full, m = z.shape
    # NaN = "not observed" per column (see proxy_svar_irf docstring):
    # zero-fill for the moment computations; per-pair first-stage
    # diagnostics use each instrument column's observed rows.
    z_observed = np.isfinite(z)
    if not z_observed.any(axis=0).all():
        raise ValueError("an instrument column has no observed (non-NaN) entries")
    z = np.where(z_observed, z, 0.0)
    if not isinstance(policy_vars, (list, tuple)) or len(policy_vars) != m:
        raise ValueError(
            f"multi-instrument call needs policy_var to be a list of length "
            f"{m} (one per instrument column); got {policy_vars!r}"
        )

    policy_idxs: list[int] = []
    for pv in policy_vars:
        if isinstance(pv, str):
            if pv not in fit.variable_names:
                raise ValueError(f"policy_var {pv!r} not in variable_names")
            policy_idxs.append(fit.variable_names.index(pv))
        else:
            idx = int(pv)
            if not 0 <= idx < fit.n_vars:
                raise ValueError(f"policy_idx {idx} out of range")
            policy_idxs.append(idx)
    if len(set(policy_idxs)) != m:
        raise ValueError(
            f"policy_vars must be distinct; got {policy_vars} "
            f"(resolved to indices {policy_idxs})"
        )
    if normalization not in ("unit_policy", "one_sd"):
        raise ValueError(
            f"multi-instrument proxy SVAR supports 'unit_policy' or 'one_sd'; "
            f"'reliability' is single-instrument only (got {normalization!r})"
        )

    z_aligned = z[fit.p:, :]
    if z_aligned.shape[0] != fit.t_eff:
        raise RuntimeError("internal alignment error")

    impact = _proxy_impact_matrix(
        fit.residuals, z_aligned, policy_idxs, normalization,
    )   # (n, m)
    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,jk->hik", Phi, impact)  # (H+1, n, m)

    fs_diag = _proxy_first_stage_multi(
        np.asarray(fit.residuals)[:, policy_idxs], z_aligned, policy_idxs,
        observed=z_observed[fit.p:, :],
    )

    if shock_name is None:
        shock_names = [f"{fit.variable_names[idx]} (proxy-SVAR)" for idx in policy_idxs]
    elif isinstance(shock_name, str):
        shock_names = [f"{shock_name}_{i}" for i in range(m)]
    else:
        shock_names = list(shock_name)
        if len(shock_names) != m:
            raise ValueError(f"shock_name list must have length {m}; got {len(shock_names)}")

    metadata = {
        "method": "VAR / proxy SVAR (multi-instrument; Mertens-Ravn 2013 §II)",
        "identification": "external_instrument_multi",
        "policy_variables": [fit.variable_names[idx] for idx in policy_idxs],
        "n_instruments": m,
        "normalization": normalization,
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
        "first_stage": fs_diag,
    }

    if bands == "none":
        return (
            IRFContainer.from_point(
                point=irf_point,
                variable_names=list(fit.variable_names),
                shock_names=shock_names,
                metadata=metadata,
            ),
            fs_diag,
        )

    if bands != "wild_bootstrap":
        raise ValueError(f"unknown bands method: {bands!r}")

    draws = _proxy_wild_bootstrap_multi(
        fit=fit, z_aligned=z_aligned, policy_idxs=policy_idxs,
        horizon=horizon, n_draws=n_draws, seed=seed,
        normalization=normalization,
    )

    return (
        IRFContainer.from_draws(
            draws=draws,
            variable_names=list(fit.variable_names),
            shock_names=shock_names,
            metadata=metadata,
        ),
        fs_diag,
    )


def _proxy_impact_matrix(
    residuals: np.ndarray,
    instrument: np.ndarray,
    policy_idxs: list[int],
    normalization: str,
) -> np.ndarray:
    """Recover the m-column impact block — Mertens-Ravn (2013) Theorem 1.

    Π = E[u_t Z_t'] (n × m). Partition: Π_p (m × m policy rows),
    Π_q ((n-m) × m other rows). Under unit-policy normalisation
    ``H_1^p = I_m``: ``H_1^q = Π_q · Π_p^{-1}``.

    Σ_p must be invertible — instruments must move distinct policy variables.
    """
    T_eff, n = residuals.shape
    m = len(policy_idxs)
    if instrument.shape != (T_eff, m):
        raise RuntimeError(
            f"instrument shape mismatch: {instrument.shape} vs ({T_eff}, {m})"
        )

    Sigma_uZ = residuals.T @ instrument / T_eff       # (n, m)
    Sigma_p = Sigma_uZ[policy_idxs, :]                # (m, m)
    # Numerical rank check BEFORE inverting: np.linalg.inv does not reliably
    # raise on a numerically singular matrix (e.g. duplicated instruments,
    # rank 1, condition number ~1e16 — inv returns garbage and the per-pair
    # F statistics can still look strong). Scale-aware SVD tolerance, as in
    # numpy.linalg.matrix_rank.
    sv = np.linalg.svd(Sigma_p, compute_uv=False)
    tol = sv.max() * max(Sigma_p.shape) * np.finfo(float).eps if sv.max() > 0 else 0.0
    rank = int(np.sum(sv > tol))
    if rank < m:
        raise RuntimeError(
            f"First-stage Σ_p is numerically rank-deficient "
            f"(rank {rank} < m={m}; singular values {sv.tolist()}). "
            f"Multi-instrument identification fails — the instruments span "
            f"only a {rank}-dimensional policy-shock subspace. Each "
            f"instrument must move a distinct policy direction; per-pair F "
            f"statistics cannot substitute for this rank requirement."
        )
    try:
        Sigma_p_inv = np.linalg.inv(Sigma_p)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(
            f"First-stage Σ_p is singular (det={np.linalg.det(Sigma_p):.2e}). "
            f"Multi-instrument identification fails — each instrument must "
            f"move its corresponding policy variable, and the m×m matrix Σ_p "
            f"(residual × instrument covariance) must be invertible."
        ) from err

    impact = np.zeros((n, m))
    impact[policy_idxs] = np.eye(m)
    other_idxs = [i for i in range(n) if i not in policy_idxs]
    if other_idxs:
        Sigma_q = Sigma_uZ[other_idxs, :]             # (n-m, m)
        impact[other_idxs] = Sigma_q @ Sigma_p_inv    # H_1^q = Σ_q Σ_p^{-1}

    if normalization == "one_sd":
        std_p = np.std(residuals[:, policy_idxs], axis=0, ddof=1)   # (m,)
        impact = impact * std_p[None, :]              # scale each shock column

    return impact


def _proxy_first_stage_multi(
    residuals_policy: np.ndarray,
    instrument: np.ndarray,
    policy_idxs: list[int],
    observed: np.ndarray | None = None,
) -> dict:
    """First-stage diagnostics for multi-instrument case.

    Reports per (instrument j, policy i) pair: γ_{ij}, R²_{ij}, F_{ij}.
    Plus min own-pair F (j == i, the "this instrument moves THIS policy
    variable" check), and the condition number of Σ_p. When ``observed``
    (a ``(T_eff, m)`` boolean mask) is given, each pair's regression runs
    on instrument j's observed rows only — zero-padded unobserved periods
    would dilute F toward zero.
    """
    T_eff, m = residuals_policy.shape
    if observed is None:
        observed = np.ones((T_eff, m), dtype=bool)
    per_pair: list[dict] = []
    for j in range(m):
        mask_j = observed[:, j]
        n_j = int(mask_j.sum())
        z_j = instrument[mask_j, j]
        Z = np.column_stack([np.ones(n_j), z_j])
        for i in range(m):
            u_i = residuals_policy[mask_j, i]
            beta, *_ = np.linalg.lstsq(Z, u_i, rcond=None)
            fitted = Z @ beta
            resid = u_i - fitted
            sse = float((resid ** 2).sum())
            sst = float(((u_i - u_i.mean()) ** 2).sum())
            r2 = 1.0 - sse / sst if sst > 0 else 0.0
            f_stat = (n_j - 2) * r2 / max(1.0 - r2, 1e-12)
            per_pair.append({
                "instrument_idx": int(j),
                "policy_idx": int(policy_idxs[i]),
                "policy_position": int(i),
                "gamma": float(beta[1]),
                "R2": float(r2),
                "F": float(f_stat),
                "n_obs": n_j,
            })

    own_F = [p["F"] for p in per_pair if p["instrument_idx"] == p["policy_position"]]
    Sigma_p = instrument.T @ residuals_policy / T_eff   # (m, m)
    cond = float(np.linalg.cond(Sigma_p))
    return {
        "per_pair": per_pair,
        "n_obs": int(T_eff),
        "weak_instrument_min_own_F": float(min(own_F)) if own_F else None,
        "weak_instrument_F_lt_10": bool(min(own_F) < 10) if own_F else None,
        "Sigma_p_cond": cond,
    }


def _proxy_wild_bootstrap_multi(
    fit: VARFit, z_aligned: np.ndarray, policy_idxs: list[int],
    horizon: int, n_draws: int, seed: int | None, normalization: str,
) -> np.ndarray:
    """Wild bootstrap for multi-instrument proxy-SVAR.

    Same Rademacher-weight construction as the single-instrument bootstrap;
    weights multiply both residuals and ALL instrument columns by the same
    sign per period (preserves cross-instrument structure).

    Returns ``(n_kept, H+1, n_vars, m)``.
    """
    rng = np.random.default_rng(seed)
    T_eff = fit.t_eff
    n = fit.n_vars
    p = fit.p
    k = fit.k
    m = z_aligned.shape[1]
    has_exog = fit.has_exog
    exog_arr = fit.exog.values if has_exog else None
    exog_lags = fit.exog_lags if has_exog else 0
    n_exog_cols = (fit.n_exog * (exog_lags + 1)) if has_exog else 0
    include_const = (k - n * p - n_exog_cols) >= 1
    include_trend = (k - n * p - n_exog_cols) >= 2

    Y_full = fit.data.values
    initial = Y_full[:p].copy()
    resid = fit.residuals
    B_hat = fit.B_hat

    draws: list[np.ndarray] = []
    for _ in range(n_draws):
        omega = rng.choice([-1.0, 1.0], size=T_eff)
        u_star = resid * omega[:, None]
        z_star = z_aligned * omega[:, None]   # (T_eff, m)

        Y_star = np.zeros_like(Y_full, dtype=float)  # float even for integer input — otherwise every simulated row is truncated
        Y_star[:p] = initial
        for t in range(p, p + T_eff):
            x_arr = np.asarray(_build_x_row(
                Y_star, t, p, include_const, include_trend, exog_arr, exog_lags,
            ))
            Y_star[t] = x_arr @ B_hat + u_star[t - p]

        Y_act_b = Y_star[p:]
        X_rows = [
            _build_x_row(Y_star, t, p, include_const, include_trend, exog_arr, exog_lags)
            for t in range(p, p + T_eff)
        ]
        X_act_b = np.asarray(X_rows)
        try:
            B_b = np.linalg.solve(X_act_b.T @ X_act_b, X_act_b.T @ Y_act_b)
        except np.linalg.LinAlgError:
            continue
        resid_b = Y_act_b - X_act_b @ B_b

        try:
            impact_b = _proxy_impact_matrix(resid_b, z_star, policy_idxs, normalization)
        except RuntimeError:
            continue

        lag_mats = extract_lag_matrices(B_b, n, p)
        if max_eigenvalue_modulus(lag_mats) >= 1.01:
            continue
        Phi_b = ma_representation(lag_mats, horizon)
        irf_b = np.einsum("hij,jk->hik", Phi_b, impact_b)   # (H+1, n, m)
        draws.append(irf_b)

    if not draws:
        raise RuntimeError("no successful wild-bootstrap replications")
    return np.stack(draws, axis=0)   # (n_kept, H+1, n, m)


# ---------------------------------------------------------------------------
# Long-run / Blanchard-Quah identification
# ---------------------------------------------------------------------------


def long_run_irf(
    fit: VARFit,
    horizon: int,
    shock_names: Sequence[str] | None = None,
    bands: str = "parameter_draws",
    n_draws: int = 2000,
    seed: int | None = None,
    sign_normalization: dict[str, dict[str, int]] | None = None,
) -> IRFContainer:
    """Long-run (Blanchard-Quah) identification.

    Imposes a lower-triangular pattern on the LONG-RUN (cumulative) impact
    matrix: shock ``j`` has no long-run effect on variables ``1, ..., j-1``.
    In the canonical two-variable Blanchard-Quah (1989) setup with
    ``[Δlog GDP, unemployment rate]``, this restricts the second shock
    (demand) to have zero long-run effect on output.

    Algorithm
    ---------
    1. Long-run multiplier ``C(1) = (I − Σ_j A_j)^{-1}`` exists ONLY if
       the VAR is stationary — the function checks the companion
       spectral radius and raises for a unit or explosive root (a
       nonsingular ``I − ΣA`` alone does not make the cumulative
       response converge).
    2. Long-run reduced-form covariance ``Ω = C(1) Σ_u C(1)'``.
    3. Cholesky decompose ``Ω = L L'`` with lower-triangular ``L``.
       Then the long-run effect of shock ``j`` on variable ``i`` is
       ``L[i, j]``, which is zero for ``j > i`` by lower-triangularity.
    4. Recover the short-run structural impact matrix ``H = C(1)^{-1} L``.
       By construction ``H H' = Σ_u`` (a valid orthogonalisation).
    5. IRF: ``Φ_h H``.

    Parameters
    ----------
    fit : VARFit
        Reduced-form VAR. For Blanchard-Quah, first variable should be
        in growth rates (e.g., Δlog real GDP) so the long-run multiplier
        on the level is meaningful.
    horizon : int
        IRF horizon.
    shock_names : sequence of str, optional
        Names for the structural shocks, in the lower-triangular order.
        For BQ: ``["supply", "demand"]``. Defaults to
        ``[f"shock_{i+1}" for i in range(n_vars)]``.
    bands : {"none", "parameter_draws"}
        Inference method.
    n_draws : int
    seed : int, optional
    sign_normalization : dict, optional
        ``{shock_name: {variable_name: +1 | -1}}`` to flip shock columns
        if needed. Cholesky-based identification produces positive-diagonal
        ``L`` by convention; for BQ this often means supply shock raises
        output but the demand shock's contemporaneous effect direction
        is sample-dependent. Specify e.g.
        ``{"demand": {"output_growth": +1}}`` to flip the demand column if
        its contemporaneous effect on output growth is negative.

    Returns
    -------
    IRFContainer with shape ``(H+1, n_vars, n_vars)``.

    Notes
    -----
    **Inference caveat — long-run fragility (Faust-Leeper 1997, JBES).**
    ``C(1) = (I - ΣA_j)^{-1}`` is a frequency-zero quantity estimated
    with vanishing precision in finite samples; with near-unit-root
    dynamics, small low-frequency specification changes can move
    BQ-style conclusions materially. Treat the permanent/transitory
    split as conditional on the maintained low-frequency specification.
    Additionally, parameter-draw bands EXCLUDE nonstationary draws and
    draws whose ``(I - ΣA)`` is numerically singular (their cumulative
    responses do not converge); retained/rejected counts are reported in
    ``metadata`` and a warning fires when more than 20% are rejected.
    The bands therefore condition on stationarity and are, if anything,
    too narrow near the unit root. See CONVENTIONS.md §4 "Inference
    caveats".
    """
    n = fit.n_vars
    if shock_names is None:
        shock_names = [f"shock_{i + 1}" for i in range(n)]
    if len(shock_names) != n:
        raise ValueError(f"shock_names has {len(shock_names)} entries, expected {n}")

    H_point = _long_run_structural_matrix(fit.lag_matrices(), fit.Sigma_u)
    if sign_normalization:
        H_point = _apply_sign_normalization(
            H_point, fit.variable_names, list(shock_names), sign_normalization,
        )
    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,jk->hik", Phi, H_point)

    metadata = {
        "method": "VAR / long-run (Blanchard-Quah)",
        "identification": "long_run_restrictions",
        "shock_order": list(shock_names),
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
    }

    if bands == "none":
        return IRFContainer.from_point(
            point=irf_point,
            variable_names=list(fit.variable_names),
            shock_names=list(shock_names),
            metadata=metadata,
        )

    if bands not in ("parameter_draws", "asymptotic_mc", "posterior"):
        raise ValueError(f"unknown bands method: {bands!r}")

    # Apply long-run identification per parameter draw. Draws that are
    # nonstationary (or otherwise fail the long-run construction) are
    # EXCLUDED — their cumulative responses do not converge — and the
    # retained/rejected counts are reported in the metadata.
    params = fit.parameter_draws(n_draws=n_draws, seed=seed)
    draws_list = []
    n_rejected = 0
    for B_d, Sigma_d in params:
        lag_mats_d = extract_lag_matrices(B_d, n, fit.p)
        try:
            H_d = _long_run_structural_matrix(lag_mats_d, Sigma_d)
        except (np.linalg.LinAlgError, ValueError):
            n_rejected += 1
            continue
        if sign_normalization:
            H_d = _apply_sign_normalization(
                H_d, fit.variable_names, list(shock_names), sign_normalization,
            )
        Phi_d = ma_representation(lag_mats_d, horizon)
        draws_list.append(np.einsum("hij,jk->hik", Phi_d, H_d))

    if not draws_list:
        raise RuntimeError("no usable parameter draws for long-run identification")
    metadata["n_draws_retained"] = len(draws_list)
    metadata["n_draws_rejected_nonstationary_or_singular"] = n_rejected
    if n_rejected > 0.2 * len(params):
        import warnings
        warnings.warn(
            f"long-run identification rejected {n_rejected} of {len(params)} "
            f"parameter draws (nonstationary or singular long-run matrix) — "
            f"the VAR is close to the unit circle and the bands condition on "
            f"stationarity.", UserWarning, stacklevel=2)
    draws = np.stack(draws_list, axis=0)

    return IRFContainer.from_draws(
        draws=draws,
        variable_names=list(fit.variable_names),
        shock_names=list(shock_names),
        metadata=metadata,
    )


def _long_run_structural_matrix(
    lag_matrices: list[np.ndarray], Sigma_u: np.ndarray,
) -> np.ndarray:
    """Compute the structural impact matrix H from BQ-style long-run restrictions.

    ``H = C(1)^{-1} L`` where ``C(1) = (I - sum A_j)^{-1}`` and
    ``L = chol(C(1) Sigma_u C(1)')``.
    """
    n = lag_matrices[0].shape[0]
    # Stationarity is REQUIRED, not just invertibility of (I - sum A_j):
    # C(1) = sum of the MA coefficients exists only when the companion
    # spectral radius is below one. An explosive VAR can still have a
    # nonsingular (I - sum A_j), and its formal "long-run effect" describes
    # a divergent cumulative response.
    radius = max_eigenvalue_modulus(lag_matrices)
    if radius >= 1.0:
        raise ValueError(
            f"long-run identification requires a stationary VAR: companion "
            f"spectral radius is {radius:.4f} >= 1, so the cumulative "
            f"impulse response does not converge and C(1) does not exist."
        )
    A_sum = sum(lag_matrices)
    I_minus_A = np.eye(n) - A_sum
    eig = np.linalg.eigvals(I_minus_A)
    if np.min(np.abs(eig)) < 1e-12:
        raise ValueError(
            "(I - sum A_j) is singular — VAR has a unit root in the wrong "
            "direction for long-run identification."
        )
    C1 = np.linalg.inv(I_minus_A)
    Omega = C1 @ Sigma_u @ C1.T
    Omega = 0.5 * (Omega + Omega.T)   # symmetrize against floating-point asymmetry
    L = np.linalg.cholesky(Omega)
    H = np.linalg.solve(C1, L)
    return H


def _apply_sign_normalization(
    H: np.ndarray, variable_names: list[str], shock_names: list[str],
    sign_norm: dict[str, dict[str, int]],
) -> np.ndarray:
    """Flip columns of H to enforce sign normalisations.

    For each ``shock_name`` in the dict, for each ``variable: sign`` entry,
    check that ``H[var_idx, shock_idx]`` has the requested sign — if not,
    multiply the column by ``-1``.
    """
    H_norm = H.copy()
    for shock, var_signs in sign_norm.items():
        if shock not in shock_names:
            raise ValueError(f"unknown shock in sign_normalization: {shock!r}")
        j = shock_names.index(shock)
        for var, sign in var_signs.items():
            if var not in variable_names:
                raise ValueError(f"unknown variable in sign_normalization: {var!r}")
            i = variable_names.index(var)
            wanted = int(np.sign(sign))
            actual = int(np.sign(H_norm[i, j])) or 1
            if wanted != 0 and wanted != actual:
                H_norm[:, j] = -H_norm[:, j]
            break   # one sign rule per shock is enough
    return H_norm


# ---------------------------------------------------------------------------
# Heteroskedasticity-based identification (Rigobon 2003)
# ---------------------------------------------------------------------------


def heteroskedasticity_irf(
    fit: VARFit,
    regime: np.ndarray | pd.Series,
    horizon: int,
    shock_names: Sequence[str] | None = None,
    sign_normalization: dict[str, dict[str, int]] | None = None,
    bands: str = "none",
    n_draws: int = 2000,
    seed: int | None = None,
) -> tuple[IRFContainer, dict]:
    """Rigobon (2003) heteroskedasticity-based identification.

    Identifies the structural matrix ``H`` (where ``u_t = H ε_t`` with
    ``E[ε_t ε_t'] = Ω_r`` differing across regimes) by exploiting the
    variance regime shift. Two regimes give a just-identified system for
    arbitrary ``n``: solve ``H Ω_1 H' = Σ_1`` and ``H Ω_2 H' = Σ_2``
    simultaneously via the generalised eigenvalue decomposition
    ``(Σ_1, Σ_2)``.

    Parameters
    ----------
    fit : VARFit
        Reduced-form VAR fit. Residuals are split by ``regime``.
    regime : array-like of int or bool
        Aligned with ``fit.data`` (length == ``len(fit.data)``). Treated as
        regime labels; the first ``fit.p`` entries are discarded for
        alignment with residuals. Must contain exactly two distinct values.
    horizon : int
        IRF horizon.
    shock_names : sequence of str, optional
        Names for the n structural shocks. Defaults to ``["shock_1", ...]``.
    sign_normalization : dict, optional
        ``{shock_name: {variable_name: +1 | -1}}`` to flip columns of ``H``.
        Recommended — heteroskedasticity-ID columns come back in arbitrary
        order/sign from the eigendecomposition.
    bands : {"none", "parameter_draws"}
        Inference method.
    n_draws : int
    seed : int, optional

    Returns
    -------
    IRFContainer of shape ``(H+1, n_vars, n_vars)``.
    dict
        Identification diagnostics: regime variance ratios, eigenvalue
        spread, and ``min_pairwise_log_separation`` — the minimum
        pairwise separation of the variance ratios on the LOG scale,
        which drives the ``weak_identification`` flag (shocks whose
        ratios nearly coincide are identified only as a joint
        subspace).

    Notes
    -----
    The generalised eigenvalue approach requires distinct regime variance
    ratios for each shock — if two shocks have the same variance ratio
    across regimes, that subspace is not identified (analogous to weak
    instruments). The function checks the eigenvalue spread and flags it
    in the diagnostics.

    **Normalisation.** The eigendecomposition fixes the shock covariance
    to the IDENTITY in the second-sorted regime label (``unique[1]`` —
    e.g. recessions when the regime dummy is 0/1): all reported IRFs are
    responses to one-standard-deviation-in-regime-2 shocks. Column order
    follows scipy's ascending generalized eigenvalues, which is
    economically arbitrary — label shocks by inspection and pin signs
    via ``sign_normalization``.

    **Inference caveat — identification uncertainty excluded.** The
    parameter-draw bands hold the structural rotation ``H`` fixed at its
    point estimate; only the VAR coefficients vary across draws. Sampling
    error in the regime covariances (Σ̂₁, Σ̂₂) — typically the dominant
    uncertainty in Rigobon applications — is NOT reflected, so the bands
    understate total uncertainty, and the weak-identification flag is a
    heuristic — the minimum pairwise LOG-scale separation of the
    variance ratios (invariant to swapping the regime labels, which an
    absolute eigen-gap is not) — not a formal test
    (Lütkepohl-Netšunajev 2017; Lewis 2021). See CONVENTIONS.md §4
    "Inference caveats".
    """
    regime = np.asarray(regime).ravel()
    if len(regime) != len(fit.data):
        raise ValueError(
            f"regime length ({len(regime)}) != data length ({len(fit.data)})"
        )
    regime_aligned = regime[fit.p:]    # drop the first p obs for residual alignment
    if len(regime_aligned) != fit.t_eff:
        raise RuntimeError("internal alignment error")

    unique = np.unique(regime_aligned)
    if len(unique) != 2:
        raise ValueError(
            f"regime must take exactly 2 distinct values; got {list(unique)}"
        )

    mask_1 = regime_aligned == unique[0]
    mask_2 = regime_aligned == unique[1]
    if mask_1.sum() < fit.n_vars + 1 or mask_2.sum() < fit.n_vars + 1:
        raise ValueError(
            "each regime must have at least n_vars + 1 observations"
        )

    resid = np.asarray(fit.residuals)
    Sigma_1 = np.cov(resid[mask_1].T, ddof=1)
    Sigma_2 = np.cov(resid[mask_2].T, ddof=1)
    Sigma_1 = 0.5 * (Sigma_1 + Sigma_1.T)
    Sigma_2 = 0.5 * (Sigma_2 + Sigma_2.T)

    H_point, eigvals = _rigobon_structural_matrix(Sigma_1, Sigma_2)
    n = fit.n_vars
    if shock_names is None:
        shock_names = [f"shock_{i + 1}" for i in range(n)]
    if len(shock_names) != n:
        raise ValueError(f"shock_names has {len(shock_names)} entries, expected {n}")

    if sign_normalization:
        H_point = _apply_sign_normalization(
            H_point, fit.variable_names, list(shock_names), sign_normalization,
        )

    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,jk->hik", Phi, H_point)

    eig_spread = float(np.max(eigvals) / max(np.min(eigvals), 1e-12))
    # Separation diagnostic on the LOG scale: |log λ_i − log λ_j| is
    # invariant to swapping the two regime labels (λ → 1/λ), which an
    # absolute eigenvalue gap is not. Shocks whose variance ratios nearly
    # coincide are identified only as a joint subspace — the max/min
    # spread cannot detect that, so it stays descriptive only. This
    # threshold is an explicit heuristic, not a formal weak-ID test.
    log_eigs = np.log(np.maximum(np.sort(eigvals), 1e-300))
    min_log_sep = float(np.min(np.diff(log_eigs))) if len(eigvals) > 1 else np.inf
    diag = {
        "regime_labels": [int(x) for x in unique],
        "n_obs_regime_1": int(mask_1.sum()),
        "n_obs_regime_2": int(mask_2.sum()),
        "eigenvalues_omega1_over_omega2": eigvals.tolist(),
        "eig_spread": eig_spread,
        "min_pairwise_log_separation": min_log_sep,
        # Heuristic flag (label-invariant): adjacent variance ratios within
        # ~10% of each other on the log scale mean the corresponding shocks
        # are close to unidentified individually (only their joint subspace
        # is pinned down).
        "weak_identification": bool(min_log_sep < 0.1),
    }

    metadata = {
        "method": "VAR / heteroskedasticity (Rigobon 2003)",
        "identification": "heteroskedasticity",
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
        "regime_diagnostics": diag,
    }

    if bands == "none":
        return (
            IRFContainer.from_point(
                point=irf_point,
                variable_names=list(fit.variable_names),
                shock_names=list(shock_names),
                metadata=metadata,
            ),
            diag,
        )

    if bands not in ("parameter_draws", "asymptotic_mc", "posterior"):
        raise ValueError(f"unknown bands method: {bands!r}")

    # For bands: sample reduced-form parameters, re-identify per draw.
    # Sigma_1, Sigma_2 are NOT re-sampled here — they come from the regime
    # split of the actual residuals. For Bayesian sampling of the regime
    # variances themselves, see Lewis (2021).
    params = fit.parameter_draws(n_draws=n_draws, seed=seed)
    draws_list = []
    for B_d, _Sigma_d in params:
        lag_mats_d = extract_lag_matrices(B_d, n, fit.p)
        Phi_d = ma_representation(lag_mats_d, horizon)
        H_d = H_point   # identification matrix held fixed across draws
        if sign_normalization:
            H_d = _apply_sign_normalization(
                H_point.copy(), fit.variable_names,
                list(shock_names), sign_normalization,
            )
        draws_list.append(np.einsum("hij,jk->hik", Phi_d, H_d))
    draws = np.stack(draws_list, axis=0)
    return (
        IRFContainer.from_draws(
            draws=draws,
            variable_names=list(fit.variable_names),
            shock_names=list(shock_names),
            metadata=metadata,
        ),
        diag,
    )


def _rigobon_structural_matrix(
    Sigma_1: np.ndarray, Sigma_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalised eigenvalue problem ``Σ_1 v = λ Σ_2 v``.

    Decomposes ``Σ_r = H Ω_r H'`` where ``H`` is the same across regimes
    and ``Ω_r`` are diagonal. The eigenvalues ``λ_k = Ω_1[k,k] / Ω_2[k,k]``
    are the variance ratios; the columns of ``H`` are recovered from the
    eigenvectors normalised so that ``H Ω_2 H' = Σ_2`` exactly.

    Returns ``(H, eigvals)``.
    """
    from scipy.linalg import eigh
    # Generalised eigenproblem: Σ_1 V = Σ_2 V diag(λ)
    eigvals, V = eigh(Sigma_1, Sigma_2)   # V is Σ_2-orthonormal
    # V satisfies V' Σ_2 V = I, V' Σ_1 V = diag(λ)
    # Want H, Ω_1, Ω_2 diagonal s.t. H Ω_1 H' = Σ_1, H Ω_2 H' = Σ_2.
    # Equivalent: H^{-1} Σ_r H^{-T} = Ω_r.
    # H^{-1} simultaneously diagonalises both Σ_r. From V' Σ_2 V = I,
    # V' Σ_1 V = diag(λ): set H^{-1} = V', so H = V^{-T}.
    # Then Ω_2 = V' Σ_2 V = I, Ω_1 = V' Σ_1 V = diag(λ).
    H = np.linalg.inv(V.T)
    return H, eigvals


# ---------------------------------------------------------------------------
# Zero + sign restrictions (Arias-Rubio-Ramirez-Waggoner 2018)
# ---------------------------------------------------------------------------


# =============================================================================
# EXPERIMENTAL
# The function below is less established than the core VAR code.
# It rests on the library's own tests on simulated data. Treat results with
# caution and confirm important ones another way before relying on them.
# See README, "Validation status and intended use".
# =============================================================================
def zero_sign_restricted_irf(
    fit: VARFit,
    horizon: int,
    sign_restrictions: dict[str, dict[str, int]],
    zero_restrictions: dict[str, list[dict]] | None = None,
    restriction_horizons: int = 1,
    target_accepted: int = 1000,
    max_rotations: int = 100_000,
    seed: int | None = None,
    parameter_uncertainty: bool = True,
    n_param_draws: int | None = None,
) -> IRFContainer:
    """Zero + sign restricted IRFs (Arias-Rubio-Ramirez-Waggoner 2018).

    Extends :func:`sign_restricted_irf` with the ability to impose zero
    restrictions on the structural impulse responses at specific horizons
    in addition to sign restrictions. ARRW 2018's Algorithm 2 generates
    the orthogonal rotation Q one column at a time: each column is drawn
    from the null space of the zero-restriction matrix for that shock,
    then orthogonalised against previously-placed columns.

    .. note::

       **Current scope**: zero restrictions are restricted to a SINGLE
       horizon per (shock, variable) entry — typically the impact period
       (``"horizon": 0``). The ARRW 2018 algorithm fully supports
       multi-horizon zero restrictions; extending requires building the
       horizon-stacked Z·MA matrix per shock, which adds complexity but
       no new conceptual machinery; it is not implemented here.

    .. note::

       **Order of shocks with the same number of zeros.** Columns are
       drawn one after another, each conditional on those already
       placed, and the importance weights of the original algorithm are
       not applied (CONVENTIONS.md §4). Shocks that carry the same number
       of zero restrictions are drawn in the order in which they appear
       in ``sign_restrictions``, and that order changes the distribution
       of the reported responses. Report the order used and check that
       the conclusions hold when it is reversed.

       **Zero and sign on the same variable.** A variable may carry a
       zero on impact and a sign restriction for the same shock (zero on
       impact, signed afterwards). The impact response is then exactly
       zero and the sign applies from horizon 1, so it has an effect
       only when ``restriction_horizons`` is larger than 1.

    Parameters
    ----------
    fit : VARFit
        Reduced-form VAR fit.
    horizon : int
        IRF horizon for the output container.
    sign_restrictions : dict
        Same format as :func:`sign_restricted_irf` —
        ``{shock_name: {variable_name: sign}}`` with ``sign ∈ {-1, 0, +1}``.
    zero_restrictions : dict, optional
        Per-shock zero restrictions:
        ``{shock_name: [{"variable": <name>, "horizon": <int>}, ...]}``.
        Each entry says "shock ``shock_name`` has zero response on variable
        ``variable`` at horizon ``horizon``". ``horizon`` must currently
        equal 0 (impact period) — see the scope note above.
        Pass ``None`` (or an empty dict) to recover pure sign-restriction
        behaviour — byte-identical to :func:`sign_restricted_irf`.
    restriction_horizons, target_accepted, max_rotations, seed,
    parameter_uncertainty, n_param_draws : as in :func:`sign_restricted_irf`.

    Returns
    -------
    IRFContainer
        ``shock_names`` matches the keys of ``sign_restrictions``.
        ``metadata["identification"]`` is ``"zero_sign_restrictions"``.

    Raises
    ------
    ValueError
        If a zero restriction has ``horizon != 0`` (current scope limit);
        if a shock referenced in ``zero_restrictions`` is not in
        ``sign_restrictions`` (every zero-restricted shock must also be
        identified); or — BEFORE any sampling — if the zero pattern is
        infeasible under the ARRW rank condition (with shocks ordered by
        decreasing zero count, position ``k`` of ``n`` admits at most
        ``n - k`` zero restrictions).
    RuntimeError
        If no rotation satisfies the joint restrictions after
        ``max_rotations`` attempts — with the rank condition checked up
        front, this signals sign restrictions too tight for the data
        (or ``max_rotations`` too low), not an infeasible zero pattern.

    Notes
    -----
    **Algorithm** (ARRW 2018 §3, Algorithm 2 — simplified for h=0 zeros):

    For each candidate rotation:

    1. For shock j = 1..n_shocks — processed MOST-ZERO-RESTRICTED FIRST
       (the ARRW 2018 Theorem 3-4 ordering; a feasible model could
       otherwise be rejected with probability one just because an
       unrestricted shock was listed first). Columns are placed back at
       the caller's requested positions, so the output ordering is
       unchanged:
       a. Let ``Z_j`` be the rows of ``P = chol(Σ_u)`` corresponding to
          zero-restricted variables for shock j (so the zero condition
          ``(P Q[:, j])[v] = 0`` becomes ``(Z_j) Q[:, j] = 0`` —
          i.e., Q[:, j] must lie in the null space of Z_j).
       b. The null-space basis additionally requires orthogonality with
          previously-placed columns. Stack the (unit-normalised) zero
          rows with the placed columns' transposes and find the null
          space via SVD with a RELATIVE rank tolerance — raw rows carry
          the variable's units, and an absolute threshold would silently
          drop a small-unit variable's restriction. Accepted draws are
          re-verified against the original zeros in scale-adjusted units.
       c. Draw a random vector ξ_j ~ N(0, I_d) where d = n − rank of the
          stacked matrix.
       d. Q[:, j] = null_basis @ ξ_j (lies in the joint null space).
       e. Normalise to unit norm.
    2. Check sign restrictions on the resulting ``A_0 = P Q``.
    3. If signs pass, accept; otherwise reject and re-draw.

    **Identification tag**: ``"zero_sign_restrictions"``.

    **Inference caveat — unweighted Algorithm-2 draws.** ARRW (2018, §5,
    Algorithm 3) show that drawing Q *conditional on the zero
    restrictions* (as Algorithm 2 does) induces a non-uniform density
    over the structural identified set; inference under the
    conditionally-agnostic prior requires importance weights built from
    the volume elements of the zero-restriction mapping. Those weights
    are NOT applied here — the cross-draw quantiles summarise the
    conditional draw measure, not ARRW's recommended posterior. The
    Fry-Pagan and Baumeister-Hamilton caveats on
    :func:`sign_restricted_irf` apply here too. See CONVENTIONS.md §4
    "Inference caveats".

    References
    ----------
    Arias, J. E., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018).
    "Inference Based on Structural Vector Autoregressions Identified
    With Sign and Zero Restrictions: Theory and Applications."
    Econometrica, 86(2), 685-720.
    """
    rng = np.random.default_rng(seed)
    n = fit.n_vars
    shock_names = list(sign_restrictions.keys())

    # Resolve zero restrictions: per-shock list of (variable_idx, horizon) tuples.
    zero_per_shock: dict[str, list[tuple[int, int]]] = {s: [] for s in shock_names}
    if zero_restrictions:
        for shock_name, entries in zero_restrictions.items():
            if shock_name not in sign_restrictions:
                raise ValueError(
                    f"zero_restrictions references shock {shock_name!r} which "
                    f"is not in sign_restrictions. Every zero-restricted shock "
                    f"must also be identified by sign restrictions (an "
                    f"un-identified zero-restricted column is unrecoverable)."
                )
            for entry in entries:
                if "variable" not in entry or "horizon" not in entry:
                    raise ValueError(
                        f"zero_restrictions entry must have 'variable' and "
                        f"'horizon' keys; got {entry!r}"
                    )
                v_name = entry["variable"]
                if v_name not in fit.variable_names:
                    raise ValueError(
                        f"unknown variable {v_name!r} in zero restriction"
                    )
                h_val = int(entry["horizon"])
                if h_val != 0:
                    raise ValueError(
                        f"zero_restrictions currently support only "
                        f"horizon=0 (impact); got horizon={h_val}. "
                        f"Multi-horizon ARRW 2018 zero restrictions are not "
                        f"implemented; see the zero_sign_restricted_irf docstring."
                    )
                v_idx = fit.variable_names.index(v_name)
                zero_per_shock[shock_name].append((v_idx, h_val))

    # If no zero restrictions, delegate to plain sign_restricted_irf — keeps
    # the behaviour byte-identical for pure sign-only calls.
    if not any(zero_per_shock.values()):
        return sign_restricted_irf(
            fit=fit, horizon=horizon, sign_restrictions=sign_restrictions,
            restriction_horizons=restriction_horizons,
            target_accepted=target_accepted, max_rotations=max_rotations,
            seed=seed, parameter_uncertainty=parameter_uncertainty,
            n_param_draws=n_param_draws,
        )

    # Build sign arrays per shock for matching
    sign_arrays: dict[str, np.ndarray] = {}
    for shock_name, var_signs in sign_restrictions.items():
        sa = np.zeros(n, dtype=int)
        for v, s in var_signs.items():
            if v not in fit.variable_names:
                raise ValueError(f"unknown variable {v!r} in sign restriction")
            sa[fit.variable_names.index(v)] = int(np.sign(s))
        sign_arrays[shock_name] = sa

    if parameter_uncertainty:
        n_param_draws = n_param_draws or max_rotations
        param_draws = fit.parameter_draws(n_draws=n_param_draws, seed=seed)
    else:
        param_draws = None

    accepted: list[np.ndarray] = []
    n_attempted = 0
    n_rejected_signs = 0

    # Process shocks MOST-ZERO-RESTRICTED FIRST (ARRW 2018, Theorems 3-4):
    # drawing an unrestricted column before a restricted one can make the
    # restricted column a probability-zero event even when the joint
    # specification is feasible — more rotations cannot fix that. Columns
    # are placed back at the caller's requested positions, so the output
    # ordering is unchanged.
    proc_order = sorted(
        range(len(shock_names)),
        key=lambda j: -len(zero_per_shock[shock_names[j]]),
    )
    # Pre-sampling rank condition: shock k in processing order faces its
    # z_k zero rows plus k orthogonality rows; a null space exists
    # generically iff z_k <= n - 1 - k.
    for k, j in enumerate(proc_order):
        z_k = len(zero_per_shock[shock_names[j]])
        if z_k > n - 1 - k:
            raise ValueError(
                f"zero restrictions are infeasible under the ARRW rank "
                f"condition: shock {shock_names[j]!r} carries {z_k} zero "
                f"restriction(s) but, with the shocks ordered by decreasing "
                f"zero count, position {k + 1} of {n} admits at most "
                f"{n - 1 - k}. Reduce the zero restrictions or the number "
                f"of jointly identified shocks."
            )

    while len(accepted) < target_accepted and n_attempted < max_rotations:
        n_attempted += 1
        if param_draws is not None:
            B_d, Sigma_d = param_draws[n_attempted % len(param_draws)]
        else:
            B_d, Sigma_d = fit.B_hat, fit.Sigma_u

        try:
            P = np.linalg.cholesky(Sigma_d)
        except np.linalg.LinAlgError:
            continue

        # Build Q one column at a time, in processing order
        Q = np.zeros((n, n))
        placed: list[int] = []   # column positions already placed
        bad_draw = False
        for j in proc_order:
            shock_name = shock_names[j]
            zero_entries = zero_per_shock[shock_name]
            # Rows of P for zero-restricted variables, NORMALISED to unit
            # length: the restriction q·P[v] = 0 is scale-free, and raw rows
            # carry the variable's units — a small-unit variable's constraint
            # would otherwise fall below any absolute rank threshold and be
            # silently dropped.
            if zero_entries:
                rows = []
                for (v_idx, _) in zero_entries:
                    r = P[v_idx]
                    r_norm = float(np.linalg.norm(r))
                    if r_norm <= 0.0:
                        bad_draw = True
                        break
                    rows.append(r / r_norm)
                if bad_draw:
                    break
                Z_j = np.stack(rows, axis=0)
            else:
                Z_j = np.zeros((0, n))
            # Stack with transposes of already-placed (unit-norm) columns.
            if placed:
                M = np.vstack([Z_j, Q[:, placed].T])
            else:
                M = Z_j
            # Null-space basis via SVD with a RELATIVE rank tolerance
            # (all rows are unit-norm, so the scale is well defined).
            if M.shape[0] > 0:
                U, s_vals, Vt = np.linalg.svd(M, full_matrices=True)
                tol = s_vals.max() * max(M.shape) * np.finfo(float).eps
                rank = int(np.sum(s_vals > tol))
                if rank >= n:
                    # No null space; can't place this column. Skip draw.
                    bad_draw = True
                    break
                # Last n - rank rows of Vt span the null space
                null_basis = Vt[rank:].T   # (n, d) where d = n - rank
            else:
                null_basis = np.eye(n)

            d = null_basis.shape[1]
            xi = rng.standard_normal(d)
            q_j = null_basis @ xi
            norm = float(np.linalg.norm(q_j))
            if norm < 1e-12:
                bad_draw = True
                break
            q_j = q_j / norm
            Q[:, j] = q_j
            placed.append(j)

        if bad_draw:
            continue

        # Verify the original zero restrictions on the finished draw in
        # scale-adjusted units — numerical slack in the null-space
        # construction must never silently weaken the model.
        zeros_ok = True
        for j, shock_name in enumerate(shock_names):
            for (v_idx, _) in zero_per_shock[shock_name]:
                row = P[v_idx]
                if abs(float(row @ Q[:, j])) > 1e-8 * float(np.linalg.norm(row)):
                    zeros_ok = False
                    break
            if not zeros_ok:
                break
        if not zeros_ok:
            continue

        # Now check sign restrictions on A_0 = P Q
        A_0 = P @ Q
        lag_mats = extract_lag_matrices(B_d, n, fit.p)
        Phi = ma_representation(lag_mats, horizon)
        irf_struct = np.einsum("hij,jk->hik", Phi, A_0)  # (H+1, n, n)

        # A zero-restricted impact cell is zero by construction, but the
        # null-space arithmetic leaves it at rounding level (about 1e-17)
        # with an arbitrary sign. Set it to exactly zero. Otherwise a sign
        # restriction on the same variable (zero on impact, signed at the
        # following horizons) is decided at horizon 0 by the sign of the
        # rounding error, which is correlated with the rotation and
        # distorts the accepted set.
        for j, shock_name in enumerate(shock_names):
            for (v_idx, _) in zero_per_shock[shock_name]:
                irf_struct[0, v_idx, j] = 0.0

        all_signs_ok = True
        flipped_sign = np.ones(n)
        for j, shock_name in enumerate(shock_names):
            sa = sign_arrays[shock_name]
            irf_j = irf_struct[:, :, j]
            # Try +column then -column at horizon 0..restriction_horizons-1
            sign_ok_plus = _matches_signs(irf_j, sa, restriction_horizons)
            sign_ok_minus = _matches_signs(-irf_j, sa, restriction_horizons)
            if sign_ok_plus:
                flipped_sign[j] = +1
            elif sign_ok_minus:
                flipped_sign[j] = -1
            else:
                all_signs_ok = False
                break

        if not all_signs_ok:
            n_rejected_signs += 1
            continue

        # Apply sign flipping to the named-shock columns only
        m = len(shock_names)
        irf_named = irf_struct[:, :, :m] * flipped_sign[None, None, :m]
        accepted.append(irf_named)

    if not accepted:
        raise RuntimeError(
            f"no rotations satisfied zero+sign restrictions after "
            f"{n_attempted} attempts (rejected by signs: {n_rejected_signs}). "
            f"Either the joint restriction set is empty, the sample is "
            f"uninformative, or max_rotations is too low."
        )

    draws = np.stack(accepted, axis=0)  # (n_accepted, H+1, n, n)

    metadata = {
        "method": "VAR / zero + sign restrictions (ARRW 2018)",
        "identification": "zero_sign_restrictions",
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "target_accepted": target_accepted,
        "n_accepted": len(accepted),
        "n_attempted": n_attempted,
        "acceptance_rate": len(accepted) / max(n_attempted, 1),
        "median_target": _median_target_meta(draws),
        "restriction_horizons": restriction_horizons,
        "sign_restrictions": {s: dict(v) for s, v in sign_restrictions.items()},
        "zero_restrictions": (
            {s: list(e) for s, e in (zero_restrictions or {}).items()}
        ),
        "n_rejected_by_signs": int(n_rejected_signs),
        "parameter_uncertainty": parameter_uncertainty,
        "seed": seed,
    }
    return IRFContainer.from_draws(
        draws=draws,
        variable_names=list(fit.variable_names),
        shock_names=shock_names,
        metadata=metadata,
    )
