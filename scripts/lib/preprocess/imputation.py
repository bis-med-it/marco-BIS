"""Missing-value imputation for multivariate time series.

Three methods:

* ``"ffill"`` — last-observation-carried-forward. Pandas one-liner.
* ``"linear"`` — linear interpolation between adjacent observations. Pandas
  one-liner.
* ``"kalman"`` — multivariate Kalman smoother via a VAR(p) cast to
  state-space form. Handles ragged-edge nowcasting (mixed observation
  patterns across series) and gives proper conditional expectations
  ``E[y_t | y_{1:T}]`` rather than naive interpolation. Supports
  optional **EM iteration** (``em_iter > 0``) to remove the bias from
  fitting the VAR on a linearly-pre-filled copy when missingness is heavy.

Scope: stateless. ``df.attrs`` from ``lib.load_dataset`` is preserved and
a ``preprocess_applied`` entry is appended.

EM iteration (``em_iter``): the initial Kalman path fits VAR(p) parameters
once on a linearly pre-filled copy, which biases the parameters when
missingness is heavy. Setting ``em_iter > 0`` alternates between Kalman
smoothing (E-step) and re-estimating ``(A_1, ..., A_p, c, Sigma_u)`` from
the smoother's sufficient statistics (Shumway-Stoffer 1982 M-step). The
log-likelihood is monotone-increasing in expectation; iteration stops
when relative log-likelihood improvement falls below ``em_tol`` or after
``em_iter`` M-steps.

Out of scope:

* Per-column method (some columns ffill, others Kalman). Pick one method
  for the whole DataFrame; for heterogeneous treatment, call ``impute``
  separately on column subsets and concat.
* Non-stationary handling. The VAR(p) state-space assumes the input is
  stationary. For unit-root data, first-difference, impute, and integrate
  back — or pre-detrend.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def impute(
    df: Union[pd.Series, pd.DataFrame],
    method: str = "linear",
    *,
    lags: int = 1,
    fill_leading: bool = False,
    em_iter: int = 0,
    em_tol: float = 1e-4,
) -> Union[pd.Series, pd.DataFrame]:
    """Fill NaN entries in a time series.

    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        Input data with a DatetimeIndex. Series are treated as a
        single-column DataFrame internally (the Kalman path then reduces
        to a univariate AR(p)).
    method : {"ffill", "linear", "kalman"}
        Imputation method.

        * ``"ffill"`` — forward-fill. Leaves leading NaNs as NaN unless
          ``fill_leading=True``.
        * ``"linear"`` — linear interpolation between adjacent observed
          values. Leaves leading / trailing NaNs as NaN unless
          ``fill_leading=True`` (trailing NaNs are not filled — those
          are the ragged-edge case the Kalman method is designed for).
        * ``"kalman"`` — multivariate Kalman smoother. Cast input as a
          VAR(p) → state-space; fit on a linearly pre-filled copy of the
          data; smooth the original (with NaNs); replace NaN entries
          with their smoothed conditional expectations. Handles all
          positions (leading / internal / trailing NaNs).
    lags : int, default 1
        VAR lag order for ``method="kalman"``. Ignored for other methods.
        Use ``lags=4`` for quarterly macro data if you want one year of
        memory; ``lags=1`` is robust to small samples.
    fill_leading : bool, default False
        For ``"ffill"`` / ``"linear"``: backfill leading NaNs from the
        first observed value (and trailing NaNs from the last for linear).
        For ``"kalman"`` this is automatic — Kalman smooths through all
        positions.
    em_iter : int, default 0
        Maximum number of EM M-step updates for ``method="kalman"``. The
        default ``0`` reproduces the single-pass fit-on-pre-filled
        behaviour bit-identically. With ``em_iter > 0`` the routine
        alternates between Kalman smoothing (E-step) and re-estimating
        ``(A_1, ..., A_p, c, Sigma_u)`` from the smoother's sufficient
        statistics (Shumway-Stoffer 1982 M-step). Iteration terminates
        early when the relative log-likelihood gain falls below ``em_tol``.
        Ignored for non-Kalman methods.
    em_tol : float, default 1e-4
        Convergence tolerance for EM: stop when
        ``(ll_new - ll_old) < em_tol * (1 + abs(ll_old))``. Smaller values
        run more iterations. Ignored for non-Kalman methods or when
        ``em_iter == 0``.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as ``df`` with NaN entries replaced (in a copy;
        input is not modified). ``df.attrs`` is preserved and a
        ``preprocess_applied`` entry recording the operation is appended.
        For ``method="kalman"`` with ``em_iter > 0`` the entry also
        records ``em_actual_iter`` (number of M-steps actually performed)
        and ``em_converged`` (bool).

    Raises
    ------
    ValueError
        If ``method`` is not one of the three names; if any column is
        fully NaN (no anchor for imputation); if ``method="kalman"`` is
        requested but fewer than ``lags + n_vars + 1`` complete-row
        observations exist after pre-filling; if ``em_iter < 0``.

    Examples
    --------
    Forward-fill a ragged-edge nowcasting panel::

        >>> filled = lib.preprocess.impute(panel, method="ffill")

    Kalman smoothing for proper conditional-expectation imputation::

        >>> filled = lib.preprocess.impute(panel, method="kalman", lags=4)

    Kalman with EM iteration for heavy missingness::

        >>> filled = lib.preprocess.impute(
        ...     panel, method="kalman", lags=2, em_iter=20,
        ... )
    """
    if method not in ("ffill", "linear", "kalman"):
        raise ValueError(
            f"Unknown method {method!r}; valid: 'ffill', 'linear', 'kalman'"
        )
    if em_iter < 0:
        raise ValueError(f"em_iter must be >= 0, got {em_iter}")

    is_series = isinstance(df, pd.Series)
    if is_series:
        wide = df.to_frame()
    else:
        wide = df.copy()

    if not isinstance(wide.index, pd.DatetimeIndex):
        raise TypeError(
            f"impute requires a DatetimeIndex, got {type(wide.index).__name__}"
        )

    all_nan = wide.columns[wide.isna().all()].tolist()
    if all_nan:
        raise ValueError(
            f"Cannot impute fully-missing columns (no anchor): {all_nan}"
        )

    em_info: dict | None = None

    if method == "ffill":
        result = wide.ffill()
        if fill_leading:
            result = result.bfill()
    elif method == "linear":
        # limit_area="inside" leaves leading/trailing NaNs untouched
        result = wide.interpolate(method="linear", limit_area="inside")
        if fill_leading:
            result = result.bfill().ffill()
    else:  # "kalman"
        result, em_info = _kalman_impute(
            wide, lags=lags, em_iter=em_iter, em_tol=em_tol,
        )

    if df.attrs:
        result.attrs = dict(df.attrs)
        applied = list(result.attrs.get("preprocess_applied", []))
        entry: dict = {
            "op": "impute",
            "method": method,
            "lags": int(lags) if method == "kalman" else None,
            "fill_leading": bool(fill_leading),
        }
        if em_info is not None:
            entry["em_iter"] = int(em_iter)
            entry["em_actual_iter"] = int(em_info["em_actual_iter"])
            entry["em_converged"] = bool(em_info["em_converged"])
        applied.append(entry)
        result.attrs["preprocess_applied"] = applied

    if is_series:
        out = result.iloc[:, 0]
        if df.name is not None:
            out.name = df.name
        return out
    return result


def _kalman_impute(
    wide: pd.DataFrame, lags: int, em_iter: int = 0, em_tol: float = 1e-4,
) -> tuple[pd.DataFrame, dict]:
    """Multivariate Kalman smoother via VAR(p) cast to state-space form.

    Steps:

    1. Pre-fill NaN positions via linear interpolation + edge fill so a
       VAR can be estimated. (Bias caveat: documented in the module
       docstring; ``em_iter > 0`` removes the bias by iterating.)
    2. Fit VAR(p) on the pre-filled copy via ``lib.fit_var``.
    3. Build a companion-form state-space model from the VAR matrices:

           state α_t = [y_t; y_{t-1}; ...; y_{t-p+1}]   (n*p × 1)
           y_t = Z α_t                                  Z = [I_n | 0 | ... | 0]
           α_{t+1} = T α_t + c + R η_t                  T = companion matrix
                                                       R = [I_n; 0; ...]
                                                       Q = Σ_u (VAR residual cov)

    4. Run the Kalman smoother on the ORIGINAL data (with NaNs).
       statsmodels handles missing entries by dropping the corresponding
       rows of the measurement update per time step.
    5. (EM mode only) Re-estimate VAR parameters from smoother's
       sufficient statistics (M-step), rebuild state-space, re-smooth.
       Repeat until log-likelihood gain falls below ``em_tol`` or
       ``em_iter`` M-steps performed.
    6. Extract smoothed state's first ``n_vars`` rows as
       ``E[y_t | y_{1:T}]``; fill ONLY the NaN positions in the output
       (observed values stay exactly as observed).

    Returns
    -------
    (df_filled, em_info) : tuple
        ``df_filled`` is the imputed DataFrame; ``em_info`` is a dict
        with ``em_actual_iter`` (M-steps performed) and ``em_converged``
        (whether tol-based early stop triggered).
    """
    # Local imports to avoid circular dependency at module load
    import lib
    from lib import state_space as ss

    Y_orig = wide.values.astype(float)
    n_obs, n_vars = Y_orig.shape
    if n_obs < lags + n_vars + 1:
        raise ValueError(
            f"Kalman imputation needs T >= lags + n_vars + 1 = "
            f"{lags + n_vars + 1}; got T={n_obs}."
        )

    # Step 1: pre-fill via linear interp + edge fill
    pre_filled = wide.interpolate(method="linear").ffill().bfill()
    if pre_filled.isna().any().any():
        raise ValueError(
            "Pre-fill failed — input has all-NaN columns or rows that "
            "can't be linearly bridged. This should have been caught earlier."
        )
    n_complete = int((~np.isnan(Y_orig).any(axis=1)).sum())
    if n_complete < lags + n_vars + 1 and em_iter == 0:
        # Pre-fill rescued the VAR estimation, but warn that the
        # parameter estimates rely heavily on interpolated rows.
        import warnings
        warnings.warn(
            f"Kalman imputation: only {n_complete} fully-observed rows "
            f"(need {lags + n_vars + 1} for unbiased VAR estimation). "
            f"Parameters are estimated on linearly-interpolated data; "
            f"pass em_iter>0 to remove the pre-fill bias via EM iteration.",
            UserWarning, stacklevel=3,
        )

    # Step 2: fit VAR(p) on pre-filled data (initial parameters)
    fit = lib.fit_var(pre_filled, lags=lags, trend="c")
    # Coerce to ndarray (lib.fit_var may return DataFrame/Series wrappers
    # for some fields; statsmodels' Kalman wants raw ndarrays).
    lag_mats = [np.asarray(A) for A in fit.lag_matrices()]
    Sigma_u = np.asarray(fit.Sigma_u)
    # Intercept lives in the last row of B_hat when trend="c"
    intercept = np.asarray(fit.B_hat)[-1, :]

    # Initial state: first `lags` rows of pre-filled data (companion-stacked)
    init_block = pre_filled.iloc[:lags].values   # (lags, n_vars)
    initial_state = np.concatenate([init_block[lags - 1 - l] for l in range(lags)])

    # EM loop. At em_iter=0 this runs exactly once (no M-step), matching
    # the prior single-pass behaviour bit-identically.
    em_actual_iter = 0
    em_converged = False
    ll_prev: float | None = None
    smoothed = None
    for k in range(em_iter + 1):
        # Build state-space with current (lag_mats, intercept, Sigma_u)
        model = _build_companion_state_space(
            lag_mats, intercept, Sigma_u, initial_state, n_vars, lags,
        )
        # E-step: smooth the ORIGINAL (with NaNs)
        smoothed = ss.kalman_filter_smooth(model, Y_orig)
        ll = smoothed.log_likelihood

        # Convergence check (only valid after at least one M-step)
        if ll_prev is not None:
            improvement = ll - ll_prev
            if improvement < em_tol * (1.0 + abs(ll_prev)):
                em_converged = True
                break
        ll_prev = ll

        # M-step: if we have iterations remaining, re-estimate parameters
        if k < em_iter:
            lag_mats, intercept, Sigma_u = _em_m_step(
                smoothed, n_vars, lags,
            )
            em_actual_iter += 1

    # Step 6: fill ONLY NaN positions; observed values stay exact
    assert smoothed is not None  # loop always runs at least once
    smoothed_y = smoothed.smoothed_state[:, :n_vars]   # E[y_t | y_{1:T}]
    out_arr = Y_orig.copy()
    nan_mask = np.isnan(Y_orig)
    out_arr[nan_mask] = smoothed_y[nan_mask]

    df_out = pd.DataFrame(out_arr, index=wide.index, columns=wide.columns)
    em_info = {
        "em_actual_iter": em_actual_iter,
        "em_converged": em_converged,
    }
    return df_out, em_info


def _build_companion_state_space(
    lag_mats: list[np.ndarray],
    intercept: np.ndarray,
    Sigma_u: np.ndarray,
    initial_state: np.ndarray,
    n_vars: int,
    lags: int,
):
    """Assemble the companion-form state-space model for VAR(p) parameters."""
    from lib import state_space as ss

    n_state = n_vars * lags
    T_mat = np.zeros((n_state, n_state))
    for l, A_l in enumerate(lag_mats):
        T_mat[:n_vars, l * n_vars:(l + 1) * n_vars] = A_l
    if lags > 1:
        T_mat[n_vars:, :-n_vars] = np.eye((lags - 1) * n_vars)
    R = np.zeros((n_state, n_vars))
    R[:n_vars] = np.eye(n_vars)
    Z = np.zeros((n_vars, n_state))
    Z[:, :n_vars] = np.eye(n_vars)
    # Tiny measurement noise for numerical stability (effectively noise-free)
    H = np.eye(n_vars) * 1e-10
    c = np.zeros(n_state)
    c[:n_vars] = intercept

    return ss.StateSpaceModel(
        Z=Z, H=H, T=T_mat, R=R, Q=Sigma_u,
        state_intercept=c,
        initial_state_mean=initial_state,
    )


def _em_m_step(
    smoothed,
    n_vars: int,
    lags: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """M-step of EM iteration — re-estimate VAR(p) coefficients from
    sufficient statistics of the Kalman smoother.

    The state α_t = [y_t; y_{t-1}; ...; y_{t-p+1}] contains the p most
    recent observations as block-rows. The VAR equation
    ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + η_t`` is equivalent to

        ``y_t = [A_1 ... A_p] · α_{t-1} + c + η_t``

    so the regressor is ``α_{t-1}`` (full state at t-1) plus a constant.

    Sufficient statistics (summed over t = 1, ..., T-1 — the T-1
    transitions implied by the VAR):

        S_yy   = Σ E[y_t y_t' | Y]               (n, n)
        S_yx   = Σ E[y_t α_{t-1}' | Y]           (n, n*p)
        S_y    = Σ E[y_t | Y]                    (n,)
        S_xx   = Σ E[α_{t-1} α_{t-1}' | Y]       (n*p, n*p)
        S_x    = Σ E[α_{t-1} | Y]                (n*p,)
        N_eff  = T - 1                          (number of transitions)

    where

        E[α_t α_t' | Y]        = M_t M_t' + V_t                (smoother)
        E[α_t α_{t-1}' | Y]    = M_t M_{t-1}' + C_t            (lag-1 cross)
        M_t = smoothed_state[t]
        V_t = smoothed_cov[t]
        C_t = smoothed_state_autocov[t]   (= Cov(α_t, α_{t-1} | Y))

    Closed-form M-step:

        [A_1 ... A_p | c] = S_yx_aug @ S_xx_aug^{-1}
        Σ_u  = (1/N_eff)  (S_yy - Φ @ S_yx_aug')

    where ``S_yx_aug`` and ``S_xx_aug`` are stacked with the constant row/col.

    Returns
    -------
    (lag_mats_new, intercept_new, Sigma_u_new)
        Same shapes / conventions as the inputs to
        ``_build_companion_state_space``.
    """
    M = smoothed.smoothed_state          # (T, n_state)
    V = smoothed.smoothed_cov            # (T, n_state, n_state)
    C = smoothed.smoothed_state_autocov  # (T, n_state, n_state); [t] = Cov(α_t, α_{t-1})

    T_obs = M.shape[0]
    n_state = n_vars * lags
    N_eff = T_obs - 1

    S_yy = np.zeros((n_vars, n_vars))
    S_yx = np.zeros((n_vars, n_state))      # E[y_t α_{t-1}']
    S_y = np.zeros(n_vars)                  # E[y_t]
    S_xx = np.zeros((n_state, n_state))     # E[α_{t-1} α_{t-1}']
    S_x = np.zeros(n_state)                 # E[α_{t-1}]

    for t in range(1, T_obs):
        M_t = M[t]
        M_tm1 = M[t - 1]
        V_t = V[t]
        V_tm1 = V[t - 1]
        C_t = C[t]   # Cov(α_t, α_{t-1} | Y)

        # E[α_t α_t' | Y] — top-left n_vars × n_vars block is E[y_t y_t']
        E_aa_t_yy = np.outer(M_t[:n_vars], M_t[:n_vars]) + V_t[:n_vars, :n_vars]
        # E[α_t α_{t-1}' | Y] — top n_vars rows give E[y_t α_{t-1}']
        E_aa_cross = np.outer(M_t, M_tm1) + C_t

        S_yy += E_aa_t_yy
        S_yx += E_aa_cross[:n_vars, :]
        S_y += M_t[:n_vars]
        S_xx += np.outer(M_tm1, M_tm1) + V_tm1
        S_x += M_tm1

    # Augment with the constant column: X_aug_t = [α_{t-1}; 1]
    S_yx_aug = np.hstack([S_yx, S_y.reshape(-1, 1)])              # (n, n_state+1)
    S_xx_aug = np.block([
        [S_xx,                  S_x.reshape(-1, 1)],
        [S_x.reshape(1, -1),    np.array([[float(N_eff)]])],
    ])

    # Closed-form OLS / GLS: Φ_aug = S_yx_aug @ S_xx_aug^{-1}
    Phi_aug = np.linalg.solve(S_xx_aug.T, S_yx_aug.T).T   # (n, n_state+1)
    Phi_lag = Phi_aug[:, :n_state]
    c_new = Phi_aug[:, n_state]

    # Σ_u: residual covariance from the same sufficient statistics
    Sigma_new = (S_yy - Phi_aug @ S_yx_aug.T) / N_eff
    Sigma_new = (Sigma_new + Sigma_new.T) / 2.0   # symmetrise (numerical)

    # Unpack [A_1, ..., A_p] from Phi_lag
    lag_mats_new = [Phi_lag[:, l * n_vars:(l + 1) * n_vars] for l in range(lags)]

    return lag_mats_new, c_new, Sigma_new
