"""
Diebold-Yilmaz (2012) connectedness based on the generalised forecast-error
variance decomposition (Pesaran-Shin 1998).

Unlike Cholesky-based FEVDs, the generalised variance decomposition is
INVARIANT to variable ordering — appropriate for connectedness measurement
across symmetric units (asset classes, countries, banks) where no natural
recursive ordering exists.

For each variable ``i`` at horizon ``H``, the generalised FEVD share
attributable to shock ``j`` is

    θ_{ij}(H) = σ_{jj}^{-1} * Σ_{h=0}^{H-1} (e_i' Φ_h Σ e_j)²
              ----------------------------------------------
                Σ_{h=0}^{H-1} (e_i' Φ_h Σ Φ_h' e_i)

where ``Φ_h`` is the MA coefficient and ``Σ`` is the reduced-form
residual covariance. Rows of the raw θ matrix do NOT sum to 1; row-
normalisation gives ``θ̃_{ij}(H)``.

The DY (2012) connectedness measures are then:

* **Total spillover index** ``C = 100 * (Σ_{i≠j} θ̃_{ij}) / N``
* **Directional FROM** ``C_{i←*} = 100 * Σ_{j≠i} θ̃_{ij} / N``
* **Directional TO**   ``C_{*←i} = 100 * Σ_{j≠i} θ̃_{ji} / N``
* **Net transmission** ``C_{i,net} = C_{*←i} − C_{i←*}``
* **Pairwise net**     ``C_{ij,net} = 100 * (θ̃_{ji} − θ̃_{ij}) / N``

For time-varying connectedness, see :func:`rolling_connectedness`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .var import VARFit, fit_var


@dataclass
class ConnectednessResult:
    """Result of a Diebold-Yilmaz (2012) connectedness computation.

    Attributes
    ----------
    fevd : np.ndarray, shape (n_vars, n_vars)
        Row-normalised generalised FEVD matrix. ``fevd[i, j]`` is the
        share of variable ``i``'s forecast-error variance attributable
        to shocks in variable ``j``. Rows sum to 1.
    total : float
        Total spillover index in [0, 100].
    directional_from : np.ndarray, shape (n_vars,)
        ``C_{i←*}``: spillovers TO variable ``i`` FROM all others.
    directional_to : np.ndarray, shape (n_vars,)
        ``C_{*←i}``: spillovers FROM variable ``i`` TO all others.
    net : np.ndarray, shape (n_vars,)
        Net transmission ``C_{*←i} − C_{i←*}``.
    pairwise_net : np.ndarray, shape (n_vars, n_vars)
        Bilateral net spillovers ``100 * (θ̃_{ji} − θ̃_{ij}) / N``.
    variable_names : list of str
    horizon : int
    """

    fevd: np.ndarray
    total: float
    directional_from: np.ndarray
    directional_to: np.ndarray
    net: np.ndarray
    pairwise_net: np.ndarray
    variable_names: list[str]
    horizon: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": float(self.total),
            "horizon": int(self.horizon),
            "variable_names": list(self.variable_names),
            "fevd": self.fevd.tolist(),
            "directional_from": self.directional_from.tolist(),
            "directional_to": self.directional_to.tolist(),
            "net": self.net.tolist(),
            "pairwise_net": self.pairwise_net.tolist(),
            "metadata": self.metadata,
        }


def generalised_fevd(
    fit: VARFit, horizon: int, row_normalise: bool = True,
) -> np.ndarray:
    """Generalised (Pesaran-Shin) forecast-error variance decomposition.

    HORIZON CONVENTION (textbook, Lutkepohl §2.2.2 / Hamilton §11.5):

      The ``H``-step-ahead forecast error is ``sum_{s=0..H-1} Phi_s u_{t+H-s}``,
      so its variance sums ``H`` terms (NOT H+1). At ``horizon=1`` (impact
      period only), the FEV is just ``Sigma_u`` and the decomposition is
      ``(Sigma_u[i,j])^2 / (Sigma_u[i,i] * Sigma_u[j,j])``.

      This function returns the single FEVD matrix at the specified horizon;
      to get a profile across horizons, loop.

    FORMULA:

        theta_{ij}(H) = sigma_{jj}^{-1} * sum_{h=0..H-1} (e_i' Phi_h Sigma e_j)^2
                      ------------------------------------------------
                                sum_{h=0..H-1} (e_i' Phi_h Sigma Phi_h' e_i)

    Returns an ``(n, n)`` matrix where entry ``(i, j)`` is the share of
    variable ``i``'s ``H``-step forecast-error variance attributable to
    variable ``j``'s shock. If ``row_normalise`` (default), rows sum to 1.
    Without row normalisation, raw rows generally don't sum to 1 because
    the generalised FEVD does not impose orthogonality.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    n = fit.n_vars
    Sigma = np.asarray(fit.Sigma_u)   # statsmodels sometimes returns a DataFrame
    Phi = fit.ma_rep(horizon - 1)   # MA(0..H-1)

    sigma_diag = np.diag(Sigma)
    # Numerator: Σ_h (e_i' Φ_h Σ e_j)^2 / σ_jj
    # Denominator: Σ_h e_i' Φ_h Σ Φ_h' e_i
    # Both are intuitive 2D matrices over (i, j) and (i,)
    theta = np.zeros((n, n))
    denom = np.zeros(n)
    for h in range(horizon):
        Phi_h = Phi[h]
        ASigma = Phi_h @ Sigma                # (n, n)
        # (Phi_h Sigma)[i, j] = (e_i' Phi_h Sigma e_j)
        theta += (ASigma ** 2) / sigma_diag[None, :]
        denom += np.einsum("ij,jk,ik->i", Phi_h, Sigma, Phi_h)

    raw = theta / denom[:, None]

    if row_normalise:
        return raw / raw.sum(axis=1, keepdims=True)
    return raw


def connectedness(
    fit: VARFit, horizon: int = 10,
) -> ConnectednessResult:
    """Compute Diebold-Yilmaz (2012) connectedness measures from a VAR fit.

    Parameters
    ----------
    fit : VARFit
        Reduced-form VAR. Connectedness is identification-free — uses
        generalised (Pesaran-Shin) FEVD rather than Cholesky.
    horizon : int, default 10
        FEVD horizon (typically 10 days for daily volatility-spillover
        applications; matches DY 2012).

    Returns
    -------
    ConnectednessResult.

    Notes
    -----
    **Scaling convention**: the directional FROM / TO indices and the
    pairwise-net matrix are divided by ``N`` (the number of variables),
    matching Diebold-Yilmaz 2012 Eq. (12). Some other implementations
    divide by ``N - 1`` instead, giving a "per-pair" interpretation. The
    ratio is ``N / (N - 1)`` — on n=4 panels the gap is ~33%, on n=10 it's
    ~11%. To compare with such numbers, multiply our directional indices by
    ``N / (N - 1)``.

    The Total Spillover Index uses the N divisor in both conventions.
    """
    fevd_n = generalised_fevd(fit, horizon=horizon, row_normalise=True)
    n = fit.n_vars

    # Total: average off-diagonal mass
    total = 100.0 * (fevd_n.sum() - np.trace(fevd_n)) / n

    # Directional FROM: spillovers RECEIVED by i (row sum minus diagonal)
    directional_from = 100.0 * (fevd_n.sum(axis=1) - np.diag(fevd_n)) / n

    # Directional TO: spillovers TRANSMITTED by i (column sum minus diagonal)
    directional_to = 100.0 * (fevd_n.sum(axis=0) - np.diag(fevd_n)) / n

    # Net transmission
    net = directional_to - directional_from

    # Pairwise net: theta_ji - theta_ij  (rescaled to 0-100 like the others)
    pairwise_net = 100.0 * (fevd_n.T - fevd_n) / n

    metadata = {
        "method": "diebold_yilmaz_2012",
        "fevd_type": "generalised_pesaran_shin",
        "horizon": int(horizon),
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
    }
    return ConnectednessResult(
        fevd=fevd_n,
        total=float(total),
        directional_from=directional_from,
        directional_to=directional_to,
        net=net,
        pairwise_net=pairwise_net,
        variable_names=list(fit.variable_names),
        horizon=int(horizon),
        metadata=metadata,
    )


def rolling_connectedness(
    data: pd.DataFrame, window: int, horizon: int = 10, lags: int = 4,
    step: int = 1, trend: str = "c",
) -> pd.DataFrame:
    """Rolling-window Diebold-Yilmaz total spillover index.

    Fits a VAR on each rolling window of ``window`` observations and
    computes the total connectedness, returning a time series.

    Parameters
    ----------
    data : pd.DataFrame
    window : int
        Rolling-window length (e.g., 200 for ~10 months daily).
    horizon : int, default 10
        FEVD horizon for each window's connectedness measure.
    lags : int, default 4
        VAR lag order.
    step : int, default 1
        Step between successive windows (1 = every observation).
    trend : str, default "c"
        VAR trend setting.

    Returns
    -------
    pd.DataFrame indexed by the LAST observation of each window with
    columns ``["total"]`` plus ``directional_from_<v>``, ``directional_to_<v>``,
    ``net_<v>`` for each variable ``v``.
    """
    n = data.shape[1]
    var_names = list(data.columns)
    if window > len(data):
        raise ValueError(f"window {window} > data length {len(data)}")

    rows = []
    end_indices = list(range(window - 1, len(data), step))
    for end_idx in end_indices:
        sub = data.iloc[end_idx - window + 1:end_idx + 1]
        try:
            fit = fit_var(sub, lags=lags, trend=trend)
            res = connectedness(fit, horizon=horizon)
        except (np.linalg.LinAlgError, ValueError):
            row = {"total": np.nan}
            for v in var_names:
                row[f"directional_from_{v}"] = np.nan
                row[f"directional_to_{v}"] = np.nan
                row[f"net_{v}"] = np.nan
            rows.append((data.index[end_idx], row))
            continue
        row = {"total": res.total}
        for j, v in enumerate(var_names):
            row[f"directional_from_{v}"] = float(res.directional_from[j])
            row[f"directional_to_{v}"] = float(res.directional_to[j])
            row[f"net_{v}"] = float(res.net[j])
        rows.append((data.index[end_idx], row))

    idx = [r[0] for r in rows]
    body = [r[1] for r in rows]
    return pd.DataFrame(body, index=pd.Index(idx, name=data.index.name))
