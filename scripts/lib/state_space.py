"""
State-space model wrappers — Kalman filter, smoother, simulation smoother.

A uniform API for state-space models in the standard form:

    y_t = Z @ alpha_t + d + epsilon_t,           epsilon_t ~ N(0, H)
    alpha_{t+1} = T @ alpha_t + c + R @ eta_t,   eta_t ~ N(0, Q)

The Kalman filter/smoother routines wrap ``statsmodels.tsa.statespace``
for the heavy lifting — we do NOT reimplement Kalman filtering. Our
value-add is a thin, VAR-friendly API plus the application-specific
helpers (Mariano-Murasawa flow aggregation, NaN-handling conventions).

Designed for use by:
* :mod:`lib.mfvar` — mixed-frequency VAR (Schorfheide-Song 2015, BM 2014)
* :mod:`lib.tvp` — TVP-VAR with stochastic volatility (Primiceri 2005)
* :mod:`lib.dfm` — Bayesian dynamic factor models

The observation design ``Z``, observation covariance ``H`` and observation
intercept ``d`` may be time varying (leading time axis). The transition
``T``, selection ``R``, state covariance ``Q`` and state intercept ``c``
are time invariant in this wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# State-space model container
# ---------------------------------------------------------------------------


@dataclass
class StateSpaceModel:
    """Linear-Gaussian state-space model.

    ``Z``, ``H`` and ``obs_intercept`` may be time varying; ``T``, ``R``,
    ``Q`` and ``state_intercept`` are time invariant.

    Parameters follow the statsmodels convention (matching Durbin-Koopman
    "Time Series Analysis by State Space Methods"):

    Measurement equation:
        ``y_t = Z @ alpha_t + d + epsilon_t``,  ``epsilon_t ~ N(0, H)``

    Transition equation:
        ``alpha_{t+1} = T @ alpha_t + c + R @ eta_t``,  ``eta_t ~ N(0, Q)``

    Attributes
    ----------
    Z : (n_obs, n_state) ndarray, or (T, n_obs, n_state) for time-varying
        Design matrix. If 3D, the leading axis is time.
    H : (n_obs, n_obs) ndarray, or (T, n_obs, n_obs) for time-varying
        Measurement-error covariance. If 3D, the leading axis is time.
        Use ``np.zeros`` for noise-free measurement.
    T : (n_state, n_state) ndarray
        Transition matrix.
    R : (n_state, n_shock) ndarray
        Selection matrix mapping shocks to states.
    Q : (n_shock, n_shock) ndarray
        State-shock covariance.
    state_intercept : (n_state,) ndarray, optional
        ``c`` in transition equation. Defaults to zeros.
    obs_intercept : (n_obs,) ndarray, or (T, n_obs) for time-varying, optional
        ``d`` in measurement equation. If 2D, the leading axis is time.
        Defaults to zeros.
    initial_state_mean : (n_state,) ndarray, optional
        Mean of α_0. Defaults to zeros (treated as known for now).
    initial_state_cov : (n_state, n_state) ndarray, optional
        Covariance of α_0. Defaults to ``1e6 * I`` (diffuse).
    """

    Z: np.ndarray
    H: np.ndarray
    T: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    state_intercept: np.ndarray | None = None
    obs_intercept: np.ndarray | None = None
    initial_state_mean: np.ndarray | None = None
    initial_state_cov: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Accept 2D (time-invariant) or 3D (time-varying first axis) for Z
        if self.Z.ndim == 2:
            n_obs, n_state = self.Z.shape
            self._z_time_varying = False
        elif self.Z.ndim == 3:
            _, n_obs, n_state = self.Z.shape
            self._z_time_varying = True
        else:
            raise ValueError(f"Z must be 2D or 3D, got {self.Z.ndim}D")

        n_shock = self.R.shape[1]

        # H may be 2D (time-invariant) or 3D (T, n_obs, n_obs) for time-varying
        if self.H.ndim == 2:
            if self.H.shape != (n_obs, n_obs):
                raise ValueError(f"H must be ({n_obs}, {n_obs}), got {self.H.shape}")
            self._h_time_varying = False
        elif self.H.ndim == 3:
            if self.H.shape[1:] != (n_obs, n_obs):
                raise ValueError(
                    f"H must be (T, {n_obs}, {n_obs}), got {self.H.shape}"
                )
            self._h_time_varying = True
        else:
            raise ValueError(f"H must be 2D or 3D, got {self.H.ndim}D")

        if self.T.shape != (n_state, n_state):
            raise ValueError(f"T must be ({n_state}, {n_state}), got {self.T.shape}")
        if self.R.shape != (n_state, n_shock):
            raise ValueError(f"R must be ({n_state}, {n_shock}), got {self.R.shape}")
        if self.Q.shape != (n_shock, n_shock):
            raise ValueError(f"Q must be ({n_shock}, {n_shock}), got {self.Q.shape}")

        if self.state_intercept is None:
            self.state_intercept = np.zeros(n_state)
        # obs_intercept may be 1D (time-invariant) or 2D (T, n_obs) for time-varying
        if self.obs_intercept is None:
            self.obs_intercept = np.zeros(n_obs)
            self._obs_intercept_time_varying = False
        elif self.obs_intercept.ndim == 1:
            self._obs_intercept_time_varying = False
        elif self.obs_intercept.ndim == 2:
            if self.obs_intercept.shape[1] != n_obs:
                raise ValueError(
                    f"obs_intercept must be (T, {n_obs}), got {self.obs_intercept.shape}"
                )
            self._obs_intercept_time_varying = True
        else:
            raise ValueError(
                f"obs_intercept must be 1D or 2D, got {self.obs_intercept.ndim}D"
            )
        if self.initial_state_mean is None:
            self.initial_state_mean = np.zeros(n_state)
        if self.initial_state_cov is None:
            self.initial_state_cov = np.eye(n_state) * 1e6

    @property
    def n_state(self) -> int:
        return self.T.shape[0]

    @property
    def n_obs(self) -> int:
        # 3D Z: (T, n_obs, n_state) -> shape[1]
        # 2D Z: (n_obs, n_state)    -> shape[0]
        return self.Z.shape[1] if self.Z.ndim == 3 else self.Z.shape[0]

    @property
    def n_shock(self) -> int:
        return self.R.shape[1]

    @property
    def z_is_time_varying(self) -> bool:
        return self.Z.ndim == 3


# ---------------------------------------------------------------------------
# Kalman filter / smoother (wraps statsmodels.tsa.statespace)
# ---------------------------------------------------------------------------


@dataclass
class KalmanResults:
    """Filter + smoother output bundle.

    Attributes
    ----------
    filtered_state : (T, n_state) ndarray
        E[alpha_t | y_{1:t}]
    filtered_cov : (T, n_state, n_state) ndarray
        Cov(alpha_t | y_{1:t})
    smoothed_state : (T, n_state) ndarray
        E[alpha_t | y_{1:T}] from RTS smoother
    smoothed_cov : (T, n_state, n_state) ndarray
        Cov(alpha_t | y_{1:T})
    smoothed_state_autocov : (T, n_state, n_state) ndarray
        Lag-1 smoothed cross-covariance ``Cov(alpha_t, alpha_{t-1} | y_{1:T})``;
        entry ``[t]`` is the matrix indexed by row=alpha_t, col=alpha_{t-1}.
        Entry ``[0]`` is undefined (no pre-sample lag) — do not use.
        Needed for EM-style ML estimation of state-space parameters
        (Shumway-Stoffer 1982 M-step).
    log_likelihood : float
        Sum of conditional log-likelihoods from the filter.
    """

    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    smoothed_state: np.ndarray
    smoothed_cov: np.ndarray
    smoothed_state_autocov: np.ndarray
    log_likelihood: float


def _to_statsmodels_repr(model: StateSpaceModel, y: np.ndarray):
    """Build a statsmodels ``SimulationSmoother`` initialised for the model + data.

    ``SimulationSmoother`` extends ``KalmanSmoother`` which extends
    ``KalmanFilter``, so a single handle supports ``.filter()``,
    ``.smooth()``, and ``.simulation_smoother()`` (which returns the
    simulation-smoother result object).
    """
    from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    T = y.shape[0]
    n_obs = model.n_obs
    if y.shape[1] != n_obs:
        raise ValueError(
            f"y has {y.shape[1]} columns; model expects n_obs={n_obs}"
        )

    # statsmodels expects (k_endog, T) Fortran-ordered; np.asfortranarray
    # handles the contiguity requirement.
    endog = np.asfortranarray(y.T.astype(float))
    kf = SimulationSmoother(
        k_endog=n_obs, k_states=model.n_state, k_posdef=model.n_shock,
        nobs=T,
    )
    kf.bind(endog)
    # statsmodels' design matrix shape: (n_obs, n_state, n_periods) where
    # n_periods is T for time-varying or 1 for time-invariant. Our Z can be
    # 2D (n_obs, n_state) or 3D (T, n_obs, n_state).
    if model.z_is_time_varying:
        if model.Z.shape[0] != T:
            raise ValueError(
                f"time-varying Z has T={model.Z.shape[0]}, "
                f"data has T={T}"
            )
        Z_sm = np.asfortranarray(np.transpose(model.Z, (1, 2, 0)))
    else:
        Z_sm = model.Z[:, :, None]
    kf["design"] = Z_sm
    # H: 2D (n_obs, n_obs) -> broadcast as (n_obs, n_obs, 1);
    #    3D (T, n_obs, n_obs) -> transpose to (n_obs, n_obs, T)
    if model._h_time_varying:
        if model.H.shape[0] != T:
            raise ValueError(
                f"time-varying H has T={model.H.shape[0]}, data has T={T}"
            )
        H_sm = np.asfortranarray(np.transpose(model.H, (1, 2, 0)))
    else:
        H_sm = model.H[:, :, None]
    kf["obs_cov"] = H_sm
    kf["transition"] = model.T[:, :, None]
    kf["selection"] = model.R[:, :, None]
    kf["state_cov"] = model.Q[:, :, None]
    kf["state_intercept"] = model.state_intercept[:, None]
    # obs_intercept: 1D (n_obs,) -> (n_obs, 1); 2D (T, n_obs) -> (n_obs, T)
    if model._obs_intercept_time_varying:
        if model.obs_intercept.shape[0] != T:
            raise ValueError(
                f"time-varying obs_intercept has T={model.obs_intercept.shape[0]}, "
                f"data has T={T}"
            )
        d_sm = np.asfortranarray(model.obs_intercept.T)
    else:
        d_sm = model.obs_intercept[:, None]
    kf["obs_intercept"] = d_sm
    kf.initialize_known(model.initial_state_mean, model.initial_state_cov)
    return kf


def kalman_filter_smooth(
    model: StateSpaceModel, y: np.ndarray,
) -> KalmanResults:
    """Run Kalman filter + RTS smoother.

    Parameters
    ----------
    model : StateSpaceModel
    y : (T, n_obs) ndarray
        Observations. NaN entries are treated as missing (statsmodels
        handles them natively via the Kalman update with the appropriate
        rows of ``Z`` and ``H`` dropped).

    Returns
    -------
    KalmanResults
    """
    kf = _to_statsmodels_repr(model, y)
    filter_results = kf.filter()
    smoother_results = kf.smooth()
    # statsmodels' smoothed_state_autocovariance(lag=1) returns shape
    # (n_state, n_state, T); entry [:, :, t] is Cov(alpha_t, alpha_{t-1} | Y).
    # Index t=0 is undefined by construction (no pre-sample lag); callers
    # must skip it.
    autocov_sm = smoother_results.smoothed_state_autocovariance(lag=1)
    # statsmodels stores states in (n_state, T); transpose for our (T, n_state)
    return KalmanResults(
        filtered_state=filter_results.filtered_state.T,
        filtered_cov=np.transpose(filter_results.filtered_state_cov, (2, 0, 1)),
        smoothed_state=smoother_results.smoothed_state.T,
        smoothed_cov=np.transpose(smoother_results.smoothed_state_cov, (2, 0, 1)),
        smoothed_state_autocov=np.transpose(autocov_sm, (2, 0, 1)),
        log_likelihood=float(filter_results.llf),
    )


def log_likelihood(model: StateSpaceModel, y: np.ndarray) -> float:
    """Compute marginal log-likelihood via the Kalman filter, no smoother."""
    kf = _to_statsmodels_repr(model, y)
    return float(kf.filter().llf)


def simulation_smoother(
    model: StateSpaceModel, y: np.ndarray,
    n_draws: int = 1, seed: int | None = None,
) -> np.ndarray:
    """Durbin-Koopman simulation smoother.

    Draws state paths from the conditional distribution ``p(alpha | y)``.
    Each draw is a realisation consistent with the observed data plus the
    state-space dynamics, including the smoothing variance.

    Parameters
    ----------
    model : StateSpaceModel
    y : (T, n_obs) ndarray
    n_draws : int, default 1
    seed : int, optional

    Returns
    -------
    ndarray of shape ``(n_draws, T, n_state)``.
    """
    kf = _to_statsmodels_repr(model, y)
    sim = kf.simulation_smoother()
    rng = np.random.default_rng(seed)
    T = y.shape[0]
    out = np.zeros((n_draws, T, model.n_state))
    for d in range(n_draws):
        # statsmodels' simulation_smoother uses statsmodels' own RNG; we seed
        # by passing a numpy-generated integer
        sim.simulate(random_state=np.random.RandomState(rng.integers(0, 2**31 - 1)))
        out[d] = sim.simulated_state.T
    return out


# ---------------------------------------------------------------------------
# Mariano-Murasawa flow aggregation (used by MF-VAR / DFM)
# ---------------------------------------------------------------------------


def mariano_murasawa_weights() -> np.ndarray:
    """Mariano-Murasawa (2003) flow-aggregation weights.

    For a flow variable observed at quarterly frequency in log-growth-rate
    form, the quarterly observation relates to five consecutive monthly
    latent states as

        y^Q_t = (1/3) y^M_t + (2/3) y^M_{t-1} + y^M_{t-2}
              + (2/3) y^M_{t-3} + (1/3) y^M_{t-4}

    This is the standard "ξ" filter for flow log-growth in MF-VAR /
    nowcasting state-space setups.

    Returns
    -------
    ndarray of shape (5,) with the weights ``[1/3, 2/3, 1, 2/3, 1/3]``.
    """
    return np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3])


def simple_flow_weights(periods_per_obs: int = 3) -> np.ndarray:
    """Simple intra-period averaging for flow level variables.

    For a flow level (not growth rate) observed at lower frequency, the
    quarterly observation is the simple intra-quarter average of the
    monthly latent state:

        y^Q_t = (1/k) Σ_{j=0}^{k-1} y^M_{t-j}

    where ``k = periods_per_obs`` (typically 3 for quarterly-from-monthly).

    For aggregating LEVELS (e.g., GDP in levels), use this. For
    aggregating LOG-GROWTH rates, use :func:`mariano_murasawa_weights`.
    """
    return np.full(periods_per_obs, 1.0 / periods_per_obs)
