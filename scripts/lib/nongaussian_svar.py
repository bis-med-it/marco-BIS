"""Non-Gaussian SVAR identification (Lanne, Meitz, Saikkonen 2017).

Exploits non-Gaussianity of the structural shocks to identify the structural
matrix :math:`B` in :math:`u_t = B \\varepsilon_t` up to column permutation
and sign — no Cholesky, sign, zero, or external-instrument restrictions
needed.

The key identifying assumption is that the structural shocks
:math:`\\varepsilon_t = (\\varepsilon_{1t}, \\ldots, \\varepsilon_{nt})` are
**mutually independent** (stronger than uncorrelated) and **at most one is
Gaussian** (need non-Gaussianity to break the rotational invariance).

This is the SVAR analogue of the ICA identification result (Comon 1994).
For methodological background see:

  Lanne, M., Meitz, M. and Saikkonen, P. (2017). "Identification and
  Estimation of Non-Gaussian Structural Vector Autoregressions."
  Journal of Econometrics, 196(2), 288-304.

Public API:

* :func:`nongaussian_svar_irf` — fit + identify + IRFs, returns
  ``(IRFContainer, dict)`` with diagnostics on the recovered shocks.

Helpers (also used by the tests):

* :func:`fastica_symmetric` — symmetric (parallel) FastICA fixed-point
  algorithm; returns the unmixing matrix and convergence info.
* :func:`whiten_residuals` — Cholesky pre-whitening.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .irf import IRFContainer
from .var import VARFit, extract_lag_matrices, ma_representation


# ---------------------------------------------------------------------------
# Pre-whitening
# ---------------------------------------------------------------------------


def whiten_residuals(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky pre-whitening of reduced-form residuals.

    Parameters
    ----------
    residuals : (T, n) array
        Reduced-form residuals (rows = observations).

    Returns
    -------
    Z : (T, n) array
        Whitened residuals with ``cov(Z) ≈ I_n``.
    L : (n, n) array
        Lower-triangular Cholesky factor of ``cov(residuals)`` such that
        ``residuals = L @ Z^T`` (column convention) or equivalently
        ``residuals[t] = L @ Z[t]`` (row convention here).

    Notes
    -----
    Uses ``np.cov(..., ddof=1)`` for the residual covariance — same
    Bessel-corrected sample covariance used throughout the framework.
    """
    if residuals.ndim != 2:
        raise ValueError(f"residuals must be 2D; got shape {residuals.shape}")
    Sigma = np.cov(residuals.T, ddof=1)
    Sigma = 0.5 * (Sigma + Sigma.T)
    L = np.linalg.cholesky(Sigma)
    L_inv = np.linalg.inv(L)
    Z = residuals @ L_inv.T
    return Z, L


# ---------------------------------------------------------------------------
# FastICA (symmetric / parallel form)
# ---------------------------------------------------------------------------


_NONLINEARITIES: dict[str, tuple[Callable[[np.ndarray], np.ndarray],
                                  Callable[[np.ndarray], np.ndarray]]] = {
    "tanh": (
        lambda u: np.tanh(u),
        lambda u: 1.0 - np.tanh(u) ** 2,
    ),
    "cube": (
        lambda u: u ** 3,
        lambda u: 3.0 * u ** 2,
    ),
    "exp": (
        lambda u: u * np.exp(-(u ** 2) / 2.0),
        lambda u: (1.0 - u ** 2) * np.exp(-(u ** 2) / 2.0),
    ),
}


def _symmetric_orthogonalise(W: np.ndarray) -> np.ndarray:
    """Symmetric (Mahalanobis) orthogonalisation: ``W ← (W W')^{-1/2} W``.

    Implemented via SVD: if ``W = U Σ V'`` then ``(W W')^{-1/2} W = U V'``.
    """
    U, _s, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt


def fastica_symmetric(
    Z: np.ndarray,
    nonlinearity: str = "tanh",
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetric FastICA on pre-whitened data ``Z`` (Hyvärinen 1999).

    All components updated in parallel each iteration, followed by
    symmetric orthogonalisation. Returns the unmixing matrix ``W`` such
    that the recovered components ``\\varepsilon_t = W z_t`` are
    approximately mutually independent.

    Parameters
    ----------
    Z : (T, n) array
        Pre-whitened residuals (``cov(Z) ≈ I``).
    nonlinearity : {"tanh", "cube", "exp"}
        Non-Gaussianity contrast function ``g``:

        * ``"tanh"``: ``g(u) = tanh(u)``, ``g'(u) = 1 - tanh^2(u)`` —
          robust default, works for many super-Gaussian distributions.
        * ``"cube"``: ``g(u) = u^3``, ``g'(u) = 3u^2`` — kurtosis-based,
          better for very super-Gaussian (heavy-tailed) sources.
        * ``"exp"``: ``g(u) = u·exp(-u²/2)``, ``g'(u) = (1-u²)·exp(-u²/2)`` —
          Gauss-like; sometimes more stable when the user is unsure.
    max_iter : int
        Fixed-point iteration limit.
    tol : float
        Convergence tolerance on ``max |1 - |diag(W_new W')||``.
    seed : int, optional
        RNG seed for the initial orthogonal matrix.

    Returns
    -------
    W : (n, n) array
        Unmixing matrix; rows are the recovered independent-component
        directions in the whitened space. ``W @ W.T ≈ I_n``.
    diag : dict
        Convergence diagnostics: ``n_iter`` (iterations performed),
        ``converged`` (bool), ``final_diff`` (last tolerance check value),
        ``nonlinearity`` (echo).

    References
    ----------
    Hyvärinen, A. (1999). "Fast and Robust Fixed-Point Algorithms for
    Independent Component Analysis." IEEE Transactions on Neural Networks,
    10(3), 626-634.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D; got shape {Z.shape}")
    if nonlinearity not in _NONLINEARITIES:
        raise ValueError(
            f"unknown nonlinearity {nonlinearity!r}; "
            f"expected one of {sorted(_NONLINEARITIES)}"
        )
    g_fn, g_prime = _NONLINEARITIES[nonlinearity]
    T, n = Z.shape
    if T < n + 1:
        raise ValueError(f"need T >= n+1 observations; got T={T}, n={n}")

    rng = np.random.default_rng(seed)

    # Initial orthogonal W: symmetric orthogonalisation of a Gaussian matrix
    W = rng.standard_normal((n, n))
    W = _symmetric_orthogonalise(W)

    diff = np.inf
    converged = False
    n_iter = 0
    for it in range(max_iter):
        # WZ has shape (n, T): the k-th row is w_k' applied to all observations
        WZ = W @ Z.T
        G = g_fn(WZ)                  # (n, T)
        Gp = g_prime(WZ)              # (n, T)
        # Fixed-point update (each row independently):
        #   w_k_new = E[z g(w_k' z)] - E[g'(w_k' z)] w_k
        # Vectorised over k:
        W_new = (G @ Z) / T - (Gp.mean(axis=1)[:, None] * W)
        W_new = _symmetric_orthogonalise(W_new)
        # Convergence: |diag(W_new W')| should be ±1 elements (W ≈ W_new
        # up to sign) when converged.
        diff = float(np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)))
        W = W_new
        n_iter += 1
        if diff < tol:
            converged = True
            break

    return W, {
        "n_iter": n_iter,
        "converged": bool(converged),
        "final_diff": diff,
        "nonlinearity": nonlinearity,
    }


# ---------------------------------------------------------------------------
# Parametric t-ML rotation estimator (~ step 2 of LMS's §4.5 three-step procedure)
# ---------------------------------------------------------------------------


def _givens_orthogonal(theta: np.ndarray, n: int) -> np.ndarray:
    """Build an ``n × n`` orthogonal matrix from ``n(n-1)/2`` Givens angles.

    Iterates over ordered pairs ``(i, j)`` with ``i < j`` and applies the
    plane rotation ``R(i, j; θ_k)`` in that plane. The result is the
    product ``Q = ∏ R(i, j; θ_k)``.

    All ``Q`` reachable this way are orthogonal; the parameterisation
    covers a dense subset of the connected component of the orthogonal
    group containing the identity (the special orthogonal group ``SO(n)``).
    Sign ambiguity in column orientation is handled separately by
    ``sign_normalization``.
    """
    n_angles = n * (n - 1) // 2
    if theta.shape != (n_angles,):
        raise ValueError(
            f"theta has shape {theta.shape}, expected ({n_angles},) "
            f"for n={n}"
        )
    Q = np.eye(n)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            c = np.cos(theta[k])
            s = np.sin(theta[k])
            R = np.eye(n)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            Q = Q @ R
            k += 1
    return Q


def _standardised_t_logpdf(x: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Log-pdf of a unit-variance scaled Student-t.

    If ``Z ~ t(ν)`` (standard, variance ``ν/(ν-2)``), then
    ``X = sqrt((ν-2)/ν) · Z`` has unit variance and pdf
    ``f(x; ν) = (1/σ) · t_ν.pdf(x/σ)`` with ``σ = sqrt((ν-2)/ν)``.

    Parameters
    ----------
    x : (T, n) array
        Observations (rows = obs, columns = independent components).
    nu : (n,) array
        Per-component degrees of freedom (each ``> 2`` required for
        unit-variance scaling to exist).

    Returns
    -------
    (T, n) array of per-(obs, component) log-pdf values.
    """
    sigma = np.sqrt((nu - 2.0) / nu)              # (n,)
    z = x / sigma[None, :]                         # (T, n)
    return stats.t.logpdf(z, df=nu[None, :]) - np.log(sigma)[None, :]


def _pml_t_neg_log_lik(params: np.ndarray, Z: np.ndarray, n: int) -> float:
    """Negative joint log-likelihood for the LMS-style t-ML rotation step.

    Parameters
    ----------
    params : (n(n-1)/2 + n,) array
        First ``n(n-1)/2`` entries are Givens angles parameterising the
        orthogonal rotation ``Q``. The remaining ``n`` entries are
        ``log(ν - 2)`` for each shock (so the optimiser's domain is
        unconstrained, with ``ν > 2`` guaranteed at all evaluation
        points).
    Z : (T, n) array
        Pre-whitened residuals.
    n : int
        Number of variables.

    Returns
    -------
    Scalar negative log-likelihood (to be minimised). The
    ``-T log |det B|`` constant in Q is dropped since ``Q`` is
    orthogonal (|det Q| = 1).
    """
    n_angles = n * (n - 1) // 2
    theta = params[:n_angles]
    log_nu_m2 = params[n_angles:]
    nu = np.exp(log_nu_m2) + 2.0                   # ν ∈ (2, ∞)
    Q = _givens_orthogonal(theta, n)
    eps = Z @ Q                                    # (T, n)
    log_pdf = _standardised_t_logpdf(eps, nu)
    return float(-log_pdf.sum())


def pml_t_estimate(
    Z: np.ndarray,
    init_Q: np.ndarray | None = None,
    n_restarts: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = None,
    nu_init: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parametric t-ML estimation of the SVAR identification rotation
    (~ step 2 of LMS's §4.5 three-step estimator, unit-variance normalisation).

    Implements LMS (2017) §3: assume each structural shock is i.i.d.
    standardised Student-t with degrees-of-freedom ``ν_j``, parameterise
    ``Q`` as a product of Givens rotations, and maximise the joint
    log-likelihood over (Q angles, log(ν - 2)) using L-BFGS-B.

    Parameters
    ----------
    Z : (T, n) array
        Pre-whitened residuals.
    init_Q : (n, n) array, optional
        Warm-start orthogonal rotation. If provided, ``Q`` is initialised
        to ``init_Q`` (Givens angles set to zero so the optimiser searches
        for small perturbations from ``init_Q`` via a fresh Givens
        parameterisation centred at the identity, then post-multiplied by
        ``init_Q``).

        If ``None``, ``Q`` is initialised to the identity and the
        optimiser explores from scratch.

        Either way, ``n_restarts`` additional random restarts are run
        and the highest-likelihood CONVERGED result is returned (a failed optimiser run can report a lower objective than a converged one and is never selected).
    n_restarts : int
        Number of random restarts (in addition to the deterministic
        ``init_Q``-or-identity start). Each restart uses a fresh random
        Givens-angle initialisation in ``[-π, π]``.
    max_iter : int
        L-BFGS-B iteration limit per restart.
    tol : float
        L-BFGS-B convergence tolerance.
    seed : int, optional
    nu_init : float
        Initial value for each ``ν_j`` at every restart. Default ``5.0``
        (moderate leptokurtic prior — neutral between Gaussian-like and
        very heavy-tailed).

    Returns
    -------
    Q : (n, n) array
        The ML-optimal orthogonal rotation.
    nu : (n,) array
        ML-estimated per-shock degrees of freedom.
    diag : dict
        Convergence diagnostics: ``log_likelihood`` (max across converged
        restarts), ``n_restarts_tried``, ``best_restart_index``,
        ``converged`` bool.

    Raises
    ------
    RuntimeError
        If NO restart converges — an unfinished iterate is not an
        estimate and is never returned.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D; got shape {Z.shape}")
    T, n = Z.shape
    if T < n + 2:
        raise ValueError(f"need T >= n+2 observations; got T={T}, n={n}")
    n_angles = n * (n - 1) // 2
    n_params = n_angles + n
    rng = np.random.default_rng(seed)

    # Bounds: angles unconstrained; log(ν - 2) ∈ [-10, 6] →
    # ν ∈ [2.000045, 405.4], spanning "very heavy-tailed" to "near-Gaussian".
    bounds = (
        [(None, None)] * n_angles
        + [(-10.0, 6.0)] * n
    )
    log_nu_m2_init = float(np.log(max(nu_init - 2.0, 1e-3)))

    # Construct starting points. The deterministic init is identity-Q.
    # If init_Q is provided, we post-process: the optimisation searches
    # over Givens(θ), and the effective Q is `Givens(θ) @ init_Q` to
    # warm-start near init_Q. (Operationally simpler: just precompose
    # the residuals: Z_eff = Z @ init_Q, then optimise as usual.)
    if init_Q is not None:
        if init_Q.shape != (n, n):
            raise ValueError(
                f"init_Q has shape {init_Q.shape}, expected ({n}, {n})"
            )
        # Use init_Q to whiten further; the optimiser searches over the
        # residual rotation, starting from identity (θ=0).
        Z_eff = Z @ init_Q
    else:
        Z_eff = Z

    best: tuple[float, np.ndarray | None, np.ndarray | None] = (np.inf, None, None)
    best_idx = -1
    convergences: list[bool] = []
    start_points: list[np.ndarray] = [
        np.concatenate([np.zeros(n_angles), np.full(n, log_nu_m2_init)])
    ]
    for _ in range(n_restarts):
        x0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, n_angles),
            np.full(n, log_nu_m2_init),
        ])
        start_points.append(x0)

    for i, x0 in enumerate(start_points):
        result = optimize.minimize(
            _pml_t_neg_log_lik,
            x0,
            args=(Z_eff, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": tol},
        )
        convergences.append(bool(result.success))
        # Only a CONVERGED, finite restart may be selected: a failed run
        # can report a lower objective than a converged one, and an
        # unfinished iterate is not an estimate.
        if (
            bool(result.success)
            and np.isfinite(result.fun)
            and result.fun < best[0]
        ):
            best = (float(result.fun), result.x.copy(), None)
            best_idx = i

    if best[1] is None:
        raise RuntimeError(
            f"pml_t optimisation did not converge at any of the "
            f"{len(start_points)} restart(s) (success flags: {convergences}); "
            f"no unfinished iterate is returned as an estimate. Increase "
            f"max_iter or the number of restarts."
        )

    params = best[1]
    theta = params[:n_angles]
    log_nu_m2 = params[n_angles:]
    Q_search = _givens_orthogonal(theta, n)
    Q = Q_search if init_Q is None else (init_Q @ Q_search)
    nu = np.exp(log_nu_m2) + 2.0

    diag = {
        "log_likelihood": -best[0],
        "n_restarts_tried": len(start_points),
        "best_restart_index": best_idx,
        "converged": convergences[best_idx],
        "nu": nu.tolist(),
    }
    return Q, nu, diag


# ---------------------------------------------------------------------------
# Shock labelling / sign normalisation utilities
# ---------------------------------------------------------------------------


def _label_columns_by_impact(
    B: np.ndarray,
    eps: np.ndarray,
    shock_names: list[str],
    variable_names: list[str],
    label_by: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Permute columns of ``B`` (and ``eps``) so each shock named in
    ``label_by`` corresponds to the column with the largest absolute
    contemporaneous impact on the specified variable.

    Greedy assignment from the strongest signal first.
    """
    n = B.shape[1]
    # Score matrix: score[shock_name, column] = |B[var_of_shock, column]|.
    used_cols: set[int] = set()
    perm: list[int] = [-1] * n

    # First, handle shocks listed in label_by (greedy, strongest first)
    requested: list[tuple[str, str]] = [
        (s, label_by[s]) for s in shock_names if s in label_by
    ]
    # Sort by the strongest DIMENSIONLESS signal: each row's impacts are
    # divided by that row's norm before comparing across rows. Raw impacts
    # carry the variable's units, so comparing them across different rows
    # would let a unit convention decide which independent component gets
    # which economic name. Within a row the comparison is unit-free.
    row_norm = {
        v: max(float(np.linalg.norm(B[variable_names.index(v), :])), 1e-300)
        for _, v in requested
        if v in variable_names
    }
    requested.sort(
        key=lambda sv: -max(
            abs(B[variable_names.index(sv[1]), c]) / row_norm[sv[1]]
            for c in range(n)
        ) if sv[1] in row_norm else 0.0,
    )
    for sh, vname in requested:
        if vname not in variable_names:
            raise ValueError(f"unknown variable {vname!r} in label_by")
        v_idx = variable_names.index(vname)
        # Pick best unused column
        best_col, best_score = None, -1.0
        for c in range(n):
            if c in used_cols:
                continue
            score = abs(B[v_idx, c])
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is None:
            raise RuntimeError("ran out of columns during labelling")
        sh_idx = shock_names.index(sh)
        perm[sh_idx] = best_col
        used_cols.add(best_col)

    # Fill remaining positions with unused columns in order
    remaining_cols = [c for c in range(n) if c not in used_cols]
    for sh_idx in range(n):
        if perm[sh_idx] == -1:
            perm[sh_idx] = remaining_cols.pop(0)

    B_perm = B[:, perm]
    eps_perm = eps[:, perm]
    return B_perm, eps_perm, perm


def _apply_sign_normalization(
    B: np.ndarray,
    eps: np.ndarray,
    variable_names: list[str],
    shock_names: list[str],
    sign_normalization: dict[str, dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Flip signs of columns of ``B`` (and rows of ``eps``) to match the
    requested contemporaneous-impact signs.

    For each ``shock_name -> {variable_name: ±1}`` entry, find the
    variable index and flip the column sign if needed.
    """
    B = B.copy()
    eps = eps.copy()
    for sh, var_signs in sign_normalization.items():
        if sh not in shock_names:
            raise ValueError(
                f"unknown shock {sh!r} in sign_normalization; "
                f"expected one of {shock_names}"
            )
        sh_idx = shock_names.index(sh)
        # Use the FIRST variable in the dict as the pinning variable
        for vname, target_sign in var_signs.items():
            if vname not in variable_names:
                raise ValueError(
                    f"unknown variable {vname!r} in sign_normalization "
                    f"for shock {sh!r}"
                )
            v_idx = variable_names.index(vname)
            current = B[v_idx, sh_idx]
            want = int(np.sign(target_sign))
            if want == 0:
                continue
            if current != 0 and int(np.sign(current)) != want:
                B[:, sh_idx] *= -1.0
                eps[:, sh_idx] *= -1.0
            break    # only the first variable pin per shock
    return B, eps


# ---------------------------------------------------------------------------
# Shock diagnostics
# ---------------------------------------------------------------------------


def _shock_diagnostics(eps: np.ndarray, shock_names: list[str]) -> dict[str, Any]:
    """Per-shock non-Gaussianity diagnostics.

    Returns a dict with:

    * ``jarque_bera``: {shock_name: {"stat": ..., "pvalue": ...}}
    * ``skewness``: {shock_name: float}
    * ``excess_kurtosis``: {shock_name: float}  (kurtosis - 3)
    * ``n_gaussian_like``: int — number of shocks whose JB p-value > 0.10
      (cannot reject Gaussianity at 10%).
    * ``weak_nongaussianity``: bool — True if TWO OR MORE shocks cannot
      reject Gaussianity. LMS identification permits AT MOST ONE Gaussian
      component (Assumption 1 / Proposition 1), so a single
      Gaussian-looking shock is admissible; and non-rejection is
      finite-sample evidence, not proof — treat the flag as an
      inconclusive / weak-identification caution, not a failure verdict.
    """
    jb_results: dict[str, dict[str, float]] = {}
    skew: dict[str, float] = {}
    kurt: dict[str, float] = {}
    for k, name in enumerate(shock_names):
        x = eps[:, k]
        jb_stat, jb_p = stats.jarque_bera(x)
        jb_results[name] = {"stat": float(jb_stat), "pvalue": float(jb_p)}
        skew[name] = float(stats.skew(x))
        kurt[name] = float(stats.kurtosis(x))     # Fisher-style (excess kurtosis)
    max_p = max(r["pvalue"] for r in jb_results.values())
    n_gaussian_like = sum(1 for r in jb_results.values() if r["pvalue"] > 0.10)
    return {
        "jarque_bera": jb_results,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "max_jb_pvalue": float(max_p),
        "n_gaussian_like": int(n_gaussian_like),
        "weak_nongaussianity": bool(n_gaussian_like >= 2),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def nongaussian_svar_irf(
    fit: VARFit,
    horizon: int,
    method: str = "fastica",
    nonlinearity: str = "tanh",
    shock_names: Sequence[str] | None = None,
    label_by: dict[str, str] | None = None,
    sign_normalization: dict[str, dict[str, int]] | None = None,
    bands: str = "none",
    n_draws: int = 1000,
    seed: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    pml_t_n_restarts: int = 3,
    pml_t_nu_init: float = 5.0,
) -> tuple[IRFContainer, dict[str, Any]]:
    """Non-Gaussian SVAR identification (Lanne-Meitz-Saikkonen 2017).

    Identifies the structural matrix :math:`B` in :math:`u_t = B \\varepsilon_t`
    up to column permutation and sign by exploiting non-Gaussianity of the
    structural shocks. No sign, zero, or external-instrument restrictions
    needed.

    Algorithm
    ---------
    Two-step:

    1. Pre-whiten reduced-form residuals via Cholesky:
       :math:`Z = L^{-1} u` with :math:`L = \\mathrm{chol}(\\Sigma_u)`.
    2. Find an orthogonal rotation :math:`Q` such that
       :math:`\\varepsilon = Q^\\top Z` are maximally non-Gaussian /
       independent. Two methods:

       * ``"fastica"`` — Hyvärinen (1999) fixed-point ICA. Fast,
         distribution-free, recovers components up to permutation +
         sign with no distributional assumption.
       * ``"pml_t"`` — LMS-style parametric t-ML (≈ LMS §4.5 step 2) with standardised
         Student-t innovations. Jointly estimates ``Q`` and per-shock
         degrees-of-freedom ``ν_j``. Warm-started from FastICA's
         solution, with additional random restarts.

    Recover :math:`B = L Q`.

    Parameters
    ----------
    fit : VARFit
    horizon : int
        IRF horizon. The container holds horizons 0..H.
    method : {"fastica", "pml_t"}
        Identification algorithm. ``"fastica"`` is the default —
        Hyvärinen (1999) fixed-point ICA, distribution-free.
        ``"pml_t"`` runs LMS-style parametric t-ML (≈ §4.5 step 2) with t-distributed
        innovations (additionally estimates per-shock degrees of freedom).
    nonlinearity : {"tanh", "cube", "exp"}
        FastICA non-Gaussianity contrast function. ``"tanh"`` is the robust
        default; ``"cube"`` is sharper for very heavy tails; ``"exp"`` is
        sometimes more stable. Ignored when ``method == "pml_t"``.
    shock_names : sequence of str, optional
        Names for the n identified shocks. Defaults to ``["shock_1", ...]``.
        Note: ICA recovers columns in arbitrary order; pair with
        ``label_by`` to assign columns to named shocks.
    label_by : dict, optional
        ``{shock_name: variable_name}`` greedy column assignment: each
        named shock is mapped to the unused column with the largest
        absolute contemporaneous impact on the named variable.
        Assignment proceeds strongest-signal-first, with the cross-shock
        priority computed on DIMENSIONLESS scores (each row's impacts
        divided by that row's norm) — a variable's unit convention can
        therefore never decide which independent component receives
        which economic name.
    sign_normalization : dict, optional
        ``{shock_name: {variable_name: ±1}}``. After column permutation,
        flip each column so the contemporaneous response of the named
        variable has the requested sign.
    bands : {"none", "residual_bootstrap"}
        ``"residual_bootstrap"`` IID-resamples the residuals
        ``n_draws`` times, re-runs ICA on each, and reports percentile
        bands. Convergence is ENFORCED per resample: an unconverged
        FastICA (or pml_t) run is retried with fresh starts up to a
        small bound, and if any resample still fails the call raises
        ``RuntimeError`` with attempted/failed counts — unfinished
        iterates are never included in the bands, and difficult
        resamples are never silently dropped. The diagnostics carry
        ``bootstrap_n_draws`` and ``bootstrap_n_retried_inits``. Captures identification-step uncertainty (the dominant
        source for LMS); does NOT re-fit reduced-form parameters
        (which is fine: reduced-form OLS is consistent by standard
        VAR theory regardless of non-Gaussianity — it is step 1 of
        LMS's §4.5 procedure, not their Theorem 1, which concerns the
        joint ML estimator).
    n_draws : int
        Number of bootstrap draws.
    seed : int, optional
        RNG seed.
    max_iter : int
        Iteration limit. For FastICA: fixed-point iterations. For
        pml_t: L-BFGS-B iterations per restart.
    tol : float
        Convergence tolerance. For FastICA: ``max |1 - |diag(W_new W')||``.
        For pml_t: L-BFGS-B ``ftol``.
    pml_t_n_restarts : int
        Only used when ``method == "pml_t"``. Number of random restarts
        in addition to the FastICA-warm-start. Default 3.
    pml_t_nu_init : float
        Only used when ``method == "pml_t"``. Initial value for each
        per-shock degrees-of-freedom ``ν_j`` at every restart. Default
        5.0 (moderate leptokurtic prior).

    Returns
    -------
    IRFContainer of shape ``(H+1, n_vars, n_shocks)`` with metadata.
    dict
        Identification diagnostics:

        * ``jarque_bera``: per-shock JB test (stat, p-value)
        * ``skewness``, ``excess_kurtosis``: per-shock sample moments
        * ``max_jb_pvalue``: worst-case (largest) p-value
        * ``n_gaussian_like`` / ``weak_nongaussianity``: see
          ``_shock_diagnostics`` — the flag fires only when two or more
          shocks cannot reject Gaussianity (one Gaussian component is
          admissible under the theorem)
        * ``fastica_converged``, ``fastica_n_iter``, ``fastica_final_diff``
          (always populated, even when ``method == "pml_t"``, since
          FastICA is used as the warm-start)
        * For ``method == "pml_t"`` additionally:
          ``pml_t_log_likelihood``, ``pml_t_nu`` (per-shock df
          estimates, ALIGNED with the final — relabelled — shock
          order), ``pml_t_converged``, ``pml_t_n_restarts_tried``,
          ``pml_t_best_restart_index``
        * ``B_point``: the point estimate of the structural matrix
          (after labelling + sign normalisation), shape ``(n, n)``.

    Notes
    -----
    LMS identification requires at most one Gaussian structural
    component. Always inspect ``diagnostics["jarque_bera"]``; treat
    ``weak_nongaussianity`` (two or more shocks cannot reject normality)
    as an inconclusive / weak-identification caution calling for
    additional evidence, not as an automatic rejection — non-rejection
    does not establish Gaussianity.

    The framework's ``identification`` metadata tag is
    ``"nongaussian_svar"`` so downstream code can detect the
    identification family.
    """
    n = fit.n_vars
    if shock_names is None:
        shock_names = [f"shock_{i + 1}" for i in range(n)]
    shock_names = list(shock_names)
    if len(shock_names) != n:
        raise ValueError(
            f"shock_names has {len(shock_names)} entries, expected {n}"
        )

    if method not in ("fastica", "pml_t"):
        raise ValueError(
            f"unknown method {method!r}; expected 'fastica' or 'pml_t'"
        )

    resid = np.asarray(fit.residuals)
    Z, L = whiten_residuals(resid)

    # Always run FastICA first — it's the warm-start for pml_t and the
    # method itself if `method == "fastica"`.
    W, ica_diag = fastica_symmetric(
        Z, nonlinearity=nonlinearity, max_iter=max_iter, tol=tol, seed=seed,
    )
    Q_fastica = W.T  # u_t = L z_t = L W' ε_t  →  Q = W'

    pml_t_extra_diag: dict[str, Any] = {}
    if method == "pml_t":
        Q, nu, pml_diag = pml_t_estimate(
            Z, init_Q=Q_fastica, n_restarts=pml_t_n_restarts,
            max_iter=max_iter, tol=tol, seed=seed, nu_init=pml_t_nu_init,
        )
        pml_t_extra_diag = {
            "pml_t_log_likelihood": pml_diag["log_likelihood"],
            "pml_t_nu": pml_diag["nu"],
            "pml_t_converged": pml_diag["converged"],
            "pml_t_n_restarts_tried": pml_diag["n_restarts_tried"],
            "pml_t_best_restart_index": pml_diag["best_restart_index"],
        }
        method_label = "LMS 2017 pml_t"
    else:
        Q = Q_fastica
        method_label = "LMS 2017 FastICA"

    eps = Z @ Q
    B_point = L @ Q

    # Column labelling
    if label_by:
        B_point, eps, _perm = _label_columns_by_impact(
            B_point, eps, shock_names, list(fit.variable_names), label_by,
        )
        # Every per-COMPONENT diagnostic must follow its column through the
        # relabeling: shock i now carries original column _perm[i], so its
        # tail parameter is nu[_perm[i]] — otherwise the printed per-shock
        # degrees of freedom describe the wrong shock.
        if "pml_t_nu" in pml_t_extra_diag:
            _nu = np.asarray(pml_t_extra_diag["pml_t_nu"], dtype=float)
            pml_t_extra_diag["pml_t_nu"] = [float(_nu[c]) for c in _perm]

    # Sign normalisation
    if sign_normalization:
        B_point, eps = _apply_sign_normalization(
            B_point, eps, list(fit.variable_names), shock_names,
            sign_normalization,
        )

    Phi = fit.ma_rep(horizon)
    irf_point = np.einsum("hij,jk->hik", Phi, B_point)

    diag = _shock_diagnostics(eps, shock_names)
    diag.update({
        "fastica_converged": ica_diag["converged"],
        "fastica_n_iter": ica_diag["n_iter"],
        "fastica_final_diff": ica_diag["final_diff"],
        "B_point": B_point.tolist(),
        **pml_t_extra_diag,
    })

    metadata: dict[str, Any] = {
        "method": f"VAR / non-Gaussian SVAR ({method_label})",
        "identification": "nongaussian_svar",
        "ica_method": method,
        "nonlinearity": nonlinearity if method == "fastica" else None,
        "n_lags": fit.p,
        "n_obs_effective": fit.t_eff,
        "bands": bands,
        "n_draws": n_draws if bands != "none" else 0,
        "seed": seed,
        "diagnostics": diag,
    }

    if bands == "none":
        return (
            IRFContainer.from_point(
                point=irf_point,
                variable_names=list(fit.variable_names),
                shock_names=shock_names,
                metadata=metadata,
            ),
            diag,
        )

    if bands != "residual_bootstrap":
        raise ValueError(
            f"unknown bands method {bands!r}; expected 'none' or "
            f"'residual_bootstrap'"
        )

    if label_by is None or sign_normalization is None:
        import warnings
        warnings.warn(
            "residual_bootstrap bands without label_by AND "
            "sign_normalization: ICA columns carry an arbitrary permutation "
            "and sign PER DRAW, so cross-draw percentile bands are "
            "meaningless unless each draw's shocks are re-labelled to the "
            "point estimate. Supply label_by={shock: variable} and "
            "sign_normalization to make the bands interpretable.",
            UserWarning, stacklevel=2,
        )

    # Residual bootstrap: re-resample residuals (preserving the empirical
    # non-Gaussian distribution), re-whiten, re-run ICA. Hold the lag
    # coefficients fixed at the point estimate — reduced-form OLS is
    # consistent by standard VAR theory independently of non-Gaussianity
    # (step 1 of LMS's §4.5 procedure), so the identification step's
    # sampling variance dominates.
    rng = np.random.default_rng(seed)
    draws_list: list[np.ndarray] = []
    # Convergence accounting: every resample must end in a CONVERGED
    # estimate or the call fails. FastICA signals an unfinished run through
    # its diagnostics (it does not raise), and an unfinished rotation is
    # not a valid uncertainty draw — silently including it corrupts the
    # bands, while silently dropping the difficult resample biases them.
    _MAX_TRIES_PER_DRAW = 3
    n_retried = 0
    failed_draws: list[str] = []
    for _ in range(n_draws):
        idx = rng.integers(0, len(resid), size=len(resid))
        resid_d = resid[idx]
        try:
            Z_d, L_d = whiten_residuals(resid_d)
        except np.linalg.LinAlgError:
            failed_draws.append("whitening: singular resampled covariance")
            continue
        # Reuse seed per draw deterministically so the bootstrap is
        # reproducible AND each draw has an independent FastICA init.
        # (Without this, all draws would start from the same orthogonal W
        # and could converge to the same fixed point.)
        W_d = None
        for _attempt in range(_MAX_TRIES_PER_DRAW):
            per_draw_seed = int(rng.integers(0, 2**31 - 1))
            try:
                W_try, ica_diag_d = fastica_symmetric(
                    Z_d, nonlinearity=nonlinearity, max_iter=max_iter,
                    tol=tol, seed=per_draw_seed,
                )
            except np.linalg.LinAlgError:
                continue
            if ica_diag_d["converged"]:
                W_d = W_try
                break
            n_retried += 1
        if W_d is None:
            failed_draws.append(
                f"FastICA did not converge in {_MAX_TRIES_PER_DRAW} attempts"
            )
            continue
        Q_d_fastica = W_d.T
        if method == "pml_t":
            # Bootstrap-specific optimisation: use the FastICA warm-start
            # ONLY (n_restarts=0). Random restarts in the bootstrap add
            # ~3-5x runtime per draw without changing the bands much —
            # the FastICA solution is already a valid local optimum
            # under any non-degenerate t-distribution log-likelihood, and
            # the headline pml_t call (above) already explores multiple
            # restarts to find the global optimum. Bootstrap variability
            # comes from residual resampling, not from the optimiser's
            # ability to find a different local optimum on each draw.
            # pml_t_estimate raises when no restart converges, so a
            # successful return is a converged draw. Retry the resample
            # with a fresh warm start before giving up on it.
            Q_d = None
            for _attempt in range(_MAX_TRIES_PER_DRAW):
                try:
                    Q_d, _nu_d, _ = pml_t_estimate(
                        Z_d, init_Q=Q_d_fastica, n_restarts=0,
                        max_iter=max_iter, tol=tol, seed=per_draw_seed,
                        nu_init=pml_t_nu_init,
                    )
                    break
                except (np.linalg.LinAlgError, RuntimeError):
                    Q_d = None
                    n_retried += 1
                    per_draw_seed = int(rng.integers(0, 2**31 - 1))
            if Q_d is None:
                failed_draws.append(
                    f"pml_t did not converge in {_MAX_TRIES_PER_DRAW} attempts"
                )
                continue
        else:
            Q_d = Q_d_fastica
        B_d = L_d @ Q_d
        eps_d = Z_d @ Q_d

        if label_by:
            B_d, eps_d, _ = _label_columns_by_impact(
                B_d, eps_d, shock_names, list(fit.variable_names), label_by,
            )
        if sign_normalization:
            B_d, eps_d = _apply_sign_normalization(
                B_d, eps_d, list(fit.variable_names), shock_names,
                sign_normalization,
            )
        draws_list.append(np.einsum("hij,jk->hik", Phi, B_d))

    if failed_draws:
        from collections import Counter
        reasons = "; ".join(f"{v}x {k}" for k, v in Counter(failed_draws).items())
        raise RuntimeError(
            f"non-Gaussian bootstrap: {len(failed_draws)} of {n_draws} "
            f"resamples did not produce a converged estimate ({reasons}). "
            f"Refusing to report bands that silently drop difficult "
            f"resamples — increase max_iter, loosen tol, or reduce the "
            f"model size."
        )
    if not draws_list:
        raise RuntimeError(
            f"residual bootstrap produced no usable draws "
            f"(all {n_draws} failed Cholesky or FastICA)"
        )

    draws = np.stack(draws_list, axis=0)
    diag["bootstrap_n_draws"] = int(len(draws_list))
    diag["bootstrap_n_retried_inits"] = int(n_retried)
    return (
        IRFContainer.from_draws(
            draws=draws,
            variable_names=list(fit.variable_names),
            shock_names=shock_names,
            metadata=metadata,
        ),
        diag,
    )
