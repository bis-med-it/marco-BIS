"""Shared test fixtures.

The tests deliberately use synthetic data with known properties wherever
possible: that way passing tests prove the library does the right math,
not that it agrees with an in-house reference implementation.

When a fixture references real FRED data, the test is skipped if the
data file isn't present in the worktree — CI-friendly default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable for `import lib`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Shared VAR(2) DGP parameters — single source of truth for the
# synthetic_var_data / synthetic_var_dgp fixtures below and for the
# Monte Carlo DGP in tests/test_coverage.py.
# ---------------------------------------------------------------------------

VAR_DGP_A1 = np.array([
    [0.55, 0.10, 0.05],
    [0.08, 0.60, 0.07],
    [0.04, 0.06, 0.62],
])
VAR_DGP_A2 = np.array([
    [0.15, 0.00, 0.02],
    [0.00, 0.10, 0.00],
    [0.02, 0.00, 0.08],
])
VAR_DGP_SIGMA = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.2],
    [0.1, 0.2, 1.0],
])


@pytest.fixture(scope="session")
def synthetic_var_data() -> pd.DataFrame:
    """Three-variable VAR(2) with a known DGP.

    DGP:
        y_t = A_1 y_{t-1} + A_2 y_{t-2} + u_t,  u_t ~ N(0, Sigma)

    Parameters chosen so the system is stationary and the cross-correlation
    is non-trivial (so identification routines have something to do).
    """
    rng = np.random.default_rng(123)
    n = 3
    T = 400
    burn = 100

    A1 = VAR_DGP_A1
    A2 = VAR_DGP_A2
    L = np.linalg.cholesky(VAR_DGP_SIGMA)
    Y = np.zeros((T + burn, n))
    for t in range(2, T + burn):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + L @ rng.standard_normal(n)
    Y = Y[burn:]
    idx = pd.period_range("1980Q1", periods=T, freq="Q").to_timestamp()
    return pd.DataFrame(Y, columns=["y1", "y2", "y3"], index=idx)


@pytest.fixture(scope="session")
def synthetic_var_dgp() -> dict:
    """Return the DGP parameters used by `synthetic_var_data`."""
    return {
        "A1": VAR_DGP_A1,
        "A2": VAR_DGP_A2,
        "Sigma": VAR_DGP_SIGMA,
        "n": 3,
        "p": 2,
    }


@pytest.fixture(scope="session")
def shock_series(synthetic_var_data) -> np.ndarray:
    """A deterministic synthetic shock series aligned with the synthetic VAR
    data. Used to test LP estimation. NOT an actual structural shock — just
    an exogenous regressor for the projection."""
    rng = np.random.default_rng(456)
    return rng.standard_normal(len(synthetic_var_data))


# ---------------------------------------------------------------------------
# Slow-test opt-in (used by tests/test_replications_smoke.py)
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add a --runslow flag for opt-in slow tests."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (e.g. tests/test_replications_smoke.py — "
             "full end-to-end replication runs, can take ~10 minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (skip by default; enable with --runslow)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow-marked tests unless --runslow was passed."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(
        reason="slow test — pass --runslow to enable"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
