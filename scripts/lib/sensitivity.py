"""Hyperparameter-sweep driver over :func:`lib.run`.

Replaces the per-replication custom sensitivity loops (BGR's brentq
calibration over ``lambda_1``, GLP's Nelder-Mead over the prior block,
AG2012's gamma sweep, Litterman's prior-tightness comparison) with a
uniform interface:

>>> base = lib.Spec.from_replication("litterman_1986_bvar").update(forecast=None)
>>> report = lib.sensitivity(base, grid={
...     "model.prior.lambda_1": [0.05, 0.1, 0.2, 0.5],
... })
>>> report.table[["model.prior.lambda_1", "max_eig"]]

The grid is a dict mapping **dotted paths** (``"model.prior.lambda_1"``,
``"irf.horizon"``) to lists of values. Combinations are taken as the
Cartesian product. Each combination is applied via
:meth:`Spec.update` and executed via :func:`lib.run`; failures are
captured as ``error`` strings in the table without killing the sweep.

For custom metrics beyond the default ``max_eig`` / ``aic`` / ``bic`` /
``hqic`` / ``log_mdd``, pass a ``metric_fn(result) -> dict`` and its
keys become extra columns in the report table.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass
class SensitivityReport:
    """Structured output of :func:`sensitivity`.

    Attributes
    ----------
    base_spec : Spec
        The starting point of the sweep.
    grid : dict[str, list]
        The grid that was swept (dotted-path -> list of values).
    table : pd.DataFrame
        One row per (grid combination, metric) — the grid values plus the
        extracted metrics. ``error`` is populated when a row failed; other
        metric columns are NaN in that case.
    results : list[SpecResult | None]
        Aligned with ``table.index``: the full SpecResult for each row, or
        ``None`` if the run failed. Empty list when ``keep_results=False``
        in the original call.

    Methods
    -------
    best(by, ascending=True) : SpecResult
        Return the SpecResult corresponding to the best row of the table
        on the given metric column.
    """

    base_spec: Any
    grid: dict[str, list]
    table: pd.DataFrame
    results: list[Any] = field(default_factory=list)

    def best(self, by: str, ascending: bool = True) -> Any:
        """Return the SpecResult for the row minimising (or maximising) ``by``.

        Raises ``ValueError`` if ``keep_results=False`` was used, or if the
        metric column doesn't exist.
        """
        if not self.results:
            raise ValueError(
                "results list is empty; pass keep_results=True to sensitivity() "
                "to retain SpecResult objects for .best() lookups."
            )
        if by not in self.table.columns:
            raise ValueError(
                f"metric {by!r} not in table columns: {list(self.table.columns)}"
            )
        valid = self.table.dropna(subset=[by])
        if valid.empty:
            raise ValueError(f"no non-NaN rows for metric {by!r}")
        idx = valid[by].idxmin() if ascending else valid[by].idxmax()
        return self.results[idx]


# ---------------------------------------------------------------------------
# sensitivity()
# ---------------------------------------------------------------------------


def sensitivity(
    base_spec,
    grid: dict[str, Sequence],
    metric_fn: Callable[[Any], dict] | None = None,
    seed: int = 42,
    keep_results: bool = True,
) -> SensitivityReport:
    """Run a Cartesian-product sweep over a base Spec.

    For each combination of grid values: apply them to ``base_spec`` via
    deep-merged ``update`` calls (dotted paths are converted to nested
    dicts), execute ``lib.run``, extract metrics, and accumulate.

    Parameters
    ----------
    base_spec : Spec
        Starting spec — all rows in the sweep start from this and diverge
        only on the grid values.
    grid : dict[str, sequence]
        ``{"model.prior.lambda_1": [0.05, 0.1, 0.2], "irf.horizon": [12, 20]}``
        sweeps the Cartesian product (here 3 × 2 = 6 combinations). Dotted
        paths are parsed into nested dicts and applied via ``spec.update``.
    metric_fn : callable, optional
        ``metric_fn(result: SpecResult) -> dict[str, scalar]``. The keys
        become extra columns in the report table. The default extracts
        ``max_eig``, ``aic``, ``bic``, ``hqic``, and ``log_mdd`` (when
        present) from the fit.
    seed : int, default 42
        Seed forwarded to ``lib.run`` (subject to its own override rules).
    keep_results : bool, default True
        If True (default), retain the list of SpecResult objects for later
        inspection (e.g., ``report.best(by="log_mdd")``). Set to False on
        large grids to avoid memory bloat.

    Returns
    -------
    SensitivityReport

    Raises
    ------
    ValueError
        If ``grid`` is empty or contains a non-sequence value.
    """
    # Local import to avoid module-load cycle with lib.specs / lib.run
    import lib
    from lib.specs import Spec

    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty dict of {dotted_path: values}")
    for path, vals in grid.items():
        if not isinstance(vals, Sequence) or isinstance(vals, str):
            raise ValueError(
                f"grid[{path!r}] must be a list/tuple of values, "
                f"got {type(vals).__name__}"
            )

    if metric_fn is None:
        metric_fn = _default_metric_fn

    paths = list(grid.keys())
    value_grids = [list(grid[p]) for p in paths]

    rows: list[dict[str, Any]] = []
    results: list[Any] = []

    for combo in itertools.product(*value_grids):
        # Build nested override dict from dotted paths
        overrides = {}
        for path, value in zip(paths, combo):
            _set_nested(overrides, path, value)

        # Apply to base_spec and run
        spec = base_spec.update(**overrides)
        row: dict[str, Any] = {p: v for p, v in zip(paths, combo)}
        try:
            result = lib.run(spec, seed=seed)
            metrics = metric_fn(result)
            row.update(metrics)
            row["error"] = None
            results.append(result if keep_results else None)
        except Exception as exc:
            row["error"] = str(exc)
            results.append(None)
        rows.append(row)

    table = pd.DataFrame(rows)
    if "error" in table.columns:
        # keep the "None on success" contract under pandas>=3, whose str
        # dtype would otherwise coerce None to NaN in mixed columns
        table["error"] = table["error"].astype(object).where(
            table["error"].notna(), None
        )
    return SensitivityReport(
        base_spec=base_spec,
        grid=dict(grid),
        table=table,
        results=results if keep_results else [],
    )


# ---------------------------------------------------------------------------
# Default metric extractor
# ---------------------------------------------------------------------------


def _default_metric_fn(result) -> dict[str, float]:
    """Default metrics extracted from a SpecResult.

    Picks up: ``max_eig`` (max companion eigenvalue), ``aic``, ``bic``,
    ``hqic`` (from ``fit.fit_info`` when present), and ``log_mdd`` (for
    BVARFit via ``lib.log_marginal_data_density``). NaN when a metric
    isn't applicable (e.g., log_mdd on a plain VAR).
    """
    import lib

    fit = result.fit
    metrics: dict[str, float] = {}
    try:
        metrics["max_eig"] = float(lib.max_eigenvalue_modulus(fit.lag_matrices()))
    except Exception:
        metrics["max_eig"] = float("nan")
    fi = getattr(fit, "fit_info", {}) or {}
    for k in ("aic", "bic", "hqic"):
        v = fi.get(k, float("nan"))
        try:
            metrics[k] = float(v)
        except (TypeError, ValueError):
            metrics[k] = float("nan")
    # BVAR-specific marginal data density
    if type(fit).__name__ == "BVARFit":
        try:
            metrics["log_mdd"] = float(lib.log_marginal_data_density(fit))
        except Exception:
            metrics["log_mdd"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_nested(d: dict, dotted_path: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` from a dotted path ``"a.b.c"``.

    Creates intermediate dicts as needed. Mutates ``d`` in place.
    """
    parts = dotted_path.split(".")
    if not parts:
        raise ValueError(f"empty dotted path")
    cursor = d
    for p in parts[:-1]:
        if p not in cursor or not isinstance(cursor[p], dict):
            cursor[p] = {}
        cursor = cursor[p]
    cursor[parts[-1]] = value
