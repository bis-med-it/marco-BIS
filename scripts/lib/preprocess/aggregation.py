"""Frequency aggregation: collapse higher-frequency data to a lower frequency.

The choice of aggregation method is per-variable and reflects whether the
series is a *flow* (sum), *rate* (mean), or *stock* (end-of-period level).
A mixed-frequency macro panel typically needs different methods for different
columns — e.g., monthly industrial production summed to quarterly, monthly
federal funds rate averaged, monthly M1 stock taken at quarter-end.

Thin wrapper around ``pd.DataFrame.resample(...).agg(...)`` with:

* per-column method dicts validated against ``df.columns``
* ``"end"`` / ``"start"`` semantic aliases for ``"last"`` / ``"first"``
* ``df.attrs`` preserved plus a ``preprocess_applied`` entry appended so
  the research log can trace what was done

Upsampling (lower → higher frequency, e.g., Q → M) is out of scope; the
library has no disaggregation method (see ``preprocess/README.md``).
"""

from __future__ import annotations

from typing import Mapping, Union

import pandas as pd

_METHOD_ALIASES = {"end": "last", "start": "first"}
_VALID_METHODS = frozenset(
    {"sum", "mean", "median", "first", "last", "min", "max", "std", "end", "start"}
)


def aggregate(
    df: Union[pd.Series, pd.DataFrame],
    target_freq: str,
    method: Union[str, Mapping[str, str]],
    allow_partial: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Aggregate a higher-frequency series/DataFrame to a lower frequency.

    Typical use: collapse monthly observations to quarterly. The method
    differs by variable type:

    * ``"sum"`` — flow variables (monthly retail sales → quarterly retail sales)
    * ``"mean"`` — rates and indices (monthly FFR → quarterly average FFR)
    * ``"end"`` (alias for ``"last"``) — stock variables at period end
      (monthly M1 level → end-of-quarter M1)
    * ``"start"`` (alias for ``"first"``) — stock at period beginning
    * ``"min"``, ``"max"``, ``"median"``, ``"std"`` — pandas pass-through

    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        Input data with a ``DatetimeIndex`` at the source frequency.
    target_freq : str
        Pandas frequency alias for the target. Examples (pandas ≥ 2.2):
        ``"QS"`` quarter-start, ``"QE"`` quarter-end, ``"YS"`` year-start,
        ``"ME"`` month-end. Legacy aliases (``"Q"``, ``"M"``, ``"A"``) also
        work but emit pandas FutureWarnings.
    method : str or Mapping[str, str]
        Aggregation method. For a Series, must be a string. For a DataFrame,
        either a string (applied to every column) or a dict mapping each
        column to its method. Dict keys must exactly match ``df.columns``.
    allow_partial : bool, default False
        By default, a target bin that is INCOMPLETE for a column — missing
        values, missing calendar rows, or a truncated boundary period — is
        returned as NaN rather than aggregated: a two-month "quarterly
        total" is not a quarterly total, and ``"end"`` on a bin whose last
        scheduled observation is missing would silently substitute an
        earlier value. Pass ``True`` to aggregate whatever observations a
        bin has (requires at least one): ``sum``/``mean`` become partial
        aggregates and ``"end"``/``"start"`` mean last/first AVAILABLE.
        Completeness is checked per column against the source calendar
        (inferred frequency), extended to whole target periods at the
        sample boundaries.

    Returns
    -------
    pd.Series or pd.DataFrame
        Aggregated output at ``target_freq``. Series ``name`` is preserved.
        DataFrame column order is preserved. ``df.attrs`` is copied to the
        result and a ``preprocess_applied`` entry recording the operation
        is appended.

    Raises
    ------
    TypeError
        If ``df.index`` is not a ``DatetimeIndex``, or if a Series is passed
        with a non-string ``method``.
    ValueError
        If ``method`` (as a dict) is missing or has extra keys relative to
        ``df.columns``, or if any method name is not recognised.

    Examples
    --------
    Monthly → quarter-start, all columns averaged::

        >>> q = lib.preprocess.aggregate(monthly_df, "QS", "mean")

    Per-column methods (mixed flow / rate / stock)::

        >>> q = lib.preprocess.aggregate(
        ...     monthly_df, "QS",
        ...     {"retail_sales": "sum", "ffr": "mean", "m1": "end"},
        ... )
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"aggregate requires a DatetimeIndex, got {type(df.index).__name__}"
        )

    is_series = isinstance(df, pd.Series)

    if is_series:
        if not isinstance(method, str):
            raise TypeError(
                "Series input requires a string method, "
                f"got {type(method).__name__}"
            )
        result = df.resample(target_freq).agg(_normalize_method(method))
    elif isinstance(method, str):
        result = df.resample(target_freq).agg(_normalize_method(method))
    else:
        missing = set(df.columns) - set(method)
        extra = set(method) - set(df.columns)
        if missing:
            raise ValueError(
                f"method dict is missing entries for columns: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"method dict has entries for unknown columns: {sorted(extra)}"
            )
        normalized = {col: _normalize_method(m) for col, m in method.items()}
        result = df.resample(target_freq).agg(normalized)
        result = result[list(df.columns)]

    # --- Completeness: don't let missing/partial bins look like data ---
    # Per-column observation counts per bin, and the number of source
    # periods each bin SHOULD contain (from the inferred source calendar,
    # extended to whole target periods at the boundaries, so a truncated
    # first/last bin counts as incomplete rather than complete-looking).
    frame = df.to_frame() if is_series else df
    obs_counts = frame.notna().resample(target_freq).sum()
    expected = _expected_slots(frame.index, target_freq)
    if expected is not None:
        obs_counts, expected = obs_counts.align(expected, axis=0,
                                                join="left", fill_value=0)
        complete = obs_counts.eq(expected, axis=0)
    else:
        # Irregular source calendar: completeness is undecidable. Without an
        # explicit opt-in, refuse rather than fabricate complete-looking bins.
        if not allow_partial:
            raise ValueError(
                "aggregate: cannot infer the source frequency from the "
                "index, so bin completeness cannot be validated. Regularise "
                "the input calendar or pass allow_partial=True to aggregate "
                "whatever observations each bin has."
            )
        complete = obs_counts > 0
    if allow_partial:
        mask = obs_counts > 0           # at least one observation per bin
    else:
        mask = complete
    if is_series:
        result = result.where(mask.iloc[:, 0].reindex(result.index,
                                                      fill_value=False))
    else:
        result = result.where(mask.reindex(result.index, fill_value=False))

    if df.attrs:
        result.attrs = dict(df.attrs)
        applied = list(result.attrs.get("preprocess_applied", []))
        applied.append(
            {"op": "aggregate", "target_freq": target_freq, "method": method}
        )
        result.attrs["preprocess_applied"] = applied

    if is_series and df.name is not None:
        result.name = df.name

    return result


def _expected_slots(idx: pd.DatetimeIndex, target_freq: str):
    """Number of source-calendar periods each target bin should contain.

    Builds the full source calendar from the inferred source frequency,
    extended to WHOLE target periods at both boundaries (so a sample that
    stops mid-quarter marks that quarter incomplete). Returns a DataFrame-
    broadcastable Series indexed like the resampled output, or ``None``
    when the source frequency cannot be inferred.
    """
    src = pd.infer_freq(idx)
    if src is None:
        return None
    try:
        step = pd.tseries.frequencies.to_offset(src)
        # Extend the source calendar well past both ends so the sample's
        # boundary bins are INTERIOR bins of the extended calendar and
        # therefore carry their full expected slot count (400 source slots
        # cover any bin up to daily-to-annual).
        ext = pd.date_range(idx.min() - 400 * step, idx.max() + 400 * step,
                            freq=src)
        expected_all = pd.Series(1.0, index=ext).resample(target_freq).count()
        sample_bins = pd.Series(1.0, index=idx).resample(target_freq).count().index
        return expected_all.reindex(sample_bins)
    except (ValueError, TypeError):
        return None


def _normalize_method(m: str) -> str:
    """Map ``"end"`` → ``"last"``, ``"start"`` → ``"first"``; validate name."""
    if m not in _VALID_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{m}'. Valid: {sorted(_VALID_METHODS)}"
        )
    return _METHOD_ALIASES.get(m, m)
