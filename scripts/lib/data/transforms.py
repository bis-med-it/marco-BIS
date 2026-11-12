"""Per-column transforms for the data layer.

The decision of which transform to apply belongs with the model that consumes
the data, not with the loader. ``lib.transform`` applies the chosen transforms
*after* ``lib.load_dataset`` returns raw, structurally-clean data.

Transform names align with the FRED-MD codes (1=level, 2=diff, 4=log, 5=dlog,
6=ddlog, 7=pct_change_diff, the first difference of proportional growth), plus scaled-percent variants (``*_x_100``, ``*_x_400``)
that are common across the replication gallery.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

#: Mapping of transform name -> a function ``pd.Series -> pd.Series``.
TRANSFORM_REGISTRY: dict[str, Callable[[pd.Series], pd.Series]] = {
    # Standard FRED-MD codes
    "level": lambda s: s,
    "none": lambda s: s,
    "log": lambda s: np.log(s),
    "diff": lambda s: s.diff(),
    "log_diff": lambda s: np.log(s).diff(),
    "dlog": lambda s: np.log(s).diff(),
    "diff_log_diff": lambda s: np.log(s).diff().diff(),
    "ddlog": lambda s: np.log(s).diff().diff(),
    "pct_change": lambda s: s.pct_change(),
    # Scaled variants (×100 = percent units; ×400 = annualised percent for quarterly)
    "log_x_100": lambda s: np.log(s) * 100.0,
    "log_pct": lambda s: np.log(s) * 100.0,
    "dlog_x_100": lambda s: np.log(s).diff() * 100.0,
    "dlog_pct": lambda s: np.log(s).diff() * 100.0,
    "log_diff_pct": lambda s: np.log(s).diff() * 100.0,
    "dlog_x_400": lambda s: np.log(s).diff() * 400.0,
    "dlog_ann_q": lambda s: np.log(s).diff() * 400.0,
    # Absolute-value variants (volatility proxies — Diebold-Yilmaz 2012)
    "abs_diff": lambda s: s.diff().abs(),
    "abs_log_diff": lambda s: np.log(s).diff().abs(),
    # Official FRED-MD/QD codes 3 and 7 (McCracken-Ng methodology, §2):
    # 3 = second difference of levels; 7 = first difference of percentage
    # growth. (The generic "pct_change" keeps its existing meaning.)
    "diff2": lambda s: s.diff().diff(),
    "pct_change_diff": lambda s: s.pct_change().diff(),
}

#: FRED-MD transform codes → registry names (McCracken-Ng methodology, §2).
FRED_MD_CODE_TO_NAME: dict[int, str] = {
    1: "level",
    2: "diff",
    3: "diff2",            # second difference of levels
    4: "log",
    5: "log_diff",
    6: "diff_log_diff",
    7: "pct_change_diff",  # first difference of percentage growth
}


def transform(
    df: pd.DataFrame,
    transforms: Mapping[str, str] | str,
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Apply per-column transforms to a DataFrame.

    The data layer's job is to deliver structured raw data. The transform
    decision (log, diff, log_diff, ...) belongs with the model that consumes
    the data — this function is the seam.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, typically the output of :func:`lib.load_dataset`.
    transforms : dict or str
        If dict: ``{column_name: transform_name}``. Columns not listed are kept
        at level (no-op).
        If str: applied uniformly to all columns.
        Available names: see :data:`TRANSFORM_REGISTRY`.
    dropna : bool, default True
        Drop rows containing any NaN after transforms. Differencing
        transforms legitimately create LEADING NaNs; note that dropping
        also removes any INTERNAL NaN row a transform produces (e.g.
        ``log`` of a non-positive value), which silently breaks calendar
        lags — the spec pipeline guards this with a contiguous-suffix
        check; direct callers who suspect internal invalid values should
        pass ``dropna=False`` and inspect.

    Returns
    -------
    pd.DataFrame
        Same columns as input, values transformed. ``df.attrs`` is carried
        forward and extended with ``"transforms_applied"`` (the resolved
        per-column transform map).

    Raises
    ------
    ValueError
        If a transform name is not in :data:`TRANSFORM_REGISTRY`.

    Examples
    --------
    >>> raw = lib.load_dataset(source="fred_qd",
    ...                        series={"gdp": "GDPC1", "unrate": "UNRATE"},
    ...                        start="1950", end="1987")
    >>> data = lib.transform(raw, {"gdp": "dlog_x_100", "unrate": "level"})
    """
    if isinstance(transforms, str):
        transform_map = {col: transforms for col in df.columns}
    else:
        transform_map = {col: transforms.get(col, "level") for col in df.columns}

    out: dict[str, pd.Series] = {}
    for col in df.columns:
        name = transform_map[col]
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform {name!r} for column {col!r}. "
                f"Available: {sorted(TRANSFORM_REGISTRY)}"
            )
        out[col] = TRANSFORM_REGISTRY[name](df[col])

    result = pd.DataFrame(out, index=df.index)
    # Carry forward provenance; record what we applied
    result.attrs = dict(df.attrs)
    result.attrs["transforms_applied"] = dict(transform_map)

    if dropna:
        result = result.dropna()
    return result
