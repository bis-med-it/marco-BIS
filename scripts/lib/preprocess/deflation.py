"""Nominal / real conversions via a price deflator.

The price-deflator pattern is common in monetary VARs (Sims 1980, CEE 1999,
GLP 2015) which want *nominal* M1 in the system, but FRED provides only
*real* M1 (M1REAL). Going the other way (nominal → real) is the workhorse
for converting nominal-currency variables to constant-base units.

These are stateless ``pd.Series -> pd.Series`` functions. The transform
*decision* (which series is nominal, which is real, which base period to use)
belongs with the model — these functions just do the arithmetic.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

PeriodLike = Union[str, pd.Timestamp]


def deflate(
    nominal: pd.Series,
    deflator: pd.Series,
    base_period: Optional[PeriodLike] = None,
    deflator_scale: float = 100.0,
) -> pd.Series:
    """Convert a nominal series to real by dividing by a price deflator.

    By convention price deflators are indices with a base value of
    ``deflator_scale`` (typically 100) at some reference period — e.g.,
    FRED's GDPCTPI is base 2017=100. The output is expressed in that same
    base-period currency unit, unless ``base_period`` rebases the deflator
    to a different reference.

    Parameters
    ----------
    nominal : pd.Series
        The nominal series to deflate (e.g., nominal GDP, nominal M1).
    deflator : pd.Series
        The price index, aligned to the same DatetimeIndex as ``nominal``.
    base_period : str or pd.Timestamp, optional
        If given, rebases the deflator to ``deflator_scale`` at this period
        so the output is in base-period currency. If None, uses the
        deflator's native base.
    deflator_scale : float, default 100.0
        The deflator's base value (FRED indices are typically 100).

    Returns
    -------
    pd.Series
        The deflated (real) series. Inherits ``nominal.name`` if set.

    Examples
    --------
    Deflate nominal M1 by the GDP deflator (FRED default 2017=100)::

        >>> real_m1 = lib.preprocess.deflate(raw["m1_nominal"], raw["gdpctpi"])

    Rebase to 2010-Q1 dollars::

        >>> real_m1_2010 = lib.preprocess.deflate(
        ...     raw["m1_nominal"], raw["gdpctpi"], base_period="2010-01-01"
        ... )
    """
    if base_period is not None:
        deflator = deflator / deflator.loc[base_period] * deflator_scale
    result = nominal * deflator_scale / deflator
    if nominal.name is not None:
        result.name = nominal.name
    return result


def nominalise(
    real: pd.Series,
    deflator: pd.Series,
    base_period: Optional[PeriodLike] = None,
    deflator_scale: float = 100.0,
) -> pd.Series:
    """Convert a real series to nominal by multiplying by a price deflator.

    Inverse of :func:`deflate`. Used in the monetary-VAR pattern where the
    source provides ``M1REAL`` but the model wants nominal M1.

    Parameters
    ----------
    real : pd.Series
        The real series to nominalise (e.g., real M1 in base-year dollars).
    deflator : pd.Series
        The price index, aligned to ``real``'s DatetimeIndex.
    base_period : str or pd.Timestamp, optional
        If given, rebases the deflator to ``deflator_scale`` at this period.
    deflator_scale : float, default 100.0
        The deflator's base value.

    Returns
    -------
    pd.Series
        The nominal series. Inherits ``real.name`` if set.

    Examples
    --------
    Compute nominal M1 from FRED's real M1 and the CPI::

        >>> nominal_m1 = lib.preprocess.nominalise(raw["m1_real"], raw["cpiaucsl"])

    Use the SAME index that built the real series: FRED defines ``M1REAL``
    as ``M1SL / CPIAUCSL`` (1982-84 dollars), so only CPI reverses it —
    multiplying by the GDP deflator instead would leave a relative-price
    component (``GDPCTPI / CPIAUCSL``) inside the money series. At a lower
    observation frequency the reversal is approximate (products of period
    averages are not averages of products).
    """
    if base_period is not None:
        deflator = deflator / deflator.loc[base_period] * deflator_scale
    result = real * deflator / deflator_scale
    if real.name is not None:
        result.name = real.name
    return result
