"""Bank of England Interactive Database (iadb) — CSV download.

API: form-encoded CSV download at
``https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp``

The endpoint returns CSV in columnar or tabular formats (parameter ``CSVF``).
We use ``CSVF=TN`` (tabular, no titles) which produces ``date,value`` rows.

Series spec shape passed via ``lib.fetch_data(api="boe_iadb", series=...)``::

    series = {"rate": "IUMABEDR", "m4": "LPMAUYN", ...}

Up to ~300 series codes can be requested in one URL, but we issue one call per
alias for simplicity and clearer error attribution. Dates default to a wide
window (1900–2100) and the load-time sample filter trims to the requested
range — the cache holds the full available history regardless of the request.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Mapping
from urllib import parse, request
from urllib.error import HTTPError, URLError

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher

_BASE_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"

# Default sample window — chosen wide enough for typical macro work but within
# the BoE API's accepted range. (Date bounds outside ~1970-2050 cause the API
# to return an HTML error page rather than CSV.)
_DEFAULT_DATE_FROM = "01/Jan/1970"
_DEFAULT_DATE_TO = "01/Jan/2050"


def _format_boe_date(d: str | pd.Timestamp | None, fallback: str) -> str:
    """Render a date as ``DD/Mon/YYYY`` for the BoE URL."""
    if d is None:
        return fallback
    try:
        ts = pd.Timestamp(d)
    except (ValueError, TypeError):
        return fallback
    return ts.strftime("%d/%b/%Y")


class BoeIadbFetcher(BaseFetcher):
    """Fetcher for the Bank of England Interactive Database."""

    name = "boe_iadb"
    description = "Bank of England Interactive Database (iadb)"
    endpoint = _BASE_URL

    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for alias, spec in series.items():
            try:
                code = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"BoE iadb series spec for {alias!r} must be a bare "
                    f"code string; got {spec!r}: {exc}"
                ) from exc

            params = {
                "Datefrom": _format_boe_date(start, _DEFAULT_DATE_FROM),
                "Dateto":   _format_boe_date(end,   _DEFAULT_DATE_TO),
                "SeriesCodes": code,
                "CSVF": "TN",          # tabular, no titles
                "UsingCodes": "Y",
                "VPD": "Y",
                "VFD": "N",
                "csv.x": "yes",
            }
            url = f"{_BASE_URL}?{parse.urlencode(params)}"
            try:
                text = self._http_get_text(url)
            except (HTTPError, URLError) as exc:
                import warnings
                warnings.warn(
                    f"BoE iadb fetch failed for alias={alias!r} (code={code}, "
                    f"url={url}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ser = self._extract_series(text)
            # BoE iadb publishes monthly series with end-of-month timestamps
            # (1992-10-31, 1992-11-30, ...). Normalise to start-of-month so
            # the joined frame aligns with FRED/ONS conventions — but ONLY
            # for a genuinely monthly series: collapsing daily/weekly data
            # to month starts would fabricate duplicate dates, and
            # quarterly data would be mislabelled as monthly observations.
            _month_idx = ser.index.to_period("M")
            if _month_idx.has_duplicates:
                raise ValueError(
                    f"boe_iadb: series {alias!r} has multiple observations "
                    f"per month (daily/weekly data) — fetch it at its "
                    f"native frequency; the monthly normalisation would "
                    f"collapse distinct dates."
                )
            # Monthly spacing must be established before relabelling; a
            # single observation cannot establish it.
            if len(_month_idx) == 1:
                raise ValueError(
                    f"boe_iadb: series {alias!r} has a single observation, so "
                    f"its frequency cannot be established; the monthly "
                    f"normalisation needs at least two."
                )
            if len(_month_idx) >= 2:
                _gaps = pd.Series(_month_idx.astype("int64")).diff().dropna()
                if _gaps.median() != 1:
                    raise ValueError(
                        f"boe_iadb: series {alias!r} is not monthly "
                        f"(median spacing {_gaps.median():.0f} months) — "
                        f"fetch it at its native frequency instead of "
                        f"relabelling it as monthly."
                    )
            ser.index = _month_idx.to_timestamp()
            ser.name = alias
            if start is not None:
                ser = ser.loc[pd.Timestamp(start):]
            if end is not None:
                ser = ser.loc[:pd.Timestamp(end)]
            out[alias] = ser
        return out

    # -------- helpers --------

    @staticmethod
    def _parse_spec(spec: Any) -> str:
        if isinstance(spec, str):
            return spec
        raise ValueError("expected a bare string series code")

    @staticmethod
    def _http_get_text(url: str, timeout: int = 60) -> str:
        req = request.Request(url, headers={
            "Accept": "text/csv,text/plain,*/*",
            "User-Agent": "Mozilla/5.0 (compatible; marco-bis/1.0)",
        })
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")

    @staticmethod
    def _extract_series(text: str) -> pd.Series:
        """Parse a BoE iadb CSV response into a Series.

        With ``CSVF=TN`` the response is::

            DATE,SERIES_CODE
            31 Jan 2020,0.75
            29 Feb 2020,0.75
            ...

        BoE iadb uses "DD Mon YYYY" with spaces (sometimes also "DD/Mon/YYYY"
        in older docs). The parser tries both. If the response is HTML — which
        happens when the request is malformed (e.g., dates out of the API's
        accepted range, ~1970-2050) — we detect that and give a clear error
        instead of "no parseable observations".
        """
        # Detect HTML error pages from the BoE site
        if text.lstrip().lower().startswith("<!doctype html") or "<html" in text[:200].lower():
            raise RuntimeError(
                "BoE iadb returned an HTML error page (likely because the "
                "request was rejected — bad series code, dates out of range, "
                "or temporary server issue). First 200 chars of response:\n"
                + text[:200]
            )

        reader = csv.reader(io.StringIO(text))
        dates: list[pd.Timestamp] = []
        values: list[float] = []
        for row in reader:
            if len(row) < 2:
                continue
            date_str, value_str = row[0].strip(), row[1].strip()
            ts = _try_parse_boe_date(date_str)
            if ts is None:
                continue  # header row or malformed
            try:
                v = float(value_str)
            except (ValueError, TypeError):
                continue
            dates.append(ts)
            values.append(v)
        if not dates:
            raise ValueError("BoE iadb CSV response had no parseable observations")
        return pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()


def _try_parse_boe_date(s: str) -> pd.Timestamp | None:
    """Try BoE iadb's ``DD/Mon/YYYY`` (then fall back to permissive parsing).

    Returns None if the string isn't a parseable date (typically the
    ``DATE`` header row).
    """
    for fmt in ("%d/%b/%Y", "%d/%B/%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, dayfirst=True))
    except (ValueError, TypeError):
        return None


register_fetcher(BoeIadbFetcher.name, BoeIadbFetcher)
