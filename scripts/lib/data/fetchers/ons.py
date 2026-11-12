"""UK Office for National Statistics — time-series API.

JSON data is served by appending ``/data`` to any ONS time-series page URL.
The page lives at ``https://www.ons.gov.uk/<category_path>/timeseries/<series>/<dataset>``
where ``<category_path>`` is series-specific. For example:

- K222 (Industrial Production) → ``economy/economicoutputandproductivity/output``, dataset ``diop``
- D7BT (CPI) → ``economy/inflationandpriceindices``, dataset ``mm23``
- IHYR (real GDP) → ``economy/grossdomesticproductgdp``, dataset (varies)

(Data is read from the ``www.ons.gov.uk`` pages; ``api.ons.gov.uk`` is not used.)

Series spec shape passed via ``lib.fetch_data(api="ons", series=...)``::

    series = {
        "indpro": "economy/economicoutputandproductivity/output/timeseries/k222/diop",
        "cpi":    "economy/inflationandpriceindices/timeseries/d7bt/mm23",
    }

Each value is the URL path *between* ``www.ons.gov.uk/`` and ``/data``.
Look up the path on the ONS Time Series Explorer — it's whatever the
public page URL is for that series.

The JSON response has ``months`` / ``quarters`` / ``years`` arrays of
``{date, value, ...}`` entries. We extract whichever block has data.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib import request
from urllib.error import HTTPError, URLError

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher

_BASE_URL = "https://www.ons.gov.uk"


class OnsFetcher(BaseFetcher):
    """Fetcher for UK ONS time-series API."""

    name = "ons"
    description = "UK Office for National Statistics — beta time-series API"
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
                path = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"ONS series spec for alias {alias!r} must be a URL path "
                    f"like 'economy/<category>/timeseries/<code>/<dataset>'; "
                    f"got {spec!r}: {exc}"
                ) from exc

            url = f"{_BASE_URL}/{path.strip('/')}/data"
            try:
                payload = self._http_get_json(url)
            except (HTTPError, URLError) as exc:
                # Don't abort the whole batch — log and skip.
                import warnings
                warnings.warn(
                    f"ONS fetch failed for alias={alias!r} (spec={spec!r}, "
                    f"url={url}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            ser = self._extract_series(payload)
            ser.name = alias
            # Optional sample filter (cheap to do here; load_dataset also filters)
            if start is not None:
                ser = ser.loc[pd.Timestamp(start):]
            if end is not None:
                ser = ser.loc[:pd.Timestamp(end)]
            out[alias] = ser
        return out

    # -------- helpers --------

    @staticmethod
    def _parse_spec(spec: Any) -> str:
        if isinstance(spec, str) and "/timeseries/" in spec:
            return spec
        # Convenience: accept a (code, dataset) tuple by guessing a top-level
        # path. Real series usually need a category prefix — prefer the
        # explicit string form. The tuple form is only useful for the
        # 'economy' top-level series and may 404 for everything else.
        if isinstance(spec, tuple) and len(spec) == 2:
            code, dataset = str(spec[0]).lower(), str(spec[1]).lower()
            return f"economy/timeseries/{code}/{dataset}"
        raise ValueError(
            "expected URL path string '<category>/timeseries/<code>/<dataset>' "
            "or a (code, dataset) tuple (which guesses 'economy' category)"
        )

    @staticmethod
    def _http_get_json(url: str, timeout: int = 30) -> dict:
        # ONS returns 403 for urllib's default User-Agent, so send an
        # explicit one that identifies this client.
        req = request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; marco-bis/1.0)",
        })
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_series(payload: dict) -> pd.Series:
        """Parse an ONS JSON payload into a pandas Series.

        ONS payloads have ``months``, ``quarters``, ``years`` arrays of
        ``{date, value}`` entries. Pick whichever has data. Date strings:
        - Monthly: e.g., "1992 OCT"
        - Quarterly: e.g., "1992 Q3"
        - Annual: e.g., "1992"
        """
        for key in ("months", "quarters", "years"):
            entries = payload.get(key) or []
            if not entries:
                continue
            dates: list[pd.Timestamp] = []
            values: list[float] = []
            for e in entries:
                if "date" not in e or "value" not in e:
                    continue
                try:
                    ts = _parse_ons_date(e["date"], key)
                except ValueError:
                    continue
                try:
                    v = float(e["value"])
                except (ValueError, TypeError):
                    continue
                dates.append(ts)
                values.append(v)
            if dates:
                ser = pd.Series(values, index=pd.DatetimeIndex(dates))
                return ser.sort_index()
        raise ValueError(
            "ONS payload has no parseable observations in months/quarters/years"
        )


_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_ons_date(date_str: str, granularity: str) -> pd.Timestamp:
    s = date_str.strip().upper()
    parts = s.split()
    if granularity == "months":
        # "1992 OCT"
        if len(parts) != 2:
            raise ValueError(f"unrecognised monthly ONS date: {date_str!r}")
        year = int(parts[0])
        month = _MONTH_MAP.get(parts[1])
        if month is None:
            raise ValueError(f"unrecognised month in {date_str!r}")
        return pd.Timestamp(year=year, month=month, day=1)
    if granularity == "quarters":
        # "1992 Q3"
        if len(parts) != 2 or not parts[1].startswith("Q"):
            raise ValueError(f"unrecognised quarterly ONS date: {date_str!r}")
        year = int(parts[0])
        q = int(parts[1][1:])
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return pd.Timestamp(year=year, month=month, day=1)
    if granularity == "years":
        # "1992"
        return pd.Timestamp(year=int(s), month=1, day=1)
    raise ValueError(f"unknown granularity {granularity!r}")


register_fetcher(OnsFetcher.name, OnsFetcher)
