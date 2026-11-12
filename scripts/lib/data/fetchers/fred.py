"""Federal Reserve Economic Data (FRED) — individual series via the REST API.

Fetches single FRED series through the public FRED REST endpoint
``https://api.stlouisfed.org/fred/series/observations``. Requires a free API
key, read from the ``FRED_API_KEY`` environment variable. Get a key at
https://fred.stlouisfed.org/docs/api/api_key.html.

This talks to the REST API directly (via ``urllib``); the ``fredapi`` package
is **not** required.

**Scope.** This is for *individual* FRED series (a custom small VAR pulling,
say, ``GDPC1`` + ``FEDFUNDS`` + ``CPIAUCSL``). The bulk McCracken **FRED-MD /
FRED-QD** panels are a different thing and are **not** fetched here — that file
server is behind Akamai Bot Manager (browser-only, HTTP 403 to automation), so
those panels are staged by a one-time manual download as pre-collected
datasets. See ``data-retrieval/SKILL.md`` ("FRED data not found").

Series-spec shape passed via ``lib.fetch_data(api="fred", series=...)``::

    series = {"rgdp": "GDPC1", "ffr": "FEDFUNDS", "cpi": "CPIAUCSL"}

i.e. ``{alias: fred_series_id}`` — a bare FRED series-ID string per alias.
FRED returns period-start dates (1959-01-01, 1959-04-01, …), matching the
start-of-period convention used elsewhere in the library; missing observations
(FRED encodes them as ``"."``) become ``NaN``.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Mapping
from urllib import parse, request
from urllib.error import HTTPError, URLError

import pandas as pd

from ..fetcher import BaseFetcher, register_fetcher

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_API_KEY_ENV = "FRED_API_KEY"
_KEY_URL = "https://fred.stlouisfed.org/docs/api/api_key.html"


class FredFetcher(BaseFetcher):
    """Fetcher for individual FRED series via the FRED REST API."""

    name = "fred"
    description = "Federal Reserve Economic Data (FRED) REST API — individual series"
    endpoint = _BASE_URL

    def fetch(
        self,
        series: Mapping[str, Any],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.Series]:
        # A missing key is a *setup* error (it affects every series), so raise a
        # clear, actionable message rather than warning per-series and falling
        # through to the dispatcher's generic "returned nothing" error.
        api_key = os.environ.get(_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"FRED requires an API key, but {_API_KEY_ENV} is not set in the "
                f"environment. Get a free key at {_KEY_URL}, then set it:\n"
                f"  macOS/Linux : export {_API_KEY_ENV}=your_key_here\n"
                f"  Windows     : setx {_API_KEY_ENV} your_key_here   (then open a new shell)\n"
                f"  or add a line {_API_KEY_ENV}=your_key_here to a .env file the runner loads.\n"
                f"(The bulk FRED-MD / FRED-QD panels do NOT need a key — they are "
                f"staged by manual download; see data-retrieval/SKILL.md.)"
            )

        out: dict[str, pd.Series] = {}
        for alias, spec in series.items():
            try:
                code = self._parse_spec(spec)
            except ValueError as exc:
                raise ValueError(
                    f"FRED series spec for {alias!r} must be a bare series-ID "
                    f"string (e.g. 'GDPC1'); got {spec!r}: {exc}"
                ) from exc

            params = {"series_id": code, "api_key": api_key, "file_type": "json"}
            if start is not None:
                params["observation_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
            if end is not None:
                params["observation_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
            url = f"{_BASE_URL}?{parse.urlencode(params)}"

            try:
                payload = self._http_get_json(url)
            except HTTPError as exc:
                hint = (
                    " — HTTP 400 from FRED usually means an invalid FRED_API_KEY "
                    "or an unknown series_id" if exc.code == 400 else ""
                )
                warnings.warn(
                    f"FRED fetch failed for alias={alias!r} (series_id={code}, "
                    f"url={url.replace(api_key, '<API_KEY>')}): {exc}{hint}",
                    UserWarning, stacklevel=2,
                )
                continue
            except URLError as exc:
                warnings.warn(
                    f"FRED fetch failed for alias={alias!r} (series_id={code}, "
                    f"url={url.replace(api_key, '<API_KEY>')}): {exc}",
                    UserWarning, stacklevel=2,
                )
                continue

            ser = self._extract_series(payload)
            if ser is None:
                warnings.warn(
                    f"FRED returned no observations for alias={alias!r} "
                    f"(series_id={code}).",
                    UserWarning, stacklevel=2,
                )
                continue
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
        if isinstance(spec, str) and spec.strip():
            return spec.strip()
        raise ValueError("expected a non-empty bare series-ID string")

    @staticmethod
    def _http_get_json(url: str, timeout: int = 60) -> dict:
        req = request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; marco-bis/1.0)",
        })
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _extract_series(payload: dict) -> pd.Series | None:
        """Parse a FRED ``series/observations`` JSON payload into a Series.

        Response shape::

            {"observations": [{"date": "1959-01-01", "value": "2.3"}, ...]}

        FRED encodes missing values as the string ``"."`` → ``NaN``.
        """
        obs = payload.get("observations")
        if not obs:
            return None
        dates: list[pd.Timestamp] = []
        values: list[float] = []
        for o in obs:
            d = o.get("date")
            if d is None:
                continue
            try:
                ts = pd.Timestamp(d)
            except (ValueError, TypeError):
                continue
            v = o.get("value")
            try:
                fv = float(v) if v not in (None, ".", "") else float("nan")
            except (ValueError, TypeError):
                fv = float("nan")
            dates.append(ts)
            values.append(fv)
        if not dates:
            return None
        return pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()


register_fetcher(FredFetcher.name, FredFetcher)
