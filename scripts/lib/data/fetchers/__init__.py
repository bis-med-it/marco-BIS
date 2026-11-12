"""Source-specific fetchers, auto-registered on import.

Each fetcher inherits from :class:`lib.data.fetcher.BaseFetcher` and registers
itself with :func:`lib.data.fetcher.register_fetcher` so that
``lib.fetch_data(api="<name>", ...)`` dispatches correctly.

Currently implemented:

| ``api`` name          | Source                                      | Series-spec shape    |
|-----------------------|---------------------------------------------|----------------------|
| ``"fred"``            | FRED REST API — individual US series        | ``series_id`` (needs ``FRED_API_KEY``) |
| ``"dbnomics"``        | DBnomics aggregator (~80 providers, no key) | ``PROVIDER/DATASET/SERIES`` |
| ``"ons"``             | UK Office for National Statistics           | URL path string      |
| ``"boe_iadb"``        | Bank of England Interactive Database        | ``code``             |
| ``"csv_file"``        | Local CSV (stage manually-downloaded data)  | ``path:column``      |
| ``"external_archive"``| Author GitHub / journal supplementary CSVs  | ``url:column`` or dict |

Note: ``"fred"`` is for *individual* FRED series and needs a free API key
(``FRED_API_KEY`` env var). The bulk McCracken FRED-MD / FRED-QD panels are
*not* fetched here — they are browser-only (Akamai-blocked) and staged by
manual download as pre-collected datasets; see ``data-retrieval/SKILL.md``.

ECB, IMF, BIS, OECD and World Bank series are reachable through
``"dbnomics"``.
"""

# Importing the submodules triggers registration as a side effect
from . import fred               # noqa: F401  registers "fred"
from . import dbnomics           # noqa: F401  registers "dbnomics"
from . import ons                # noqa: F401  registers "ons"
from . import boe_iadb           # noqa: F401  registers "boe_iadb"
from . import csv_file           # noqa: F401  registers "csv_file"
from . import external_archive   # noqa: F401  registers "external_archive"
