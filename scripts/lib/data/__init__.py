"""Data layer for the econometric library.

Two-step pattern:

>>> from lib import load_dataset, transform
>>> raw = load_dataset(
...     source="fred_md",
...     series={"indpro": "INDPRO", "cpi": "CPIAUCSL"},
...     start="1965-01", end="2003-12",
... )
>>> data = transform(raw, {"indpro": "log", "cpi": "log"})

``load_dataset`` delivers structurally-clean raw data with provenance metadata
in ``df.attrs``. ``transform`` applies per-column transforms — the decision of
which transform to use belongs with the model, not the loader.

Pre-collected CSVs are indexed in ``data/registry.yaml``; runtime fetchers
(``lib.fetch_data``, see ``fetchers/``) cache further series under
``data/<id>/``.
"""

from .loader import load_dataset
from .registry import get_dataset_path, load_registry
from .transforms import TRANSFORM_REGISTRY, FRED_MD_CODE_TO_NAME, transform
from .fetcher import fetch_data, BaseFetcher, register_fetcher, get_fetcher
from . import fetchers  # noqa: F401  triggers source-specific fetcher registration

__all__ = [
    "load_dataset",
    "transform",
    "fetch_data",
    "BaseFetcher",
    "register_fetcher",
    "get_fetcher",
    "TRANSFORM_REGISTRY",
    "FRED_MD_CODE_TO_NAME",
    "load_registry",
    "get_dataset_path",
]
