"""Registry lookup for pre-collected datasets.

``data/registry.yaml`` indexes datasets by id (e.g., ``fred_md``, ``fred_qd``)
with metadata including the on-disk path. Cache-on-fetch datasets registered
by ``lib.fetch_data`` live in a gitignored overlay, ``data/registry.local.yaml``,
merged in at load time — so the tracked registry never churns (and never
loses its comments) as local fetches accumulate. This module resolves a
source id to a concrete CSV path.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Cache for repeated calls within a session
_PROJECT_ROOT_KEY = "_project_root"


def _find_default_registry() -> Path:
    """Walk up from this file to find ``data/registry.yaml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "registry.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "data/registry.yaml not found by walking up from "
        f"{here}. Pass registry_path explicitly."
    )


def load_registry(registry_path: str | Path | None = None) -> dict:
    """Load the dataset registry from ``data/registry.yaml``.

    The returned dict has the registry contents plus a ``_project_root`` key
    pointing to the registry's parent of parent (so dataset paths can be
    resolved as ``project_root / entry["path"]``).
    """
    if registry_path is None:
        registry_path = _find_default_registry()
    registry_path = Path(registry_path)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    # Merge the gitignored local overlay (cache-on-fetch entries), if any.
    local_path = registry_path.parent / "registry.local.yaml"
    if local_path.exists():
        local = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        main_ids = {e.get("id") for e in data.get("datasets", [])}
        extra = [e for e in local.get("datasets", []) if e.get("id") not in main_ids]
        if extra:
            data.setdefault("datasets", []).extend(extra)
    # registry is at <project_root>/data/registry.yaml
    data[_PROJECT_ROOT_KEY] = registry_path.parent.parent
    return data


def get_dataset_path(
    source: str,
    registry: dict | None = None,
    registry_path: str | Path | None = None,
) -> Path:
    """Resolve a source id to its ``dataset.csv`` path.

    Parameters
    ----------
    source : str
        Dataset id (e.g., ``"fred_md"``).
    registry : dict, optional
        Pre-loaded registry. If None, loaded from ``registry_path``.
    registry_path : path, optional
        Path to ``registry.yaml``. If None, found by walking up from this file.

    Returns
    -------
    Path
        Absolute path to ``<project_root>/<entry path>/dataset.csv``.

    Raises
    ------
    KeyError
        If ``source`` is not in the registry.
    """
    if registry is None:
        registry = load_registry(registry_path)
    project_root: Path = registry[_PROJECT_ROOT_KEY]
    for entry in registry.get("datasets", []):
        if entry["id"] == source:
            return project_root / entry["path"] / "dataset.csv"
    available = [e["id"] for e in registry.get("datasets", [])]
    raise KeyError(
        f"source {source!r} not in registry; available: {available}"
    )
