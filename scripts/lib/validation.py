"""Paper-target validation: side-by-side comparison of replication
output against paper-declared target values.

Pattern: each replication ships a ``paper_targets.yaml`` listing the
specific values the paper reports (e.g., "energy FEV at 2y horizon =
19%"). At the end of ``replicate.py``, call :func:`compare_to_paper`
with a dict of our computed values; the function prints a comparison
table and returns a :class:`ComparisonReport` with pass/fail status.

The act of writing the YAML forces a careful read of the paper before
declaring replication success. The diff column surfaces mismatches at
build time — at the same place the convention contracts catch code
drift.

YAML schema (one entry per target)::

    targets:
      - description: "Short label shown in the comparison table"
        location: "Where in the paper (e.g., 'Footnote 5', 'Table 1 row 2')"
        paper_value: 0.19
        paper_units: "share (0-1)"      # descriptive only
        our_key: "energy_fev_2y"         # key in the `results` dict
        tolerance: 0.10                  # max acceptable diff
        tolerance_kind: "absolute"        # or "relative" (|diff/paper| <= tol)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TargetEntry:
    """One paper-target row."""
    description: str
    location: str
    paper_value: float
    paper_units: str
    our_key: str
    tolerance: float
    tolerance_kind: str = "absolute"

    def assess(self, our_value: float | None) -> dict[str, Any]:
        """Compute pass/fail and the diff for our_value vs paper_value."""
        import math
        if our_value is None:
            return {
                "ok": False,
                "diff": None,
                "reason": "our_key not in results",
            }
        if isinstance(our_value, float) and math.isnan(our_value):
            return {
                "ok": False,
                "diff": None,
                "reason": "our_value is NaN — replication did not produce this metric",
            }
        diff = our_value - self.paper_value
        if self.tolerance_kind == "absolute":
            ok = abs(diff) <= self.tolerance
        elif self.tolerance_kind == "relative":
            if self.paper_value == 0:
                ok = abs(diff) <= self.tolerance
            else:
                ok = abs(diff / self.paper_value) <= self.tolerance
        else:
            raise ValueError(
                f"tolerance_kind must be 'absolute' or 'relative'; "
                f"got {self.tolerance_kind!r}"
            )
        return {"ok": ok, "diff": diff, "reason": None}


@dataclass
class ComparisonReport:
    """Result of comparing replication output to paper targets.

    Attributes
    ----------
    targets : list[TargetEntry]
        The declared targets from paper_targets.yaml.
    rows : list[dict]
        Per-target row: description, location, paper_value, our_value,
        diff, tolerance, ok, reason.
    """

    targets: list[TargetEntry]
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.rows if r["ok"])

    @property
    def n_fail(self) -> int:
        return self.n_targets - self.n_pass

    @property
    def all_pass(self) -> bool:
        return self.n_fail == 0

    def table(self, width: int = 90) -> str:
        """Render a human-readable comparison table."""
        if not self.rows:
            return "(no targets compared)"
        lines = ["=" * width, "Paper-target comparison", "=" * width]
        lines.append(
            f"{'Status':<7} {'Description':<35} {'Paper':>10} {'Ours':>10} "
            f"{'Diff':>10} {'Tol':>8}"
        )
        lines.append("-" * width)
        for r in self.rows:
            mark = " OK  " if r["ok"] else "FAIL "
            ours_str = (
                f"{r['our_value']:>10.4f}" if r["our_value"] is not None
                else f"{'(missing)':>10}"
            )
            diff_str = (
                f"{r['diff']:>+10.4f}" if r["diff"] is not None else f"{'-':>10}"
            )
            lines.append(
                f"{mark:<7} {r['description'][:35]:<35} "
                f"{r['paper_value']:>10.4f} {ours_str} "
                f"{diff_str} {r['tolerance']:>8.4f}"
            )
            if r.get("reason"):
                lines.append(f"        ^ {r['reason']}")
        lines.append("-" * width)
        lines.append(
            f"Summary: {self.n_pass}/{self.n_targets} pass"
            + (f" ({self.n_fail} fail)" if self.n_fail else "")
        )
        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for inclusion in research_log.json."""
        return {
            "n_targets": self.n_targets,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "all_pass": self.all_pass,
            "rows": self.rows,
        }


def _load_targets(targets: str | Path | list) -> list[TargetEntry]:
    """Parse targets from a YAML file path or a list-of-dicts."""
    if isinstance(targets, (str, Path)):
        path = Path(targets)
        if not path.exists():
            raise FileNotFoundError(f"paper_targets file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "targets" not in data:
            raise ValueError(
                f"{path}: expected top-level mapping with 'targets' key"
            )
        raw = data["targets"]
    elif isinstance(targets, list):
        raw = targets
    else:
        raise TypeError(
            f"targets must be a YAML path or list; got {type(targets).__name__}"
        )

    out: list[TargetEntry] = []
    required = {"description", "location", "paper_value", "our_key", "tolerance"}
    for i, entry in enumerate(raw):
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(
                f"target #{i}: missing required keys {missing}; "
                f"got {list(entry.keys())}"
            )
        out.append(TargetEntry(
            description=str(entry["description"]),
            location=str(entry["location"]),
            paper_value=float(entry["paper_value"]),
            paper_units=str(entry.get("paper_units", "")),
            our_key=str(entry["our_key"]),
            tolerance=float(entry["tolerance"]),
            tolerance_kind=str(entry.get("tolerance_kind", "absolute")),
        ))
    return out


def compare_to_paper(
    results: dict[str, float],
    targets: str | Path | list[dict],
) -> ComparisonReport:
    """Compare replication output to paper-declared targets.

    Parameters
    ----------
    results : dict[str, float]
        Numerical results from the replication, keyed by the same names
        used in the YAML's ``our_key`` fields.
    targets : path or list of dict
        Either a path to a ``paper_targets.yaml`` file, or a pre-parsed
        list of target dicts.

    Returns
    -------
    ComparisonReport with one row per target. Per-row entries:
      ``description, location, paper_value, paper_units, our_value, diff,
      tolerance, tolerance_kind, ok, reason``.

    Example
    -------
    >>> from lib import compare_to_paper                # doctest: +SKIP
    >>> our = {"core_cpi_peak": 0.13, "energy_fev_2y": 0.49}
    >>> report = compare_to_paper(our, "paper_targets.yaml")
    >>> print(report.table())                            # doctest: +SKIP
    """
    target_list = _load_targets(targets)
    rows: list[dict[str, Any]] = []
    for t in target_list:
        our_val = results.get(t.our_key)
        assessment = t.assess(our_val)
        rows.append({
            "description": t.description,
            "location": t.location,
            "paper_value": t.paper_value,
            "paper_units": t.paper_units,
            "our_value": our_val,
            "our_key": t.our_key,
            "diff": assessment["diff"],
            "tolerance": t.tolerance,
            "tolerance_kind": t.tolerance_kind,
            "ok": assessment["ok"],
            "reason": assessment["reason"],
        })
    return ComparisonReport(targets=target_list, rows=rows)
