"""Render each model-dictionary entry into an AUTHOR-FACING dossier.

One markdown file per entry, organised by field provenance (see
skills/model-dictionary/SKILL.md §"Field provenance"): what the framework
attributes to the paper (with source references where the entry records
them), the declared validation targets and
how closely the replication meets them, what it deviates on, and what is
the framework's own editorial content. Entry fields the provenance table
does not classify are shown in a separate, labelled section, so nothing
in an entry is silently dropped or silently attributed.

The working defensibility test: an entry is in good standing when its
dossier could be sent to the paper's original author as-is.

Usage:
    python scripts/tools/make_author_dossiers.py [entry_id ...]

Writes to scripts/tools/outputs/author_dossiers/<id>.md (gitignored,
regenerable). With no arguments, renders all entries. If the paired
replication has been run locally, the dossier also shows our latest
values next to the declared targets (from outputs/research_log.json).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ENTRIES = ROOT / "skills" / "model-dictionary" / "entries"
REPLICATIONS = ROOT / "scripts" / "replications"
OUT = HERE / "outputs" / "author_dossiers"

PAPER_FIELDS_NOTE = (
    "Sections 1-2 render the entry's specification and findings fields. "
    "Source references (page / table / figure / equation) appear where the "
    "entry records them in those fields; YAML comments are not carried "
    "over, and replication proxies or sample adaptations are labelled "
    "inline. Section 5 is the framework's own editorial judgment and is not "
    "attributed to the paper or its authors. Section 6 lists entry fields "
    "whose provenance the dictionary does not classify."
)

TARGETS_NOTE = (
    "A target value is the paper's own number only where its location cites "
    "the paper. Other targets are configuration checks, structural "
    "identities, or the authors' own results from the same replication, "
    "which catch regressions but do not by themselves establish agreement "
    "with the paper; each location says which."
)

# Top-level fields by provenance (skills/model-dictionary/SKILL.md).
PAPER_TOP = ("model", "identification", "sample")
EDITORIAL_TOP = ("research_questions", "tags", "recommend_when",
                 "not_recommended_when", "alternative_approaches")
HANDLED_TOP = {"id", "citation", "verified", "expected_results",
               "implementation_notes", "caveats", *PAPER_TOP, *EDITORIAL_TOP}

# implementation_notes keys that record departures from the paper's data,
# sample or specification (rendered under deviations, not editorial).
_DEVIATION_KEY = re.compile(
    r"replication|data|fred|align|mapping|caveat|vintage|lag_order|"
    r"requirement|warning")


def _variable_proxies(variables) -> dict:
    """Map each variable to its stand-in series where the entry records one."""
    found: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            name, proxy = node.get("name"), node.get("proxy")
            if (isinstance(name, str) and isinstance(proxy, str)
                    and proxy.strip().lower() != name.strip().lower()):
                found[name] = proxy
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(variables)
    return found


def _drop_empty(obj):
    if isinstance(obj, dict):
        out = {k: _drop_empty(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    return obj


def _fmt_citation(cit: dict) -> str:
    bits = [cit.get("authors", "?"), f"({cit.get('year', '?')})",
            f"\"{cit.get('title', '?')}\""]
    issue = cit.get("issue") or cit.get("number")
    tail = ", ".join(
        str(v) for v in (cit.get("journal"), cit.get("volume"), issue,
                         cit.get("pages")) if v
    )
    return " ".join(bits) + (f", {tail}" if tail else "")


def _yaml_block(obj) -> str:
    obj = _drop_empty(obj)
    if not obj:
        return "_None recorded in the entry._"
    return "```yaml\n" + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True,
                                        width=78).strip() + "\n```"


def _load_latest_comparison(rid: str) -> dict | None:
    log = REPLICATIONS / rid / "outputs" / "research_log.json"
    if not log.exists():
        return None
    try:
        data = json.loads(log.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("paper_target_comparison")


def _targets_section(rid: str) -> str:
    tpath = REPLICATIONS / rid / "paper_targets.yaml"
    if not tpath.exists():
        return "_No paper_targets.yaml found for this entry._"
    targets = yaml.safe_load(tpath.read_text(encoding="utf-8")).get("targets", [])
    comp = _load_latest_comparison(rid)
    ours = {}
    if comp:
        for row in comp.get("rows", []):
            ours[row.get("our_key") or row.get("description")] = row
    lines = [
        f"_{TARGETS_NOTE}_",
        "",
        "| Target (location) | Target value | Tolerance | Our latest run |",
        "|---|---|---|---|",
    ]
    for t in targets:
        row = ours.get(t.get("our_key")) or ours.get(t.get("description")) or {}
        ours_cell = (
            f"{row['our_value']:+.4g} ({'OK' if row.get('ok') else 'FAIL'})"
            if "our_value" in row else "run replicate.py to fill"
        )
        lines.append(
            f"| {t.get('description', '?')} ({t.get('location', '?')}) "
            f"| {t.get('paper_value')} {t.get('paper_units', '')} "
            f"| ±{t.get('tolerance')} ({t.get('tolerance_kind', 'absolute')}) "
            f"| {ours_cell} |"
        )
    return "\n".join(lines)


def render(entry_path: Path) -> str:
    e = yaml.safe_load(entry_path.read_text(encoding="utf-8"))
    rid = e["id"]
    cit = e.get("citation", {})
    verified = e.get("verified", {})
    ident = dict(e.get("identification") or {})
    headline = ident.pop("headline_finding", None)
    sample = e.get("sample") or {}
    impl = e.get("implementation_notes") or {}
    impl_dev = {k: v for k, v in impl.items()
                if _DEVIATION_KEY.search(k) and k != "replication"}
    impl_ed = {k: v for k, v in impl.items()
               if k not in impl_dev and k != "replication"}
    other = {k: v for k, v in e.items() if k not in HANDLED_TOP}

    parts = [
        f"# Author dossier — {_fmt_citation(cit)}",
        "",
        f"Dictionary entry: `skills/model-dictionary/entries/{rid}.yaml` — "
        f"checked against the paper on {verified.get('date', '?')}.",
        "",
        f"> {PAPER_FIELDS_NOTE}",
        "",
        "## 1. The specification we attribute to the paper",
        "",
        _yaml_block({"model": e.get("model"), "identification": ident,
                     "sample": sample}),
        "",
        "## 2. The findings we attribute to the paper",
        "",
        _yaml_block({"headline_finding": headline,
                     "expected_results": e.get("expected_results")}),
        "",
        "## 3. Declared validation targets and our latest run",
        "",
        _targets_section(rid),
        "",
        "## 4. Documented deviations from the paper",
        "",
        "_`sample_notes` may restate the paper's own sample. Further "
        "departures — for example a replication that demonstrates the method "
        "on different data — are recorded in the verification notes "
        "(section 7) and the replication README._",
        "",
        _yaml_block({
            "variable_proxies": _variable_proxies(
                (e.get("model") or {}).get("variables")),
            "sample_notes": sample.get("notes"),
            "implementation_notes": impl_dev,
            "caveats": e.get("caveats"),
        }),
        "",
        f"Replication: `{impl.get('replication') or f'scripts/replications/{rid}/'}`"
        f" (README: `scripts/replications/{rid}/README.md`)",
        "",
        "## 5. Framework editorial content — ours, NOT attributed to the paper",
        "",
        _yaml_block({**{k: e.get(k) for k in EDITORIAL_TOP},
                     "implementation_notes": impl_ed}),
        "",
        "## 6. Other entry fields — provenance not classified",
        "",
        _yaml_block(other),
        "",
        "## 7. Verification record",
        "",
        # The internal status (verified.status) steers the agent's entry choice;
        # it is not shown to the paper's authors.
        _yaml_block({"verified": {k: v for k, v in verified.items()
                                  if k != "status"}}),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    ids = argv or sorted(p.stem for p in ENTRIES.glob("*.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    for rid in ids:
        path = ENTRIES / f"{rid}.yaml"
        if not path.exists():
            print(f"  !! no entry {rid}")
            continue
        (OUT / f"{rid}.md").write_text(render(path), encoding="utf-8")
        print(f"  -> {rid}.md")
    print(f"\n{len(ids)} dossier(s) in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
