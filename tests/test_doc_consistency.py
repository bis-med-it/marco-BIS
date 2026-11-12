"""Framework consistency checks — guard against documentation / spec drift.

This is the *automated* half of the consistency-audit mechanism; the
*on-demand semantic* half lives in ``skills/consistency-audit/``. These tests
run with the normal suite, so mechanical drift between the three layers
(skills ↔ lib ↔ dictionary) fails loudly instead of rotting silently.

Each check pins a class of mechanical drift this repo is prone to:

* a dictionary entry YAML that doesn't parse (e.g. a ``1./2./3.``
  enumeration that YAML reads as floats);
* ``index.yaml`` out of sync with the entry files (count or id drift);
* a stale "N-module library" count in a skill that lags the actual
  module count;
* an invalid ``verified.status`` level;
* a broken skill→file markdown link;
* a skills-prose reference to a ``lib`` symbol that no longer exists;
* a replication folder missing its core runnable files.

Keep these checks *mechanical and precise* — semantic consistency (stage
numbering, propose/confirm choreography, doc-vs-practice) is the job of the
on-demand ``consistency-audit`` skill, not of brittle string assertions here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import lib

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
ENTRIES = SKILLS / "model-dictionary" / "entries"
INDEX = SKILLS / "model-dictionary" / "index.yaml"
LIB_DIR = ROOT / "scripts" / "lib"
REPLICATIONS = ROOT / "scripts" / "replications"

VALID_VERIFIED_STATUS = {"draft", "checked_against_paper", "signed_off"}


def _skill_md_files() -> list[Path]:
    return sorted(SKILLS.glob("**/*.md"))


def _entry_files() -> list[Path]:
    return sorted(ENTRIES.glob("*.yaml"))


def _load_index() -> dict:
    return yaml.safe_load(INDEX.read_text(encoding="utf-8"))


def _actual_lib_module_count() -> int:
    return len([p for p in LIB_DIR.glob("*.py") if p.name != "__init__.py"])


# ---------------------------------------------------------------------------
# 1. Every dictionary entry parses as YAML and is self-identifying
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _entry_files(), ids=lambda p: p.stem)
def test_entry_yaml_parses(path: Path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        pytest.fail(f"{path.name} does not parse as YAML: {exc}")
    assert isinstance(data, dict), f"{path.name} did not parse to a mapping"
    assert data.get("id") == path.stem, (
        f"{path.name}: 'id' ({data.get('id')!r}) must equal the filename stem"
    )


# ---------------------------------------------------------------------------
# 2. index.yaml is in sync with the entry files
# ---------------------------------------------------------------------------


def test_index_matches_entry_files():
    idx = _load_index()
    index_ids = {e["id"] for e in idx["entries"]}
    file_ids = {p.stem for p in _entry_files()}
    assert not (file_ids - index_ids), (
        f"entry files with no index row: {sorted(file_ids - index_ids)}"
    )
    assert not (index_ids - file_ids), (
        f"index rows with no entry file: {sorted(index_ids - file_ids)}"
    )


def test_index_total_entries_header_is_accurate():
    text = INDEX.read_text(encoding="utf-8")
    m = re.search(r"#\s*Total entries:\s*(\d+)", text)
    assert m, "index.yaml is missing the '# Total entries: N' header"
    claimed, actual = int(m.group(1)), len(_entry_files())
    assert claimed == actual, (
        f"index.yaml header says {claimed} entries but {actual} entry files exist"
    )


# ---------------------------------------------------------------------------
# 3. verified.status uses a known level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _entry_files(), ids=lambda p: p.stem)
def test_entry_verified_status_is_valid(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    verified = data.get("verified") or {}
    assert "by" not in verified, f"{path.name}: use verified.status, not verified.by"
    if "status" not in verified:
        pytest.skip(f"{path.name} has no verified.status")
    status = verified["status"]
    assert status in VALID_VERIFIED_STATUS, (
        f"{path.name}: verified.status={status!r} not in {sorted(VALID_VERIFIED_STATUS)}"
    )


def test_index_verified_status_is_valid():
    idx = _load_index()
    bad = {
        e["id"]: e.get("verified_status")
        for e in idx["entries"]
        if e.get("verified_status") not in VALID_VERIFIED_STATUS
    }
    assert not bad, f"index rows with invalid verified_status: {bad}"


# ---------------------------------------------------------------------------
# 4. lib module-count claims in prose match the real module count
# ---------------------------------------------------------------------------


def test_module_count_claims_match_reality():
    actual = _actual_lib_module_count()
    # "24 modules" must be exact; "20+ modules" is a lower-bound claim and
    # must not exceed the actual count (the "+" previously let stale claims
    # like "25+ modules" evade this test entirely).
    pat = re.compile(r"(\d+)(\+?)[\s-]module")
    bad = []
    for f in _skill_md_files() + [LIB_DIR / "README.md", ROOT / "CLAUDE.md",
                                  ROOT / "README.md"]:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            low = line.lower()
            if "library" in low or "scripts/lib" in low or "lib/" in low or "modules" in low:
                for n, plus in pat.findall(line):
                    ok = (int(n) <= actual) if plus else (int(n) == actual)
                    if not ok:
                        bad.append(
                            (f.relative_to(ROOT).as_posix(), n + plus,
                             line.strip()[:90])
                        )
    assert not bad, (
        f"stale lib module-count claim(s) (actual={actual} modules): {bad}"
    )


# ---------------------------------------------------------------------------
# 4b. dictionary entry / replication count claims in prose match reality
# ---------------------------------------------------------------------------


def test_entry_count_claims_match_reality():
    """A hardcoded total like "All 29 shipped entries" or "currently 29
    canonical paper replications" rots the moment an entry is added (both
    exact examples shipped once). Totals in prose must match the folders;
    lower-bound "N+" claims must not exceed them. Scans whitespace-normalised
    text with markdown markers stripped, so bold and line wraps cannot hide
    a claim. Mirrors the module-count check above."""
    actual_entries = len(_entry_files())
    actual_repls = len([d for d in (ROOT / "scripts" / "replications").iterdir()
                        if d.is_dir()])
    # Explicit totality claims only. Cohort counts ("15 dictionary entries
    # drafted from ...") and years ("CEE 1999 replications") must not match.
    pat = re.compile(
        r"(?:[Aa]ll|currently|total of|holds)\s+(\d+)(\+?)\s+(?:\w+\s+){0,2}?(entries|replications)"
        r"|(\d+)(\+?)\s+(shipped entries)"
        r"|(\d+)(\+?)\s+(canonical (?:paper )?(?:specifications|replications))"
        r"|(\d+)(\+?)\s+(paper-by-paper (?:YAML )?specifications)"
        r"|\((\d+)()( papers)\)"
    )
    bad = []
    files = _skill_md_files() + [
        ROOT / "CLAUDE.md", ROOT / "README.md", ROOT / "DEVELOPMENT.md",
        ROOT / "TEST_SUITE.md",
        ROOT / "scripts" / "applications" / "README.md",
        ROOT / "scripts" / "replications" / "README.md",
    ]
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", text.replace("*", "").replace("`", ""))
        for m in pat.finditer(flat):
            groups = [g for g in m.groups() if g is not None]
            n, plus, noun = groups[0], groups[1], groups[2]
            target = actual_repls if "replication" in noun else actual_entries
            ok = (int(n) <= target) if plus else (int(n) == target)
            if not ok:
                bad.append((f.relative_to(ROOT).as_posix(),
                            n + plus + " " + noun, m.group(0)[:90]))
    assert not bad, (
        f"stale entry/replication count claim(s) "
        f"(actual: {actual_entries} entries, {actual_repls} replications): {bad}"
    )


# ---------------------------------------------------------------------------
# 4c. replication configs name real data sources
# ---------------------------------------------------------------------------


def test_replication_data_sources_are_valid():
    """A typo in a config's data.source must fail here, deterministically —
    at runtime it raises the same "not in registry" message as a genuinely
    unfetched dataset, which the smoke test must not treat as a skip.
    Valid sources: dataset ids in data/registry.yaml, and fetcher names in
    scripts/lib/data/fetchers/ (the staging convention uses "csv_file")."""
    import yaml as _yaml
    reg = _yaml.safe_load((ROOT / "data" / "registry.yaml").read_text(encoding="utf-8"))
    ds = reg.get("datasets") or []
    valid = set(ds.keys()) if isinstance(ds, dict) else {
        d.get("id") for d in ds if isinstance(d, dict) and d.get("id")}
    fdir = ROOT / "scripts" / "lib" / "data" / "fetchers"
    valid |= {f.stem for f in fdir.glob("*.py") if f.stem != "__init__"}
    bad = []
    for cfg in (ROOT / "scripts" / "replications").glob("*/config.yaml"):
        c = _yaml.safe_load(cfg.read_text(encoding="utf-8"))
        src = (c.get("data") or {}).get("source")
        if isinstance(src, str) and src not in valid:
            bad.append((cfg.parent.name, src))
    assert not bad, (
        f"replication config data.source not in registry or fetchers "
        f"(valid: {sorted(valid)}): {bad}"
    )


# ---------------------------------------------------------------------------
# 5. Markdown links inside skills resolve to real files
# ---------------------------------------------------------------------------


_LINK = re.compile(r"\]\(([^)]+)\)")


def test_skill_markdown_links_resolve():
    bad = []
    for f in _skill_md_files():
        for target in _LINK.findall(f.read_text(encoding="utf-8")):
            path_part = target.split("#", 1)[0].strip()
            if not path_part or path_part.startswith(("http://", "https://", "mailto:")):
                continue
            if not (f.parent / path_part).resolve().exists():
                bad.append((f.relative_to(ROOT).as_posix(), target))
    assert not bad, f"broken markdown links in skills: {bad}"


_ROOT_DOCS = ["README.md", "CLAUDE.md", "DEVELOPMENT.md", "TEST_SUITE.md",
              "CONVENTIONS.md"]


def test_root_doc_links_resolve():
    """The five root documents are the public front door — their relative
    links must resolve just like the skills' links do."""
    bad = []
    for name in _ROOT_DOCS:
        f = ROOT / name
        if not f.exists():
            continue
        for target in _LINK.findall(f.read_text(encoding="utf-8")):
            path_part = target.split("#", 1)[0].strip()
            if not path_part or path_part.startswith(("http://", "https://", "mailto:")):
                continue
            if not (f.parent / path_part).resolve().exists():
                bad.append((name, target))
    assert not bad, f"broken markdown links in root docs: {bad}"


# Controlled vocabulary for entries' identification.method (documented in
# skills/model-dictionary/SKILL.md §Entry schema). Nuance belongs in a
# same-line comment or the rationale field, never in the value.
_CANONICAL_IDENT_METHODS = {
    "cholesky", "long_run", "sign_restrictions", "narrative_sign_restrictions",
    "sign_restrictions_plus_higher_moments", "external_instrument",
    "heteroskedasticity", "non_gaussian_independent_components",
    "generalised_variance_decomposition", "none", "varies",
}


@pytest.mark.parametrize("path", _entry_files(), ids=lambda p: p.stem)
def test_entry_identification_method_is_canonical(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    method = (data.get("identification") or {}).get("method")
    if method is None:
        pytest.skip(f"{path.name} has no identification.method")
    assert method in _CANONICAL_IDENT_METHODS, (
        f"{path.name}: identification.method {method!r} is not in the "
        f"controlled vocabulary {sorted(_CANONICAL_IDENT_METHODS)} — put "
        f"nuance in a comment or the rationale, not the value"
    )


# ---------------------------------------------------------------------------
# 6. lib symbols referenced in skills prose actually exist
# ---------------------------------------------------------------------------


_LIB_REF = re.compile(r"`lib\.([A-Za-z_][A-Za-z0-9_]*)`")


def test_lib_references_in_skills_resolve():
    known = set(lib.__all__) | {p.stem for p in LIB_DIR.glob("*.py")}
    bad = []
    for f in _skill_md_files():
        for name in _LIB_REF.findall(f.read_text(encoding="utf-8")):
            if name not in known:
                bad.append((f.relative_to(ROOT).as_posix(), f"lib.{name}"))
    assert not bad, (
        f"skills reference lib symbols that don't exist in lib.__all__: {bad}"
    )


_LIB_REF_PLAIN = re.compile(r"\blib\.([A-Za-z_][A-Za-z0-9_]*)")


def test_lib_references_in_entries_resolve():
    """Dictionary entries name lib functions in implementation_notes (plain
    ``lib.fit_var`` prose, no backticks) — those must exist too."""
    known = (set(lib.__all__) | {p.stem for p in LIB_DIR.glob("*.py")}
             | {"data", "preprocess"})   # subpackages referenced as lib.data.*
    bad = []
    for f in _entry_files():
        for name in _LIB_REF_PLAIN.findall(f.read_text(encoding="utf-8")):
            if name not in known:
                bad.append((f.name, f"lib.{name}"))
    assert not bad, (
        f"entries reference lib symbols that don't exist: {bad}"
    )


# ---------------------------------------------------------------------------
# 7. checked_against_paper entries actually carry page references
# ---------------------------------------------------------------------------
# The status means "cross-checked field-by-field against the source PDF with
# page references" (index.yaml header; model-dictionary SKILL.md §Field
# provenance). This pins the mechanical half of that promise: an entry
# claiming the status without in-text paper references fails loudly.

_PAPER_REF = re.compile(
    r"p{1,2}\.\s*\d{1,4}|Table\s+[A-Z0-9]|Figure\s+\d|Fig\.\s*\d|eq\.\s*\("
)
_MIN_PAPER_REFS = 5


@pytest.mark.parametrize("path", _entry_files(), ids=lambda p: p.stem)
def test_checked_entries_have_page_refs(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (data.get("verified") or {}).get("status") != "checked_against_paper":
        pytest.skip(f"{path.name} is not checked_against_paper")
    n_refs = len(_PAPER_REF.findall(path.read_text(encoding="utf-8")))
    assert n_refs >= _MIN_PAPER_REFS, (
        f"{path.name}: verified.status=checked_against_paper but only {n_refs} "
        f"page/table/figure/equation references found (need >= "
        f"{_MIN_PAPER_REFS}) — the status promises page-cited verification"
    )


# ---------------------------------------------------------------------------
# 8. Dictionary entries and replication configs agree on the runnable spec
# ---------------------------------------------------------------------------
# The entry documents the PAPER's spec; the paired config documents what the
# replication actually RUNS. Machine-comparable fields must be EQUAL, or the
# divergence must be DECLARED below — mirroring, in one reviewable place, the
# deviation that the replication's own config comments / README document in
# prose. An undeclared mismatch is silent drift and fails.

_IDENT_SYNONYMS = {
    "choleski": "cholesky", "cholesky": "cholesky", "recursive": "cholesky",
    "long_run": "long_run", "sign": "sign",
    "external_instrument": "proxy", "proxy": "proxy",
    "instrumental_variables": "proxy", "heteroskedasticity": "het",
}

# Stated deterministic specification -> config trend code. The entry's
# statement is reduced to its leading clause (text before any "(" or ";",
# which carry page citations and side remarks), lowercased and
# whitespace-normalised, then looked up EXACTLY: "constant" alone must not
# match "constant + linear time trend". An unrecognised phrase fails, so
# a new wording is added here deliberately instead of passing by default.
_DETERMINISTIC_PHRASES = {
    "none": "n",
    "no constant": "n",
    "constant": "c",
    "constant only": "c",
    "intercept": "c",
    "constant + linear time trend": "ct",
    "constant and linear time trend": "ct",
    "constant + trend": "ct",
    "constant and trend": "ct",
    "constant + linear trend": "ct",
    "constant + linear + quadratic time trend": "ctt",
}


def _deterministic_code(statement: str) -> str | None:
    head = re.split(r"[(;]", statement, maxsplit=1)[0]
    head = " ".join(head.lower().split())
    return _DETERMINISTIC_PHRASES.get(head)


# (entry_id, field) -> the exact entry/config values the deviation
# authorises, plus a short reason; each mirrors documentation that already
# exists in the replication's config.yaml comments and/or README. A
# declaration covers only those two values: any other value on either side
# is undeclared drift.
DECLARED_SPEC_DEVIATIONS = {
    ("glp_2015_prior_selection", "lags"): {
        "entry": 5, "config": 4,
        "reason": "method-demo runs the CEE-1999 4-lag system (config "
                  "header: 'DIVERGES on purpose'; README spec table "
                  "paper-vs-demo)"},
    ("glp_2015_prior_selection", "n_endogenous"): {
        "entry": 3, "config": 7,
        "reason": "demo runs the CEE 7-var system, not GLP's small-3 "
                  "(config header)"},
    ("lenza_primiceri_2022_covid_var", "lags"): {
        "entry": 13, "config": 5,
        "reason": "quarterly demo, 5 lags; the paper's illustration is "
                  "monthly with 13 (declared inline at config model.lags)"},
    ("lenza_primiceri_2022_covid_var", "n_endogenous"): {
        "entry": 7, "config": 4,
        "reason": "4-var quarterly demo vs the paper's 7 monthly variables "
                  "(README)"},
    ("schorfheide_song_2015_mfvar", "lags"): {
        "entry": 6, "config": 4,
        "reason": "FRED demo uses 4 VAR lags + 5 Mariano-Murasawa state "
                  "lags; the paper's p(m)=6 (README spec table)"},
    ("banbura_giannone_lenza_2015_conditional_forecasts", "n_endogenous"): {
        "entry": 13, "config": 7,
        "reason": "7-var demo of the paper's 26-var system; the entry lists "
                  "representative variables"},
    ("banbura_modugno_2014_bdfm", "n_endogenous"): {
        "entry": 7, "config": 15,
        "reason": "15-series FRED panel vs the entry's representative "
                  "composition list"},
    ("cee_1996_flow_of_funds", "n_endogenous"): {
        "entry": 7, "config": 10,
        "reason": "config fetches the D_t-variant (UNRATE) and fn-11 "
                  "alternative-PCOM series beyond the 6-var benchmark "
                  "(README 'NOT asserted')"},
    ("lanne_meitz_saikkonen_2017_nongaussian_svar", "n_endogenous"): {
        "entry": 4, "config": 3,
        "reason": "3-var system because the paper's KCFCI is not publicly "
                  "available (README)"},
}


def _norm_ident(s: str) -> str:
    s = (s or "").lower()
    for key, val in _IDENT_SYNONYMS.items():
        if key in s:
            return val
    return s


def _entries_with_configs() -> list[Path]:
    return [p for p in _entry_files()
            if (REPLICATIONS / p.stem / "config.yaml").exists()]


def _spec_comparisons(entry: dict, config: dict) -> dict:
    """Machine-comparable fields: field -> (entry_value, config_value, equal)."""
    e_model = entry.get("model") or {}
    c_model = config.get("model") or {}
    out = {}

    # Lag order (only when both sides are concrete integers)
    e_lags, c_lags = e_model.get("lags"), c_model.get("lags")
    if isinstance(e_lags, int) and isinstance(c_lags, int):
        out["lags"] = (e_lags, c_lags, e_lags == c_lags)

    # Deterministics: exact stated terms vs the config trend code
    e_det = e_model.get("deterministics") or e_model.get("deterministic")
    c_trend = c_model.get("trend")
    if isinstance(e_det, str) and isinstance(c_trend, str):
        out["deterministics"] = (
            e_det, c_trend, _deterministic_code(e_det) == c_trend)

    # Identification method (normalised vocabulary)
    e_ident = (entry.get("identification") or {}).get("method")
    c_ident = (config.get("identification") or {}).get("method")
    if isinstance(e_ident, str) and isinstance(c_ident, str):
        en, cn = _norm_ident(e_ident), _norm_ident(c_ident)
        if en not in ("none", "varies"):
            out["identification"] = (e_ident, c_ident, en == cn)

    # Endogenous-variable count vs the config's MODEL variables. The
    # variable_order block names what actually enters the VAR; the raw
    # series list may carry auxiliary inputs used only in preprocessing
    # (e.g. the CPI column that nominalises M1REAL) and must not count.
    e_vars = (e_model.get("variables") or {}).get("endogenous")
    data_cfg = config.get("data") or {}
    c_model_vars = data_cfg.get("variable_order")
    if not isinstance(c_model_vars, list):
        c_model_vars = data_cfg.get("series")
    if isinstance(e_vars, list) and isinstance(c_model_vars, list):
        named = [v for v in e_vars
                 if isinstance(v, dict) and v.get("name") not in (None, "varies")]
        if named:
            out["n_endogenous"] = (
                len(named), len(c_model_vars), len(named) == len(c_model_vars))
    return out


def _spec_drift_problems(rid: str, entry: dict, config: dict) -> list[str]:
    problems = []
    for field, (e_val, c_val, ok) in _spec_comparisons(entry, config).items():
        if ok:
            continue
        if field == "deterministics" and _deterministic_code(e_val) is None:
            problems.append(
                f"deterministics: entry statement {e_val!r} is not a "
                f"recognised phrase: add it to _DETERMINISTIC_PHRASES"
            )
            continue
        declared = DECLARED_SPEC_DEVIATIONS.get((rid, field))
        if declared is not None:
            if e_val == declared["entry"] and c_val == declared["config"]:
                continue
            problems.append(
                f"{field}: entry says {e_val!r}, config runs {c_val!r}, but "
                f"the declared deviation authorises only entry="
                f"{declared['entry']!r} / config={declared['config']!r}"
            )
            continue
        problems.append(
            f"{field}: entry says {e_val!r}, config runs {c_val!r} "
            f"— fix one side or declare it in DECLARED_SPEC_DEVIATIONS"
        )
    return problems


def _load_entry_and_config(rid: str) -> tuple[dict, dict]:
    entry = yaml.safe_load(
        (ENTRIES / f"{rid}.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (REPLICATIONS / rid / "config.yaml").read_text(encoding="utf-8"))
    return entry, config


@pytest.mark.parametrize("path", _entries_with_configs(), ids=lambda p: p.stem)
def test_entry_and_replication_config_agree(path: Path):
    rid = path.stem
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (REPLICATIONS / rid / "config.yaml").read_text(encoding="utf-8"))
    problems = _spec_drift_problems(rid, entry, config)
    assert not problems, f"{rid}: entry/config drift:\n  " + "\n  ".join(problems)


def test_declared_spec_deviations_are_not_stale():
    """Every declared deviation must point at an existing entry+config pair
    that still differs on that field with exactly the declared values; a
    registry row that no longer resolves, or no longer describes the
    current pair, is itself drift."""
    ids = {p.stem for p in _entries_with_configs()}
    stale = []
    for (rid, field), declared in DECLARED_SPEC_DEVIATIONS.items():
        if rid not in ids:
            stale.append((rid, field, "no entry+config pair"))
            continue
        cmp = _spec_comparisons(*_load_entry_and_config(rid)).get(field)
        if cmp is None:
            stale.append((rid, field, "field not compared"))
        elif cmp[2]:
            stale.append((rid, field, "sides now agree"))
        elif (cmp[0], cmp[1]) != (declared["entry"], declared["config"]):
            stale.append((rid, field, f"values now {cmp[0]!r}/{cmp[1]!r}"))
    assert not stale, f"stale DECLARED_SPEC_DEVIATIONS rows: {stale}"


@pytest.mark.parametrize("rid,field,value", [
    # Dropping the linear trend the Sims entry requires.
    ("sims_1980_var", "trend", "c"),
    # Lag drift on an entry with no declared lag deviation.
    ("sims_1980_var", "lags", 40),
    # Lag drift beyond the 5 -> 4 demo pair GLP declares.
    ("glp_2015_prior_selection", "lags", 40),
])
def test_spec_drift_negative_controls(rid, field, value):
    """In-memory config changes that the guard must reject."""
    entry, config = _load_entry_and_config(rid)
    assert not _spec_drift_problems(rid, entry, config)
    config["model"][field] = value
    problems = _spec_drift_problems(rid, entry, config)
    assert problems and any(
        p.startswith("deterministics" if field == "trend" else field)
        for p in problems
    ), problems


def test_deterministic_phrases_are_exact():
    assert _deterministic_code("constant") == "c"
    assert _deterministic_code(
        "constant + linear time trend (paper p. 18); no seasonals") == "ct"
    assert _deterministic_code("none") == "n"
    # "constant" inside a longer statement is not the constant-only spec.
    assert _deterministic_code("constant + linear time trend") != "c"
    assert _deterministic_code("constant plus something new") is None


# ---------------------------------------------------------------------------
# 9. Every replication folder has its core runnable files
# ---------------------------------------------------------------------------


def test_replication_dirs_have_core_files():
    if not REPLICATIONS.exists():
        pytest.skip("no replications directory")
    bad = []
    for d in sorted(REPLICATIONS.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        for req in ("config.yaml", "replicate.py"):
            if not (d / req).exists():
                bad.append(f"{d.name}/{req}")
    assert not bad, f"replication folders missing core files: {bad}"
