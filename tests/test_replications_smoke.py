"""End-to-end smoke tests for every canonical replication and application.

Each test runs ``scripts/<replications|applications>/<id>/replicate.py``
in a fresh Python subprocess, treating the script's exit code as the
pass/fail signal. The script must print "Done." in stdout to count as
successful.

These tests close the silent-regression failure mode where a change to
``scripts/lib/`` keeps all unit tests green but breaks a replication.
With minimum-viable bootstrap settings, the full suite runs in about
20 minutes.

Marked ``slow`` so the default ``pytest tests/`` skips them. To run:

  pytest tests/test_replications_smoke.py --runslow

To run a single replication:

  pytest tests/test_replications_smoke.py --runslow -k "litterman"

Behaviour:

* **Pass**: subprocess exits 0 AND prints "Done." in stdout AND every
  paper-target comparison it prints reports ``passed == total``.
* **Fail**: subprocess exits non-zero AND no stderr line is a known
  "data missing" message; OR the run completes but a
  ``compare_to_paper`` summary reports fewer passes than targets
  (numerical regression against the source paper). The last 30-40
  lines of stderr + stdout are included in the failure message for
  debugging.
* **Skip**: any referenced ``data/staging/<file>.csv`` doesn't exist,
  OR a stderr line is exactly one of the known data-missing messages
  in ``DATA_MISSING_SIGNATURES`` (e.g., "*** Staging CSV not found:
  <path>").

Skipped tests are reported with the staging-file path so the user
knows which data to fetch / stage.

Per-script timeout: 600 seconds (10 minutes), with explicit per-id
overrides in ``TIMEOUT_OVERRIDES`` for replications whose verified
settings are legitimately heavier (document the reason next to each
entry). Anything else exceeding the default should be split or sped up.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLICATIONS_DIR = PROJECT_ROOT / "scripts" / "replications"
APPLICATIONS_DIR = PROJECT_ROOT / "scripts" / "applications"
PER_SCRIPT_TIMEOUT_SEC = 600

# Per-id overrides for replications whose PINNED settings are legitimately
# heavier than the default budget. Keep each entry justified.
TIMEOUT_OVERRIDES = {
    # Higher-moment screening over sign-passing rotations: 800k-rotation
    # budget for 2x500 accepted draws (acceptance rate ~0.005 is itself a
    # paper target). ~890s measured single-run under the fixed-column
    # rotation measure; results are seed-pinned, so the remedy is
    # budget, not fewer draws.
    "andrade_ferroni_melosi_2023_higher_moments": 1500,
}


def _discover_scripts() -> list[tuple[str, str, Path]]:
    """Yield (kind, id, replicate.py path) for each replication and application."""
    scripts: list[tuple[str, str, Path]] = []
    for kind, parent in (
        ("replication", REPLICATIONS_DIR),
        ("application", APPLICATIONS_DIR),
    ):
        if not parent.exists():
            continue
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir():
                continue
            replicate_py = folder / "replicate.py"
            if replicate_py.exists():
                scripts.append((kind, folder.name, replicate_py))
    return scripts


def _config_staging_paths(folder: Path) -> list[Path]:
    """Inspect the folder's config.yaml for any referenced staging-area paths.

    Recognised patterns:
      * ``data.staging_path`` (Canova-Ciccarelli)
      * ``data.shocks_path`` (Romer-Romer, similar)
      * any top-level ``staging_path`` / ``data_file`` fields
    """
    config_path = folder / "config.yaml"
    if not config_path.exists():
        return []
    try:
        cfg = yaml.safe_load(config_path.read_text())
    except Exception:
        return []
    if not isinstance(cfg, dict):
        return []
    paths: list[Path] = []

    def _maybe_add(value):
        if isinstance(value, str) and (
            "data/staging" in value or value.endswith(".csv")
        ):
            paths.append(PROJECT_ROOT / value)

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("staging_path", "shocks_path", "data_file", "file"):
                    _maybe_add(v)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(cfg)
    return paths


# Complete data-missing messages, one regex per message, each matched
# against a WHOLE stripped stderr line (``re.fullmatch``). They are the
# exact exit messages the replication scripts write before
# ``SystemExit(1)`` when staged / user-supplied data is absent:
#   canova_ciccarelli_2013_panel_var   "*** Staging CSV not found: <path>"
#   mertens_ravn_2013_tax_proxy_svar   "*** Mertens-Ravn 2013 staging CSV not found ***"
#   romer_romer_2004_narrative_mp      "*** Romer-Romer shock CSV not found ***"
# Fragments elsewhere in the output (significance stars in a results
# table, an unrelated "config not found") must not turn a failure into
# a skip, so there is no generic "*** ... not found" fallback. A new
# script with a staged-data exit adds its exact message here.
DATA_MISSING_SIGNATURES = [
    r"\*\*\* Staging CSV not found: \S.*",
    r"\*\*\* Mertens-Ravn 2013 staging CSV not found \*\*\*",
    r"\*\*\* Romer-Romer shock CSV not found \*\*\*",
]
# NB deliberately NOT a skip signature: "not in registry". A config typo
# raises the same message as an unfetched cache-on-fetch dataset, and a
# typo must fail, not skip; dataset-id validity is checked statically in
# tests/test_doc_consistency.py, and genuinely missing cached data
# surfaces through the explicit file-not-found signatures above.


def _looks_like_missing_data(stderr: str) -> str | None:
    """Return the matched stderr line if one is a complete known
    data-missing message (see ``DATA_MISSING_SIGNATURES``); None
    otherwise."""
    for line in (stderr or "").splitlines():
        line = line.strip()
        for sig in DATA_MISSING_SIGNATURES:
            if re.fullmatch(sig, line):
                return line
    return None


_SUMMARY_RE = re.compile(
    r"Summary:\s*(\d+)/(\d+)\s*pass(?:\s*\((\d+)\s*fail\))?")


def _paper_summary_failure(kind: str, rid: str, stdout: str) -> str | None:
    """Return a failure message if the printed paper-target summaries do
    not show a complete pass; None if they do.

    Every summary must have ``passed == total``; a parenthesised fail
    count, when present, must equal ``total - passed`` (hence zero). A
    canonical replication must print at least one nonempty summary.
    """
    summaries = [
        (int(m.group(1)), int(m.group(2)),
         None if m.group(3) is None else int(m.group(3)))
        for m in _SUMMARY_RE.finditer(stdout or "")
    ]
    if kind == "replication":
        if not summaries:
            return (
                f"{kind} {rid!r}: ran to 'Done.' but printed no paper-target "
                f"comparison — a vacuous pass. Every canonical replication "
                f"must run lib.compare_to_paper against its "
                f"paper_targets.yaml."
            )
        if not any(tot > 0 for _n, tot, _f in summaries):
            return (
                f"{kind} {rid!r}: paper-target comparison was empty "
                f"(0 targets)."
            )
    for n_pass, n_tot, n_fail in summaries:
        if n_pass > n_tot:
            return (
                f"{kind} {rid!r}: malformed summary {n_pass}/{n_tot} pass."
            )
        if n_fail is not None and n_fail != n_tot - n_pass:
            return (
                f"{kind} {rid!r}: inconsistent summary {n_pass}/{n_tot} pass "
                f"({n_fail} fail)."
            )
        if n_pass != n_tot:
            return (
                f"{kind} {rid!r}: ran to completion but "
                f"{n_tot - n_pass} of {n_tot} paper target(s) FAILED "
                f"(summary {n_pass}/{n_tot} pass): numerical regression "
                f"against the source paper."
            )
    return None


_SCRIPTS = _discover_scripts()


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind,rid,replicate_py",
    _SCRIPTS,
    ids=[f"{k}:{rid}" for (k, rid, _) in _SCRIPTS],
)
def test_replication_runs_to_done(
    kind: str, rid: str, replicate_py: Path,
) -> None:
    """The replication / application script must run cleanly and print 'Done.'.

    Skips with a clear message if a referenced staging-data file is missing.
    """
    folder = replicate_py.parent

    # Pre-flight: skip if any referenced staging path is absent
    missing = [p for p in _config_staging_paths(folder) if not p.exists()]
    if missing:
        pretty = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        pytest.skip(f"missing staging data ({pretty})")

    # Run the script in a fresh subprocess. Use UTF-8 to handle any non-ASCII
    # output (e.g., ε in some replication printouts).
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    timeout_sec = TIMEOUT_OVERRIDES.get(rid, PER_SCRIPT_TIMEOUT_SEC)
    try:
        result = subprocess.run(
            [sys.executable, str(replicate_py)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{kind} {rid!r}: timed out after {timeout_sec}s"
        )

    # If exit code != 0, check whether stderr matches a known
    # data-missing signature → skip rather than fail
    if result.returncode != 0:
        matched = _looks_like_missing_data(result.stderr or "")
        if matched:
            pytest.skip(
                f"{kind} {rid!r} reports missing data (matched '{matched}')"
            )
        # Otherwise it's a real failure
        stderr_tail = "\n".join((result.stderr or "").splitlines()[-30:])
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-30:])
        pytest.fail(
            f"\n{kind} {rid!r}: exit code {result.returncode}.\n"
            f"  script: {replicate_py.relative_to(PROJECT_ROOT)}\n"
            f"--- stderr (last 30 lines) ---\n{stderr_tail}\n"
            f"--- stdout (last 30 lines) ---\n{stdout_tail}"
        )

    # Exit code 0 but never printed Done. → silent failure
    if "Done." not in (result.stdout or ""):
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-30:])
        pytest.fail(
            f"\n{kind} {rid!r}: exited 0 but did not print 'Done.' — "
            f"likely silent failure.\n"
            f"--- stdout (last 30 lines) ---\n{stdout_tail}"
        )

    # Paper-target enforcement: replicate.py prints lib.compare_to_paper's
    # table when paper_targets.yaml exists ("Summary: N/M pass (K fail)").
    # A run that completes but reports a failing target is a numerical
    # regression, not a success — fail loudly instead of relying on a
    # human to read the table. (Honest divergences are encoded as targets
    # with adjusted expectations, so a fail here is never intentional.)
    #
    # A canonical replication that runs to "Done." without printing ANY
    # nonempty comparison validates nothing — require the summary. (A pass
    # must mean "targets checked", not "script exited".) The renderer
    # omits the "(K fail)" suffix when nothing fails, so the pass count
    # itself is checked against the total.
    failure = _paper_summary_failure(kind, rid, result.stdout or "")
    if failure:
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-40:])
        pytest.fail(
            f"\n{failure}\n"
            f"--- stdout (last 40 lines) ---\n{stdout_tail}"
        )


# ---------------------------------------------------------------------------
# Fast unit tests of the harness helpers (no subprocess; not slow)
# ---------------------------------------------------------------------------


class TestPaperSummaryParser:
    @pytest.mark.parametrize("stdout", [
        "Summary: 2/2 pass\nDone.",
        "Summary: 2/2 pass (0 fail)\nDone.",
        "Summary: 3/3 pass\nSummary: 5/5 pass\nDone.",
        "Summary: 0/0 pass\nSummary: 4/4 pass\nDone.",
    ])
    def test_complete_pass_accepted(self, stdout):
        assert _paper_summary_failure("replication", "x", stdout) is None

    @pytest.mark.parametrize("stdout", [
        "Summary: 1/2 pass (1 fail)\nDone.",
        "Summary: 1/2 pass\nDone.",            # no parenthesised count
        "Summary: 0/8 pass\nDone.",
        "Summary: 4/4 pass\nSummary: 2/3 pass (1 fail)\nDone.",
        "Summary: 2/2 pass (1 fail)\nDone.",   # inconsistent counts
        "Summary: 1/3 pass (1 fail)\nDone.",   # inconsistent counts
        "Summary: 3/2 pass\nDone.",            # malformed
    ])
    def test_incomplete_or_inconsistent_rejected(self, stdout):
        assert _paper_summary_failure("replication", "x", stdout)
        assert _paper_summary_failure("application", "x", stdout)

    def test_replication_requires_nonempty_summary(self):
        assert _paper_summary_failure("replication", "x", "Done.")
        assert _paper_summary_failure(
            "replication", "x", "Summary: 0/0 pass\nDone.")
        # Applications need not compare to a paper.
        assert _paper_summary_failure("application", "x", "Done.") is None


class TestMissingDataSignatures:
    @pytest.mark.parametrize("stderr", [
        # Exact messages written by the replication scripts.
        "\n*** Staging CSV not found: C:/repo/data/staging/cc.csv\n"
        "Run prepare_data.py first.\n\n",
        "\n*** Mertens-Ravn 2013 staging CSV not found ***\n\n"
        "Expected file:   C:/repo/data/staging/mr.csv\n",
        "\n*** Romer-Romer shock CSV not found ***\n\n"
        "Expected file:   C:/repo/data/staging/rr.csv\n",
    ])
    def test_known_messages_skip(self, stderr):
        assert _looks_like_missing_data(stderr)

    @pytest.mark.parametrize("stderr", [
        # Configuration errors are failures, not missing data.
        "FileNotFoundError: config not found: missing-config.yaml",
        "Per-pair diagnostics: F=40 ***\n"
        "FileNotFoundError: config not found: missing-config.yaml",
        "*** something else ***\nKeyError: column not found",
        "FileNotFoundError: paper_targets file not found: x.yaml",
        # A known message quoted inside other text is not the message.
        "warning: '*** Romer-Romer shock CSV not found ***' was expected",
        "",
    ])
    def test_unrelated_output_does_not_skip(self, stderr):
        assert _looks_like_missing_data(stderr) is None

    def test_signatures_match_script_sources(self):
        """Each signature corresponds to a message still present in a
        replication script, so a reworded exit message is noticed."""
        sources = "\n".join(
            p.read_text(encoding="utf-8")
            for p in REPLICATIONS_DIR.glob("*/replicate.py")
        )
        for literal in (
            "*** Staging CSV not found: {path}",
            "*** Mertens-Ravn 2013 staging CSV not found ***",
            "*** Romer-Romer shock CSV not found ***",
        ):
            assert literal in sources, literal
