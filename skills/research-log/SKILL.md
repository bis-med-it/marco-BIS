---
name: research-log
description: >
  Automatic documentation of every decision, iteration, and rationale in the econometric
  analysis workflow. Use whenever any other skill in the MaRCo@BIS suite completes
  a stage, or when the user asks for an audit trail, reproducibility log, or session summary.
---

# Research log

## Purpose

Maintain a complete, timestamped record of the analysis for reproducibility. This log
becomes the basis for a replication appendix in the published paper.

## What gets logged

Every orchestrator stage writes an entry. Each entry contains:

```
log_entry:
  timestamp: ISO 8601
  stage: "knowledge_base_init" | "intake_triage" | "model_selection" |
         "specification" | "data_retrieval" | "data_disposition" |
         "estimation" | "diagnostics" | "sensibility_check" |
         "revision" | "final_results" | "tool_review" | "disposition"
         # A session paused before estimation still closes with a
         # "disposition" entry (decision: continue) whose next_steps
         # records where to resume.
  iteration: integer (1 for first attempt, increments on revisions)
  inputs: summary of what was provided to this stage
  outputs: summary of what was produced
  decisions:
    - description: what was decided
      rationale: why (economic or statistical reasoning)
      alternatives_considered: what else could have been done
      knowledge_source: user | dictionary | knowledge_base | parametric | data_driven
                        # which level of the specification skill's source-priority
                        # order informed the decision (CLAUDE.md rule 3)
      human_confirmed: bool
  flags: any warnings or concerns raised
  code_hash: SHA-256 of the script used (optional; always record the script
             PATH; hash too when convenient, e.g.
             hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
```

## Storage

- **During session:** append stage entries to the `entries` array of
  `session_research_log.json` in the run's `outputs/` folder (shape below).
- **Filename layering:** replication/application scripts write a *run-level*
  `research_log.json` (provenance of one execution) into their `outputs/`,
  and a re-run replaces it. Keep that name for the script's run provenance
  only; session decisions never go in it.
- **Session records are not regenerable:** `session_research_log.json`,
  `research_log.md` and `run_state.json` sit in the gitignored `outputs/`
  folder, and no re-run rebuilds them. Copy them out before deleting
  `outputs/` (see the orchestrator's Stage 7 "Discard" option).
- **Multi-session continuation:** when a later session resumes work
  disposed as "continue iterating", EXTEND the existing log rather than
  starting over: append the new session's stage entries, keep the prior
  disposition in a `disposition_history` array (the latest disposition is
  the one of record), and continue the `version_history` numbering (a
  resumed session's first new spec is v(n+1), not v1). Decisions the prior
  session confirmed are *settled*: re-confirm, don't re-decide, and log
  anything actually re-opened as a `revision` with its trigger.
- **Version tracking:** each revision creates a new version. Previous versions are
  never deleted. The session log is one object holding the stage `entries`,
  the `disposition_history`, the `version_history` and any `branches`:

  ```json
  {
    "entries": [...],
    "disposition_history": [...],
    "version_history": [
      {
        "version": "v1",
        "created": "2026-04-11T10:30:00Z",
        "status": "superseded by v2",
        "changed_in_next": ["added exchange rate variable"],
        "stages": [...]
      },
      {
        "version": "v2",
        "created": "2026-04-11T11:15:00Z",
        "parent": "v1",
        "status": "current",
        "changes_from_parent": ["added NEER to endogenous variables",
                                 "updated Cholesky ordering"],
        "stages": [...]
      }
    ],
    "branches": {
      "main": "v2",
      "robustness_8lags": "v2-alt1"
    }
  }
  ```

- **On completion:** generate a human-readable markdown summary (`research_log.md`)
  suitable for inclusion as an appendix or supplementary material. Include a version
  history section showing what was tried, what was changed, and why.
- **Code archive:** the runnable script lives at its tiered home
  (`scripts/<tier>/<id>/replicate.py`), *not* duplicated in the log directory.
  Record the script path and a per-version content hash in the log, and store
  only the diffs between versions, never parallel copies of the script.
- **Diff generation:** when presenting a revision to the user, the agent writes
  a human-readable diff between versions, highlighting what changed in
  specification, results, and interpretation.

## Run state file (`run_state.json`)

The research log is the *record*; `run_state.json` is the *pointer*: a tiny,
machine-readable sensor of where the run stands, designed to survive context
compaction, interruptions, and topic pivots. The orchestrator's
[run-completion contract](../orchestrator/SKILL.md#run-completion-contract)
reads it before concluding any piece of work; this section owns the schema.

### Location

Next to the session research log in the run's co-located outputs folder:
`scripts/<tier>/<id>/outputs/run_state.json`, gitignored like everything
else in `outputs/`. One file per run.

### Schema

```json
{
  "run_id": "eu_commodity_spillover",
  "script_dir": "scripts/applications/eu_commodity_spillover/",
  "mode": "C",
  "current_stage": 5,
  "stages_completed": [1, 2, 3, 4],
  "status": "open",
  "disposition": null,
  "next_action": "present IRF sensibility verdict to the user",
  "updated_at": "2026-07-04T09:12:00Z"
}
```

- `current_stage` / `stages_completed` use the log's numeric stage numbering
  (the 0a/0b steps fold into stage 1, as in the markdown report format).
- `status` is `"open"` until Stage 7 records a disposition, then `"closed"`.
  Every disposition closes the file, including `continue` (the resuming
  session re-opens it: `status: "open"`, appending to the same research log
  per the multi-session continuation rule above).
- `disposition` mirrors the log's `disposition.decision` once decided.
- `next_action` is one sentence: what the next turn should do. Keep it
  current: it is the first thing a resumed session acts on.

### Lifecycle

| Moment | Write |
|---|---|
| Stage 1: mode confirmed, working folder exists | Create with `status: "open"`, `current_stage: 1` |
| Every stage gate (same moment the stage's log entry is appended) | Update `current_stage`, `stages_completed`, `next_action`, `updated_at` |
| Stage 7: disposition recorded | Set `status: "closed"`, fill `disposition` |

### Read triggers

- **Session start / after context compaction:** read it before doing
  anything else on a resumed run; `next_action` says where to pick up.
- **Before concluding any reply that ends a piece of work:** if `status`
  is `"open"` and `current_stage < 7`, surface the pending stages: proceed,
  or propose an explicit deferral that gets logged. Never let a run go
  silent.
- **Start of Stage 7:** to enumerate what was completed before proposing a
  disposition.

## Tool promotion queue

The orchestrator's Stage 7 tool-review sub-step (see
[`skills/orchestrator/SKILL.md`](../orchestrator/SKILL.md#tool-review-sub-step))
emits "flag for promotion" candidates: inline helper functions written
in this session's `replicate.py` that look reusable but haven't appeared
in a second caller yet. These get appended to a session-spanning queue
managed here.

### Storage

`skills/research-log/tools_to_promote.yaml`: one YAML file at the skill's root
holding flagged candidates across sessions. It is gitignored.

### Schema

```yaml
candidates:
  - function: extract_cholesky_shock
    first_seen:
      session: "<date>-<run-id>"
      location: scripts/applications/<id>/replicate.py
      commit: "abc1234"
    appearances:
      - session: "<date>-<run-id>"
        location: scripts/applications/<id>/replicate.py
        commit: "abc1234"
      - session: "<later-date>-<run-id>"
        location: scripts/applications/<other-id>/replicate.py
        commit: "def5678"
    proposed_lib_home: scripts/lib/var.py
    proposed_signature: |
      def extract_cholesky_shock(fit: VARFit, shock_var: str) -> pd.Series:
          """Recover the structural shock for ``shock_var`` from a fitted VAR."""
          ...
    status: "queued"   # queued | promoted | rejected
    rationale: "Generic structural-shock extraction; 2 callers already, candidate for lib.var"
    notes: ""
```

### Consolidation trigger

When a Stage 7 tool review identifies a candidate whose name + signature
match an existing queue entry, the agent flags this as a **consolidation
opportunity** and proposes promote-now rather than flag-again. The
threshold defaults to "second caller": when N ≥ 2 callers exist for a
function with matching signature, propose promotion.

A "matching signature" is conservative: name match plus parameter-name
match plus same number of positional parameters. If the names are similar
but the parameters drift (e.g., `extract_cholesky_shock(fit, shock_var)`
vs `extract_chol_shock(fit, var_name, sign=+1)`), the agent surfaces the
mismatch and asks whether to unify the API before promotion.

### Query interface

The `tools_to_promote.yaml` file is queried by the orchestrator at:

- **Start of Stage 7:** to surface "you have N queued candidates from
  prior sessions; any of them match helpers in today's script?"
- **Start of Stage 4:** to suggest reusable lib candidates the user
  might want to use directly as primitives in the generated
  `replicate.py`, rather than rewriting the same helper inline. (This
  is the proactive consolidation step that catches "I'm about to write
  the third copy of `extract_cholesky_shock`".)

### Lifecycle

- **Queued.** Flagged but no promotion yet. Stays in the file
  indefinitely until either promoted or explicitly rejected.
- **Promoted.** Function added to lib. Status updated; the entry
  retains the historical record (when promoted, by which session, which
  commit).
- **Rejected.** Explicitly marked as "do not promote, keep ad-hoc
  forever", e.g., the user decided the function is paper-specific
  despite a generic-looking signature, or the function's logic is
  better delivered as a sub-skill rather than a lib primitive. Status
  preserved so the agent doesn't re-propose it in a future session.

### When the queue itself gets reviewed

At the start of any developer session (the user is working ON the
framework; see CLAUDE.md Step 1), the agent
should optionally surface the queue contents (*"you have N queued tool
candidates from prior sessions, most-promotion-overdue: …"*) so that
deferred consolidation doesn't get forgotten across many sessions. This
is a hint, not a mandatory step.

## Markdown report format

```markdown
# Research log: [analysis title]

## Session metadata
- Date: ...
- Tool version: ...
- Language: Python 3.x / R 4.x

(Stage numbers track the orchestrator: the 0a/0b steps (knowledge-base
initialisation, then intake & triage) fold into the Stage-1 log entry;
1 = mode-specific entry flow, 2 = specification, 3 = data, 4 = estimation, 5 = sensibility,
6 = results, 7 = disposition.)

## Stage 1: Intake, triage & entry flow
[mode (A/B/C), dictionary match if any, model-selection outcome]

## Stage 2: Specification
[narrative summary of decisions]

## Stage 3: Data retrieval
[variable mapping table, transformations, any discrepancies noted]

## Stage 4: Estimation (iteration 1)
[model details, key results]

## Stage 5: Sensibility check (iteration 1)
[what passed, what failed, what was revised]

...

## Stage 6: Results and reporting
[final results: summary of accepted specification and key findings]

## Stage 7: Disposition & curation
[archive only / save as application / promote to canonical / continue
iterating / discard:
chosen option + rationale; tool-promotion flags if any]

## Appendix: code listing
[or reference to scripts/ directory]
```

## Usage by other skills

The agent appends records in the `log_entry` schema defined above to the
session log, and attaches the log to reports with `VARReport.add_research_log`.

The research-log skill handles serialisation and formatting. Other skills do not need
to manage the log file directly.
