# Applications

This tier holds **worked applications**: analyses built *with* the framework
that go beyond canonical replication: variants of canonical specs (new
samples, new economies), adaptations to local data, and comparative or
methodological studies. Each folder is a self-contained, re-runnable piece of
research: read the README, run `replicate.py`, get the outputs.

## Starting point

This folder ships empty and fills through use: every Stage 7 "save as
application" disposition creates a folder here. For worked examples of
the folder shape and standards, see the canonical replications, e.g.
[`scripts/replications/uhlig_2005_sign_restrictions/`](../replications/uhlig_2005_sign_restrictions/)
(a verified dictionary entry run end to end, with documented FRED substitutes for the paper's data) or
[`scripts/replications/gali_1999_technology_var/`](../replications/gali_1999_technology_var/)
(a fully reproducible end-to-end example that runs on the pre-collected
FRED-QD panel alone, after the one-time download in `data/fred_qd/UPDATE.md`).

## When work goes here vs `scripts/replications/`

| Folder | What goes here |
|---|---|
| [`scripts/replications/<id>/`](../replications/) | Canonical replications, the public-good gallery. Each pairs with a dictionary entry checked against the paper. |
| `scripts/applications/<id>/` | Your own analyses: variants, local adaptations, comparative studies. A dictionary entry, when one exists, usually carries a spec-deviation note. |

## How new folders arrive: Stage 7 disposition

When a run completes, the orchestrator's
[Stage 7 (Disposition & curation)](../../skills/orchestrator/SKILL.md)
asks how to handle the code that was just produced. **"Save as
application"**, the default for successful Mode B and Mode C runs, is
what creates a folder in this tier: the run's README, config, and script
are kept here. The generated outputs stay regenerable; the session records
in `outputs/` do not (see "What to keep vs what regenerates" below).

## Folder convention (same shape as replications)

```
applications/
└── <id>/
    ├── README.md            question, spec + deviations, results, caveats, how to run
    ├── config.yaml          spec choices (mirrors dictionary entry if one exists)
    ├── replicate.py         thin orchestration script; calls scripts/lib/
    ├── paper_targets.yaml   (optional) declared paper values for validation
    └── outputs/             generated reports + numerical snapshots (gitignored)
```

The id should describe the analysis: `uhlig_2005_updated_us`,
`fiscal_lp_state_dependence`, etc. Use snake_case.

## What to keep vs what regenerates

- **Keep** (and commit, if you put your copy under version control):
  `README.md`, `config.yaml`, `replicate.py` (and `paper_targets.yaml`
  if present). This is your intellectual work, recoverable across machines and
  sessions.
- **Regenerable**: the generated files in `outputs/` (reports, numerical
  snapshots, the run-level `research_log.json`), rebuilt from the script +
  data. The shipped [.gitignore](../../.gitignore) already excludes nested
  `outputs/` directories, so no special setup is needed.
- **Not regenerable (copy out before deleting `outputs/`)**: the session
  records the agent writes there (`session_research_log.json`, the optional
  `research_log.md`, and `run_state.json`). They hold the decisions, gates and
  continuation state of the conversation, which no re-run can reconstruct,
  and because `outputs/` is excluded from version control they are not
  committed with the rest of your work.

## Orchestrator behaviour

Mode A's lookup searches both gallery tiers:

1. [`scripts/replications/<id>/`](../replications/): canonical (ranked first).
2. `scripts/applications/<id>/`: your past work (surfaced as "you've done
   something similar; want to start from there?").

A clean canonical match takes precedence. A personal application surfaces as a
suggestion the user can pick, not the default.

## Promotion to canonical

If an application matures into something paper-faithful and rigorously
verified (e.g., you replicate a canonical paper exactly), promote it: move the
folder from `scripts/applications/<id>/` to `scripts/replications/<id>/` and
update the dictionary entry's verification record (see the
[model dictionary](../../skills/model-dictionary/SKILL.md)). This is rare; most
applications stay applications.
