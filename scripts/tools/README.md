# Developer tools

Small maintenance utilities that operate on the framework's own files
(not estimation code, which lives in [`scripts/lib/`](../lib/)).

| Tool | Purpose |
|---|---|
| [`make_author_dossiers.py`](make_author_dossiers.py) | Render each model-dictionary entry into an author-facing dossier (what we attribute to the paper with page cites, what we reproduce, what we changed, what is our editorial content). The defensibility test: an entry is in good standing when its dossier could be sent to the paper's original author as-is. See the field-provenance section in [`model-dictionary/SKILL.md`](../../skills/model-dictionary/SKILL.md). |

Outputs go to the co-located `outputs/` subfolder (gitignored,
regenerable), following the repo-wide convention.
