---
name: consistency-audit
description: >
  On-demand framework self-audit. Checks the three layers (skills ↔ lib ↔
  dictionary) for drift, internal inconsistency, and doc-vs-practice mismatch.
  Use when the user asks to "audit the framework", "check for consistency /
  drift", "is the workflow consistent", "go through the end-to-end process",
  before a release or promotion, or after a batch of edits to skills, lib, or
  the dictionary. This is NOT a stage of the 7-stage research loop; it is a
  maintenance pass run occasionally, on demand, by a framework developer.
---

# Consistency audit

The framework's three layers (the `skills/` workflow + knowledge, the
`scripts/lib/` implementation, and the `skills/model-dictionary/` specs) must
not silently drift out of agreement.

There are **two layers of defence**, and you run both:

| Layer | Where | Cadence | Catches |
|---|---|---|---|
| **Mechanical** | `tests/test_doc_consistency.py` | every test run (automated) | YAML parse breaks, index↔entries drift, stale module/entry counts, invalid `verified.status`, broken skill→file links, dangling `lib.*` references, replication folders missing core files |
| **Semantic** | this skill | on demand | stage-numbering coherence, propose/confirm choreography, doc-vs-practice, orphaned/overlapping skills, terminology drift (the judgement calls a regex can't make) |

The mechanical layer is necessary but **not sufficient**: a green test suite
with an incoherent workflow is still broken. Run the semantic pass too.

## When to run

- The user asks to audit / check the framework, or to "go through the
  end-to-end process".
- Before promoting work to canonical, or before a release.
- **After a batch of edits to skills, lib, or the dictionary**, especially a
  paper rebuild. Rebuilds can silently break dependent tests and docs.

## Procedure

### Step 1: Mechanical checks (fast, deterministic)

```
python -m pytest tests/test_doc_consistency.py -q
```

Every failure is a concrete drift with a `file:line`. Fix or report each. If
during the audit you find a new class of drift that is *mechanically* checkable
(a count, a reference, a parse, a path), **encode it as a new test here** so it
can never regress. That is how the mechanical layer grows.

### Step 2: Semantic audit (judgement)

A regex can't read meaning. By reading the skills, check:

1. **Stage-numbering coherence.** The orchestrator is the source of truth:
   Stages 0a (knowledge-base initialisation) and 0b (intake & triage) fold
   into the Stage-1 log entry; Stage 1 = mode-specific entry flow, 2 = specification, 3 = data,
   4 = estimation, 5 = sensibility, 6 = results, 7 = disposition. Every
   sub-skill that names a stage must match.
2. **Propose → confirm choreography.** Every decision-point skill must *propose
   a specific choice with rationale and wait for confirmation*, never
   blank-ask (design principle 3). Flag any skill that merely describes.
3. **Doc-vs-practice.** Spot-check that what the skills / CLAUDE.md claim
   matches what the code does: output paths, default hyperparameters, file
   locations, method availability.
4. **Orphaned / overlapping skills.** Is every skill in the orchestrator's
   invocation table reachable from a stage (or clearly marked optional/on
   request)? Do two skills run the same check without a stated division of
   labour?
5. **Terminology / vocabulary.** The same concept uses the same name across
   skills, e.g. the three `verified.status` levels
   (`draft | checked_against_paper | signed_off`), the mode labels (A/B/C).
6. **Source-priority agreement.** Every statement of the knowledge-source
   cascade uses the one canonical order, `user > model dictionary > knowledge
   base > parametric > data-driven`, with economic logic checked at every
   level, defined once in `specification/SKILL.md`.

### Step 3: Live execute smoke (prove the path, not just the prose)

Run one fast verified replication end-to-end and confirm it still passes its
paper targets:

```
python scripts/replications/blanchard_quah_1989/replicate.py
```

Expect a `6/6` paper-target comparison table and output files under the script's
co-located `outputs/`. A clean mechanical + semantic audit with a broken
execute path is still a broken framework.

### Step 4: Report & fix (propose, confirm)

Compile a prioritized findings list (**P1** misleads the agent or user · **P2**
real friction or latent contradiction · **P3** polish), each item with a
`file:line` and a one-line fix. Per design principle 3, **propose the fixes and
wait for the user's confirmation** before editing; then re-run Steps 1–3 to
green. Record the audit (date, findings, dispositions) in your own notes, and
update CONVENTIONS.md or docstrings where a finding changes behaviour.

## Adding new invariants

When the audit surfaces a new class of drift, decide where it belongs:

- **Mechanically checkable** (counts, references, parse-ability, paths, schema)
  → add a test to `tests/test_doc_consistency.py`. **Prefer this**: it runs
  continuously and can't be forgotten.
- **Judgement-only** (semantics, choreography, design coherence) → add a bullet
  to Step 2 above so the next on-demand audit covers it.
