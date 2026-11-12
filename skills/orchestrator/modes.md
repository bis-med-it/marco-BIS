# Three modes of entry: reference

This document expands the brief overview in [SKILL.md](SKILL.md#three-modes-of-entry).
Read it when triaging a new session, drafting a new dictionary entry from a paper,
or deciding whether a mode transition is warranted.

## Mode A: Dictionary execution (deterministic)

The user names a paper, method, or specification that has a verified entry in
[`skills/model-dictionary/entries/`](../model-dictionary/entries/). Examples:

- "Replicate Sims 1980."
- "Run the CEE 1999 VAR on FRED data."
- "Estimate a sign-restricted VAR with Uhlig's restrictions."

**Path:**

1. Look up the matching entry by id or citation. If multiple candidates, present
   them and ask the user to pick.
2. **Search both tiers** for a runnable folder:
   - [`scripts/replications/<id>/`](../../scripts/replications/) (canonical, ranked first)
   - [`scripts/applications/<id>/`](../../scripts/applications/) (your past work)

   If a folder exists in either tier, the deterministic path is
   `python scripts/<tier>/<id>/replicate.py`.
   If both tiers match, present them and let the user pick; canonical is the
   default. Confirm any data substitutions the user wants, then run.
3. If no folder exists in either tier, generate a thin `replicate.py` following
   the [conventions in `scripts/replications/`](../../scripts/replications/README.md):
   load config, call [`scripts/lib/`](../../scripts/lib/README.md) functions,
   save outputs via `VARReport`. The new folder goes to `scripts/replications/`
   only when it's a clean canonical replication of a verified dictionary entry
   (`verified.status: checked_against_paper` or `signed_off`); otherwise to
   `scripts/applications/`.
4. Stage 1 is the brief 1A confirmation of the entry; skip Stage 2 (specification
   design), because the specification is already determined by the dictionary entry. The user only confirms data
   substitutions and any intended deviations.

Everything from variable selection to identification to expected results comes
from an entry already checked against the paper.

## Mode B: Paper submission (parse, then execute)

The user submits a paper (PDF path, URL, citation, or plain text) that does not
yet have a dictionary entry. Examples:

- "Here's a PDF. Replicate this."
- "Replicate Smith and Jones (2024, JME) on monetary policy."
- "Take this arXiv paper and apply it to euro area data."

**Path:** follow Stage 1B in
[SKILL.md](SKILL.md#stage-1b-paper-submission): duplicate check, verify the
staged file (look in `papers/` first; files are named `<id>.pdf`), draft the entry,
present ambiguities (for example, a sample start the paper does not pin to a
quarter, or an ordering given only in prose), escalate a novel method to
`advisor-protocol`, then fall through to Mode A. Stage 4 writes the working
script to `scripts/applications/<id>/`; Stage 7 confirms the final tier.

Each completed paper adds a verified dictionary entry that future sessions can
reuse.

## Mode C: Research design (parametric)

The user asks an open research question without naming a specific method or
paper. Examples:

- "How does monetary policy affect output and inflation?"
- "What's the best way to identify a fiscal multiplier in euro area data?"
- "I want to study the transmission of commodity price shocks."

**Path:**

1. Run `model-selection` against the question, the knowledge base, and the
   dictionary.
2. **If model-selection finds one or more matching dictionary entries** with
   high confidence, present them and offer to upgrade to Mode A.
3. **If no dictionary match exists**, use parametric knowledge to design a
   specification: pick a model class, propose variables and identification,
   recommend a sample period. Flag every decision's knowledge source.
4. After spec design and successful execution, Stage 4 writes the working
   `replicate.py` to `scripts/applications/<id>/` by default. The final
   disposition (archive / save as application / promote canonical / continue /
   discard) is confirmed in **Stage 7 (Disposition & curation)**. Mode C runs
   that pass sensibility checks typically save as application + optionally
   draft a dictionary entry. A new entry stays `verified.status: draft` until
   its paper-attributed fields are checked against the source; `signed_off`
   also requires the user's approval of that checked entry. Approving the
   application or its results does not change an entry's status.

This mode is where the agent does the most independent reasoning. The dictionary
serves as a reference; parametric knowledge fills the gaps.

## Mode transitions (extended)

Modes are not one-way. A session can move between them mid-flow:

- **C → A**: model-selection finds a clean dictionary match; promote to Mode A.
  *Example:* user asks "how does MP affect output?", search finds
  `cee_1999_monetary_svar` and `sims_1980_var`, user picks CEE → switch to Mode A.
- **B → C**: paper extraction fails or the method is too novel to map cleanly;
  ask the user to describe the research question instead. *Example:* user
  submits a 60-page DSGE paper with a Bayesian estimation routine we can't
  implement → ask "what specifically do you want from this paper?" and design
  from there.
- **A → C**: user wants to deviate so much from the dictionary entry that
  they're effectively designing a new spec; demote to Mode C with the entry as
  a starting reference. *Example:* user picks CEE (1999) but asks to swap 5 of
  7 variables; that's a new spec, not CEE.
- **C → B**: user finds a paper mid-design and wants to anchor to it.
  *Example:* user is designing from scratch, then says "actually here's a paper
  that did exactly this" and uploads it.

Always announce the transition explicitly: *"Switching to Mode A: your question
maps onto CEE (1999), which is in the dictionary. Want me to run that
specification?"* Log every transition in the research log with the trigger.
