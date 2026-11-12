---
name: advisor-protocol
description: >
  Model-agnostic escalation protocol for complex mathematical reasoning, derivation
  verification, and theoretical validation. Use when the executing agent encounters
  dense mathematical content from a paper, unfamiliar identification schemes, non-standard
  prior derivations, or any situation where the agent's confidence in correctly translating
  mathematics to code is low. This skill routes the query to a more capable reasoning
  model; the specific provider and model are configurable and not hardcoded.
---

# Advisor protocol

## Purpose

The executing agent handles the routine workflow. When mathematical content
exceeds its confidence, it asks a more capable reasoning model to check the
work before proceeding, as a researcher would consult a senior colleague on a
tricky derivation.

## When to escalate

The executing agent should invoke the advisor when:

1. **Unfamiliar derivations.** A paper introduces notation, distributional assumptions,
   or algebraic steps the agent cannot confidently verify.
2. **Novel identification schemes.** Sign restrictions with complex penalty functions,
   narrative identification with inequality constraints, or set-identified models.
3. **Non-standard priors.** Any prior structure not covered by the
   `scripts/lib/bvar.py` module docstrings or `CONVENTIONS.md`.
4. **Distributional theory.** Posterior derivations involving non-conjugate cases,
   marginalisation over nuisance parameters, or asymptotic approximations.
5. **Ambiguous mathematical notation.** When a paper's notation is inconsistent or
   context-dependent and the agent is unsure of the correct interpretation.
6. **Cross-checking implementations.** After coding a complex formula, the agent can
   ask the advisor to verify the code matches the mathematics.

**Do not escalate for:** routine matrix algebra, standard OLS/GLS, well-documented
Minnesota prior computation, basic time-series transformations, or anything already
covered in verified reference documents.

## Advisor configuration

The advisor is off by default. Before any call, read `advisor.enabled` in the
repo-root `config.yaml`. If it is not `true`, do not call the advisor: tell
the user the step would normally be escalated, continue without it, and
suggest checking the derivation by hand or enabling the advisor.

The advisor is configured in the repo-root `config.yaml` under `advisor:`
(`enabled`, provider, model, endpoint, `api_key_env`, an optional fallback provider, and
settings: `max_attempts`, `timeout_seconds`, `log_all_exchanges`,
`require_human_review`). The model named there is an example; set one the
user has access to. Users or administrators set this once; the agent reads it
at runtime.

The agent sends the query through the configured provider's API; the
library implements no advisor calls.

## Escalation protocol

### Step 1: Prepare the query

The executing agent constructs a structured prompt for the advisor:

```
CONTEXT:
I am implementing [model/method] from [paper reference].
The executing environment is [Python/R/Julia].

MATHEMATICAL CONTENT:
[Extract the relevant equations, definitions, and notation from the paper.
Include surrounding context so the advisor understands the setup.]

SPECIFIC QUESTION:
[One of:]
- "Verify that the following derivation is correct: [steps]"
- "Translate this mathematical expression into pseudocode: [expression]"
- "This paper uses notation X for Y. Please confirm my interpretation: [interpretation]"
- "I have implemented this formula as [code]. Verify it matches the mathematics."
- "Derive the posterior distribution for [parameter] given [prior] and [likelihood]."

CONSTRAINTS:
- Provide step-by-step reasoning.
- Flag any assumptions that are implicit in the paper but should be made explicit.
- If the derivation has multiple valid interpretations, list them.
- Express the final result in explicit matrix notation with dimensions annotated.
```

### Step 2: Call the advisor

Send the query through the configured provider's API as a standard request,
with no provider-specific logic beyond API format translation. If the primary
provider fails and a fallback is set, try the fallback. If no provider answers,
tell the user to verify the mathematical content manually before proceeding.

### Step 3: Process the response

1. Parse the advisor's response.
2. Extract: verified derivation steps, pseudocode, flagged assumptions, and any
   alternative interpretations.
3. **Present to the human for review** (if `require_human_review` is true).
   The review happens after the call and before the response is used.
4. Log the full exchange via the research-log skill.

### Step 4: Incorporate into workflow

Once the human confirms the advisor's output:

1. Save the verified derivation as a reference document in the relevant skill's
   `references/` directory (created on first use; e.g., `<skill-name>/references/verified_xyz_derivation.md`)
   or alongside the lib module that needs it
   (e.g., `scripts/lib/references/verified_xyz_derivation.md`).
2. Use the pseudocode to implement the formula.
3. The reference document persists, so future runs of the same method do not need
   to re-escalate.

This creates a growing library of verified mathematical content over time.

## Cost management

Advisor calls are expensive relative to routine execution. For each session,
the agent records in the research log the number of advisor calls, tokens
per call, and a cost estimate from the provider's published pricing. If costs
are a concern, the user can:

- Ask the agent to confirm with the user before each advisor call (an
  explicit step before invoking; `require_human_review` reviews the response
  after the call, so it saves no cost)
- Use a local model as the advisor for cost-free but potentially lower-quality review
- Pre-populate `references/` with verified content to reduce future escalations

## Institutional deployment considerations

Central banks and research institutions may have:

- **Air-gapped networks.** Use a locally hosted model as the advisor.
- **Data confidentiality constraints.** The advisor prompt should never include
  proprietary data, only mathematical content from published or working papers.
  The skill instructs the agent to send only equations and derivations, never
  datasets or proprietary data. This is a policy, not a technical filter.
- **Approved vendor lists.** The configurable provider system accommodates this:
  set only the approved provider(s) in the configuration.
- **Audit requirements.** The agent logs every advisor exchange in the
  research log, with timestamps, prompts, responses, and human confirmation
  status.
