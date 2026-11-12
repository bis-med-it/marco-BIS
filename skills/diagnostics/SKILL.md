---
name: econometric-diagnostics
description: >
  Residual diagnostics and model adequacy checks for VAR/BVAR models. Use when the
  orchestrator reaches the diagnostics step of Stage 4 (estimation), whose results
  feed the Stage 5 sensibility check, or when the user asks to check residuals,
  test for autocorrelation, heteroskedasticity, normality, or structural breaks.
---

# Econometric diagnostics

## Tests performed

**Built-in bundle.** `lib.diagnostics(fit)` provides:

- per equation: Ljung-Box (one or more lags), Jarque-Bera and ARCH-LM;
- system level: the maximum companion-matrix eigenvalue modulus, with a
  status label (stability);
- an AIC/BIC/HQ lag-order comparison: it refits an OLS VAR at
  p = 1..`max_lag_search` and reports which p each criterion picks.

The other tests below are marked *additional*. They are not part of the
built-in bundle; the agent can run them separately (e.g. via statsmodels)
when the question calls for them.

### Residual-level diagnostics

For each equation in the VAR system:

1. **Autocorrelation.** Ljung-Box test at lags 4, 8, 12 (quarterly) or 6, 12, 24 (monthly).
   (`lib.diagnostics(fit, lb_lags=[...])` computes all requested lags in one
   call; the report's diagnostics table labels each row with the lag actually
   used.) *Additional:* Breusch-Godfrey LM test for the system.
2. **Heteroskedasticity.** ARCH-LM test (Engle, 1982) to detect conditional
   heteroskedasticity. *Additional:* White test for unconditional
   heteroskedasticity.
3. **Normality.** Jarque-Bera test per equation. *Additional:*
   Lomnicki-Jarque-Bera for the system.
   Note: non-normality is common in macro data and is not a reason to reject the model
   unless extreme.

### System-level diagnostics

1. **Stability.** Companion matrix eigenvalues. Report the maximum root and classify the dynamics as stationary (below 1),
   unit-root (at 1 within sampling error) or explosive (above 1). Apply the
   declared model-class and replication criteria consistently, treat
   near-unit-root sampling uncertainty as a disclosed diagnostic, and do not
   certify stability from finite-horizon plots.
   (`lib.diagnostics` reports the maximum modulus; plotting the eigenvalues
   on the unit circle diagram is an additional step.)
   **Arbitration for LEVELS systems:** `lib.diagnostics` labels any
   max |eig| ≥ 1.0 "non-stationary / unstable", but a VAR in log-levels
   (or with a random-walk Minnesota prior) legitimately sits at or just
   above the unit circle; several canonical replications validate
   against an expected band like 1.0 ± 0.05. Reconcile in this order:
   (i) does the spec intend unit-root-like persistence (levels +
   shrinkage)? then near-1 is expected, not a failure; (ii) apply the
   Stage-5 standard for the IRFs: responses to transitory shocks should
   die out and responses to permanent shocks flatten to a finite limit;
   growth at long horizons is a failure signal, but a flat-looking response
   within a finite horizon does not show that a root above one is stable;
   (iii) treat the lib label as a prompt to run checks (i)-(ii) and to
   disclose the classification, never as an auto-reject or an auto-pass.
2. **Lag-order comparison.** AIC, BIC and HQ picks across p = 1..`max_lag_search`
   (built in; a pick that differs from the fitted p is listed in the warnings).
3. **Granger causality** (*additional*). Wald tests for block exogeneity.
   Useful for interpreting the system structure.
4. **Lag exclusion** (*additional*). Joint significance of each lag across all equations.
5. **Set-identified models: acceptance rate.** For sign/zero-restriction
   schemes, track accepted draws / attempts against the canonical run or
   the entry's expected range (the Uhlig replication validates ~0.06 ±
   0.04). Low acceptance or a missed `target_accepted` quota is a
   computational and precision warning. Acceptance depends on the number of
   variables, the restrictions, the proposal measure and the sampler, so
   compare like with like, inspect the accepted and effective draw counts,
   and check stability separately. Do not infer invalid restrictions from a
   universal acceptance threshold, and do not silently report results from
   too few accepted draws.

### Structural break detection (optional, additional)

1. CUSUM and CUSUM-of-squares tests.
2. If the sample is long, consider a split-sample stability check at known break dates
   (e.g., 1979Q4 for Volcker, 2008Q3 for GFC).

**Scope note: statistical vs economic anomalies.** This skill covers the
statistical gate. *Economic* anomalies (price puzzle, wrong-signed
responses, implausible magnitudes) are Stage-5 territory: the remedy
ladder lives in `skills/specification/SKILL.md` §"Iterative revision after
failed sensibility checks". Go there, not here, when the numbers are
statistically clean but economically wrong.

## Output format

**For the research log**, record `lib.diagnostics(...).to_log_dict()`, plus
any additional tests run separately and the agent's recommendation:

```
diagnostics:
  residuals:                                  # keyed by variable name
    M1:
      ljung_box: {4: {stat, p_value}, 8: ..., 12: ...}   # keyed by lag
      jarque_bera_stat, jarque_bera_pvalue
      arch_stat, arch_pvalue
      breusch_godfrey: {stat, p_value}        # additional; only if run separately
  stability: {max_eigenvalue_modulus, stable: bool, status}   # status: stationary | marginal (near unit root) | non-stationary / unstable
  lag_selection: {current_p, max_lag_search, aic_recommended, bic_recommended, hqic_recommended}
  portmanteau: {stat, p_value, lags}          # additional; only if run separately
  eigenvalues: list of complex numbers        # additional; only if run separately
  warnings:                                   # lib.diagnostics(...).warnings
    - "Autocorrelation detected in M1 equation at lag 4 (p=0.02)"
    - "ARCH effects in GNP deflator equation"
  recommendation: "Consider increasing lag order or adding conditional heteroskedasticity"   # written by the agent; not a lib field
```

**For the report**, do not build the table by hand.
`VARReport.add_standard_sections` passes the fit's `DiagnosticReport`
through the report layer's adapter (`_build_diagnostics_dict_from_report`
in `scripts/lib/report.py`). A direct call to `VARReport.add_diagnostics`
takes a reduced view, keyed by variable name with numeric Ljung-Box lags:

```yaml
per_equation:
  M1:
    ljung_box:
      4: {p_value: 0.30}
      8: {p_value: 0.40}
    arch_lm: {p_value: 0.50}
    jarque_bera: {p_value: 0.60}
system:
  stability:
    max_eigenvalue_modulus: 0.90
    note: stationary
```

Additional tests, warnings and recommendations stay in the log; the report
table shows only these fields.

## Propose → confirm (handing back to Stage 5 / Stage 4)

When presenting results to the user, lead with whether the model passes or
fails in plain language. Diagnostics never silently re-specifies the model. On a **critical** failure
(instability, or severe autocorrelation indicating misspecification):

1. **State the failed test in plain language** and why it matters economically,
   e.g. "the companion matrix has a root at 1.02, so the VAR is explosive and
   impulse responses won't die out."
2. **Propose one specific remedy with rationale**, drawn from the usual levers
   (+1 lag, reorder / add a variable, tighten the prior, shorten the sample),
   e.g. "I propose adding a 5th lag; the Ljung-Box failure at lag 4 points to
   unmodelled own dynamics."
3. **Wait for the user to confirm or override** before routing back to Stage 4
   for re-estimation. (The orchestrator's 5-iteration automatic budget covers
   only implementation/data-error fixes and pre-specified statistical rules;
   economically substantive revisions like this one are each gated.)

For **soft** warnings (mild non-normality, marginal ARCH), state them and
recommend proceeding; don't block. Either way the flag and its disposition are
logged via the research-log skill. Never auto-respecify without surfacing the
choice. This is design principle 3 (active suggestion, human confirmation)
applied to diagnostics.
