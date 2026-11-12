"""Tests for the report layer's economic-narrative support.

Two things are covered:

1. ``lib.draft_narrative`` — turns IRF / FEVD / stability / Stage-5 flags into a
   factual narrative *skeleton* (signs, magnitudes, peaks, convergence, variance
   shares, stability, flagged concerns) that the agent then refines. Pure logic,
   tested with a duck-typed IRF stand-in.
2. ``VARReport.add_standard_sections`` narrative wiring — ``narrative=None`` now
   *warns* (the report would otherwise carry no economic interpretation),
   ``narrative="auto"`` injects a drafted skeleton, and a string is verbatim.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

import lib
from lib.irf import IRFContainer


# ---------------------------------------------------------------------------
# draft_narrative — pure logic
# ---------------------------------------------------------------------------

def _fake_irf():
    """A duck-typed IRF: output dips then recovers (converges); prices rise and
    do NOT return to zero (so the non-convergence caveat should fire)."""
    H = 8
    median = np.zeros((H + 1, 2, 1))
    median[:, 0, 0] = [-0.3, -0.45, -0.5, -0.4, -0.25, -0.12, -0.05, -0.02, -0.01]
    median[:, 1, 0] = [0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.36, 0.38, 0.40]
    return types.SimpleNamespace(
        median=median, variable_names=["output", "prices"],
        shock_names=["mp"], horizons=np.arange(H + 1),
    )


def test_draft_narrative_core():
    txt = lib.draft_narrative(
        _fake_irf(), max_eigenvalue=0.86, spec={"Model": "VAR(2)"},
        flags=["Price puzzle: prices rise after the MP shock"],
    )
    assert "output" in txt and "prices" in txt
    assert "mp" in txt
    assert "down on impact" in txt                 # output impact = -0.3
    assert "0.86" in txt and "mean-reverting" in txt
    assert "Price puzzle" in txt                   # Stage-5 flag surfaced
    assert "[DRAFT NARRATIVE" in txt               # un-refined skeleton is obvious
    # prices never return to zero -> the non-convergence caveat names them
    caveats = txt.split("Points to interpret")[1]
    assert "have not returned toward zero" in caveats
    assert "prices" in caveats


def test_draft_narrative_flags_instability():
    txt = lib.draft_narrative(_fake_irf(), max_eigenvalue=1.02)
    assert "NOT stable" in txt


def test_draft_narrative_fevd_names_dominant_shock():
    H = 4
    median = np.zeros((H + 1, 2, 2))
    median[:, 0, 0] = 1.0
    median[:, 1, 1] = 1.0
    irf = types.SimpleNamespace(
        median=median, variable_names=["a", "b"],
        shock_names=["sa", "sb"], horizons=np.arange(H + 1),
    )
    # (n_resp, n_horizon, n_shock); at the last horizon sa has the largest share of a, sb of b
    fevd = np.array([
        [[0.5, 0.5], [0.8, 0.2]],
        [[0.5, 0.5], [0.1, 0.9]],
    ])
    txt = lib.draft_narrative(irf, fevd=fevd)
    assert "variance at the longest horizon" in txt
    assert "a (sa, 80%)" in txt
    assert "b (sb, 90%)" in txt


def test_draft_narrative_shock_filter():
    H = 4
    median = np.zeros((H + 1, 1, 2))
    median[:, 0, 0] = 0.5
    median[:, 0, 1] = -0.5
    irf = types.SimpleNamespace(
        median=median, variable_names=["y"],
        shock_names=["s1", "s2"], horizons=np.arange(H + 1),
    )
    txt = lib.draft_narrative(irf, shock="s2")
    assert "Response to a s2 shock" in txt
    assert "Response to a s1 shock" not in txt


def test_draft_narrative_rejects_non_3d():
    bad = types.SimpleNamespace(median=np.zeros((5, 2)), variable_names=["a", "b"],
                                shock_names=["s"], horizons=np.arange(5))
    with pytest.raises(ValueError, match="3-D"):
        lib.draft_narrative(bad)


# ---------------------------------------------------------------------------
# add_standard_sections narrative wiring
# ---------------------------------------------------------------------------

def _real_irf(n=2, H=8):
    med = np.zeros((H + 1, n, n))
    for s in range(n):
        med[:, s, s] = np.linspace(1.0, 0.0, H + 1)   # decaying diagonal response
    return IRFContainer(
        median=med, lower_68=med - 0.1, upper_68=med + 0.1,
        lower_90=med - 0.2, upper_90=med + 0.2,
        variable_names=[f"y{i + 1}" for i in range(n)],
        shock_names=[f"s{i + 1}" for i in range(n)],
    )


def _stub_fit(n=2):
    # Only .variable_names is needed up front; diagnostics/FEVD/summary will
    # raise on this stub and be warn-skipped — which is fine for these tests.
    return types.SimpleNamespace(variable_names=[f"y{i + 1}" for i in range(n)])


def _narrative_html(report):
    return [html for title, html in report.sections if title == "Narrative"]


def test_add_standard_sections_warns_when_no_narrative():
    rep = lib.VARReport("t")
    with pytest.warns(UserWarning) as rec:
        rep.add_standard_sections(_stub_fit(), _real_irf())
    messages = [str(w.message) for w in rec]
    assert any("no narrative provided" in m for m in messages)
    # The stub fit lacks FEVD inputs; the graceful-degradation path must
    # warn-and-skip rather than abort the report build. Recording the
    # warnings here also keeps them out of the suite-level summary.
    assert any("FEVD skipped" in m for m in messages)
    assert not _narrative_html(rep)        # and no narrative section was added


def test_add_standard_sections_auto_narrative():
    rep = lib.VARReport("t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # ignore the diagnostics/FEVD skip warnings
        rep.add_standard_sections(_stub_fit(), _real_irf(), narrative="auto")
    narr = _narrative_html(rep)
    assert narr and "[DRAFT" in narr[0]


def test_add_standard_sections_verbatim_narrative():
    rep = lib.VARReport("t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep.add_standard_sections(_stub_fit(), _real_irf(),
                                  narrative="My economic story.")
    narr = _narrative_html(rep)
    assert narr and "My economic story." in narr[0]


# ---------------------------------------------------------------------------
# Standard-section truthful-label contracts
# ---------------------------------------------------------------------------
#
# The report is the diagnostics gate the user actually reads, so its labels
# must describe what was computed: the summary section names the shock it
# summarises; cumulative columns exist only at horizons the IRF covers;
# Ljung-Box rows carry the lag the test was run at; FEVD columns are labelled
# with SHOCK names (response variables label the rows).


def _full_report_sections(synthetic_var_data):
    fit = lib.fit_var(data=synthetic_var_data, lags=2, trend="c")
    irf = lib.cholesky_irf(fit, horizon=12, bands="parameter_draws",
                           n_draws=60, seed=7)
    rep = lib.VARReport("t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep.add_standard_sections(
            fit, irf, narrative="auto",
            var_labels=["Output", "Prices", "Rate"],
        )
    return dict(rep.sections), irf


def test_summary_title_names_the_shock(synthetic_var_data):
    sections, irf = _full_report_sections(synthetic_var_data)
    shock0 = str(irf.shock_names[0]).replace("_", " ").title()
    # default recursive shock names already end in "shock": no doubled word
    assert shock0.endswith("Shock") and f"{shock0} Summary" in sections
    assert not any("Shock Shock" in title for title in sections)
    assert "Monetary Shock Summary" not in sections


def test_summary_cum_columns_match_available_horizon(synthetic_var_data):
    sections, _ = _full_report_sections(synthetic_var_data)
    summary_html = next(html for title, html in sections.items()
                        if title.endswith("Shock Summary"))
    assert "Cum(4)" in summary_html and "Cum(8)" in summary_html
    # horizon is 12 — columns beyond it must not be fabricated
    assert "Cum(20)" not in summary_html and "Cum(40)" not in summary_html


def test_diagnostics_rows_carry_actual_lb_lag(synthetic_var_data):
    sections, _ = _full_report_sections(synthetic_var_data)
    diag_html = sections["Diagnostics"]
    assert "Ljung-Box(8)" in diag_html          # lib.diagnostics default
    assert "Ljung-Box(4)" not in diag_html      # nothing was computed at 4/12
    assert "Ljung-Box(12)" not in diag_html


def test_fevd_columns_are_shock_names(synthetic_var_data):
    sections, irf = _full_report_sections(synthetic_var_data)
    fevd_html = sections["Forecast Error Variance Decomposition (%)"]
    for s in irf.shock_names:
        assert f"<th>{s}</th>" in fevd_html
    # response variables label the rows (display labels), not the columns
    assert "<strong>Output</strong>" in fevd_html


def test_add_summary_legacy_rows_and_default_title():
    rep = lib.VARReport("t")
    rep.add_summary([{
        "variable": "y", "peak_response": -0.5, "peak_horizon": 3,
        "ci68_excludes_zero": True, "ci90_excludes_zero": False,
        "cum_4": -1.0, "cum_8": -1.5, "cum_20": -1.7, "cum_40": -1.7,
    }])
    title, html = rep.sections[-1]
    assert title == "Shock Summary"
    assert "Cum(40)" in html                     # legacy shape keeps its columns


def test_add_fevd_explicit_shock_names_and_note():
    rep = lib.VARReport("t")
    fevd = np.full((2, 2, 2), 0.5)
    rep.add_fevd(fevd, ["Output", "Prices"], [1, 4],
                 shock_names=["technology", "nontechnology"],
                 note="shares rescaled")
    _, html = rep.sections[-1]
    assert "<th>technology</th>" in html and "<th>nontechnology</th>" in html
    assert "shares rescaled" in html


# ---------------------------------------------------------------------------
# add_irf: y-axis unit label and custom section title
# ---------------------------------------------------------------------------
#
# Published IRF figures label the y-axis with the response unit ("Percent"),
# not the variable name, and a single report may carry more than one IRF block
# (e.g. an agnostic-sign identification beside a recursive one). ``units`` and
# ``section_title`` support both; the defaults must stay byte-for-byte the same.


def test_add_irf_units_label_and_section_title():
    rep = lib.VARReport("t")
    irf = _real_irf(n=2, H=8)
    rep.add_irf(
        irf.median, irf.lower_68, irf.upper_68, irf.lower_90, irf.upper_90,
        var_names=["Output", "Prices"], shock_labels=["s1", "s2"],
        units="Percent", section_title="IRF (agnostic)",
    )
    titles = [t for t, _ in rep.sections]
    assert "IRF (agnostic)" in titles                   # custom heading used
    assert "Impulse Response Functions" not in titles   # default not emitted
    html = next(h for t, h in rep.sections if t == "IRF (agnostic)")
    assert '"text":"Percent"' in html                   # unit lives on the y-axis
    assert "Output" in html and "Prices" in html         # names live in subplot titles


def test_add_irf_defaults_unchanged():
    # Backward compatibility: with no ``units`` the y-axis carries the variable
    # name and the section keeps its canonical title.
    rep = lib.VARReport("t")
    irf = _real_irf(n=2, H=8)
    rep.add_irf(
        irf.median, irf.lower_68, irf.upper_68, irf.lower_90, irf.upper_90,
        var_names=["Output", "Prices"], shock_labels=["s1", "s2"],
    )
    title, html = rep.sections[-1]
    assert title == "Impulse Response Functions"
    assert '"text":"Percent"' not in html               # no unit leaks in
    assert '"text":"Output"' in html                    # variable name on the y-axis


# ---------------------------------------------------------------------------
# Regression tests for report-layer presentation bugs
# ---------------------------------------------------------------------------
# Distinct shares at non-consecutive horizons: equal-share fixtures cannot
# detect a label/value mismatch.

def _distinct_fevd(n_h):
    """(1 response, n_h horizons, 2 shocks); slice k has shares (k+1)/10."""
    out = np.zeros((1, n_h, 2))
    for k in range(n_h):
        out[0, k] = [(k + 1) / 10, 1 - (k + 1) / 10]
    return out


def _fevd_rows(html):
    import re
    return re.findall(r"<td class='num'>(\d+)</td><td class=\"num\"[^>]*>([\d.]+)</td>", html)


def test_add_fevd_compact_table_pairs_each_label_with_its_slice():
    rep = lib.VARReport("t")
    rep.add_fevd(_distinct_fevd(3), ["y"], [1, 4, 8], shock_names=["s1", "s2"])
    _, html = rep.sections[-1]
    assert _fevd_rows(html) == [("1", "10.0"), ("4", "20.0"), ("8", "30.0")]


def test_add_fevd_full_path_is_indexed_by_horizon():
    rep = lib.VARReport("t")
    rep.add_fevd(_distinct_fevd(10), ["y"], [1, 4, 8], shock_names=["s1", "s2"])
    _, html = rep.sections[-1]
    assert _fevd_rows(html) == [("1", "20.0"), ("4", "50.0"), ("8", "90.0")]


def test_add_fevd_rejects_horizon_beyond_full_path():
    rep = lib.VARReport("t")
    with pytest.raises(ValueError):
        rep.add_fevd(_distinct_fevd(5), ["y"], [1, 4, 8], shock_names=["s1", "s2"])


def test_standard_fevd_table_matches_structural_fevd_per_horizon():
    rng = np.random.default_rng(0)
    T, A = 400, np.array([[0.5, 0.4], [0.0, 0.5]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(2)
    import pandas as pd
    fit = lib.fit_var(pd.DataFrame(y, columns=["a", "b"]), lags=1)
    from lib.report import _compute_fevd_table
    S = np.linalg.cholesky(np.asarray(fit.Sigma_u))
    horizons = [1, 4, 8]
    table = _compute_fevd_table(fit, S, horizons)
    rep = lib.VARReport("t")
    rep.add_fevd(table, ["a", "b"], horizons, shock_names=["s1", "s2"])
    _, html = rep.sections[-1]
    rows = _fevd_rows(html)[:3]                     # response "a"
    for (label, shown), h in zip(rows, horizons):
        expect = 100 * lib.structural_fevd(fit, S, horizon=h)[0, 0]
        assert label == str(h) and abs(float(shown) - expect) < 0.051


def test_draft_narrative_keeps_partial_shares_of_total_variance():
    H = 4
    irf = types.SimpleNamespace(
        median=np.ones((H + 1, 2, 1)), variable_names=["a", "b"],
        shock_names=["mp"], horizons=np.arange(H + 1),
    )
    fevd = np.array([[[0.30], [0.36]], [[0.90], [0.96]]])   # one identified shock
    txt = lib.draft_narrative(irf, fevd=fevd)
    assert "a (mp, 36%)" in txt and "b (mp, 96%)" in txt
    assert "100%" not in txt
    assert "not identified here" in txt


def test_point_only_irf_has_no_band_traces_or_band_claim():
    point = np.ones((5, 2, 2))
    irf = IRFContainer.from_point(point, ["a", "b"], ["s1", "s2"],
                                  metadata={"bands": "none"})
    rep = lib.VARReport("t")
    rep.add_irf(point, point, point, point, point, ["a", "b"], ["s1", "s2"])
    _, html = rep.sections[-1]
    assert "90% CI" not in html and "68% CI" not in html
    assert "uncertainty intervals were not computed" in html
    txt = lib.draft_narrative(irf)
    assert "surrounded by the plotted confidence" not in txt
    assert "Point estimates only" in txt


@pytest.mark.parametrize("norm, expect_table", [
    ("reliability", True), ("one_sd", False), ("unit_policy", False)])
def test_proxy_fevd_only_for_unit_variance_normalisation(norm, expect_table):
    """The standard FEVD table needs a unit-variance shock. Of the proxy-SVAR
    normalisations only ``reliability`` provides one: ``one_sd`` scales the
    impact to the policy residual's standard deviation, which would attribute
    nearly all of the policy variable's impact variance to the policy shock."""
    import pandas as pd
    rng = np.random.default_rng(1)
    T = 600
    B0 = np.array([[0.5, 0.8, 0.3], [0.4, 1.0, 0.2], [0.3, 0.1, 1.0]])
    E = rng.standard_normal((T, 3))
    U = E @ B0.T
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = 0.5 * Y[t - 1] + U[t]
    idx = pd.date_range("1970-01-01", periods=T, freq="MS")
    fit = lib.fit_var(pd.DataFrame(Y, index=idx, columns=["policy", "y", "p"]), lags=1)
    z = pd.Series(E[:, 0] + 0.5 * rng.standard_normal(T), index=idx)
    irf, _ = lib.proxy_svar_irf(fit, instrument=z, policy_var="policy",
                                horizon=4, bands="none", normalization=norm)
    rep = lib.VARReport("t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep.add_standard_sections(fit, irf, narrative="auto")
    html = dict(rep.sections)["Forecast Error Variance Decomposition (%)"]
    assert ("FEVD omitted" not in html) is expect_table
    if expect_table:
        # in this DGP the policy shock explains about a quarter of the policy
        # variable's impact variance (0.25 / 0.98), nowhere near all of it
        share = 100 * lib.structural_fevd(fit, irf.median[0][:, :1], horizon=1)[0, 0]
        assert 15 < share < 45

