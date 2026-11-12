"""
Reusable HTML report generator for VAR/BVAR estimation results.

Produces a single HTML file (charts load Plotly from a CDN) with interactive Plotly charts,
styled tables, and collapsible metadata sections.

Usage:
    from lib.report import VARReport

    report = VARReport(title="My VAR", subtitle="Details...")
    report.add_irf(irf_point, ci_68_lo, ci_68_hi, ci_90_lo, ci_90_hi, ...)
    report.add_diagnostics(diag_dict, var_names)
    report.save("report.html")
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #ffffff;
    --fg: #1a1a2e;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #4a6fa5;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --section-bg: #f9fafb;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: "Georgia", "Times New Roman", serif;
    color: var(--fg);
    background: var(--bg);
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2rem 4rem;
}
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
h2 {
    font-size: 1.25rem;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--accent);
}
h3 { font-size: 1.05rem; margin: 1.5rem 0 0.5rem; color: var(--accent); }
.subtitle { color: var(--muted); font-size: 1rem; margin-bottom: 0.25rem; }
.timestamp { color: var(--muted); font-size: 0.85rem; margin-bottom: 2rem; }
.narrative { margin: 1rem 0; line-height: 1.7; }
.narrative p { margin-bottom: 0.8rem; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 600; background: var(--section-bg); }
tr:hover { background: #f3f4f6; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: var(--green); font-weight: 600; }
.warn { color: var(--amber); font-weight: 600; }
.fail { color: var(--red); font-weight: 600; }

/* Spec table */
.spec-table { max-width: 600px; }
.spec-table td:first-child { font-weight: 600; width: 40%; color: var(--muted); }

/* Tabs */
.tab-container { margin: 1rem 0; }
.tab-buttons { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 1rem; }
.tab-btn {
    padding: 0.5rem 1.2rem;
    border: none;
    background: none;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.9rem;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.15s;
}
.tab-btn:hover { color: var(--fg); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.tab-panel { display: none !important; }
.tab-panel.active { display: block !important; }

/* Collapsible */
details { margin: 1rem 0; }
summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--accent);
    padding: 0.5rem 0;
}
details pre {
    background: var(--section-bg);
    padding: 1rem;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.8rem;
    line-height: 1.4;
    max-height: 500px;
    overflow-y: auto;
}

/* Print */
@media print {
    body { max-width: none; padding: 1rem; }
    .tab-btn { display: none; }
    .tab-panel { display: block !important; page-break-inside: avoid; }
    details { open; }
}
"""

# ---------------------------------------------------------------------------
# Tab JS
# ---------------------------------------------------------------------------

_JS = """
function switchTab(groupId, tabIdx) {
    const buttons = document.querySelectorAll(`[data-group="${groupId}"] .tab-btn`);
    const panels = document.querySelectorAll(`[data-group="${groupId}"] .tab-panel`);
    buttons.forEach((b, i) => b.classList.toggle('active', i === tabIdx));
    panels.forEach((p, i) => p.classList.toggle('active', i === tabIdx));
    // Trigger Plotly resize for charts that were hidden
    const activePanel = panels[tabIdx];
    if (activePanel) {
        const plots = activePanel.querySelectorAll('.js-plotly-plot');
        plots.forEach(p => Plotly.Plots.resize(p));
    }
}
"""


# ---------------------------------------------------------------------------
# Adaptive layout helpers
# ---------------------------------------------------------------------------
#
# These compute Plotly figure layout parameters that scale gracefully with
# the number of subplots — grid dimensions, panel sizes, font sizes, spacing,
# legend placement, and margins are all derived from n_panels rather than
# hardcoded. Goal: a single VARReport call should produce a clean figure
# whether n_vars is 1 (single panel), 6 (typical monetary VAR), 12 (large
# BVAR), or 16+ (FAVAR-scale system).


def _smart_grid(n_panels):
    """Pick (n_rows, n_cols) for n_panels subplots.

    Optimises for a near-square arrangement when n_panels > 3, minimises
    empty cells, and avoids degenerate aspect ratios (e.g., 3+1 for n=4).
    """
    if n_panels <= 1:
        return (1, 1)
    if n_panels == 2:
        return (1, 2)
    if n_panels == 3:
        return (1, 3)
    if n_panels == 4:
        return (2, 2)
    if n_panels <= 6:
        return (2, 3)
    if n_panels <= 9:
        return (3, 3)
    if n_panels <= 12:
        return (3, 4)
    if n_panels <= 16:
        return (4, 4)
    # General: as square as possible
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    return (n_rows, n_cols)


def _smart_layout(n_panels, has_overlay=False):
    """Compute adaptive layout parameters for a multi-panel IRF figure.

    Parameters
    ----------
    n_panels : int
        Number of subplots to render (one per response variable).
    has_overlay : bool, default False
        Whether multiple IRF series are being overlaid in each panel
        (e.g., LP vs VAR comparison). Used to widen horizontal spacing
        slightly so legend / line styles don't crowd.

    Returns
    -------
    dict with keys:
        n_rows, n_cols              — grid dimensions
        panel_w_px, panel_h_px      — target panel size in pixels
        chart_height_px             — total figure height
        font_size_main              — base font (axis titles, etc.)
        font_size_subplot           — subplot title font (smaller for dense grids)
        font_size_tick              — tick label font
        h_spacing, v_spacing        — Plotly normalised spacing (from absolute px target)
        legend_orient               — 'h' (below) or 'v' (right side)
        legend_y                    — y-position for horizontal legend below
        margin_t, margin_b          — top/bottom margins (px)
        margin_l, margin_r          — left/right margins (px)
        subplot_title_max_chars     — wrap threshold for long variable names
    """
    n_rows, n_cols = _smart_grid(n_panels)

    # Per-panel size + font scaling by n_panels density
    if n_panels <= 1:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 380, 320, 13, 13, 11
    elif n_panels <= 2:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 340, 280, 12, 12, 10
    elif n_panels <= 4:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 300, 260, 12, 11, 10
    elif n_panels <= 6:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 280, 240, 11, 11, 10
    elif n_panels <= 9:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 260, 220, 11, 10, 9
    elif n_panels <= 12:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 240, 200, 10, 10, 9
    elif n_panels <= 16:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 220, 180, 10, 9, 8
    else:
        panel_w_px, panel_h_px, f_main, f_sub, f_tick = 200, 170, 9, 9, 8

    # Margins (absolute pixels)
    margin_t = 60                          # space for figure-level title
    margin_b = 70 if n_rows < 3 else 50    # bigger if horizontal legend below
    margin_l = 80                          # y-axis title room
    margin_r = 140 if n_rows >= 3 else 30  # right-side legend for tall grids

    # Inter-panel gaps (pixels). Plotly's vertical_spacing / horizontal_spacing
    # are fractions of the PLOT AREA (the figure MINUS its margins), so the gap
    # targets below must be divided by the plot-area dimensions, not the full
    # figure size. Dividing by the full size would shrink every gap by the
    # margin fraction and crowd each row's subplot titles against the axes of
    # the row above.
    title_gap_px = 58           # subplot title above each row
    label_gap_px = 38           # x-axis tick labels (+ axis title on the bottom row)
    h_gap_px = 30 if has_overlay else 25

    total_panel_h = n_rows * panel_h_px
    total_v_gap = (n_rows - 1) * (title_gap_px + label_gap_px) if n_rows > 1 else 0
    chart_height_px = total_panel_h + total_v_gap + margin_t + margin_b

    plot_area_h = total_panel_h + total_v_gap          # = chart_height_px - margins
    v_spacing = (title_gap_px + label_gap_px) / plot_area_h if n_rows > 1 else 0.0

    plot_area_w = n_cols * panel_w_px + (n_cols - 1) * h_gap_px
    chart_width_estimate = plot_area_w + margin_l + margin_r
    h_spacing = h_gap_px / plot_area_w if n_cols > 1 else 0.0

    # Legend placement
    if n_rows >= 3:
        legend_orient = "v"
        legend_y = 0.5
    else:
        legend_orient = "h"
        # Push below the bottom x-axis labels by enough for n_rows
        legend_y = -(0.05 + 0.03 * n_rows)

    # Subplot title wrap threshold — heuristic: ~9 px per char at base font
    max_subplot_chars = max(15, int(panel_w_px / 9))

    return dict(
        n_rows=n_rows,
        n_cols=n_cols,
        panel_w_px=panel_w_px,
        panel_h_px=panel_h_px,
        chart_height_px=chart_height_px,
        font_size_main=f_main,
        font_size_subplot=f_sub,
        font_size_tick=f_tick,
        h_spacing=h_spacing,
        v_spacing=v_spacing,
        legend_orient=legend_orient,
        legend_y=legend_y,
        margin_t=margin_t,
        margin_b=margin_b,
        margin_l=margin_l,
        margin_r=margin_r,
        subplot_title_max_chars=max_subplot_chars,
    )


def _fit_subplot_title(name, max_chars):
    """Wrap a long subplot title at word boundaries to fit max_chars per line.

    Uses Plotly's HTML-style ``<br>`` line break. Single very-long words are
    truncated with an ellipsis rather than mid-word split.
    """
    if not name:
        return ""
    if len(name) <= max_chars:
        return name
    words = name.replace("_", " ").split()
    if len(words) == 1:
        return name[: max_chars - 1] + "…"
    lines = []
    current = []
    current_len = 0
    for word in words:
        proposed = current_len + len(word) + (1 if current else 0)
        if proposed > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = proposed if current_len else len(word)
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def _apply_panel_axis_titles(fig, n_rows, n_cols, n_panels, var_names, layout,
                             x_axis_label, y_axis_label=None):
    """Place y-axis titles on leftmost column only; x-axis titles on bottom-most
    panel of each column only. This avoids the "Quarters" appearing under every
    panel (which collides with the next row's subplot title).

    ``y_axis_label`` controls what the leftmost-column y-axes show. When it is
    ``None`` (the default) the response-variable name lives on the y-axis. When a
    string is supplied (e.g. ``"Percent"``) that unit label is shown instead and
    the variable name is carried by the subplot title alone — matching the
    convention of published IRF figures (where the axis reads "Percent", not the
    variable name) and removing the variable-name-twice redundancy.
    """
    for j in range(n_panels):
        row, col = divmod(j, n_cols)
        row += 1
        col += 1
        # Y-axis title only on leftmost column
        if col == 1:
            fig.update_yaxes(
                title_text=y_axis_label if y_axis_label is not None else var_names[j],
                title_font=dict(size=layout["font_size_subplot"]),
                row=row, col=col,
            )
        # X-axis title only when no panel lies below this one in the same column
        if j + n_cols >= n_panels:
            fig.update_xaxes(
                title_text=x_axis_label,
                title_font=dict(size=layout["font_size_subplot"]),
                row=row, col=col,
            )


def _apply_layout_finalize(fig, layout):
    """Apply the layout dict's figure-level settings: height, margins, font,
    legend, plot background, grid styling, and subplot-title annotation fonts.
    Idempotent and safe to call once per figure.
    """
    legend_kwargs = dict(font=dict(size=layout["font_size_subplot"]))
    if layout["legend_orient"] == "h":
        legend_kwargs.update(
            orientation="h",
            y=layout["legend_y"],
            x=0.5,
            xanchor="center",
        )
    else:
        legend_kwargs.update(
            orientation="v",
            y=0.5,
            x=1.02,
            xanchor="left",
            yanchor="middle",
        )

    fig.update_layout(
        height=layout["chart_height_px"],
        margin=dict(
            t=layout["margin_t"],
            b=layout["margin_b"],
            l=layout["margin_l"],
            r=layout["margin_r"],
        ),
        font=dict(family="Georgia, serif", size=layout["font_size_main"]),
        legend=legend_kwargs,
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        gridcolor="#eee",
        tickfont=dict(size=layout["font_size_tick"]),
    )
    fig.update_yaxes(
        gridcolor="#eee",
        tickfont=dict(size=layout["font_size_tick"]),
    )
    # Plotly's subplot titles are annotations — restyle their fonts
    for annotation in fig.layout.annotations:
        annotation.font = dict(
            size=layout["font_size_subplot"], family="Georgia, serif"
        )


# ---------------------------------------------------------------------------
# Standard-section helpers (used by VARReport.add_standard_sections)
# ---------------------------------------------------------------------------
#
# These adapt the framework's existing data structures (DiagnosticReport,
# IRFContainer, the structural_fevd primitive) to the dict shapes that the
# individual VARReport.add_* methods expect. Letting one helper bridge them
# avoids each replicate.py rebuilding the adapter logic by hand.


def _build_diagnostics_dict_from_report(dr):
    """Adapt a :class:`lib.diagnostics.DiagnosticReport` instance to the
    ``{per_equation, system}`` dict that :meth:`VARReport.add_diagnostics`
    expects.

    Ljung-Box rows are passed through at the lags the diagnostics were
    actually computed at (``residuals[var]["ljung_box"]`` maps lag →
    stat/p-value); the rendered table labels each row with its true lag.
    """
    per_eq = {}
    for var, r in dr.residuals.items():
        lb_map = r.get("ljung_box") or {}
        if not lb_map:
            # Older DiagnosticReport shape: a single unlabelled LB p-value.
            lb_map = {"n/a": {"p_value": float(r.get("ljung_box_pvalue", float("nan")))}}
        jb_p = float(r.get("jarque_bera_pvalue", float("nan")))
        arch_p = float(r.get("arch_pvalue", float("nan")))
        per_eq[var] = {
            "ljung_box": {
                lag: {"p_value": float(v["p_value"])} for lag, v in lb_map.items()
            },
            "jarque_bera": {"p_value": jb_p},
            "arch_lm": {"p_value": arch_p},
        }
    system = {
        "stability": {
            "max_eigenvalue_modulus": float(dr.stability["max_eigenvalue_modulus"]),
            "note": str(dr.stability.get("status", "")),
        },
    }
    return {"per_equation": per_eq, "system": system}


def _compute_fevd_table(fit, structural_impact, horizons):
    """Compute FEVD shares at multiple horizons via ``lib.structural_fevd``.

    ``lib.structural_fevd`` returns a single-horizon ``(n, m)`` array; this
    helper loops over a list of horizons and returns a 3D array shape
    ``(n_response_vars, n_horizons, n_shocks)`` matching what
    :meth:`VARReport.add_fevd` expects.
    """
    from .irf import structural_fevd as _structural_fevd

    n_vars = fit.n_vars
    structural_impact = np.asarray(structural_impact)
    if structural_impact.ndim == 1:
        n_shocks = 1
        S = structural_impact
    else:
        n_shocks = structural_impact.shape[1]
        S = structural_impact

    table = np.zeros((n_vars, len(horizons), n_shocks))
    for i, h in enumerate(horizons):
        h_clip = max(1, int(h))
        shares = _structural_fevd(fit, S, horizon=h_clip)
        if shares.ndim == 1:
            table[:, i, 0] = shares
        else:
            table[:, i, :] = shares
    return table


def _irf_has_inference(irf):
    """True when the IRF carries uncertainty bands. Point-only output
    (``bands="none"`` or bands identical to the point) does not."""
    bands_meta = str((getattr(irf, "metadata", {}) or {}).get("bands", "")).lower()
    bounds = [getattr(irf, a, None) for a in ("lower_68", "upper_68", "lower_90", "upper_90")]
    if any(b is None for b in bounds):
        return bands_meta not in ("none", "")
    lo68, hi68, lo90, hi90 = (np.asarray(b) for b in bounds)
    return bands_meta not in ("none", "") or not (
        np.array_equal(lo68, hi68) and np.array_equal(lo90, hi90)
    )


def _build_summary_rows(irf, shock_idx, var_names):
    """Compute peak / horizon / significance / cumulative columns for the
    summary table, one row per response variable.

    Significance flags come from whether the CI bands share a sign at the
    peak horizon (i.e., the band excludes zero). Cumulative columns are
    reported only at horizons the IRF actually covers (``"cums"``: dict
    mapping horizon → cumulative sum through that horizon) — no silent
    truncation to a longer nominal horizon.
    """
    median = np.asarray(irf.median)
    lo68 = np.asarray(irf.lower_68)
    hi68 = np.asarray(irf.upper_68)
    lo90 = np.asarray(irf.lower_90)
    hi90 = np.asarray(irf.upper_90)

    H_max = median.shape[0] - 1                      # last available horizon
    cum_horizons = [h for h in (4, 8, 20, 40) if h <= H_max] or [H_max]

    # Point-only output (bands="none" or degenerate lower==upper) carries no
    # inference: significance must read N/A, not Yes. And a [0, 0] interval
    # must not count as sign agreement (sign(0) == sign(0)).
    has_inference = _irf_has_inference(irf)

    def _excludes_zero(lo, hi):
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        return (lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)

    rows = []
    for j, v in enumerate(var_names):
        path = median[:, j, shock_idx]
        peak_idx = int(np.argmax(np.abs(path)))
        peak_val = float(path[peak_idx])
        if has_inference:
            ci68_excl = _excludes_zero(
                lo68[peak_idx, j, shock_idx], hi68[peak_idx, j, shock_idx])
            ci90_excl = _excludes_zero(
                lo90[peak_idx, j, shock_idx], hi90[peak_idx, j, shock_idx])
        else:
            ci68_excl = None
            ci90_excl = None
        rows.append(
            {
                "variable": v,
                "peak_response": peak_val,
                "peak_horizon": peak_idx,
                "ci68_excludes_zero": ci68_excl,
                "ci90_excludes_zero": ci90_excl,
                "cums": {h: float(path[: h + 1].sum()) for h in cum_horizons},
            }
        )
    return rows


class _FEVDUnsupported(Exception):
    """Internal: automatic FEVD not computable for this normalization."""


class VARReport:
    """Builds an HTML report from VAR estimation results."""

    def __init__(self, title, subtitle="", config=None):
        self.title = title
        self.subtitle = subtitle
        self.sections = []
        self._tab_counter = 0

        # Chart style from config
        cfg = (config or {}).get("presentation", {}).get("chart", {})
        self.line_color = cfg.get("line_color", "#1a1a2e")
        self.band_68_color = cfg.get("band_68_color", "#4a6fa5")
        self.band_68_alpha = cfg.get("band_68_alpha", 0.35)
        self.band_90_color = cfg.get("band_90_color", "#4a6fa5")
        self.band_90_alpha = cfg.get("band_90_alpha", 0.15)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_specification(self, spec):
        """Add a model specification summary table.

        spec: dict of {label: value} pairs, e.g.
              {"Model": "VAR(4)", "Variables": "m1, rgdp, ...", ...}
        """
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in spec.items()
        )
        html = f'<table class="spec-table">{rows}</table>'
        self.sections.append(("Specification", html))

    def add_irf(self, irf_point, ci_68_lo, ci_68_hi, ci_90_lo, ci_90_hi,
                var_names, shock_labels, horizons=None,
                x_axis_label="Quarters", units=None,
                section_title="Impulse Response Functions"):
        """Add tabbed IRF charts (one tab per shock, panels per tab).

        All arrays shape: (H+1, n_vars, n_vars).

        Layout adapts to ``n_vars`` via :func:`_smart_layout`:
        grid dimensions, panel sizes, font sizes, inter-panel spacing,
        legend placement, and margins are all derived from the panel count
        rather than hardcoded. The same call renders cleanly whether
        ``n_vars`` is 1 (single panel) or 16+ (FAVAR-scale system).

        Variable names live on the subplot titles. By default the y-axis
        (leftmost column only) repeats the variable name; pass ``units``
        (e.g. ``"Percent"``) to show a shared unit label there instead, and
        ``section_title`` to override the section heading (useful when one
        report carries more than one IRF block, e.g. an agnostic-sign and a
        recursive identification side by side). The x-axis label appears only
        on the bottom-most panel of each column.
        """
        n_horizons, n_vars, n_shocks = irf_point.shape
        if horizons is None:
            horizons = np.arange(n_horizons)
        # Point-only input (bands equal to each other) has no uncertainty:
        # draw no band traces and say so, instead of labelling empty bands.
        point_only = (np.array_equal(ci_68_lo, ci_68_hi)
                      and np.array_equal(ci_90_lo, ci_90_hi))

        layout = _smart_layout(n_vars, has_overlay=False)
        n_rows = layout["n_rows"]
        n_cols = layout["n_cols"]

        self._tab_counter += 1
        group_id = f"irf-{self._tab_counter}"

        btns = "".join(
            f'<button class="tab-btn{"  active" if i == 0 else ""}" '
            f'onclick="switchTab(\'{group_id}\', {i})">{shock_labels[i]}</button>'
            for i in range(n_shocks)
        )

        # Wrap long subplot titles; pad to fill the grid (Plotly requires
        # len(subplot_titles) == n_rows * n_cols)
        wrapped_titles = [
            _fit_subplot_title(v, layout["subplot_title_max_chars"]) for v in var_names
        ]
        wrapped_titles += [""] * (n_rows * n_cols - n_vars)

        panels = []
        for s in range(n_shocks):
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=wrapped_titles,
                horizontal_spacing=layout["h_spacing"],
                vertical_spacing=layout["v_spacing"],
            )
            for j in range(n_vars):
                row, col = divmod(j, n_cols)
                row += 1
                col += 1
                show_legend = (j == 0)

                if not point_only:
                    # 90% band
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([horizons, horizons[::-1]]),
                        y=np.concatenate([ci_90_hi[:, j, s], ci_90_lo[::-1, j, s]]),
                        fill="toself", fillcolor=f"rgba(74,111,165,{self.band_90_alpha})",
                        line=dict(width=0), name="90% CI",
                        showlegend=show_legend, legendgroup="ci90",
                    ), row=row, col=col)

                    # 68% band
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([horizons, horizons[::-1]]),
                        y=np.concatenate([ci_68_hi[:, j, s], ci_68_lo[::-1, j, s]]),
                        fill="toself", fillcolor=f"rgba(74,111,165,{self.band_68_alpha})",
                        line=dict(width=0), name="68% CI",
                        showlegend=show_legend, legendgroup="ci68",
                    ), row=row, col=col)

                # Point estimate
                fig.add_trace(go.Scatter(
                    x=horizons, y=irf_point[:, j, s],
                    line=dict(color=self.line_color, width=2),
                    name="Point estimate",
                    showlegend=show_legend, legendgroup="point",
                ), row=row, col=col)

                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="#999999",
                              line_width=0.8, row=row, col=col)

            _apply_panel_axis_titles(
                fig, n_rows, n_cols, n_vars, var_names, layout, x_axis_label,
                y_axis_label=units,
            )
            _apply_layout_finalize(fig, layout)

            chart_html = fig.to_html(full_html=False, include_plotlyjs=False,
                                     config={"responsive": True})
            panels.append(
                f'<div class="tab-panel{"  active" if s == 0 else ""}">{chart_html}</div>'
            )

        html = (
            f'<div class="tab-container" data-group="{group_id}">'
            f'<div class="tab-buttons">{btns}</div>'
            f'{"".join(panels)}'
            f'</div>'
        )
        if point_only:
            html += ('<p style="font-size:0.8rem;color:var(--muted);margin-top:0.5rem">'
                     'Point estimates only; uncertainty intervals were not computed.</p>')
        self.sections.append((section_title, html))

    def add_diagnostics(self, diag_dict, var_names):
        """Add a color-coded diagnostics table.

        diag_dict: same structure as research_log["diagnostics"]
        """
        def pval_cell(p):
            if p > 0.10:
                cls = "pass"
            elif p > 0.05:
                cls = "warn"
            else:
                cls = "fail"
            return f'<td class="num {cls}">{p:.3f}</td>'

        header = "<tr><th>Test</th>" + "".join(f"<th>{v}</th>" for v in var_names) + "</tr>"

        # Ljung-Box rows: one per lag actually computed (union across
        # equations, in first-seen order) — labels reflect the true lags.
        lb_lags_present: list = []
        for v in var_names:
            for lag in diag_dict["per_equation"].get(v, {}).get("ljung_box", {}):
                if lag not in lb_lags_present:
                    lb_lags_present.append(lag)
        test_rows = [(f"Ljung-Box({lag})", ("ljung_box", lag))
                     for lag in lb_lags_present]
        test_rows += [
            ("Jarque-Bera", ("jarque_bera", None)),
            ("ARCH-LM", ("arch_lm", None)),
        ]

        rows = []
        for test_name, test_key in test_rows:
            cells = f"<td>{test_name}</td>"
            for v in var_names:
                eq = diag_dict["per_equation"][v]
                if test_key[1] is not None:
                    p = eq[test_key[0]].get(test_key[1], {}).get("p_value", float("nan"))
                else:
                    p = eq[test_key[0]]["p_value"]
                cells += pval_cell(p)
            rows.append(f"<tr>{cells}</tr>")

        # System-level
        sys_rows = ""
        if "portmanteau" in diag_dict.get("system", {}):
            p = diag_dict["system"]["portmanteau"]["p_value"]
            sys_rows += f'<tr><td>Portmanteau(12)</td><td colspan="{len(var_names)}" class="num {"pass" if p > 0.10 else "warn" if p > 0.05 else "fail"}">{p:.3f} (system)</td></tr>'
        if "stability" in diag_dict.get("system", {}):
            ev = diag_dict["system"]["stability"]["max_eigenvalue_modulus"]
            note = diag_dict["system"]["stability"].get("note", "")
            cls = "pass" if ev < 1.0 else "warn"
            sys_rows += f'<tr><td>Max eigenvalue</td><td colspan="{len(var_names)}" class="num {cls}">{ev:.4f} <span style="color:var(--muted);font-size:0.8rem">({note})</span></td></tr>'

        html = (
            f"<table>{header}{''.join(rows)}"
            f'<tr><td colspan="{len(var_names)+1}" style="border:none;height:0.5rem"></td></tr>'
            f"{sys_rows}</table>"
            f'<p style="font-size:0.8rem;color:var(--muted);margin-top:0.5rem">'
            f'<span class="pass">Green</span> p &gt; 0.10 &nbsp; '
            f'<span class="warn">Amber</span> 0.05 &lt; p &lt; 0.10 &nbsp; '
            f'<span class="fail">Red</span> p &lt; 0.05</p>'
        )
        self.sections.append(("Diagnostics", html))

    def add_fevd(self, fevd_data, var_names, horizons, shock_names=None,
                 note=None):
        """Add a FEVD table.

        fevd_data: array shape (n_vars, H, n_shocks) — [response, horizon, shock].
            If the horizon axis has one slice per entry of ``horizons``
            (what ``_compute_fevd_table`` returns), slice ``k`` is shown for
            ``horizons[k]``. Otherwise the axis is read as a full path
            indexed by horizon and must cover every requested horizon;
            a horizon beyond it raises instead of borrowing another row.
        horizons: list of horizons to display (labels of the rows)
        shock_names: column labels for the SHOCKS. Defaults to ``var_names``
            (correct only for recursive identifications where shocks are
            named after equations — pass explicit shock names otherwise).
        note: optional caveat sentence rendered under the table.
        """
        shock_labels = list(shock_names) if shock_names is not None else list(var_names)
        header = ("<tr><th>Response</th><th>h</th>"
                  + "".join(f"<th>{s}</th>" for s in shock_labels) + "</tr>")

        fevd_data = np.asarray(fevd_data)
        horizons = list(horizons)
        n_slices = fevd_data.shape[1]
        if n_slices == len(horizons):
            slice_of = list(range(len(horizons)))        # one slice per label
        else:
            bad = [h for h in horizons if not 0 <= int(h) < n_slices]
            if bad:
                raise ValueError(
                    f"add_fevd: fevd_data has {n_slices} horizon slices, which "
                    f"matches neither the {len(horizons)} requested horizons nor "
                    f"a full path covering horizons {bad}")
            slice_of = [int(h) for h in horizons]          # full path by index
        rows = []
        for i, vname in enumerate(var_names):
            for h_idx, h in enumerate(horizons):
                shares = fevd_data[i, slice_of[h_idx], :] * 100
                label = f"<td><strong>{vname}</strong></td>" if h_idx == 0 else "<td></td>"
                cells = label + f"<td class='num'>{h}</td>"
                for s in shares:
                    bg_intensity = min(s / 100, 1.0) * 0.15
                    cells += f'<td class="num" style="background:rgba(74,111,165,{bg_intensity})">{s:.1f}</td>'
                rows.append(f"<tr>{cells}</tr>")

        note_html = (
            f'<p style="font-size:0.8rem;color:var(--muted);margin-top:0.5rem">{note}</p>'
            if note else ""
        )
        html = f"<table>{header}{''.join(rows)}</table>{note_html}"
        self.sections.append(("Forecast Error Variance Decomposition (%)", html))

    def add_summary(self, summary_rows, title=None):
        """Add a summary results table.

        summary_rows: list of dicts with keys:
            variable, peak_response, peak_horizon, ci68_excludes_zero,
            ci90_excludes_zero, and either ``cums`` (dict horizon → cum
            value; preferred) or legacy ``cum_4 / cum_8 / cum_20 / cum_40``.
            Significance flags require finite interval bounds of the
            same sign; pass ``None`` for a run without interval
            inference (``bands="none"``) and the cell renders
            "N/A (no inference)" — significance is never fabricated
            from point estimates.
        title: section title; defaults to "Shock Summary". Callers that
            know the shock should pass e.g. "Technology Shock Summary".
        """
        legacy = summary_rows and "cums" not in summary_rows[0]
        if legacy:
            cum_cols = [(f"Cum({h})", f"cum_{h}") for h in (4, 8, 20, 40)]
        else:
            cum_hs = sorted(summary_rows[0]["cums"]) if summary_rows else []
            cum_cols = [(f"Cum({h})", h) for h in cum_hs]

        header = (
            "<tr><th>Variable</th><th>Peak</th><th>Horizon</th>"
            "<th>Sig. 68%</th><th>Sig. 90%</th>"
            + "".join(f"<th>{label}</th>" for label, _ in cum_cols)
            + "</tr>"
        )
        rows = []
        for r in summary_rows:
            def _sig(v):
                if v is None:
                    return '<span class="na">N/A (no inference)</span>'
                return '<span class="pass">Yes</span>' if v else '<span class="fail">No</span>'
            sig68 = _sig(r["ci68_excludes_zero"])
            sig90 = _sig(r["ci90_excludes_zero"])
            cum_cells = "".join(
                f'<td class="num">{(r[key] if legacy else r["cums"][key]):.4f}</td>'
                for _, key in cum_cols
            )
            rows.append(
                f'<tr><td>{r["variable"]}</td>'
                f'<td class="num">{r["peak_response"]:.4f}</td>'
                f'<td class="num">{r["peak_horizon"]}</td>'
                f'<td class="num">{sig68}</td><td class="num">{sig90}</td>'
                f'{cum_cells}</tr>'
            )
        html = f"<table>{header}{''.join(rows)}</table>"
        self.sections.append((title or "Shock Summary", html))

    def add_robustness(self, baseline_irf, alt_irf, labels, var_names,
                       shock_idx=0, horizons=None,
                       x_axis_label="Quarters"):
        """Add a robustness comparison chart (baseline vs alternative).

        baseline_irf, alt_irf: shape (H+1, n_vars, n_vars)
        labels: tuple of (baseline_label, alt_label)

        Layout adapts to ``len(var_names)`` via :func:`_smart_layout` with
        ``has_overlay=True`` (slightly wider horizontal spacing so the two
        overlaid lines don't crowd the legend). Variable names live on the
        y-axis; x-axis label appears only on bottom-most panel of each
        column.
        """
        n_horizons, n_vars_total, _ = baseline_irf.shape
        if horizons is None:
            horizons = np.arange(n_horizons)

        n_vars = len(var_names)
        layout = _smart_layout(n_vars, has_overlay=True)
        n_rows = layout["n_rows"]
        n_cols = layout["n_cols"]

        wrapped_titles = [
            _fit_subplot_title(v, layout["subplot_title_max_chars"]) for v in var_names
        ]
        wrapped_titles += [""] * (n_rows * n_cols - n_vars)

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=wrapped_titles,
            horizontal_spacing=layout["h_spacing"],
            vertical_spacing=layout["v_spacing"],
        )

        for j in range(n_vars):
            row, col = divmod(j, n_cols)
            row += 1
            col += 1
            show_legend = (j == 0)

            fig.add_trace(go.Scatter(
                x=horizons, y=baseline_irf[:, j, shock_idx],
                line=dict(color=self.line_color, width=2),
                name=labels[0], showlegend=show_legend, legendgroup="base",
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=horizons, y=alt_irf[:, j, shock_idx],
                line=dict(color="#c44e52", width=2, dash="dash"),
                name=labels[1], showlegend=show_legend, legendgroup="alt",
            ), row=row, col=col)

            fig.add_hline(y=0, line_dash="dash", line_color="#999999",
                          line_width=0.8, row=row, col=col)

        _apply_panel_axis_titles(
            fig, n_rows, n_cols, n_vars, var_names, layout, x_axis_label
        )
        _apply_layout_finalize(fig, layout)

        chart_html = fig.to_html(full_html=False, include_plotlyjs=False,
                                 config={"responsive": True})
        self.sections.append(("Robustness", chart_html))

    def add_narrative(self, text):
        """Add a free-form narrative section (plain text with paragraph breaks)."""
        paragraphs = "".join(f"<p>{p.strip()}</p>" for p in text.strip().split("\n\n") if p.strip())
        self.sections.append(("Narrative", f'<div class="narrative">{paragraphs}</div>'))

    def add_research_log(self, log_dict):
        """Add a collapsible research log (JSON)."""
        log_json = json.dumps(log_dict, indent=2, default=str)
        html = (
            f"<details><summary>Full research log (JSON)</summary>"
            f"<pre>{_escape_html(log_json)}</pre></details>"
        )
        self.sections.append(("Research Log", html))

    # ------------------------------------------------------------------
    # One-shot standard-report helper
    # ------------------------------------------------------------------

    def add_standard_sections(
        self,
        fit,
        irf,
        *,
        spec=None,
        log=None,
        var_labels=None,
        shock_idx_for_summary=0,
        fevd_horizons=(1, 4, 8, 20, 40),
        x_axis_label="Quarters",
        units=None,
        alt_irf=None,
        alt_label="Alternative",
        baseline_label="Baseline",
        narrative=None,
        flags=None,
    ):
        """Add the canonical set of report sections in standard order.

        Given a fitted model and an :class:`~lib.irf.IRFContainer`, adds:

        1. **Specification** (from ``spec`` dict if provided)
        2. **Impulse Response Functions** (from ``irf`` — always)
        3. **Diagnostics** (auto-derived via ``lib.diagnostics(fit)``)
        4. **FEVD** (auto-derived via ``lib.structural_fevd``, at the
           horizons specified in ``fevd_horizons``)
        5. **Summary** (peak / horizon / significance / cumulative table)
        6. **Robustness** (only if ``alt_irf`` is provided)
        7. **Narrative** — ``narrative='auto'`` drafts an economics-tied
           skeleton (signs / magnitudes / FEVD / stability / ``flags``) to
           refine; a string is rendered verbatim; ``None`` warns (otherwise the
           report carries no economic interpretation at all).
        8. **Research Log** (from ``log`` dict if provided)

        Sections 3-5 require the fit to expose the standard ``VARFit``
        interface (``residuals``, ``Sigma_u``, ``ma_rep``, ``variable_names``).
        Any section that raises during construction is skipped with a
        ``warnings.warn`` message — the report build still completes.
        This makes the helper safe to call on fit types where not every
        section applies (panel VAR diagnostics, etc.). One exception is
        made VISIBLE in the report itself: when the IRF's normalisation
        does not give a unit-variance shock (proxy-SVAR ``unit_policy``
        or ``one_sd``), the FEVD is not identified and the section is replaced by an
        explanatory "FEVD omitted" note rather than silently dropped.

        Parameters
        ----------
        fit : VARFit
            The fitted model.
        irf : IRFContainer
            The impulse response container.
        spec : dict, optional
            ``{label: value}`` pairs for the specification table.
        log : dict, optional
            Research-log dictionary (rendered as collapsible JSON).
        var_labels : list[str], optional
            Display labels for response variables. Defaults to
            ``fit.variable_names``.
        shock_idx_for_summary : int, default 0
            Which shock's IRFs to summarise in the summary table.
        fevd_horizons : tuple[int], default (1, 4, 8, 20, 40)
            Horizons at which to display FEVD shares.
        x_axis_label : str, default "Quarters"
            X-axis label for IRF charts.
        units : str, optional
            Y-axis unit label for IRF charts (e.g. ``"Percent"``). When set, the
            leftmost-column y-axes read this unit and the variable name is shown
            only in the subplot title. When ``None`` (default) the y-axis repeats
            the variable name, preserving the prior behaviour.
        alt_irf : ndarray or IRFContainer, optional
            Alternative IRFs for the robustness section. Either a raw
            ``(H+1, n_vars, n_vars)`` ndarray or another IRFContainer.
        alt_label, baseline_label : str
            Labels for the robustness comparison.
        narrative : str or "auto", optional
            ``"auto"`` drafts an economics-tied skeleton via
            :func:`draft_narrative` (which the agent should then refine); a
            plain string is rendered verbatim; ``None`` emits a warning, since a
            report with no narrative has no economic interpretation.
        flags : list[str], optional
            Economic-sensibility findings (Stage 5) forwarded to the auto-draft,
            e.g. ``["Price puzzle: deflator rises after a contraction"]``.
        """
        import warnings

        var_names = list(fit.variable_names) if var_labels is None else list(var_labels)

        # Captured below so an optional auto-drafted narrative (narrative="auto")
        # can reuse the stability + FEVD that the diagnostics / FEVD steps build.
        max_eig = None
        fevd_table = None

        # 1. Specification
        if spec is not None:
            self.add_specification(spec)

        # 2. IRF (always)
        self.add_irf(
            irf_point=irf.median,
            ci_68_lo=irf.lower_68,
            ci_68_hi=irf.upper_68,
            ci_90_lo=irf.lower_90,
            ci_90_hi=irf.upper_90,
            var_names=var_names,
            shock_labels=list(irf.shock_names),
            horizons=irf.horizons,
            x_axis_label=x_axis_label,
            units=units,
        )

        # 3. Diagnostics — rekey per_equation to display labels if needed
        try:
            from .diagnostics import diagnostics as _diagnostics
            dr = _diagnostics(fit)
            diag_dict = _build_diagnostics_dict_from_report(dr)
            raw_names = list(fit.variable_names)
            if var_names != raw_names:
                # Rekey per_equation entries from raw names to display labels
                relabeled = {}
                for raw, label in zip(raw_names, var_names):
                    if raw in diag_dict["per_equation"]:
                        relabeled[label] = diag_dict["per_equation"][raw]
                diag_dict = {
                    "per_equation": relabeled,
                    "system": diag_dict["system"],
                }
            self.add_diagnostics(diag_dict, var_names)
            max_eig = (diag_dict.get("system", {}).get("stability", {})
                       .get("max_eigenvalue_modulus"))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"add_standard_sections: diagnostics skipped — "
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )

        # 4. FEVD — use IRF at h=0 as the structural impact matrix.
        # Guard: the FEVD formula needs unit-variance structural shocks. Only
        # the proxy-SVAR routine sets a normalisation, and of its options only
        # "reliability" scales the impact to a unit-variance shock.
        # "unit_policy" (one unit of policy response) and "one_sd" (one
        # standard deviation of the policy residual) do not pin the shock's
        # variance, so their impact would mis-scale the variance shares.
        # Omit the automatic table in those cases.
        _norm = str(((getattr(irf, "metadata", {}) or {})
                     .get("normalization", ""))).lower()
        _fevd_supported = _norm in ("", "reliability")
        if not _fevd_supported:
            self.sections.append((
                "Forecast Error Variance Decomposition (%)",
                f"<p>FEVD omitted: the IRF is normalised for reporting "
                f"({_norm!r}), which does not identify the structural "
                f"shock's variance. A forecast-error variance decomposition "
                f"needs the impact of a unit-variance shock; a single-"
                f"instrument proxy SVAR provides one with "
                f"<code>normalization='reliability'</code>.</p>",
            ))
        try:
            if not _fevd_supported:
                raise _FEVDUnsupported()
            usable_horizons = [h for h in fevd_horizons if h <= irf.horizon + 1]
            structural_impact = np.asarray(irf.median)[0]
            fevd_table = _compute_fevd_table(fit, structural_impact, usable_horizons)
            fevd_note = None
            if structural_impact.ndim == 2 and structural_impact.shape[1] == fit.n_vars:
                # Full identification: shares must sum to 1 per (response,
                # horizon) row. With banded IRFs the h=0 impact is a
                # draw-wise median, so SS' only approximates Sigma_u —
                # renormalise for display and say so when it mattered.
                row_sums = fevd_table.sum(axis=2)
                max_dev = float(np.max(np.abs(row_sums - 1.0)))
                if max_dev > 1e-12:
                    fevd_table = fevd_table / row_sums[:, :, None]
                if max_dev > 0.01:
                    fevd_note = (
                        "Impact matrix taken from the pointwise-median IRF at "
                        "h=0 (a draw-wise median, so it reproduces the "
                        "innovation covariance only approximately); raw shares "
                        f"summed to {100 * float(row_sums.min()):.0f}–"
                        f"{100 * float(row_sums.max()):.0f}% "
                        "and have been rescaled to 100%."
                    )
            elif structural_impact.ndim == 2:
                # Partial identification (m < n): rows legitimately sum to
                # less than 100% — say so instead of leaving readers to
                # wonder where the rest went.
                fevd_note = (
                    f"Partial identification ({structural_impact.shape[1]} of "
                    f"{fit.n_vars} shocks identified): shares are of TOTAL "
                    "forecast-error variance, so rows need not sum to 100% — "
                    "the remainder is attributable to the unidentified shocks."
                )
            self.add_fevd(fevd_table, var_names, list(usable_horizons),
                          shock_names=list(irf.shock_names), note=fevd_note)
        except _FEVDUnsupported:
            pass  # already explained in the report body above
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"add_standard_sections: FEVD skipped — "
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )

        # 5. Summary — section title carries the actual shock name
        try:
            summary_rows = _build_summary_rows(
                irf, shock_idx_for_summary, var_names
            )
            shock_label = str(irf.shock_names[shock_idx_for_summary])
            shock_title = shock_label.replace('_', ' ').title()
            if "shock" not in shock_title.lower().split():
                shock_title += " Shock"
            summary_title = f"{shock_title} Summary"
            self.add_summary(summary_rows, title=summary_title)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"add_standard_sections: summary skipped — "
                f"{type(exc).__name__}: {exc}",
                stacklevel=2,
            )

        # 6. Robustness (only if alt_irf provided)
        if alt_irf is not None:
            try:
                alt_array = (
                    np.asarray(alt_irf)
                    if isinstance(alt_irf, np.ndarray)
                    else np.asarray(alt_irf.median)
                )
                self.add_robustness(
                    baseline_irf=np.asarray(irf.median),
                    alt_irf=alt_array,
                    labels=(baseline_label, alt_label),
                    var_names=var_names,
                    shock_idx=shock_idx_for_summary,
                    horizons=irf.horizons,
                    x_axis_label=x_axis_label,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"add_standard_sections: robustness skipped — "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )

        # 7. Narrative
        if narrative == "auto":
            try:
                self.add_narrative(draft_narrative(
                    irf, fevd=fevd_table, max_eigenvalue=max_eig,
                    spec=spec, flags=flags,
                ))
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"add_standard_sections: auto-narrative skipped — "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
        elif narrative is not None:
            self.add_narrative(narrative)
        else:
            warnings.warn(
                "add_standard_sections: no narrative provided — the report has "
                "charts and tables but NO economic interpretation. Pass "
                "narrative='auto' for a draft skeleton to refine, or "
                "narrative=<your text>. See results-presentation/SKILL.md.",
                stacklevel=2,
            )

        # 8. Research log
        if log is not None:
            self.add_research_log(log)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def save(self, path):
        """Render and save the HTML report."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        body_sections = ""
        for title, content in self.sections:
            body_sections += f"<h2>{title}</h2>\n{content}\n"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape_html(self.title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<h1>{_escape_html(self.title)}</h1>
<div class="subtitle">{_escape_html(self.subtitle)}</div>
<div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
{body_sections}
<script>{_JS}</script>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        return path


def _escape_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def draft_narrative(
    irf,
    *,
    shock=None,
    fevd=None,
    max_eigenvalue=None,
    spec=None,
    flags=None,
):
    """Draft an economic-narrative *skeleton* from estimation results.

    Returns blank-line-separated plain-text paragraphs ready for
    :meth:`VARReport.add_narrative`. This is the factual backbone the agent
    **refines** — it states the signs, magnitudes, peak horizons, convergence,
    variance shares, stability, and any economic-sensibility ``flags`` (the
    Stage-5 findings: price puzzle, explosive roots, …), and ends with an
    explicit ``[DRAFT …]`` marker so an un-refined skeleton is obvious. The
    agent supplies the economic interpretation and the link to the research
    question; this routine guarantees the report never ships with *no*
    interpretation and that the interpretation is tied to the actual numbers.

    Parameters
    ----------
    irf : IRFContainer
        Must expose ``.median`` ``(H+1, n_response, n_shock)``,
        ``.variable_names``, ``.shock_names`` and ``.horizons``.
    shock : str or int, optional
        Restrict the dynamics paragraph(s) to one shock (name or index).
        Default: narrate every shock.
    fevd : ndarray, optional
        ``(n_response, n_horizon, n_shock)`` variance shares; if given, names
        the dominant shock per variable at the longest available horizon.
    max_eigenvalue : float, optional
        Max companion-matrix eigenvalue modulus, for the stability sentence.
    spec : dict, optional
        ``{label: value}`` pairs for a one-line framing sentence.
    flags : list[str], optional
        Economic-sensibility findings from Stage 5 to surface as caveats,
        e.g. ``["Price puzzle: the deflator rises for 6 quarters after a
        contraction"]``.

    Returns
    -------
    str
        Plain-text paragraphs (blank-line separated).
    """
    median = np.asarray(irf.median, dtype=float)
    if median.ndim != 3:
        raise ValueError(
            f"irf.median must be 3-D (H+1, n_response, n_shock); got {median.shape}"
        )
    n_resp, n_shock = median.shape[1], median.shape[2]
    resp_names = list(getattr(irf, "variable_names", None) or
                      [f"y{i + 1}" for i in range(n_resp)])
    shock_names = list(getattr(irf, "shock_names", None) or
                       [f"shock {i + 1}" for i in range(n_shock)])
    horizons = list(getattr(irf, "horizons", None) if getattr(irf, "horizons", None)
                    is not None else range(median.shape[0]))
    h_last = horizons[-1]

    def _dir(x, eps=1e-9):
        return "up" if x > eps else "down" if x < -eps else "flat"

    paras = []
    if spec:
        paras.append("Specification — " +
                     "; ".join(f"{k}: {v}" for k, v in spec.items()) + ".")

    if shock is None:
        shock_idxs = list(range(n_shock))
    elif isinstance(shock, int):
        shock_idxs = [shock]
    else:
        shock_idxs = [shock_names.index(shock)] if shock in shock_names else [0]

    nonconverged = []
    for si in shock_idxs:
        sname = shock_names[si]
        sentences = []
        for ri, rname in enumerate(resp_names):
            path = median[:, ri, si]
            if not np.isfinite(path).any():
                continue
            impact = path[0]
            k = int(np.nanargmax(np.abs(path)))
            peak, last = path[k], path[-1]
            converges = abs(peak) < 1e-12 or abs(last) <= 0.2 * abs(peak)
            if not converges:
                nonconverged.append(f"{rname} (to {sname})")
            sentences.append(
                f"{rname} goes {_dir(impact)} on impact ({impact:+.3g}), "
                f"{'peaks' if peak >= 0 else 'troughs'} at {peak:+.3g} by horizon "
                f"{horizons[k]}, and is {last:+.3g} by horizon {h_last} "
                f"({'≈ back to zero' if converges else 'still away from zero'})."
            )
        if sentences:
            paras.append(f"Response to a {sname} shock. " + " ".join(sentences))

    if fevd is not None:
        try:
            fa = np.asarray(fevd, dtype=float)
            long_h = fa[:, -1, :]
            bits = []
            partial = False
            for ri, rname in enumerate(resp_names):
                row = long_h[ri]
                if float(row.sum()) > 0:
                    j = int(np.argmax(row))
                    # Entries are shares of TOTAL forecast-error variance;
                    # never renormalise over the identified shocks only.
                    bits.append(f"{rname} ({shock_names[j]}, {row[j]:.0%})")
                    partial = partial or float(row.sum()) < 0.995
            if bits:
                paras.append(
                    "Largest identified-shock share of forecast-error variance "
                    "at the longest horizon: " + "; ".join(bits) + "."
                    + (" Shares are of total variance; the remainder is due to "
                       "shocks not identified here." if partial else "")
                )
        except Exception:  # noqa: BLE001
            pass

    if max_eigenvalue is not None:
        stable = max_eigenvalue < 1.0
        paras.append(
            f"Stability: the largest companion eigenvalue modulus is "
            f"{max_eigenvalue:.3f} — "
            + ("inside the unit circle, so responses are mean-reverting."
               if stable else
               "on or outside the unit circle: the system is NOT stable and "
               "responses will not die out.")
        )

    caveats = list(flags or [])
    if nonconverged:
        shown = ", ".join(nonconverged[:4]) + ("…" if len(nonconverged) > 4 else "")
        caveats.append(
            f"Some responses have not returned toward zero by the final horizon "
            f"({shown}) — check the horizon is long enough and the system stable."
        )
    if _irf_has_inference(irf):
        caveats.append(
            "All magnitudes are point estimates surrounded by the plotted "
            "confidence bands; treat them as indicative, not exact."
        )
    else:
        caveats.append(
            "Point estimates only; uncertainty intervals were not computed, "
            "so treat magnitudes as indicative."
        )
    paras.append("Points to interpret and caveat: " +
                 " ".join(f"({i + 1}) {c}" for i, c in enumerate(caveats)))

    paras.append(
        "[DRAFT NARRATIVE — refine the economic interpretation, connect the "
        "findings to the research question, and confirm/expand the flagged items "
        "before publishing.]"
    )
    return "\n\n".join(paras)
