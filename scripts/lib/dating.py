"""Algorithmic business-cycle dating (Bry-Boschan / Harding-Pagan).

Turning-point detection on a univariate series:

* :func:`harding_pagan_dating` — Harding-Pagan (2002) quarterly adaptation
  of Bry-Boschan. Local extrema in a 2-quarter window; alternation +
  minimum-phase-length (2 quarters) + minimum-cycle-length (5 quarters)
  censoring.

* :func:`bry_boschan_dating` — simplified Bry-Boschan (1971)-style monthly
  dating. Local extrema of the raw series in a 5-month window; phase
  length >= 5 months; cycle length >= 15 months. The original
  procedure's extreme-value treatment and smoothed-series stages
  (12-month moving average, Spencer curve) are omitted, which can move
  turning dates.

Both return a :class:`DatingResult` with peak/trough dates and a
per-period phase label ('expansion' / 'recession'). The phase variable
can be plugged into state-dependent local projections (e.g., to pair
with `auerbach_gorodnichenko_2012`).

CONVENTION:

  A peak is a LOCAL MAXIMUM (level higher than k neighbours each side).
  A trough is a LOCAL MINIMUM. The series between a trough and the next
  peak is an expansion; between a peak and the next trough, a recession.

  The phase label is forward-looking up to the next turning point. For
  the period before the first identified turn or after the last, the
  phase is set to None (unknown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatingResult:
    """Output of a business-cycle dating algorithm.

    Attributes
    ----------
    data : pd.Series
        Original input series.
    peaks : pd.DatetimeIndex
        Dates classified as cyclical peaks.
    troughs : pd.DatetimeIndex
        Dates classified as cyclical troughs.
    phase : pd.Series
        String labels per period: 'expansion', 'recession', or None for
        periods outside the dated interval.
    metadata : dict
        Algorithm parameters and intermediate diagnostics.
    """

    data: pd.Series
    peaks: pd.DatetimeIndex
    troughs: pd.DatetimeIndex
    phase: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_recessions(self) -> int:
        """Number of complete peak-to-trough phases."""
        return len(self.troughs)

    def recession_dummy(self) -> pd.Series:
        """Binary dummy: 1 in recession, 0 in expansion, NaN outside dated range."""
        out = pd.Series(np.nan, index=self.data.index, dtype=float)
        out[self.phase == "recession"] = 1.0
        out[self.phase == "expansion"] = 0.0
        return out


def _find_local_extrema(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Find indices of local maxima and minima within +/- `window` periods.

    A point at index t is a local maximum if values[t] > values[t-k] for
    all k in 1..window AND values[t] > values[t+k] for all k in 1..window.
    Strict inequality — plateaus do not count.
    """
    T = len(values)
    if T < 2 * window + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    peaks: list[int] = []
    troughs: list[int] = []
    for t in range(window, T - window):
        v = values[t]
        left = values[t - window:t]
        right = values[t + 1:t + window + 1]
        if (v > left).all() and (v > right).all():
            peaks.append(t)
        elif (v < left).all() and (v < right).all():
            troughs.append(t)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


def _enforce_alternation(
    peaks: np.ndarray, troughs: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce alternation between peaks and troughs (no two consecutive of
    the same type). On consecutive same-type runs, keep the most extreme.
    """
    # Merge into a single sorted list of (index, kind, value)
    events = (
        [(t, "P", values[t]) for t in peaks]
        + [(t, "T", values[t]) for t in troughs]
    )
    events.sort(key=lambda e: e[0])

    cleaned: list[tuple[int, str, float]] = []
    for ev in events:
        if not cleaned or cleaned[-1][1] != ev[1]:
            cleaned.append(ev)
        else:
            # Same kind as previous: keep more extreme
            kind = ev[1]
            prev = cleaned[-1]
            if kind == "P" and ev[2] > prev[2]:
                cleaned[-1] = ev
            elif kind == "T" and ev[2] < prev[2]:
                cleaned[-1] = ev
            # else keep prev
    peaks_out = np.array([e[0] for e in cleaned if e[1] == "P"], dtype=int)
    troughs_out = np.array([e[0] for e in cleaned if e[1] == "T"], dtype=int)
    return peaks_out, troughs_out


def _enforce_phase_length(
    peaks: np.ndarray, troughs: np.ndarray, values: np.ndarray,
    min_phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop turning points that violate the minimum-phase-length rule.

    When two consecutive turning points (peak->trough or trough->peak)
    are closer than ``min_phase`` periods, drop one. Heuristic: drop the
    one that makes the shorter phase to the OTHER side.
    """
    # Iteratively prune the closest violating pair until none remain
    while True:
        events = (
            [(t, "P", values[t]) for t in peaks]
            + [(t, "T", values[t]) for t in troughs]
        )
        events.sort(key=lambda e: e[0])
        if len(events) < 2:
            break
        gaps = [events[i + 1][0] - events[i][0] for i in range(len(events) - 1)]
        violations = [i for i, g in enumerate(gaps) if g < min_phase]
        if not violations:
            break
        # Drop the FIRST violator (simple, conservative)
        i = violations[0]
        # Of the two events i and i+1, drop the one with the less extreme value
        a = events[i]
        b = events[i + 1]
        # If a is a peak and b is a trough (or vice versa), drop the one closer
        # to the global average; here we just drop the one with smaller |value-avg|
        avg = values.mean()
        drop_i = i if abs(a[2] - avg) < abs(b[2] - avg) else i + 1
        events.pop(drop_i)
        peaks = np.array([e[0] for e in events if e[1] == "P"], dtype=int)
        troughs = np.array([e[0] for e in events if e[1] == "T"], dtype=int)
    return peaks, troughs


def _enforce_min_cycle(
    peaks: np.ndarray, troughs: np.ndarray, values: np.ndarray,
    min_cycle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the minimum FULL-CYCLE rule (same-kind turn spacing).

    Bry-Boschan (1971) / Harding-Pagan (2002): peak-to-peak and
    trough-to-trough distances must be at least ``min_cycle`` periods.
    This does NOT follow from alternation + the minimum-phase rule under
    either default rule set (HP: 2+2 < 5; BB: 5+5 < 15), so it needs its
    own censoring pass: drop the SHALLOWER of a violating same-kind pair
    (the lower peak / the higher trough), restore alternation, repeat.
    """
    while True:
        changed = False
        for kind, arr in (("P", peaks), ("T", troughs)):
            if len(arr) < 2:
                continue
            gaps = np.diff(arr)
            viol = np.where(gaps < min_cycle)[0]
            if len(viol) == 0:
                continue
            i = int(viol[0])
            a, b = arr[i], arr[i + 1]
            if kind == "P":
                drop = a if values[a] < values[b] else b   # lower peak
            else:
                drop = a if values[a] > values[b] else b   # higher trough
            arr = arr[arr != drop]
            if kind == "P":
                peaks = arr
            else:
                troughs = arr
            # Removing one of a same-kind pair leaves consecutive
            # opposite-kind turns around it — restore alternation before
            # re-checking distances.
            peaks, troughs = _enforce_alternation(peaks, troughs, values)
            changed = True
            break
        if not changed:
            return peaks, troughs


def _enforce_amplitude_sign(
    peaks: np.ndarray, troughs: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Censor adjacent turn pairs with the wrong amplitude sign.

    A retained recession must END BELOW its starting peak, and a retained
    expansion must end above its starting trough — local extrema found in a
    rolling window do not guarantee this, and labelling a net RISE a
    recession fabricates a regime regressor (RBA quarterly dating,
    Appendix B step 4). When a pair violates the sign, drop the turning
    point whose ADJACENT-PHASE AMPLITUDE is smaller and let the caller
    re-impose alternation and duration rules. Amplitudes are level
    DIFFERENCES, so the censoring is invariant to adding a constant to
    the series (rebasing a logged index must never change the
    chronology) — a raw-level comparison across event types would not be.
    """
    while True:
        events = (
            [(int(t), "P", float(values[t])) for t in peaks]
            + [(int(t), "T", float(values[t])) for t in troughs]
        )
        events.sort(key=lambda e: e[0])
        bad = None
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            if a[1] == "P" and b[1] == "T" and b[2] >= a[2]:
                bad = (i, i + 1)          # "recession" that ends higher
                break
            if a[1] == "T" and b[1] == "P" and b[2] <= a[2]:
                bad = (i, i + 1)          # "expansion" that ends lower
                break
        if bad is None:
            break
        i, j = bad

        def _adjacent_amplitude(idx: int) -> float:
            """|level difference| to the event's valid neighbouring turns,
            EXCLUDING the violating partner — shift-invariant by
            construction. An event with no other neighbour scores 0."""
            amps = []
            for k in (idx - 1, idx + 1):
                if k in (i, j) or k < 0 or k >= len(events):
                    continue
                amps.append(abs(events[idx][2] - events[k][2]))
            return max(amps) if amps else 0.0

        events.pop(i if _adjacent_amplitude(i) < _adjacent_amplitude(j) else j)
        peaks = np.array([e[0] for e in events if e[1] == "P"], dtype=int)
        troughs = np.array([e[0] for e in events if e[1] == "T"], dtype=int)
    return peaks, troughs


def _build_phase_series(
    index: pd.DatetimeIndex, peaks_idx: np.ndarray, troughs_idx: np.ndarray,
) -> pd.Series:
    """Assign expansion / recession labels based on turning points.

    Between a trough and the next peak: expansion.
    Between a peak and the next trough: recession.
    Before the first turning point and after the last: None.
    """
    phase = pd.Series([None] * len(index), index=index, dtype=object)
    if len(peaks_idx) == 0 and len(troughs_idx) == 0:
        return phase
    # Merge into chronological sequence of turn dates
    turns = sorted(
        [(t, "P") for t in peaks_idx] + [(t, "T") for t in troughs_idx],
        key=lambda x: x[0],
    )
    for i in range(len(turns) - 1):
        start_idx, start_kind = turns[i]
        end_idx, _ = turns[i + 1]
        label = "recession" if start_kind == "P" else "expansion"
        # Period AFTER the peak/trough up to (and including) the next turn
        phase.iloc[start_idx + 1:end_idx + 1] = label
    return phase


def harding_pagan_dating(
    series: pd.Series,
    window: int = 2,
    min_phase: int = 2,
    min_cycle: int = 5,
) -> DatingResult:
    """Harding-Pagan (2002) quarterly business-cycle dating.

    Default censoring rules from Harding-Pagan (2002, JME):
      * Local extrema in a 2-quarter window (peak: y_t > y_{t±1, t±2}).
      * Minimum phase length: 2 quarters.
      * Minimum cycle length (trough-to-trough or peak-to-peak): 5 quarters.
      * Amplitude sign: a retained recession must END BELOW its starting
        peak and an expansion above its starting trough — local extrema
        alone do not guarantee this, and a net rise must never be
        labelled a recession. Wrong-signed phases are censored and the
        full rule set re-applied to a fixed point.

    Parameters
    ----------
    series : pd.Series
        Quarterly time series (real GDP, IPI in log levels, etc.). NaN
        values are dropped before detection.
    window : int
        Window size for local-extrema detection (default 2 = HP convention).
    min_phase : int
        Minimum periods between consecutive turning points.
    min_cycle : int
        Minimum periods between same-kind turning points (peak-to-peak
        or trough-to-trough). Enforced by its own censoring pass — it
        does NOT follow from alternation + min_phase under the default
        rule sets (2 + 2 < 5 quarterly; 5 + 5 < 15 monthly).

    Returns
    -------
    DatingResult
    """
    return _date_series(series, window, min_phase, min_cycle, method="harding_pagan")


def bry_boschan_dating(
    series: pd.Series,
    window: int = 5,
    min_phase: int = 5,
    min_cycle: int = 15,
) -> DatingResult:
    """Simplified Bry-Boschan (1971)-style monthly business-cycle dating.

    Extrema are found on the raw series; the original procedure's
    extreme-value treatment and smoothed-series stages are omitted.
    Default censoring rules from Bry-Boschan (1971, NBER):
      * Local extrema in a 5-month window.
      * Minimum phase length: 5 months.
      * Minimum cycle length: 15 months.

    Returns
    -------
    DatingResult
    """
    return _date_series(series, window, min_phase, min_cycle, method="bry_boschan")


def _date_series(
    series: pd.Series, window: int, min_phase: int, min_cycle: int,
    method: str,
) -> DatingResult:
    """Shared dating logic."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be pd.Series, got {type(series).__name__}")
    s = series.dropna()
    if len(s) < 2 * window + 1:
        raise ValueError(
            f"series too short ({len(s)}) for window={window}; "
            f"need at least {2 * window + 1} observations"
        )

    values = s.values
    peaks_raw, troughs_raw = _find_local_extrema(values, window)
    peaks, troughs = _enforce_alternation(peaks_raw, troughs_raw, values)
    peaks, troughs = _enforce_phase_length(peaks, troughs, values, min_phase)
    # Phase-length pruning can leave consecutive same-kind turns —
    # restore alternation before the cycle-length censoring pass.
    peaks, troughs = _enforce_alternation(peaks, troughs, values)
    peaks, troughs = _enforce_min_cycle(peaks, troughs, values, min_cycle)
    # Amplitude-sign censoring interacts with alternation/duration, so
    # iterate the full rule set to a fixed point before building phases.
    for _ in range(len(values)):
        p0, t0 = peaks.copy(), troughs.copy()
        peaks, troughs = _enforce_amplitude_sign(peaks, troughs, values)
        peaks, troughs = _enforce_alternation(peaks, troughs, values)
        peaks, troughs = _enforce_phase_length(peaks, troughs, values, min_phase)
        peaks, troughs = _enforce_alternation(peaks, troughs, values)
        peaks, troughs = _enforce_min_cycle(peaks, troughs, values, min_cycle)
        if np.array_equal(peaks, p0) and np.array_equal(troughs, t0):
            break

    phase = _build_phase_series(s.index, peaks, troughs)

    return DatingResult(
        data=s,
        peaks=s.index[peaks] if len(peaks) > 0 else pd.DatetimeIndex([]),
        troughs=s.index[troughs] if len(troughs) > 0 else pd.DatetimeIndex([]),
        phase=phase,
        metadata={
            "method": method,
            "window": int(window),
            "min_phase": int(min_phase),
            "min_cycle": int(min_cycle),
            "n_peaks_raw": int(len(peaks_raw)),
            "n_troughs_raw": int(len(troughs_raw)),
            "n_peaks_final": int(len(peaks)),
            "n_troughs_final": int(len(troughs)),
        },
    )
