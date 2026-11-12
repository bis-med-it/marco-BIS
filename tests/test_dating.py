"""Tests for the business-cycle dating algorithms (Bry-Boschan / Harding-Pagan)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import lib


def _make_sine_cycle(T: int = 100, period: int = 20, amp: float = 1.0,
                     phase: float = 0.0, freq: str = "QS") -> pd.Series:
    """Generate a clean sine wave for deterministic turning-point tests."""
    t = np.arange(T)
    y = amp * np.sin(2 * np.pi * (t + phase) / period)
    return pd.Series(y, index=pd.date_range("1970-01-01", periods=T, freq=freq))


class TestHardingPagan:

    def test_clean_sine_identifies_correct_turning_points(self):
        """On a clean sine wave with period 20q, we expect peaks at quarter-mark
        positions [5, 25, 45, 65, 85] and troughs at [15, 35, 55, 75, 95]."""
        s = _make_sine_cycle(T=100, period=20)
        result = lib.harding_pagan_dating(s)
        # Allow ±1 quarter slack due to discrete sampling
        peak_positions = np.array([s.index.get_loc(d) for d in result.peaks])
        trough_positions = np.array([s.index.get_loc(d) for d in result.troughs])
        # First peak should be near index 5
        assert (peak_positions[0] - 5) in (-1, 0, 1)
        # Peaks should be ~20 apart
        if len(peak_positions) > 1:
            np.testing.assert_array_less(np.abs(np.diff(peak_positions) - 20), 2)

    def test_alternation_enforced(self):
        """Peaks and troughs must alternate strictly."""
        s = _make_sine_cycle(T=120, period=24)
        result = lib.harding_pagan_dating(s)
        # Sort all turns and check alternation
        all_turns = sorted(
            [(d, "P") for d in result.peaks] + [(d, "T") for d in result.troughs],
            key=lambda x: x[0],
        )
        for i in range(1, len(all_turns)):
            assert all_turns[i][1] != all_turns[i - 1][1], (
                "two consecutive turns of the same kind"
            )

    def test_phase_label_consistency(self):
        """Period after a peak should be 'recession' until next trough."""
        s = _make_sine_cycle(T=80, period=20)
        result = lib.harding_pagan_dating(s)
        # Sample a period right after a peak: should be 'recession'
        if len(result.peaks) > 0 and len(result.troughs) > 0:
            first_peak = result.peaks[0]
            first_trough = result.troughs[result.troughs > first_peak][0] if (result.troughs > first_peak).any() else None
            if first_trough is not None:
                mid_idx = s.index[(s.index > first_peak) & (s.index < first_trough)]
                if len(mid_idx) > 0:
                    assert result.phase.loc[mid_idx[0]] == "recession"

    def test_minimum_phase_length(self):
        """Short noisy ripples within a long cycle shouldn't produce turning points
        spaced < min_phase apart."""
        s = _make_sine_cycle(T=120, period=30)
        result = lib.harding_pagan_dating(s, min_phase=2)
        all_turns = sorted(list(result.peaks) + list(result.troughs))
        if len(all_turns) > 1:
            gaps_quarters = np.diff([s.index.get_loc(d) for d in all_turns])
            assert (gaps_quarters >= 2).all()

    def test_recession_dummy_format(self):
        s = _make_sine_cycle(T=80, period=20)
        result = lib.harding_pagan_dating(s)
        dummy = result.recession_dummy()
        assert dummy.index.equals(s.index)
        valid = dummy.dropna()
        assert valid.isin([0.0, 1.0]).all()

    def test_too_short_series_raises(self):
        s = _make_sine_cycle(T=3)
        with pytest.raises(ValueError, match="series too short"):
            lib.harding_pagan_dating(s)

    def test_non_series_input_raises(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="must be pd.Series"):
            lib.harding_pagan_dating(arr)


class TestBryBoschan:

    def test_monthly_period_30_returns_correct_count(self):
        """A monthly sine of period 30 over 120 months should produce 4 cycles."""
        s = _make_sine_cycle(T=120, period=30, freq="MS")
        result = lib.bry_boschan_dating(s)
        # Roughly 4 peaks and 4 troughs (allow some boundary loss)
        assert 3 <= result.metadata["n_peaks_final"] <= 5
        assert 3 <= result.metadata["n_troughs_final"] <= 5

    def test_minimum_phase_length_for_monthly_is_stricter(self):
        """min_phase=5 (BB default) drops more turns than min_phase=2 (HP default)."""
        # Create a noisy signal with both short and long-period components
        rng = np.random.default_rng(0)
        T = 240
        long_cycle = np.sin(2 * np.pi * np.arange(T) / 60)
        short_noise = 0.3 * np.sin(2 * np.pi * np.arange(T) / 8)   # high-frequency ripple
        signal = long_cycle + short_noise
        s = pd.Series(signal, index=pd.date_range("1990-01-01", periods=T, freq="MS"))
        result_bb = lib.bry_boschan_dating(s, window=5, min_phase=5, min_cycle=15)
        result_lax = lib.bry_boschan_dating(s, window=2, min_phase=2, min_cycle=5)
        # Stricter rules → fewer or equal turns identified
        n_bb = result_bb.metadata["n_peaks_final"] + result_bb.metadata["n_troughs_final"]
        n_lax = result_lax.metadata["n_peaks_final"] + result_lax.metadata["n_troughs_final"]
        assert n_bb <= n_lax


class TestDatingResultContainer:

    def test_metadata_carries_method_and_params(self):
        s = _make_sine_cycle(T=80, period=20)
        result = lib.harding_pagan_dating(s, window=2, min_phase=2, min_cycle=5)
        assert result.metadata["method"] == "harding_pagan"
        assert result.metadata["window"] == 2
        assert result.metadata["min_phase"] == 2
        assert result.metadata["min_cycle"] == 5

    def test_n_recessions_property(self):
        s = _make_sine_cycle(T=80, period=20)
        result = lib.harding_pagan_dating(s)
        # n_recessions = number of troughs (=number of completed recession phases)
        assert result.n_recessions == len(result.troughs)


_FRED_QD_CSV = Path(__file__).resolve().parents[1] / "data" / "fred_qd" / "dataset.csv"
requires_fred_qd = pytest.mark.skipif(
    not _FRED_QD_CSV.exists(),
    reason="fred_qd dataset.csv not downloaded - see data/fred_qd/UPDATE.md",
)


@requires_fred_qd
class TestRealGDPSanityCheck:
    """Sanity-check on actual US real GDP — algorithm should identify recessions
    that broadly match NBER reference dates (within 1-2 quarters)."""

    def test_finds_major_postwar_recessions(self):
        raw = lib.load_dataset(source="fred_qd", series={"gdp": "GDPC1"},
                                start="1960-01", end="2019-12")
        log_gdp = np.log(raw["gdp"])
        result = lib.harding_pagan_dating(log_gdp)
        # Should find roughly 6-8 recessions in the postwar US (NBER: 1969-70,
        # 1973-75, 1980, 1981-82, 1990-91, 2001 (mild, often missed by GDP-only),
        # 2008-09). We exclude COVID by cutting the sample at 2019Q4.
        assert 5 <= result.n_recessions <= 9


class TestMinCycleEnforcement:
    """Min-cycle censoring contract (Bry-Boschan / Harding-Pagan):
    peak-to-peak and trough-to-trough distances must be >= min_cycle.
    This does NOT follow from alternation + min_phase under either
    default rule set (HP: 2+2 < 5; BB: 5+5 < 15), so it must be
    enforced, not merely recorded in metadata.
    """

    @staticmethod
    def _gaps_ok(result, min_cycle):
        import numpy as np
        for idx in (result.peaks, result.troughs):
            pos = result.data.index.get_indexer(idx)
            if len(pos) >= 2:
                if np.diff(np.sort(pos)).min() < min_cycle:
                    return False
        return True

    def test_short_spurious_cycle_censored(self):
        """A 2+2-quarter blip creates a P-T-P cycle of 4 < 5 that
        alternation + min_phase alone would accept; the shallower peak
        must be censored."""
        rng = np.random.default_rng(3)
        t = np.arange(120)
        base = np.sin(2 * np.pi * t / 24)            # genuine 24-quarter cycles
        blip = np.zeros_like(base, dtype=float)
        blip[60] = -0.6                              # carve a 4-quarter mini-cycle
        s = pd.Series(base + blip + 0.01 * rng.standard_normal(120),
                      index=pd.period_range("1980Q1", periods=120,
                                            freq="Q").to_timestamp())
        res = lib.harding_pagan_dating(s, window=2, min_phase=2, min_cycle=5)
        assert self._gaps_ok(res, 5), "peak-to-peak / trough-to-trough < min_cycle survived"

    def test_min_cycle_holds_under_both_rule_sets(self):
        rng = np.random.default_rng(9)
        t = np.arange(300)
        s_q = pd.Series(np.sin(2 * np.pi * t / 20) + 0.15 * rng.standard_normal(300),
                        index=pd.period_range("1950Q1", periods=300,
                                              freq="Q").to_timestamp())
        res_q = lib.harding_pagan_dating(s_q)
        assert self._gaps_ok(res_q, 5)
        s_m = pd.Series(np.sin(2 * np.pi * t / 60) + 0.15 * rng.standard_normal(300),
                        index=pd.period_range("1990-01", periods=300,
                                              freq="M").to_timestamp())
        res_m = lib.bry_boschan_dating(s_m)
        assert self._gaps_ok(res_m, 15)

    def test_alternation_preserved_after_censoring(self):
        rng = np.random.default_rng(9)
        t = np.arange(300)
        s = pd.Series(np.sin(2 * np.pi * t / 20) + 0.15 * rng.standard_normal(300),
                      index=pd.period_range("1950Q1", periods=300,
                                            freq="Q").to_timestamp())
        res = lib.harding_pagan_dating(s)
        pos_p = res.data.index.get_indexer(res.peaks)
        pos_t = res.data.index.get_indexer(res.troughs)
        events = sorted([(p, "P") for p in pos_p] + [(q, "T") for q in pos_t])
        kinds = [k for _, k in events]
        assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)), \
            "consecutive same-kind turning points survived"


class TestAmplitudeSignCensor:
    """Acceptance check: a net RISE must never be labelled a recession."""

    def test_net_rise_not_a_recession(self):
        vals = [0, 1, 5, 4, 4.5, 6, 7, 8, 9, 11, 12, 10, 13, 14, 15]
        idx = pd.period_range("2000Q1", periods=len(vals), freq="Q").to_timestamp()
        res = lib.harding_pagan_dating(pd.Series(vals, index=idx, dtype=float))
        # Every retained recession must end BELOW its starting peak
        for p, t in zip(res.peaks, res.troughs):
            assert res.data[t] < res.data[p]
        assert int((res.phase == "recession").sum()) == 0

    def test_genuine_recession_retained(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 6, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        idx = pd.period_range("2000Q1", periods=len(vals), freq="Q").to_timestamp()
        res = lib.harding_pagan_dating(pd.Series(vals, index=idx, dtype=float))
        assert int((res.phase == "recession").sum()) == 4
        for p, t in zip(res.peaks, res.troughs):
            assert res.data[t] < res.data[p]


class TestAdditiveShiftInvariance:
    """Fix acceptance: turning points, censoring and tie-breaks must depend
    only on level DIFFERENCES, never on the level's origin — dating y and
    y ± 100 must give identical dates."""

    @staticmethod
    def _dates(res):
        return (list(res.peaks), list(res.troughs))

    def test_noisy_cycle_dates_invariant(self):
        rng = np.random.default_rng(9)
        t = np.arange(300)
        base = np.sin(2 * np.pi * t / 20) + 0.15 * rng.standard_normal(300)
        idx = pd.period_range("1950Q1", periods=300, freq="Q").to_timestamp()
        res0 = lib.harding_pagan_dating(pd.Series(base, index=idx))
        res_up = lib.harding_pagan_dating(pd.Series(base + 100.0, index=idx))
        res_dn = lib.harding_pagan_dating(pd.Series(base - 100.0, index=idx))
        assert self._dates(res0) == self._dates(res_up) == self._dates(res_dn)

    def test_amplitude_censor_invariant_under_negative_levels(self):
        # Engages the amplitude-sign censor (net-rise pseudo-recession).
        # Shifting all levels negative flips every raw-level sign, which the
        # old raw-level tie-break would have turned into different censoring.
        vals = np.array([0, 1, 5, 4, 4.5, 6, 7, 8, 9, 11, 12, 10, 13, 14, 15],
                        dtype=float)
        idx = pd.period_range("2000Q1", periods=len(vals), freq="Q").to_timestamp()
        res0 = lib.harding_pagan_dating(pd.Series(vals, index=idx))
        res_dn = lib.harding_pagan_dating(pd.Series(vals - 100.0, index=idx))
        assert self._dates(res0) == self._dates(res_dn)
