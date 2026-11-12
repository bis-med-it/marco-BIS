"""Contracts between replication scripts and the library return values.

A replication script that reads a diagnostic under the wrong key silently
turns a valid run into a reported failure (missing keys default to inf or
NaN). These tests feed the library's actual return value through the
script's own extraction.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lib.identification import _proxy_first_stage_multi

ROOT = Path(__file__).resolve().parent.parent
_MR = ROOT / "scripts" / "replications" / "mertens_ravn_2013_tax_proxy_svar" / "replicate.py"
_spec = importlib.util.spec_from_file_location("mr_replicate", _MR)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)


def _diagnostics():
    rng = np.random.default_rng(0)
    T = 200
    z = rng.standard_normal((T, 2))
    u = z @ np.array([[1.0, 0.2], [0.1, 1.0]]) + 0.5 * rng.standard_normal((T, 2))
    return _proxy_first_stage_multi(u, z, policy_idxs=[0, 1])


def test_mr_first_stage_summary_reads_library_keys():
    fs = _diagnostics()
    cond, personal_F = mr.first_stage_summary(fs)
    assert cond == pytest.approx(fs["Sigma_p_cond"])
    own = [p["F"] for p in fs["per_pair"]
           if p["instrument_idx"] == 0 and p["policy_position"] == 0]
    assert personal_F == pytest.approx(own[0])
    assert np.isfinite(cond) and np.isfinite(personal_F) and personal_F > 10


def test_mr_first_stage_summary_raises_on_missing_keys():
    with pytest.raises(KeyError):
        mr.first_stage_summary({"per_pair": []})
