"""Tests for design-matrix construction and lag selection."""
from __future__ import annotations

import numpy as np

from qiyasi_bootardl.core.lag_selection import build_design, select_diff_lags


def test_build_design_shapes_and_names(cointegrated_data):
    X, y, info = build_design(cointegrated_data, [1, 1, 1], case=3, conditional=True, trim=2)
    assert "const" in X.columns
    assert info.y_level == "y.l1"
    assert info.x_levels == ["x1.l1", "x2.l1"]
    # contemporaneous diffs of X present only in conditional model
    assert "D_x1" in X.columns and "D_x2" in X.columns
    assert len(X) == len(y)
    assert X.notna().all().all()


def test_unconditional_has_no_contemp(cointegrated_data):
    X, _, info = build_design(cointegrated_data, [1, 1, 1], case=3, conditional=False, trim=2)
    assert info.contemp_names == []
    assert not any(c == "D_x1" for c in X.columns)


def test_trend_added_for_case_ge_4(cointegrated_data):
    X, _, _ = build_design(cointegrated_data, [1, 1, 1], case=4, conditional=True, trim=2)
    assert "trend" in X.columns


def test_select_diff_lags_returns_valid(cointegrated_data):
    lags, ic, crit = select_diff_lags(cointegrated_data, case=3, max_lag=3, criterion="AIC")
    assert len(lags) == 3
    assert all(0 <= L <= 3 for L in lags)
    assert np.isfinite(ic)
    assert crit == "AIC"
