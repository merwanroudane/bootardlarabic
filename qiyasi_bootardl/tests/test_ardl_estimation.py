"""Tests for ARDL-ECM estimation and test statistics."""
from __future__ import annotations

import numpy as np

from qiyasi_bootardl.core.ardl import estimate_ardl
from qiyasi_bootardl.core.statistics import compute_statistics


def test_estimate_ardl_basic(cointegrated_data):
    fit = estimate_ardl(cointegrated_data, [1, 1, 1], case=3, conditional=True, trim=2)
    assert fit.nobs > 0
    assert fit.k == fit.X.shape[1]
    assert set(["y.l1", "x1.l1", "x2.l1", "const"]).issubset(set(fit.names))
    assert 0.0 <= fit.rsquared <= 1.0


def test_statistics_finite_and_positive_F(cointegrated_data):
    fit = estimate_ardl(cointegrated_data, [1, 1, 1], case=3, conditional=True, trim=2)
    st = compute_statistics(fit, case=3)
    assert np.isfinite(st.f_overall) and st.f_overall > 0
    assert np.isfinite(st.f_ind) and st.f_ind > 0
    assert np.isfinite(st.t_dep)
    assert st.f_overall_q >= st.f_ind_q


def test_cointegrated_rejects_more_than_independent(cointegrated_data, independent_data):
    fit_c = estimate_ardl(cointegrated_data, [1, 1, 1], case=3, conditional=True, trim=2)
    fit_i = estimate_ardl(independent_data, [1, 1, 1], case=3, conditional=True, trim=2)
    st_c = compute_statistics(fit_c, case=3)
    st_i = compute_statistics(fit_i, case=3)
    # cointegrated system should show a stronger overall F
    assert st_c.f_overall > st_i.f_overall
