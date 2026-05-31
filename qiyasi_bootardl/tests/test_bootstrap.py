"""Tests for the VECM-free fixed-X residual bootstrap."""
from __future__ import annotations

import numpy as np

from qiyasi_bootardl.core.engine import run_bootstrap_ardl


def test_bootstrap_detects_cointegration(cointegrated_data):
    r = run_bootstrap_ardl(
        cointegrated_data, case=3, max_lag=3, n_boot=400,
        random_state=2024, progress=False,
    )
    assert r.bootstrap.n_boot_effective > 0
    # strong cointegration -> small p-values
    assert r.bootstrap.pvalue_f_overall < 0.10
    assert r.decision.code in ("تكامل_مشترك", "تكامل_زائف_متدهور")


def test_bootstrap_no_false_cointegration(independent_data):
    r = run_bootstrap_ardl(
        independent_data, case=3, max_lag=3, n_boot=400,
        random_state=7, progress=False,
    )
    # independent walks should usually NOT be flagged as clean cointegration
    assert r.decision.code != "تكامل_مشترك" or r.bootstrap.pvalue_f_overall > 0.01


def test_reproducible_with_seed(cointegrated_data):
    r1 = run_bootstrap_ardl(cointegrated_data, case=3, max_lag=2, n_boot=200,
                            random_state=123, progress=False)
    r2 = run_bootstrap_ardl(cointegrated_data, case=3, max_lag=2, n_boot=200,
                            random_state=123, progress=False)
    assert r1.bootstrap.crit_f_overall == r2.bootstrap.crit_f_overall


def test_critical_values_monotonic(cointegrated_data):
    r = run_bootstrap_ardl(cointegrated_data, case=3, max_lag=2, n_boot=400,
                           random_state=5, progress=False)
    cf = r.bootstrap.crit_f_overall
    # higher significance (smaller alpha) -> larger F critical value
    assert cf[0.10] <= cf[0.05] <= cf[0.01]
