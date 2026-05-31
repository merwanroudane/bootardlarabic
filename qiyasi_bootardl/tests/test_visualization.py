"""Tests for the Arabic visualization layer (headless)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import pytest

from qiyasi_bootardl.core.engine import run_bootstrap_ardl


@pytest.fixture(scope="module")
def result(cointegrated_data):
    return run_bootstrap_ardl(cointegrated_data, case=3, max_lag=2, n_boot=200,
                              random_state=2024, progress=False)


def test_distributions_returns_figure(result):
    from matplotlib.figure import Figure
    fig = result.رسم_توزيعات_البوتستراب()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_decision_plot(result):
    from matplotlib.figure import Figure
    assert isinstance(result.رسم_القرار(), Figure)


def test_fake_cointegration_plot(result):
    from matplotlib.figure import Figure
    assert isinstance(result.رسم_التكامل_الزائف(), Figure)


def test_series_plot_uses_source_data(result):
    from matplotlib.figure import Figure
    assert result.source_data is not None
    assert isinstance(result.رسم_السلاسل_الزمنية(), Figure)


def test_save_all_plots(result, tmp_path):
    paths = result.حفظ_كل_الرسوم(str(tmp_path), "t")
    assert len(paths) == 4
    for p in paths:
        import os
        assert os.path.exists(p) and os.path.getsize(p) > 0


def test_arabic_shaping_helper():
    from qiyasi_bootardl.reports.visualization import ع
    out = ع("الإحصائية")
    assert isinstance(out, str) and len(out) > 0
