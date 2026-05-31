"""Tests for the Arabic result object and its methods."""
from __future__ import annotations

import pandas as pd

from qiyasi_bootardl.core.engine import run_bootstrap_ardl


def _result(data):
    return run_bootstrap_ardl(data, case=3, max_lag=2, n_boot=200,
                              random_state=2024, progress=False)


def test_summary_is_arabic_text(cointegrated_data):
    r = _result(cointegrated_data)
    s = r.ملخص()
    assert isinstance(s, str) and len(s) > 100
    assert "التكامل" in s or "تكامل" in s


def test_tables_are_dataframes(cointegrated_data):
    r = _result(cointegrated_data)
    for tbl in (r.جدول_المعاملات(), r.جدول_الإحصائيات(),
                r.جدول_البوتستراب(), r.جدول_القيم_الاحتمالية(),
                r.جدول_حدود_PSS()):
        assert isinstance(tbl, pd.DataFrame)


def test_decision_and_interpretation(cointegrated_data):
    r = _result(cointegrated_data)
    dec = r.قرار()
    assert dec.code in ("تكامل_مشترك", "لا_تكامل_مشترك", "غير_حاسم", "تكامل_زائف_متدهور")
    assert isinstance(r.تفسير(), str)


def test_html_report_rtl(cointegrated_data, tmp_path):
    r = _result(cointegrated_data)
    out = tmp_path / "report.html"
    html = r.تقرير_HTML(str(out))
    assert 'dir="rtl"' in html and 'lang="ar"' in html
    assert out.exists()


def test_warnings_present(cointegrated_data):
    r = _result(cointegrated_data)
    # n_boot=200 < 1000 triggers a low-replications warning
    codes = [w.code for w in r.تحذيرات()]
    assert "low_nboot" in codes
