"""Tests for the Arabic and English public APIs (shared engine)."""
from __future__ import annotations

import pytest

from qiyasi_bootardl import اختبار_ARDL_بالبوتستراب, bootstrap_ardl_test
from qiyasi_bootardl.results.arabic_result import ArabicBootARDLResult
from qiyasi_bootardl.results.english_result import EnglishBootARDLResult
from qiyasi_bootardl.utils.exceptions import DataValidationError


def test_arabic_api_returns_arabic_result(cointegrated_data):
    r = اختبار_ARDL_بالبوتستراب(
        cointegrated_data, الحالة=3, أقصى_إبطاء=2,
        عدد_تكرارات_البوتستراب=200, عشوائية=1, طباعة_التقدم=False,
    )
    assert isinstance(r, ArabicBootARDLResult)


def test_english_api_returns_english_result(cointegrated_data):
    r = bootstrap_ardl_test(
        cointegrated_data, case=3, max_lag=2, n_boot=200,
        random_state=1, progress=False,
    )
    assert isinstance(r, EnglishBootARDLResult)
    assert isinstance(r.arabic, ArabicBootARDLResult)


def test_both_apis_agree(cointegrated_data):
    ar = اختبار_ARDL_بالبوتستراب(cointegrated_data, الحالة=3, أقصى_إبطاء=2,
                                  عدد_تكرارات_البوتستراب=200, عشوائية=42, طباعة_التقدم=False)
    en = bootstrap_ardl_test(cointegrated_data, case=3, max_lag=2, n_boot=200,
                             random_state=42, progress=False)
    assert ar.statistics.f_overall == en.arabic.statistics.f_overall
    assert ar.decision.code == en.arabic.decision.code


def test_fixed_lags_wrong_length_raises(cointegrated_data):
    with pytest.raises(ValueError):
        bootstrap_ardl_test(cointegrated_data, fixed_ardl_lags=[1, 1],
                            n_boot=50, progress=False)


def test_tiny_sample_raises():
    import pandas as pd
    small = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
    with pytest.raises(DataValidationError):
        bootstrap_ardl_test(small, n_boot=50, progress=False)
