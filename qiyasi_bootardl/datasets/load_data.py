"""Bundled example datasets. مجموعات البيانات التوضيحية المرفقة."""
from __future__ import annotations

import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def تحميل_بيانات_كلية() -> pd.DataFrame:
    """تحميل بيانات اقتصاد كلي توضيحية (متكاملة مشتركاً) بأعمدة عربية.

    الأعمدة: الناتج (تابع)، الاستثمار، الانفتاح. n=120.
    """
    return pd.read_csv(os.path.join(_HERE, "macro_example.csv"), encoding="utf-8-sig")


def load_macro_example() -> pd.DataFrame:
    """English alias for :func:`تحميل_بيانات_كلية`."""
    return تحميل_بيانات_كلية()


def تحميل_بيانات_ألمانيا() -> pd.DataFrame:
    """تحميل بيانات الاقتصاد الكلي الألماني الحقيقية (Lütkepohl 2007).

    سلاسل ربعية موسمياً معدّلة لألمانيا الغربية من 1960Q1 إلى 1982Q4
    (مليار مارك ألماني)، مصدرها بنك ألمانيا الفيدرالي عبر ملف E1.

    الأعمدة: التاريخ، الاستهلاك (تابع)، الدخل، الاستثمار. n=92.
    """
    df = pd.read_csv(os.path.join(_HERE, "ger_macro_ar.csv"), encoding="utf-8-sig")
    return df


def load_german_macro() -> pd.DataFrame:
    """English alias for :func:`تحميل_بيانات_ألمانيا`."""
    return تحميل_بيانات_ألمانيا()
