"""الواجهة العربية لاختبار ARDL بالبوتستراب.

الدالة الرئيسية ``اختبار_ARDL_بالبوتستراب`` تستدعي المحرّك المركزي المشترك
وتُرجع كائن النتائج العربي ``ArabicBootARDLResult``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .core.engine import run_bootstrap_ardl
from .results.arabic_result import ArabicBootARDLResult


def اختبار_ARDL_بالبوتستراب(
    البيانات: pd.DataFrame,
    المتغير_التابع: Optional[str] = None,
    المتغيرات_المستقلة: Optional[Sequence[str]] = None,
    إبطاءات_ARDL_ثابتة: Optional[Sequence[int]] = None,
    معيار_اختيار_ARDL: str = "AIC",
    أقصى_إبطاء: int = 5,
    الحالة: int = 3,
    عدد_تكرارات_البوتستراب: int = 2000,
    مستويات_البوتستراب: Sequence[float] = (0.10, 0.05, 0.01),
    مستوى_القرار: float = 0.05,
    طباعة_التقدم: bool = True,
    عشوائية: Optional[int] = None,
) -> ArabicBootARDLResult:
    """تشغيل اختبار حدود ARDL بالبوتستراب (بدون VECM) وإرجاع النتيجة العربية.

    المعاملات
    ---------
    البيانات: إطار بيانات pandas يحتوي السلاسل الزمنية.
    المتغير_التابع: اسم العمود التابع (الافتراضي: العمود الأول).
    المتغيرات_المستقلة: أسماء أعمدة المتغيرات المستقلة (الافتراضي: بقية الأعمدة).
    إبطاءات_ARDL_ثابتة: رتب الإبطاء الثابتة لكل متغير [y, x1, ...]؛ إن تُركت None
        يُجرى اختيار تلقائي بمعيار المعلومات.
    معيار_اختيار_ARDL: AIC / AICc / BIC / SC / HQ / adjR2.
    أقصى_إبطاء: أقصى رتبة إبطاء عند الاختيار التلقائي.
    الحالة: الحالة الحتمية (1 إلى 5) حسب Pesaran-Shin-Smith.
    عدد_تكرارات_البوتستراب: عدد عينات البوتستراب.
    مستويات_البوتستراب: مستويات المعنوية للقيم الحرجة.
    مستوى_القرار: مستوى المعنوية المعتمد في القرار النهائي.
    طباعة_التقدم: إظهار شريط التقدم.
    عشوائية: بذرة العشوائية لإعادة الإنتاج.
    """
    return run_bootstrap_ardl(
        data=البيانات,
        yvar=المتغير_التابع,
        xvar=المتغيرات_المستقلة,
        fixed_ardl_lags=إبطاءات_ARDL_ثابتة,
        ardl_ic=معيار_اختيار_ARDL,
        max_lag=أقصى_إبطاء,
        case=الحالة,
        n_boot=عدد_تكرارات_البوتستراب,
        bootstrap_levels=مستويات_البوتستراب,
        decision_level=مستوى_القرار,
        progress=طباعة_التقدم,
        random_state=عشوائية,
    )
