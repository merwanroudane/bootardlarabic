"""قياسي BootARDL — مكتبة عربية لاختبار التكامل المشترك ARDL بالبوتستراب.

qiyasi_bootardl: an Arabic-first Python library for the bootstrap ARDL bounds
test for cointegration (single-equation, VECM-free), inspired by R's bootCT.

المؤلف: د. مروان رودان | Author: Dr. Merwan Roudane.

الواجهة العربية:
    >>> from qiyasi_bootardl import اختبار_ARDL_بالبوتستراب
    >>> النتيجة = اختبار_ARDL_بالبوتستراب(البيانات)
    >>> print(النتيجة.ملخص())

English interface:
    >>> from qiyasi_bootardl import bootstrap_ardl_test
    >>> result = bootstrap_ardl_test(data)
    >>> print(result.summary())
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Dr. Merwan Roudane"

from .api_ar import اختبار_ARDL_بالبوتستراب
from .api_en import bootstrap_ardl_test
from .results.arabic_result import ArabicBootARDLResult
from .results.english_result import EnglishBootARDLResult
from .utils.arabic_text import REFERENCES, CASE_NAMES_AR, CASE_NAMES_EN

__all__ = [
    "اختبار_ARDL_بالبوتستراب",
    "bootstrap_ardl_test",
    "ArabicBootARDLResult",
    "EnglishBootARDLResult",
    "REFERENCES",
    "CASE_NAMES_AR",
    "CASE_NAMES_EN",
    "__version__",
    "__author__",
]
