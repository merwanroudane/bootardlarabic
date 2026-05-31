"""Arabic text resources and translation dictionaries.

All user-facing Arabic strings live here so that computation stays separate
from presentation (design principle 17 of the specification).
موارد النصوص العربية وقواميس الترجمة.
"""
from __future__ import annotations

# --- Eastern-Arabic digit mapping (optional cosmetic) ---
_WESTERN = "0123456789"
_EASTERN = "٠١٢٣٤٥٦٧٨٩"
_E_MAP = str.maketrans(_WESTERN, _EASTERN)


def to_arabic_digits(text: str) -> str:
    """Convert Western digits to Eastern-Arabic digits. تحويل الأرقام."""
    return str(text).translate(_E_MAP)


# --- Deterministic case descriptions (Arabic) ---
CASE_NAMES_AR = {
    1: "الحالة الأولى: بدون ثابت وبدون اتجاه زمني",
    2: "الحالة الثانية: ثابت مقيد وبدون اتجاه زمني",
    3: "الحالة الثالثة: ثابت غير مقيد وبدون اتجاه زمني",
    4: "الحالة الرابعة: ثابت غير مقيد واتجاه زمني مقيد",
    5: "الحالة الخامسة: ثابت غير مقيد واتجاه زمني غير مقيد",
}

CASE_NAMES_EN = {
    1: "Case I: no intercept, no trend",
    2: "Case II: restricted intercept, no trend",
    3: "Case III: unrestricted intercept, no trend",
    4: "Case IV: unrestricted intercept, restricted trend",
    5: "Case V: unrestricted intercept, unrestricted trend",
}

# --- Statistic labels ---
STAT_LABELS_AR = {
    "f_overall": "إحصائية F الكلية",
    "t_dep": "إحصائية t للمتغير التابع المتأخر",
    "f_ind": "إحصائية F للمتغيرات المستقلة المتأخرة",
}

STAT_LABELS_EN = {
    "f_overall": "F-overall statistic",
    "t_dep": "t-statistic (lagged dependent)",
    "f_ind": "F-independent statistic",
}

# --- Information criteria names ---
IC_NAMES_AR = {
    "AIC": "معيار أكايكي AIC",
    "AICc": "معيار أكايكي المصحح AICc",
    "BIC": "معيار شوارتز BIC",
    "SC": "معيار شوارتز SC",
    "HQ": "معيار هانان-كوين HQ",
    "adjR2": "معامل التحديد المعدل",
}

# --- Generic terms ---
TERMS_AR = {
    "test_name": "اختبار ARDL بالبوتستراب للتكامل المشترك",
    "dependent": "المتغير التابع",
    "independent": "المتغيرات المستقلة",
    "n_obs": "عدد المشاهدات المستخدمة",
    "ardl_ic": "معيار اختيار ARDL",
    "max_lag": "أقصى إبطاء",
    "n_boot": "عدد تكرارات البوتستراب",
    "selected_case": "الحالة المحددة",
    "selected_lags": "فترات الإبطاء المختارة",
    "test_statistics": "إحصائيات الاختبار",
    "boot_crit": "القيم الحرجة بالبوتستراب",
    "boot_pval": "القيم الاحتمالية بالبوتستراب",
    "final_decision": "القرار النهائي",
    "conclusion": "الاستنتاج",
    "level": "المستوى",
    "statistic": "الإحصائية",
    "pvalue": "القيمة الاحتمالية",
    "warnings": "التحذيرات المنهجية",
    "interpretation": "التفسير الاقتصادي والمنهجي",
    "pss_bounds": "اختبار الحدود PSS",
    "smg": "اختبار SMG للمتغيرات المستقلة",
    "lower_bound": "الحد الأدنى I(0)",
    "upper_bound": "الحد الأعلى I(1)",
    "fstat": "إحصائية F",
}

REFERENCES = [
    "Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing "
    "approaches to the analysis of level relationships. Journal of Applied "
    "Econometrics, 16(3), 289-326.",
    "Sam, C. Y., McNown, R., & Goh, S. K. (2019). An augmented autoregressive "
    "distributed lag bounds test for cointegration. Economic Modelling, 80, 130-141.",
    "McNown, R., Sam, C. Y., & Goh, S. K. (2018). Bootstrapping the "
    "autoregressive distributed lag test for cointegration. Applied Economics, "
    "50(13), 1509-1521.",
    "Narayan, P. K. (2005). The saving and investment nexus for China: evidence "
    "from cointegration tests. Applied Economics, 37(17), 1979-1990.",
    "Vacca, G., & Bertelli, S. (2024). bootCT: Bootstrapping the ARDL Tests for "
    "Cointegration. R package version 2.1.0. (computational inspiration only).",
]
