"""Excel report generator (multi-sheet, Arabic headers).

مولّد تقرير Excel متعدّد الأوراق بعناوين عربية. يتطلب openpyxl.
"""
from __future__ import annotations

import pandas as pd

from ..utils.arabic_text import CASE_NAMES_AR, TERMS_AR, IC_NAMES_AR


def build_excel_report(result, path: str) -> str:
    """Write a multi-sheet Excel workbook and return ``path``."""
    r = result

    meta = pd.DataFrame(
        {
            "البند": [
                TERMS_AR["dependent"], TERMS_AR["independent"],
                TERMS_AR["selected_case"], TERMS_AR["n_obs"],
                TERMS_AR["ardl_ic"], TERMS_AR["selected_lags"],
                TERMS_AR["n_boot"], TERMS_AR["final_decision"],
            ],
            "القيمة": [
                r.yvar, "، ".join(r.xvar),
                CASE_NAMES_AR.get(r.case, r.case), r.n_obs,
                IC_NAMES_AR.get(r.ardl_ic, r.ardl_ic), str(r.diff_lags),
                f"{r.n_boot} (فعّالة {r.bootstrap.n_boot_effective})",
                r.decision.label_ar,
            ],
        }
    )

    warnings_df = pd.DataFrame(
        {
            "الخطورة": [w.severity for w in r.warnings_list],
            "التحذير": [w.message_ar for w in r.warnings_list],
            "الإرشاد": [w.advice_ar for w in r.warnings_list],
        }
    )

    try:
        engine = "openpyxl"
        with pd.ExcelWriter(path, engine=engine) as xl:
            meta.to_excel(xl, sheet_name="ملخص", index=False)
            r.جدول_الإحصائيات().to_excel(xl, sheet_name="الإحصائيات")
            r.جدول_البوتستراب().to_excel(xl, sheet_name="القيم_الحرجة")
            r.جدول_القيم_الاحتمالية().to_excel(xl, sheet_name="القيم_الاحتمالية")
            r.جدول_المعاملات().to_excel(xl, sheet_name="المعاملات_مشروط")
            r.جدول_المعاملات(مشروط=False).to_excel(xl, sheet_name="المعاملات_غير_مشروط")
            r.جدول_حدود_PSS().to_excel(xl, sheet_name="حدود_PSS")
            smg = r.جدول_حدود_SMG()
            if not smg.empty:
                smg.to_excel(xl, sheet_name="حدود_SMG")
            if not warnings_df.empty:
                warnings_df.to_excel(xl, sheet_name="التحذيرات", index=False)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "تصدير Excel يتطلب openpyxl. ثبّته عبر: pip install openpyxl"
        ) from exc
    return path
