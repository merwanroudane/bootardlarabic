"""Arabic economic/methodological interpretation generator.

يولّد فقرة تفسيرية عربية تربط نتيجة الاختبار بالمعنى الاقتصادي والمنهجي.
"""
from __future__ import annotations

from typing import List

from ..utils.formatting import fmt, pct
from .decisions import (
    Decision,
    COINTEGRATION,
    NO_COINTEGRATION,
    INCONCLUSIVE,
    DEGENERATE,
)


def interpret(
    decision: Decision,
    yvar: str,
    xvar: List[str],
    f_overall_obs: float,
    t_obs: float,
    f_ind_obs: float,
) -> str:
    """Return a multi-line Arabic interpretation paragraph."""
    xs = "، ".join(xvar)
    lvl = pct(decision.level)
    lines: List[str] = []

    if decision.code == COINTEGRATION:
        lines.append(
            f"عند مستوى معنوية {lvl}، توجد علاقة توازنية طويلة الأجل (تكامل مشترك) "
            f"بين المتغير التابع «{yvar}» والمتغيرات المستقلة ({xs})."
        )
        lines.append(
            "يمكن المضي قُدماً في تقدير معاملات الأجل الطويل ونموذج تصحيح الخطأ "
            "(ECM) لتقدير سرعة التعديل نحو التوازن."
        )
    elif decision.code == NO_COINTEGRATION:
        lines.append(
            f"عند مستوى معنوية {lvl}، لا يوجد دليل على علاقة توازنية طويلة الأجل "
            f"بين «{yvar}» و({xs})."
        )
        lines.append(
            "يُفضّل في هذه الحالة نمذجة العلاقة بدلالة الفروق الأولى (نموذج قصير الأجل) "
            "دون فرض علاقة تكامل مشترك."
        )
    elif decision.code == DEGENERATE:
        lines.append(
            f"عند مستوى معنوية {lvl}، النتيجة تشير إلى تكامل مشترك زائف/متدهور: "
            "ترفض الإحصائية المشروطة الفرضيةَ الصفرية بينما لا ترفضها غير المشروطة."
        )
        lines.append(
            f"قد يعني ذلك أن «{yvar}» لا يُعدّل فعلياً نحو ({xs})، أو أن أحد المتغيرات "
            "المستقلة ليس خارجياً ضعيفاً. يُنصح بإعادة فحص اتجاه العلاقة."
        )
    else:  # INCONCLUSIVE
        lines.append(
            f"عند مستوى معنوية {lvl}، النتيجة غير حاسمة؛ لا تتفق إحصائيات الاختبار "
            "الثلاث على قرار موحّد."
        )
        lines.append(
            "يُنصح بإعادة اختيار رتب الإبطاء، أو تغيير الحالة الحتمية، أو زيادة حجم "
            "العينة، قبل إصدار حكم نهائي."
        )

    # statistic recap
    lines.append(
        "ملخص الإحصائيات المرصودة: "
        f"F الكلية = {fmt(f_overall_obs)}، "
        f"t = {fmt(t_obs)}، "
        f"F للمتغيرات المستقلة = {fmt(f_ind_obs)}."
    )
    return "\n".join(lines)
