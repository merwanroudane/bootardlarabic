"""Arabic decision engine for the bootstrap ARDL bounds test.

محرك القرار: يحوّل الإحصائيات والقيم الحرجة بالبوتستراب إلى قرار عربي واضح
حول وجود التكامل المشترك، مع رصد حالات التكامل الزائف/المتدهور.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..core.bootstrap import BootstrapResult
from ..core.deterministic_cases import get_case_spec


# Decision codes
COINTEGRATION = "تكامل_مشترك"
NO_COINTEGRATION = "لا_تكامل_مشترك"
INCONCLUSIVE = "غير_حاسم"
DEGENERATE = "تكامل_زائف_متدهور"


@dataclass
class Decision:
    code: str
    level: float
    label_ar: str
    detail_ar: str
    f_overall_reject: bool
    t_reject: bool
    f_ind_reject: bool
    degenerate: bool
    per_level: Dict[float, str] = field(default_factory=dict)


def _reject_f(stat: float, crit: Dict[float, float], level: float) -> bool:
    return (level in crit) and (stat is not None) and (stat > crit[level])


def _reject_t(stat: float, crit: Dict[float, float], level: float) -> bool:
    # t-statistic: rejection of the unit-root null on lagged y is a large negative value
    return (level in crit) and (stat is not None) and (stat < crit[level])


def decide(
    boot: BootstrapResult,
    f_overall_obs: float,
    t_obs: float,
    f_ind_obs: float,
    case: int,
    level: float = 0.05,
) -> Decision:
    """Derive the Arabic cointegration decision at the given significance level.

    منطق McNown-Sam-Goh: التكامل المشترك المؤكَّد يتطلب رفض كلٍّ من F الكلية
    و t (على المتغير التابع المتأخر) و F للمتغيرات المستقلة. رفض F الكلية وحدها
    دون رفض t أو F المستقلة يشير إلى علاقة زائفة/متدهورة.
    """
    spec = get_case_spec(case)
    f_rej = _reject_f(f_overall_obs, boot.crit_f_overall, level)
    fi_rej = _reject_f(f_ind_obs, boot.crit_f_ind, level)
    t_rej = _reject_t(t_obs, boot.crit_t_dep, level) if spec.t_test_applicable else False

    degenerate = bool(boot.fake_cointegration.get(level, False))

    # Per-level quick summary
    per_level: Dict[float, str] = {}
    for a in boot.levels:
        fr = _reject_f(f_overall_obs, boot.crit_f_overall, a)
        ir = _reject_f(f_ind_obs, boot.crit_f_ind, a)
        tr = _reject_t(t_obs, boot.crit_t_dep, a) if spec.t_test_applicable else False
        if boot.fake_cointegration.get(a, False):
            per_level[a] = DEGENERATE
        elif fr and ir and (tr or not spec.t_test_applicable):
            per_level[a] = COINTEGRATION
        elif fr or ir or tr:
            per_level[a] = INCONCLUSIVE
        else:
            per_level[a] = NO_COINTEGRATION

    if degenerate:
        code = DEGENERATE
        label = "تكامل مشترك زائف/متدهور"
        detail = (
            "ترفض إحصائية F للمتغيرات المستقلة في النموذج المشروط الفرضيةَ الصفرية، "
            "بينما لا ترفضها في النموذج غير المشروط. هذا يشير إلى أن العلاقة طويلة "
            "الأجل قد تكون زائفة (degenerate case)، ويُنصح بإعادة فحص اتجاه السببية "
            "وكون المتغيرات المستقلة فعلاً خارجية وضعيفة."
        )
    elif f_rej and fi_rej and (t_rej or not spec.t_test_applicable):
        code = COINTEGRATION
        label = "وجود تكامل مشترك"
        detail = (
            "تتفق الإحصائيات الثلاث (F الكلية، t، F للمتغيرات المستقلة) على رفض "
            "فرضية عدم وجود تكامل مشترك. توجد علاقة توازنية طويلة الأجل."
            if spec.t_test_applicable else
            "تتفق إحصائيتا F (الكلية والمستقلة) على رفض فرضية عدم وجود تكامل مشترك."
        )
    elif not f_rej and not fi_rej and not t_rej:
        code = NO_COINTEGRATION
        label = "عدم وجود تكامل مشترك"
        detail = (
            "لا ترفض أيٌّ من الإحصائيات الفرضيةَ الصفرية عند هذا المستوى. لا يوجد "
            "دليل على علاقة توازنية طويلة الأجل."
        )
    else:
        code = INCONCLUSIVE
        label = "نتيجة غير حاسمة"
        parts = []
        parts.append("ترفض" if f_rej else "لا ترفض")
        detail = (
            "لا تتفق الإحصائيات على قرار واحد (بعضها يرفض وبعضها لا يرفض). النتيجة "
            "غير حاسمة عند هذا المستوى؛ يُنصح بإعادة النظر في رتبة الإبطاء أو الحالة "
            "أو حجم العينة."
        )

    return Decision(
        code=code, level=level, label_ar=label, detail_ar=detail,
        f_overall_reject=f_rej, t_reject=t_rej, f_ind_reject=fi_rej,
        degenerate=degenerate, per_level=per_level,
    )
