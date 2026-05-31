"""Arabic methodology advisor.

المستشار المنهجي العربي: يرصد المخاطر المنهجية الشائعة في تطبيق اختبار ARDL
بالبوتستراب ويصدر تحذيرات قابلة للتنفيذ (لا مجرد رسائل خطأ).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.bootstrap import BootstrapResult
from ..core.diagnostics import DiagnosticResult


SEVERITY_ORDER = {"خطأ": 0, "تحذير": 1, "ملاحظة": 2}


@dataclass
class MethodologyWarning:
    severity: str   # "خطأ" | "تحذير" | "ملاحظة"
    code: str
    message_ar: str
    advice_ar: str


class ArabicMethodologyAdvisor:
    """يجمع التحذيرات المنهجية بناءً على المواصفات ونتائج التقدير."""

    def __init__(self):
        self._warnings: List[MethodologyWarning] = []

    def _add(self, severity: str, code: str, message: str, advice: str) -> None:
        self._warnings.append(MethodologyWarning(severity, code, message, advice))

    def review(
        self,
        n_obs: int,
        n_boot: int,
        max_lag: int,
        diff_lags: List[int],
        case: int,
        boot: BootstrapResult,
        decision_code: str,
        diagnostics: Optional[Dict[str, DiagnosticResult]] = None,
        smg_available: bool = True,
    ) -> List[MethodologyWarning]:
        self._warnings = []

        # --- bootstrap replications ---
        if n_boot < 1000:
            self._add(
                "تحذير", "low_nboot",
                f"عدد تكرارات البوتستراب ({n_boot}) منخفض.",
                "يُنصح باستخدام 2000 تكرار على الأقل (ويفضّل 5000) لقيم حرجة مستقرة.",
            )

        # --- effective draws lost to singular fits ---
        if boot.n_boot_effective < 0.9 * boot.n_boot:
            lost = boot.n_boot - boot.n_boot_effective
            self._add(
                "تحذير", "lost_draws",
                f"فُقد {lost} من تكرارات البوتستراب بسبب تفرّد المصفوفة.",
                "قد يدل ذلك على ضعف تحديد النموذج أو تعدّد خطّي؛ راجع رتب الإبطاء.",
            )

        # --- sample size ---
        if n_obs < 30:
            self._add(
                "تحذير", "small_sample",
                f"حجم العينة ({n_obs}) صغير.",
                "اختبار الحدود حساس لصغر العينة؛ فسّر النتائج بحذر واعتمد على "
                "القيم الحرجة بالبوتستراب لا التقاربية.",
            )

        # --- lag length relative to sample ---
        if max_lag >= 1 and n_obs > 0 and (max_lag / n_obs) > 0.2:
            self._add(
                "ملاحظة", "high_maxlag",
                f"أقصى إبطاء ({max_lag}) كبير نسبياً مقارنة بحجم العينة ({n_obs}).",
                "إبطاء مرتفع يستهلك درجات حرية ويضعف قوة الاختبار.",
            )

        # --- degenerate / inconclusive decision ---
        if decision_code == "تكامل_زائف_متدهور":
            self._add(
                "تحذير", "degenerate",
                "رُصدت حالة تكامل مشترك زائف/متدهور.",
                "تأكّد من أن المتغيرات المستقلة خارجية ضعيفة، وافحص اتجاه السببية "
                "وإمكانية عكس دور المتغير التابع.",
            )
        elif decision_code == "غير_حاسم":
            self._add(
                "ملاحظة", "inconclusive",
                "النتيجة غير حاسمة (الإحصائيات لا تتفق على قرار واحد).",
                "جرّب معياراً مختلفاً لاختيار الإبطاء أو حالة حتمية أخرى، أو وسّع العينة.",
            )

        # --- SMG availability ---
        if not smg_available and case in (2, 4):
            self._add(
                "ملاحظة", "smg_case",
                f"اختبار SMG للمتغيرات المستقلة غير متاح للحالة {case}.",
                "اختبار SMG معرّف للحالات 1 و3 و5 فقط؛ اعتمد على F بالبوتستراب.",
            )

        # --- diagnostics ---
        if diagnostics:
            sc = diagnostics.get("serial_correlation")
            if sc and sc.ok is False:
                self._add(
                    "تحذير", "serial_corr",
                    f"رُصد ارتباط ذاتي في البواقي (p={sc.pvalue:.3f}).",
                    "زِد رتبة الإبطاء للتخلص من الارتباط الذاتي قبل الاعتماد على النتائج.",
                )
            nm = diagnostics.get("normality")
            if nm and nm.ok is False:
                self._add(
                    "ملاحظة", "non_normal",
                    f"بواقي غير موزّعة طبيعياً (p={nm.pvalue:.3f}).",
                    "البوتستراب أكثر متانة من القيم التقاربية في هذه الحالة، لكن افحص "
                    "القيم الشاذة والكسور الهيكلية.",
                )
            ht = diagnostics.get("heteroskedasticity")
            if ht and ht.ok is False:
                self._add(
                    "ملاحظة", "heterosk",
                    f"رُصد عدم ثبات في تباين البواقي (p={ht.pvalue:.3f}).",
                    "فكّر في بوتستراب متماثل التباين (wild bootstrap) كامتداد مستقبلي.",
                )

        # --- I(2) caution (always a methodological reminder) ---
        self._add(
            "ملاحظة", "i2_caution",
            "اختبار الحدود يفترض أن المتغيرات متكاملة من رتبة لا تتجاوز I(1).",
            "تحقّق من عدم وجود متغيرات I(2) باختبارات جذر الوحدة قبل الاعتماد على النتائج.",
        )

        self._warnings.sort(key=lambda w: SEVERITY_ORDER.get(w.severity, 99))
        return self._warnings
