"""Arabic-first result object for the bootstrap ARDL bounds test.

كائن النتائج العربي: يحمل كل مخرجات الاختبار ويوفّر دوالّ عربية للملخص
والتفسير والقرار والتحذيرات والجداول والتقارير.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..core.ardl import ARDLFit
from ..core.statistics import TestStatistics
from ..core.bootstrap import BootstrapResult
from ..core.bounds import BoundsResult, SMGResult
from ..core.diagnostics import DiagnosticResult
from ..utils.arabic_text import (
    CASE_NAMES_AR,
    STAT_LABELS_AR,
    TERMS_AR,
    IC_NAMES_AR,
    REFERENCES,
)
from ..utils.formatting import fmt, pct, center_block, rule
from .decisions import Decision, decide
from .interpretation import interpret
from .warnings import ArabicMethodologyAdvisor, MethodologyWarning
from . import tables


@dataclass
class ArabicBootARDLResult:
    """نتيجة اختبار ARDL بالبوتستراب — الواجهة العربية الأساسية."""

    yvar: str
    xvar: List[str]
    case: int
    k: int
    n_obs: int
    diff_lags: List[int]
    ardl_ic: str
    max_lag: int
    n_boot: int
    levels: List[float]
    decision_level: float

    cond_fit: ARDLFit
    uncond_fit: ARDLFit
    statistics: TestStatistics
    f_ind_uncond: float
    bounds: BoundsResult
    smg: SMGResult
    bootstrap: BootstrapResult
    diagnostics: Dict[str, DiagnosticResult]

    source_data: Optional[pd.DataFrame] = None

    decision: Decision = field(init=False)
    warnings_list: List[MethodologyWarning] = field(init=False)

    def __post_init__(self) -> None:
        self.decision = decide(
            self.bootstrap,
            self.statistics.f_overall,
            self.statistics.t_dep,
            self.statistics.f_ind,
            self.case,
            level=self.decision_level,
        )
        advisor = ArabicMethodologyAdvisor()
        self.warnings_list = advisor.review(
            n_obs=self.n_obs,
            n_boot=self.n_boot,
            max_lag=self.max_lag,
            diff_lags=self.diff_lags,
            case=self.case,
            boot=self.bootstrap,
            decision_code=self.decision.code,
            diagnostics=self.diagnostics,
            smg_available=self.smg.available,
        )

    # ------------------------------------------------------------------ tables
    def جدول_المعاملات(self, مشروط: bool = True) -> pd.DataFrame:
        """جدول معاملات النموذج (المشروط افتراضياً)."""
        return tables.coefficients_table(self.cond_fit if مشروط else self.uncond_fit)

    def جدول_الإحصائيات(self) -> pd.DataFrame:
        """جدول إحصائيات الاختبار الثلاث وقيمها الاحتمالية التقاربية."""
        return tables.statistics_table(self.statistics)

    def جدول_البوتستراب(self) -> pd.DataFrame:
        """القيم الحرجة بالبوتستراب لكل مستوى معنوية."""
        return tables.bootstrap_table(self.bootstrap)

    def جدول_القيم_الاحتمالية(self) -> pd.DataFrame:
        """القيم الاحتمالية بالبوتستراب."""
        return tables.pvalues_table(self.bootstrap)

    def جدول_حدود_PSS(self) -> pd.DataFrame:
        return tables.pss_table(self.bounds)

    def جدول_حدود_SMG(self) -> pd.DataFrame:
        return tables.smg_table(self.smg)

    # --------------------------------------------------------------- decisions
    def قرار(self, مستوى: Optional[float] = None) -> Decision:
        """القرار النهائي حول التكامل المشترك عند مستوى معنوية معيّن."""
        if مستوى is None or مستوى == self.decision_level:
            return self.decision
        return decide(
            self.bootstrap,
            self.statistics.f_overall,
            self.statistics.t_dep,
            self.statistics.f_ind,
            self.case,
            level=مستوى,
        )

    def تفسير(self) -> str:
        """فقرة تفسيرية اقتصادية ومنهجية للنتيجة."""
        return interpret(
            self.decision,
            self.yvar,
            self.xvar,
            self.statistics.f_overall,
            self.statistics.t_dep,
            self.statistics.f_ind,
        )

    def تحذيرات(self) -> List[MethodologyWarning]:
        """قائمة التحذيرات المنهجية مرتّبة حسب الخطورة."""
        return self.warnings_list

    # ----------------------------------------------------------------- summary
    def ملخص(self) -> str:
        """نص ملخّص عربي شامل (يُطبع عبر print)."""
        L: List[str] = []
        L.append(center_block(TERMS_AR["test_name"]))
        L.append(f"{TERMS_AR['dependent']}: {self.yvar}")
        L.append(f"{TERMS_AR['independent']}: {'، '.join(self.xvar)}")
        L.append(f"{TERMS_AR['selected_case']}: {CASE_NAMES_AR.get(self.case, self.case)}")
        L.append(f"{TERMS_AR['n_obs']}: {self.n_obs}")
        L.append(f"{TERMS_AR['ardl_ic']}: {IC_NAMES_AR.get(self.ardl_ic, self.ardl_ic)}")
        L.append(f"{TERMS_AR['selected_lags']}: {self.diff_lags}")
        L.append(f"{TERMS_AR['n_boot']}: {self.n_boot} (فعّالة: {self.bootstrap.n_boot_effective})")
        L.append(rule())

        L.append(TERMS_AR["test_statistics"] + ":")
        L.append(f"  {STAT_LABELS_AR['f_overall']} = {fmt(self.statistics.f_overall)}")
        L.append(f"  {STAT_LABELS_AR['t_dep']} = {fmt(self.statistics.t_dep)}")
        L.append(f"  {STAT_LABELS_AR['f_ind']} = {fmt(self.statistics.f_ind)} "
                 f"(غير مشروط = {fmt(self.f_ind_uncond)})")
        L.append(rule())

        L.append(f"{TERMS_AR['boot_pval']}:")
        L.append(f"  F الكلية: {fmt(self.bootstrap.pvalue_f_overall)}")
        L.append(f"  t: {fmt(self.bootstrap.pvalue_t_dep)}")
        L.append(f"  F المستقلة: {fmt(self.bootstrap.pvalue_f_ind)}")
        L.append(rule())

        L.append(f"{TERMS_AR['boot_crit']} (عند {pct(self.decision_level)}):")
        a = self.decision_level
        L.append(f"  F الكلية: {fmt(self.bootstrap.crit_f_overall.get(a))}")
        L.append(f"  t: {fmt(self.bootstrap.crit_t_dep.get(a))}")
        L.append(f"  F المستقلة: {fmt(self.bootstrap.crit_f_ind.get(a))}")
        L.append(rule())

        L.append(f"{TERMS_AR['final_decision']} (عند {pct(self.decision_level)}): "
                 f"{self.decision.label_ar}")
        L.append(self.decision.detail_ar)
        L.append(rule())

        if self.warnings_list:
            L.append(TERMS_AR["warnings"] + ":")
            for w in self.warnings_list:
                L.append(f"  [{w.severity}] {w.message_ar}")
                L.append(f"        ← {w.advice_ar}")
        return "\n".join(L)

    # ------------------------------------------------------------------ reports
    def تقرير_HTML(self, مسار: Optional[str] = None) -> str:
        from ..reports.html_report import build_html_report
        return build_html_report(self, مسار)

    def تقرير_Word(self, مسار: str) -> str:
        from ..reports.word_report import build_word_report
        return build_word_report(self, مسار)

    def تقرير_Excel(self, مسار: str) -> str:
        from ..reports.excel_report import build_excel_report
        return build_excel_report(self, مسار)

    # ----------------------------------------------------------------- plotting
    def رسم_توزيعات_البوتستراب(self, مسار: Optional[str] = None):
        """مدرّجات التوزيعات الصفرية الثلاثة مع الإحصائية المحسوبة والقيم الحرجة."""
        from ..reports.visualization import plot_bootstrap_distributions
        return plot_bootstrap_distributions(self, مسار)

    def رسم_القرار(self, مسار: Optional[str] = None):
        """أعمدة تقارن الإحصائيات المحسوبة بالقيم الحرجة بالبوتستراب عند مستوى القرار."""
        from ..reports.visualization import plot_decision
        return plot_decision(self, مسار)

    def رسم_السلاسل_الزمنية(self, مسار: Optional[str] = None):
        """رسم السلاسل الزمنية للمتغيرات (يتطلب توفّر البيانات الأصلية)."""
        from ..reports.visualization import plot_series
        return plot_series(self, مسار)

    def رسم_التكامل_الزائف(self, مسار: Optional[str] = None):
        """مقارنة F للمتغيرات المستقلة بين النموذج المشروط وغير المشروط."""
        from ..reports.visualization import plot_fake_cointegration
        return plot_fake_cointegration(self, مسار)

    def حفظ_كل_الرسوم(self, مجلد: str = ".", بادئة: str = "رسم") -> List[str]:
        """حفظ جميع الرسوم في مجلد وإرجاع مسارات الملفات."""
        import os

        os.makedirs(مجلد, exist_ok=True)
        outputs = []
        specs = [
            (self.رسم_توزيعات_البوتستراب, "توزيعات"),
            (self.رسم_القرار, "القرار"),
            (self.رسم_التكامل_الزائف, "تكامل_زائف"),
        ]
        if self.source_data is not None:
            specs.append((self.رسم_السلاسل_الزمنية, "سلاسل"))
        for func, name in specs:
            p = os.path.join(مجلد, f"{بادئة}_{name}.png")
            func(p)
            outputs.append(p)
        return outputs

    @staticmethod
    def المراجع() -> List[str]:
        return list(REFERENCES)

    # ------------------------------------------------------------------ dunder
    def __str__(self) -> str:
        return self.ملخص()

    def __repr__(self) -> str:
        return (f"<ArabicBootARDLResult: {self.yvar} ~ {', '.join(self.xvar)} | "
                f"الحالة {self.case} | {self.decision.label_ar}>")
