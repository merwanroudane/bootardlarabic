"""English-facing thin wrapper over :class:`ArabicBootARDLResult`.

Exposes English-named accessors that delegate to the same underlying Arabic
result object. Computation is shared; only the presentation labels differ.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..utils.arabic_text import CASE_NAMES_EN, STAT_LABELS_EN
from ..utils.formatting import fmt, pct
from .arabic_result import ArabicBootARDLResult
from .decisions import (
    Decision,
    COINTEGRATION,
    NO_COINTEGRATION,
    INCONCLUSIVE,
    DEGENERATE,
)

_DECISION_EN = {
    COINTEGRATION: "Cointegration",
    NO_COINTEGRATION: "No cointegration",
    INCONCLUSIVE: "Inconclusive",
    DEGENERATE: "Degenerate / fake cointegration",
}


class EnglishBootARDLResult:
    """English presentation layer wrapping a shared ArabicBootARDLResult."""

    def __init__(self, ar: ArabicBootARDLResult):
        self._ar = ar

    @property
    def arabic(self) -> ArabicBootARDLResult:
        return self._ar

    def decision(self, level: Optional[float] = None) -> Decision:
        return self._ar.قرار(level)

    def decision_label(self, level: Optional[float] = None) -> str:
        return _DECISION_EN.get(self.decision(level).code, "Unknown")

    def coefficients(self, conditional: bool = True) -> pd.DataFrame:
        return self._ar.جدول_المعاملات(مشروط=conditional)

    def statistics_table(self) -> pd.DataFrame:
        return self._ar.جدول_الإحصائيات()

    def bootstrap_table(self) -> pd.DataFrame:
        return self._ar.جدول_البوتستراب()

    def pvalues(self) -> pd.DataFrame:
        return self._ar.جدول_القيم_الاحتمالية()

    def pss_bounds(self) -> pd.DataFrame:
        return self._ar.جدول_حدود_PSS()

    def smg_bounds(self) -> pd.DataFrame:
        return self._ar.جدول_حدود_SMG()

    def warnings(self) -> List[str]:
        return [f"[{w.severity}] {w.message_ar}" for w in self._ar.تحذيرات()]

    # --- plots (Arabic-labelled, shared with the Arabic result) ---
    def plot_distributions(self, path: Optional[str] = None):
        return self._ar.رسم_توزيعات_البوتستراب(path)

    def plot_decision(self, path: Optional[str] = None):
        return self._ar.رسم_القرار(path)

    def plot_series(self, path: Optional[str] = None):
        return self._ar.رسم_السلاسل_الزمنية(path)

    def plot_fake_cointegration(self, path: Optional[str] = None):
        return self._ar.رسم_التكامل_الزائف(path)

    def save_all_plots(self, folder: str = ".", prefix: str = "plot") -> List[str]:
        return self._ar.حفظ_كل_الرسوم(folder, prefix)

    def summary(self) -> str:
        s = self._ar
        st = s.statistics
        lvl = s.decision_level
        lines = [
            "Bootstrap ARDL Bounds Test for Cointegration",
            "=" * 52,
            f"Dependent: {s.yvar}",
            f"Independent: {', '.join(s.xvar)}",
            f"Case: {CASE_NAMES_EN.get(s.case, s.case)}",
            f"Observations: {s.n_obs}",
            f"Selected lags: {s.diff_lags}  (IC: {s.ardl_ic})",
            f"Bootstrap reps: {s.n_boot} (effective: {s.bootstrap.n_boot_effective})",
            "-" * 52,
            f"{STAT_LABELS_EN['f_overall']} = {fmt(st.f_overall)} "
            f"(boot p = {fmt(s.bootstrap.pvalue_f_overall)})",
            f"{STAT_LABELS_EN['t_dep']} = {fmt(st.t_dep)} "
            f"(boot p = {fmt(s.bootstrap.pvalue_t_dep)})",
            f"{STAT_LABELS_EN['f_ind']} = {fmt(st.f_ind)} "
            f"(boot p = {fmt(s.bootstrap.pvalue_f_ind)})",
            "-" * 52,
            f"Decision @ {pct(lvl)}: {self.decision_label()}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<EnglishBootARDLResult: {self._ar.yvar} ~ "
                f"{', '.join(self._ar.xvar)} | case {self._ar.case} | "
                f"{self.decision_label()}>")

    def __str__(self) -> str:
        return self.summary()
