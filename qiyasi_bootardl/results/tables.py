"""Tidy Arabic-labelled result tables (pandas DataFrames).

جداول النتائج بعناوين عربية: المعاملات، الإحصائيات، القيم الحرجة بالبوتستراب،
حدود PSS، وحدود SMG.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..core.ardl import ARDLFit
from ..core.statistics import TestStatistics
from ..core.bootstrap import BootstrapResult
from ..core.bounds import BoundsResult, SMGResult
from ..utils.arabic_text import STAT_LABELS_AR
from ..utils.formatting import pct


def coefficients_table(fit: ARDLFit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "المعامل": fit.params.values,
            "الخطأ_المعياري": fit.bse.values,
            "إحصائية_t": (fit.params / fit.bse).values,
            "القيمة_الاحتمالية": fit.pvalues.values,
        },
        index=fit.params.index,
    )


def statistics_table(stats_obj: TestStatistics) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "الإحصائية": [stats_obj.f_overall, stats_obj.t_dep, stats_obj.f_ind],
            "القيمة_الاحتمالية_التقاربية": [
                stats_obj.f_overall_p, stats_obj.t_dep_p, stats_obj.f_ind_p,
            ],
        },
        index=[STAT_LABELS_AR["f_overall"], STAT_LABELS_AR["t_dep"], STAT_LABELS_AR["f_ind"]],
    )


def bootstrap_table(boot: BootstrapResult) -> pd.DataFrame:
    """Bootstrap critical values per significance level for the three statistics."""
    rows = []
    index = []
    for a in boot.levels:
        index.append(pct(a))
        rows.append(
            [
                boot.crit_f_overall.get(a, np.nan),
                boot.crit_t_dep.get(a, np.nan),
                boot.crit_f_ind.get(a, np.nan),
                boot.crit_f_ind_uncond.get(a, np.nan),
            ]
        )
    return pd.DataFrame(
        rows,
        index=index,
        columns=[
            "القيمة_الحرجة_F_الكلية",
            "القيمة_الحرجة_t",
            "القيمة_الحرجة_F_المستقلة_مشروط",
            "القيمة_الحرجة_F_المستقلة_غير_مشروط",
        ],
    )


def pvalues_table(boot: BootstrapResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "القيمة_الاحتمالية_بالبوتستراب": [
                boot.pvalue_f_overall, boot.pvalue_t_dep, boot.pvalue_f_ind,
            ],
        },
        index=[STAT_LABELS_AR["f_overall"], STAT_LABELS_AR["t_dep"], STAT_LABELS_AR["f_ind"]],
    )


def pss_table(bounds: BoundsResult) -> pd.DataFrame:
    rows = []
    index = []
    for a, (lo, hi) in sorted(bounds.f_bounds.items()):
        index.append(pct(a))
        trow = bounds.t_bounds.get(a) if bounds.t_bounds else None
        rows.append([lo, hi, trow[0] if trow else np.nan, trow[1] if trow else np.nan])
    return pd.DataFrame(
        rows, index=index,
        columns=["F_الحد_الأدنى_I0", "F_الحد_الأعلى_I1", "t_الحد_الأدنى_I0", "t_الحد_الأعلى_I1"],
    )


def smg_table(smg: SMGResult) -> pd.DataFrame:
    if not smg.available or not smg.bounds:
        return pd.DataFrame(columns=["F_الحد_الأدنى_I0", "F_الحد_الأعلى_I1"])
    rows = []
    index = []
    for a, (lo, hi) in sorted(smg.bounds.items()):
        index.append(pct(a))
        rows.append([lo, hi])
    return pd.DataFrame(rows, index=index, columns=["F_الحد_الأدنى_I0", "F_الحد_الأعلى_I1"])
