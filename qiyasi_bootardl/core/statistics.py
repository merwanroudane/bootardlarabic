"""Test statistics for the conditional ARDL bounds test.

إحصائيات الاختبار: F الكلية، t للمتغير التابع المتأخر، F للمتغيرات المستقلة.

All three follow Pesaran-Shin-Smith (2001): they are Wald tests on the lagged
*level* coefficients of the ARDL-ECM. The reported F-form equals the Wald
chi-square divided by the number of restrictions (matching bootCT / dynamac).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .ardl import ARDLFit
from .deterministic_cases import get_case_spec


@dataclass
class TestStatistics:
    f_overall: float
    t_dep: float
    f_ind: float
    f_overall_p: float
    t_dep_p: float
    f_ind_p: float
    f_overall_q: int   # restrictions in F-overall
    f_ind_q: int       # restrictions in F-independent


def _wald(params: pd.Series, cov: pd.DataFrame, names: Sequence[str]) -> Tuple[float, int]:
    """Return (Wald chi-square statistic, number of restrictions)."""
    names = [n for n in names if n in params.index]
    q = len(names)
    if q == 0:
        return float("nan"), 0
    b = params.loc[names].values.astype(float)
    V = cov.loc[names, names].values.astype(float)
    try:
        w = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        w = float(b @ np.linalg.pinv(V) @ b)
    return w, q


def f_overall_terms(fit: ARDLFit, case: int) -> List[str]:
    spec = get_case_spec(case)
    terms = list(fit.info.level_names)
    if spec.intercept_restricted and fit.info.const:
        terms = [fit.info.const] + terms
    if spec.trend_restricted and fit.info.trend:
        terms = terms + [fit.info.trend]
    return terms


def compute_statistics(fit: ARDLFit, case: int) -> TestStatistics:
    """Compute F-overall, t (on lagged y level) and F-independent."""
    # F-overall
    fo_terms = f_overall_terms(fit, case)
    w_fo, q_fo = _wald(fit.params, fit.cov, fo_terms)
    f_overall = w_fo / q_fo if q_fo else float("nan")
    p_fo = float(stats.chi2.sf(w_fo, q_fo)) if q_fo else float("nan")

    # t-statistic on the lagged dependent level
    yl = fit.info.y_level
    t_dep = float(fit.params[yl] / fit.bse[yl])
    p_t = float(2 * stats.norm.sf(abs(t_dep)))

    # F-independent (joint test on lagged X levels)
    w_fi, q_fi = _wald(fit.params, fit.cov, fit.info.x_levels)
    f_ind = w_fi / q_fi if q_fi else float("nan")
    p_fi = float(stats.chi2.sf(w_fi, q_fi)) if q_fi else float("nan")

    return TestStatistics(
        f_overall=f_overall,
        t_dep=t_dep,
        f_ind=f_ind,
        f_overall_p=p_fo,
        t_dep_p=p_t,
        f_ind_p=p_fi,
        f_overall_q=q_fo,
        f_ind_q=q_fi,
    )


def statistics_frame(stats_obj: TestStatistics) -> pd.DataFrame:
    """Tidy statistics table with Arabic row labels."""
    from ..utils.arabic_text import STAT_LABELS_AR

    return pd.DataFrame(
        {
            "الإحصائية": [
                stats_obj.f_overall,
                stats_obj.t_dep,
                stats_obj.f_ind,
            ],
            "القيمة_الاحتمالية_التقاربية": [
                stats_obj.f_overall_p,
                stats_obj.t_dep_p,
                stats_obj.f_ind_p,
            ],
        },
        index=[
            STAT_LABELS_AR["f_overall"],
            STAT_LABELS_AR["t_dep"],
            STAT_LABELS_AR["f_ind"],
        ],
    )
