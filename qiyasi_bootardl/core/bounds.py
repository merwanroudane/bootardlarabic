"""Asymptotic bounds: PSS (2001) F/t bounds and SMG (2019) F-independent bounds.

اختبار الحدود PSS واختبار SMG على المتغيرات المستقلة.

NOTE / ملاحظة منهجية:
The PSS critical values embedded here are the *asymptotic* (large-sample) bounds
from Pesaran, Shin & Smith (2001), Tables CI(i)-CI(v) and CII(i)-CII(v). They are
supplied as a convenience wrapper. The package's primary decision basis is the
*bootstrap* critical values, which are exact for the sample at hand. For
publication-grade work, verify these tables against the source and consider the
finite-sample (Narayan, 2005) bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from . import _smk_data as _smk

# ---------------------------------------------------------------------------
# PSS (2001) asymptotic critical value bounds.
# Layout: PSS_F[case][k] = {alpha: (I0_lower, I1_upper)}; k = number of
# independent regressors (= d - 1). Significance alphas: .10 .05 .025 .01.
# ---------------------------------------------------------------------------
_A = (0.10, 0.05, 0.025, 0.01)

PSS_F: Dict[int, Dict[int, Dict[float, Tuple[float, float]]]] = {
    1: {  # Case I: no intercept, no trend
        1: {0.10: (2.44, 3.28), 0.05: (3.15, 4.11), 0.025: (3.88, 4.92), 0.01: (4.81, 6.02)},
        2: {0.10: (2.17, 3.19), 0.05: (2.72, 3.83), 0.025: (3.22, 4.50), 0.01: (3.88, 5.30)},
        3: {0.10: (2.01, 3.10), 0.05: (2.45, 3.63), 0.025: (2.87, 4.16), 0.01: (3.42, 4.84)},
        4: {0.10: (1.90, 3.01), 0.05: (2.26, 3.48), 0.025: (2.62, 3.90), 0.01: (3.07, 4.44)},
        5: {0.10: (1.81, 2.93), 0.05: (2.14, 3.34), 0.025: (2.44, 3.71), 0.01: (2.82, 4.21)},
        6: {0.10: (1.75, 2.87), 0.05: (2.04, 3.24), 0.025: (2.32, 3.59), 0.01: (2.66, 4.05)},
        7: {0.10: (1.70, 2.83), 0.05: (1.98, 3.18), 0.025: (2.22, 3.49), 0.01: (2.54, 3.91)},
    },
    2: {  # Case II: restricted intercept, no trend
        1: {0.10: (3.02, 3.51), 0.05: (3.62, 4.16), 0.025: (4.18, 4.79), 0.01: (4.94, 5.58)},
        2: {0.10: (2.63, 3.35), 0.05: (3.10, 3.87), 0.025: (3.55, 4.38), 0.01: (4.13, 5.00)},
        3: {0.10: (2.37, 3.20), 0.05: (2.79, 3.67), 0.025: (3.15, 4.08), 0.01: (3.65, 4.66)},
        4: {0.10: (2.20, 3.09), 0.05: (2.56, 3.49), 0.025: (2.88, 3.87), 0.01: (3.29, 4.37)},
        5: {0.10: (2.08, 3.00), 0.05: (2.39, 3.38), 0.025: (2.70, 3.73), 0.01: (3.06, 4.15)},
        6: {0.10: (1.99, 2.94), 0.05: (2.27, 3.28), 0.025: (2.55, 3.61), 0.01: (2.88, 3.99)},
        7: {0.10: (1.92, 2.89), 0.05: (2.17, 3.21), 0.025: (2.43, 3.51), 0.01: (2.73, 3.90)},
    },
    3: {  # Case III: unrestricted intercept, no trend
        1: {0.10: (3.02, 3.51), 0.05: (3.62, 4.16), 0.025: (4.18, 4.79), 0.01: (4.94, 5.58)},
        2: {0.10: (2.63, 3.35), 0.05: (3.10, 3.87), 0.025: (3.55, 4.38), 0.01: (4.13, 5.00)},
        3: {0.10: (2.37, 3.20), 0.05: (2.79, 3.67), 0.025: (3.15, 4.08), 0.01: (3.65, 4.66)},
        4: {0.10: (2.20, 3.09), 0.05: (2.56, 3.49), 0.025: (2.88, 3.87), 0.01: (3.29, 4.37)},
        5: {0.10: (2.08, 3.00), 0.05: (2.39, 3.38), 0.025: (2.70, 3.73), 0.01: (3.06, 4.15)},
        6: {0.10: (1.99, 2.94), 0.05: (2.27, 3.28), 0.025: (2.55, 3.61), 0.01: (2.88, 3.99)},
        7: {0.10: (1.92, 2.89), 0.05: (2.17, 3.21), 0.025: (2.43, 3.51), 0.01: (2.73, 3.90)},
    },
    4: {  # Case IV: unrestricted intercept, restricted trend
        1: {0.10: (4.05, 4.49), 0.05: (4.68, 5.15), 0.025: (5.30, 5.83), 0.01: (6.10, 6.73)},
        2: {0.10: (3.38, 4.02), 0.05: (3.88, 4.61), 0.025: (4.37, 5.16), 0.01: (5.00, 5.78)},
        3: {0.10: (2.97, 3.74), 0.05: (3.38, 4.23), 0.025: (3.80, 4.68), 0.01: (4.30, 5.23)},
        4: {0.10: (2.68, 3.53), 0.05: (3.05, 3.97), 0.025: (3.40, 4.36), 0.01: (3.81, 4.92)},
        5: {0.10: (2.49, 3.38), 0.05: (2.81, 3.76), 0.025: (3.11, 4.13), 0.01: (3.50, 4.63)},
        6: {0.10: (2.33, 3.25), 0.05: (2.62, 3.61), 0.025: (2.89, 3.95), 0.01: (3.25, 4.38)},
        7: {0.10: (2.22, 3.17), 0.05: (2.48, 3.50), 0.025: (2.73, 3.82), 0.01: (3.05, 4.24)},
    },
    5: {  # Case V: unrestricted intercept and trend
        1: {0.10: (4.05, 4.49), 0.05: (4.68, 5.15), 0.025: (5.30, 5.83), 0.01: (6.10, 6.73)},
        2: {0.10: (3.38, 4.02), 0.05: (3.88, 4.61), 0.025: (4.37, 5.16), 0.01: (5.00, 5.78)},
        3: {0.10: (2.97, 3.74), 0.05: (3.38, 4.23), 0.025: (3.80, 4.68), 0.01: (4.30, 5.23)},
        4: {0.10: (2.68, 3.53), 0.05: (3.05, 3.97), 0.025: (3.40, 4.36), 0.01: (3.81, 4.92)},
        5: {0.10: (2.49, 3.38), 0.05: (2.81, 3.76), 0.025: (3.11, 4.13), 0.01: (3.50, 4.63)},
        6: {0.10: (2.33, 3.25), 0.05: (2.62, 3.61), 0.025: (2.89, 3.95), 0.01: (3.25, 4.38)},
        7: {0.10: (2.22, 3.17), 0.05: (2.48, 3.50), 0.025: (2.73, 3.82), 0.01: (3.05, 4.24)},
    },
}

# PSS (2001) t-bounds (Tables CII). The I(0) bound is constant in k (DF case);
# the I(1) bound grows with k. Provided for cases I, III, V only.
PSS_T: Dict[int, Dict[int, Dict[float, Tuple[float, float]]]] = {
    1: {  # Case I
        1: {0.10: (-1.62, -1.95), 0.05: (-1.95, -2.28), 0.025: (-2.24, -2.58), 0.01: (-2.58, -2.93)},
        2: {0.10: (-1.62, -2.30), 0.05: (-1.95, -2.60), 0.025: (-2.24, -2.90), 0.01: (-2.58, -3.22)},
        3: {0.10: (-1.62, -2.59), 0.05: (-1.95, -2.91), 0.025: (-2.24, -3.19), 0.01: (-2.58, -3.49)},
        4: {0.10: (-1.62, -2.85), 0.05: (-1.95, -3.16), 0.025: (-2.24, -3.44), 0.01: (-2.58, -3.74)},
        5: {0.10: (-1.62, -3.09), 0.05: (-1.95, -3.41), 0.025: (-2.24, -3.66), 0.01: (-2.58, -3.96)},
        6: {0.10: (-1.62, -3.31), 0.05: (-1.95, -3.61), 0.025: (-2.24, -3.88), 0.01: (-2.58, -4.18)},
        7: {0.10: (-1.62, -3.51), 0.05: (-1.95, -3.83), 0.025: (-2.24, -4.10), 0.01: (-2.58, -4.37)},
    },
    3: {  # Case III
        1: {0.10: (-2.57, -2.91), 0.05: (-2.86, -3.22), 0.025: (-3.13, -3.50), 0.01: (-3.43, -3.82)},
        2: {0.10: (-2.57, -3.21), 0.05: (-2.86, -3.53), 0.025: (-3.13, -3.80), 0.01: (-3.43, -4.10)},
        3: {0.10: (-2.57, -3.46), 0.05: (-2.86, -3.78), 0.025: (-3.13, -4.05), 0.01: (-3.43, -4.37)},
        4: {0.10: (-2.57, -3.66), 0.05: (-2.86, -3.99), 0.025: (-3.13, -4.26), 0.01: (-3.43, -4.60)},
        5: {0.10: (-2.57, -3.86), 0.05: (-2.86, -4.19), 0.025: (-3.13, -4.46), 0.01: (-3.43, -4.79)},
        6: {0.10: (-2.57, -4.04), 0.05: (-2.86, -4.38), 0.025: (-3.13, -4.66), 0.01: (-3.43, -4.99)},
        7: {0.10: (-2.57, -4.23), 0.05: (-2.86, -4.57), 0.025: (-3.13, -4.85), 0.01: (-3.43, -5.19)},
    },
    5: {  # Case V
        1: {0.10: (-3.13, -3.40), 0.05: (-3.41, -3.69), 0.025: (-3.65, -3.96), 0.01: (-3.96, -4.26)},
        2: {0.10: (-3.13, -3.63), 0.05: (-3.41, -3.95), 0.025: (-3.65, -4.20), 0.01: (-3.96, -4.53)},
        3: {0.10: (-3.13, -3.84), 0.05: (-3.41, -4.16), 0.025: (-3.65, -4.42), 0.01: (-3.96, -4.73)},
        4: {0.10: (-3.13, -4.04), 0.05: (-3.41, -4.36), 0.025: (-3.65, -4.62), 0.01: (-3.96, -4.96)},
        5: {0.10: (-3.13, -4.21), 0.05: (-3.41, -4.52), 0.025: (-3.65, -4.79), 0.01: (-3.96, -5.13)},
        6: {0.10: (-3.13, -4.37), 0.05: (-3.41, -4.69), 0.025: (-3.65, -4.96), 0.01: (-3.96, -5.31)},
        7: {0.10: (-3.13, -4.53), 0.05: (-3.41, -4.85), 0.025: (-3.65, -5.12), 0.01: (-3.96, -5.47)},
    },
}

PSS_NOTE = (
    "القيم الحرجة لاختبار PSS هي قيم تقاربية (Asymptotic) من Pesaran, Shin & Smith "
    "(2001). يُنصح بالاعتماد على القيم الحرجة بالبوتستراب في القرار النهائي."
)


@dataclass
class BoundsResult:
    fstat: float
    tstat: Optional[float]
    case: int
    k: int
    f_bounds: Dict[float, Tuple[float, float]]
    t_bounds: Optional[Dict[float, Tuple[float, float]]]
    note: str = PSS_NOTE


def _bounds_zone(stat: float, lower: float, upper: float, is_t: bool) -> str:
    """Return the Arabic decision zone for a statistic against (lower, upper)."""
    if stat is None or np.isnan(stat):
        return "غير متاح"
    if is_t:
        # t-bounds are negative; rejection (cointegration) when t < I(1) lower.
        if stat < upper:
            return "رفض الفرضية الصفرية (تكامل مشترك)"
        if stat > lower:
            return "عدم الرفض (لا يوجد تكامل مشترك)"
        return "غير حاسم"
    if stat > upper:
        return "رفض الفرضية الصفرية (تكامل مشترك)"
    if stat < lower:
        return "عدم الرفض (لا يوجد تكامل مشترك)"
    return "غير حاسم"


def pss_bounds(fstat: float, tstat: Optional[float], case: int, k: int) -> BoundsResult:
    """Look up PSS asymptotic F/t bounds for the given case and k = d - 1."""
    f_bounds = PSS_F.get(case, {}).get(k, {})
    t_bounds = PSS_T.get(case, {}).get(k)
    return BoundsResult(
        fstat=fstat,
        tstat=tstat,
        case=case,
        k=k,
        f_bounds=f_bounds,
        t_bounds=t_bounds,
    )


def bounds_zone(stat: float, lower: float, upper: float, is_t: bool = False) -> str:
    return _bounds_zone(stat, lower, upper, is_t)


# ---------------------------------------------------------------------------
# SMG (Sam-McNown-Goh 2019) F-independent critical values (cases 1, 3, 5).
# ---------------------------------------------------------------------------
@dataclass
class SMGResult:
    available: bool
    case: int
    k: int
    sample_size: int
    used_num: Optional[int]
    bounds: Dict[float, Tuple[float, float]]  # alpha -> (I0, I1)
    note: str = ""


def smg_bounds(case: int, k: int, sample_size: int) -> SMGResult:
    """Look up SMG F-independent bounds. k = number of independent vars (1..7)."""
    if case not in (1, 3, 5):
        return SMGResult(False, case, k, sample_size, None, {},
                         note=f"اختبار SMG غير معرّف للحالة {case} (متاح للحالات 1، 3، 5).")
    if not (1 <= k <= 7):
        return SMGResult(False, case, k, sample_size, None, {},
                         note=f"عدد المتغيرات المستقلة ({k}) خارج نطاق جداول SMG (1 إلى 7).")

    # choose grid sample size (largest grid value <= sample_size, else asymptotic 81)
    grid = [g for g in _smk.NUM_GRID if g <= 80]
    eligible = [g for g in grid if g <= sample_size]
    used_num = max(eligible) if eligible else (81 if sample_size > 80 else 30)
    if sample_size > 80:
        used_num = 81

    col_I0 = getattr(_smk, f"I0_{k}")
    col_I1 = getattr(_smk, f"I1_{k}")

    bounds: Dict[float, Tuple[float, float]] = {}
    for i in range(len(_smk.CASE)):
        if _smk.CASE[i] == case and _smk.NUM[i] == used_num:
            bounds[_smk.PROB[i]] = (col_I0[i], col_I1[i])

    return SMGResult(
        available=True,
        case=case,
        k=k,
        sample_size=sample_size,
        used_num=used_num,
        bounds=bounds,
    )
