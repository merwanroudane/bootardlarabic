"""Central orchestration engine (shared by the Arabic and English APIs).

المحرّك المركزي: نقطة الدخول الموحّدة التي تستدعيها الواجهتان العربية
والإنجليزية. يربط: التحقق ← اختيار الإبطاء ← تقدير ARDL (مشروط وغير مشروط)
← الإحصائيات ← حدود PSS/SMG ← البوتستراب (بدون VECM) ← التشخيص ← كائن النتائج.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .validation import validate_and_prepare
from .lag_selection import select_diff_lags
from .ardl import estimate_ardl
from .statistics import compute_statistics
from .bounds import pss_bounds, smg_bounds
from .bootstrap import run_bootstrap
from .diagnostics import run_diagnostics
from ..results.arabic_result import ArabicBootARDLResult


def run_bootstrap_ardl(
    data: pd.DataFrame,
    yvar: Optional[str] = None,
    xvar: Optional[Sequence[str]] = None,
    fixed_ardl_lags: Optional[Sequence[int]] = None,
    ardl_ic: str = "AIC",
    max_lag: int = 5,
    case: int = 3,
    n_boot: int = 2000,
    bootstrap_levels: Sequence[float] = (0.10, 0.05, 0.01),
    decision_level: float = 0.05,
    progress: bool = True,
    random_state: Optional[int] = None,
) -> ArabicBootARDLResult:
    """Run the full conditional bootstrap ARDL bounds test and return the result.

    تشغيل الاختبار الكامل وإرجاع كائن النتائج العربي. هذه الدالة محايدة لغوياً
    وتُستدعى من الواجهتين.
    """
    # --- validation ---
    clean, yvar, xvar, case, n_boot = validate_and_prepare(
        data, yvar, xvar, case, max_lag, n_boot
    )
    numdata = clean[[yvar] + list(xvar)]
    d = numdata.shape[1]
    k = d - 1

    # --- lag selection (or fixed) ---
    if fixed_ardl_lags is not None:
        diff_lags = list(int(v) for v in fixed_ardl_lags)
        if len(diff_lags) != d:
            raise ValueError(
                f"عدد رتب الإبطاء الثابتة ({len(diff_lags)}) يجب أن يساوي عدد "
                f"المتغيرات ({d})."
            )
    else:
        diff_lags, _, ardl_ic = select_diff_lags(numdata, case, max_lag, ardl_ic)

    trim = 1 + int(max(diff_lags)) if any(diff_lags) else 1
    trim = max(trim, 1)

    # --- estimation (conditional + unconditional on common sample) ---
    cond_fit = estimate_ardl(numdata, diff_lags, case, conditional=True, trim=trim)
    uncond_fit = estimate_ardl(numdata, diff_lags, case, conditional=False, trim=trim)

    # --- observed statistics ---
    stats_cond = compute_statistics(cond_fit, case)
    stats_uncond = compute_statistics(uncond_fit, case)
    f_ind_uncond = stats_uncond.f_ind

    # --- asymptotic bounds (reference only) ---
    bounds = pss_bounds(stats_cond.f_overall, stats_cond.t_dep, case, k)
    smg = smg_bounds(case, k, sample_size=cond_fit.nobs)

    # --- bootstrap (VECM-free, fixed-X) ---
    rng = np.random.default_rng(random_state)
    boot = run_bootstrap(
        numdata=numdata,
        diff_lags=diff_lags,
        case=case,
        n_boot=n_boot,
        levels=list(bootstrap_levels),
        cond_fit=cond_fit,
        uncond_fit=uncond_fit,
        f_ind_obs_cond=stats_cond.f_ind,
        f_ind_obs_uncond=f_ind_uncond,
        f_overall_obs=stats_cond.f_overall,
        t_obs=stats_cond.t_dep,
        rng=rng,
        progress=progress,
    )

    # --- diagnostics ---
    diagnostics = run_diagnostics(cond_fit, bg_lags=max(1, int(max(diff_lags)) or 1))

    return ArabicBootARDLResult(
        yvar=yvar,
        xvar=list(xvar),
        case=case,
        k=k,
        n_obs=cond_fit.nobs,
        diff_lags=diff_lags,
        ardl_ic=ardl_ic,
        max_lag=max_lag,
        n_boot=n_boot,
        levels=list(bootstrap_levels),
        decision_level=decision_level,
        cond_fit=cond_fit,
        uncond_fit=uncond_fit,
        statistics=stats_cond,
        f_ind_uncond=f_ind_uncond,
        bounds=bounds,
        smg=smg,
        bootstrap=boot,
        diagnostics=diagnostics,
        source_data=numdata,
    )
