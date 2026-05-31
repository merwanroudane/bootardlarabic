"""Conditional and unconditional ARDL-ECM estimation (single-equation OLS).

تقدير نموذج تصحيح الخطأ ARDL المشروط وغير المشروط بطريقة المربعات الصغرى.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .lag_selection import DesignInfo, build_design


@dataclass
class ARDLFit:
    params: pd.Series
    cov: pd.DataFrame
    resid: np.ndarray
    fitted: np.ndarray
    sigma2: float
    nobs: int
    k: int
    rsquared: float
    rsquared_adj: float
    pvalues: pd.Series
    bse: pd.Series
    info: DesignInfo
    conditional: bool
    X: pd.DataFrame
    y: pd.Series

    @property
    def names(self) -> List[str]:
        return list(self.params.index)


def estimate_ardl(
    numdata: pd.DataFrame,
    diff_lags: Sequence[int],
    case: int,
    conditional: bool,
    trim: int,
) -> ARDLFit:
    """Estimate the (un)conditional ARDL-ECM by OLS on a fixed sample."""
    X, y, info = build_design(numdata, diff_lags, case, conditional=conditional, trim=trim)
    model = sm.OLS(y.values, X.values)
    res = model.fit()

    params = pd.Series(res.params, index=X.columns)
    cov = pd.DataFrame(res.cov_params(), index=X.columns, columns=X.columns)
    bse = pd.Series(res.bse, index=X.columns)
    pvals = pd.Series(res.pvalues, index=X.columns)

    return ARDLFit(
        params=params,
        cov=cov,
        resid=np.asarray(res.resid),
        fitted=np.asarray(res.fittedvalues),
        sigma2=float(res.scale),
        nobs=int(res.nobs),
        k=int(X.shape[1]),
        rsquared=float(res.rsquared) if case >= 2 else float(res.rsquared),
        rsquared_adj=float(res.rsquared_adj),
        pvalues=pvals,
        bse=bse,
        info=info,
        conditional=conditional,
        X=X,
        y=y,
    )


def coefficients_frame(fit: ARDLFit) -> pd.DataFrame:
    """Tidy coefficient table (Arabic-friendly column keys)."""
    return pd.DataFrame(
        {
            "المعامل": fit.params.values,
            "الخطأ_المعياري": fit.bse.values,
            "إحصائية_t": (fit.params / fit.bse).values,
            "القيمة_الاحتمالية": fit.pvalues.values,
        },
        index=fit.params.index,
    )
