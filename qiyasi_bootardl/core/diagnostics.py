"""Residual diagnostics for the conditional ARDL-ECM.

الاختبارات التشخيصية لبواقي النموذج: الارتباط الذاتي، التوزيع الطبيعي،
ثبات التباين، والصيغة الدالية.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from .ardl import ARDLFit


@dataclass
class DiagnosticResult:
    name_ar: str
    statistic: float
    pvalue: float
    df: Optional[int] = None
    ok: Optional[bool] = None  # True when the desirable null is NOT rejected at 5%
    note: str = ""


def _ols_resid(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def breusch_godfrey(fit: ARDLFit, nlags: int = 1) -> DiagnosticResult:
    """LM test for serial correlation up to ``nlags`` (Breusch-Godfrey)."""
    u = np.asarray(fit.resid, dtype=float)
    X = fit.X.values
    n = len(u)
    nlags = max(1, min(nlags, n - X.shape[1] - 1))

    Z = [X]
    for L in range(1, nlags + 1):
        lag = np.zeros(n)
        lag[L:] = u[:-L]
        Z.append(lag.reshape(-1, 1))
    Zmat = np.hstack(Z)

    e = _ols_resid(Zmat, u)
    sse = float(e @ e)
    sst = float(((u - u.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    lm = n * r2
    p = float(stats.chi2.sf(lm, nlags))
    return DiagnosticResult(
        name_ar=f"اختبار الارتباط الذاتي (Breusch-Godfrey, إبطاء={nlags})",
        statistic=lm, pvalue=p, df=nlags, ok=(p > 0.05),
    )


def jarque_bera(fit: ARDLFit) -> DiagnosticResult:
    """Jarque-Bera test for residual normality."""
    u = np.asarray(fit.resid, dtype=float)
    n = len(u)
    u = u - u.mean()
    s2 = float((u ** 2).mean())
    if s2 <= 0:
        return DiagnosticResult("اختبار التوزيع الطبيعي (Jarque-Bera)", float("nan"),
                                float("nan"), df=2, ok=None, note="تباين صفري")
    skew = float((u ** 3).mean()) / s2 ** 1.5
    kurt = float((u ** 4).mean()) / s2 ** 2
    jb = n / 6.0 * (skew ** 2 + 0.25 * (kurt - 3.0) ** 2)
    p = float(stats.chi2.sf(jb, 2))
    return DiagnosticResult(
        name_ar="اختبار التوزيع الطبيعي (Jarque-Bera)",
        statistic=jb, pvalue=p, df=2, ok=(p > 0.05),
    )


def breusch_pagan(fit: ARDLFit) -> DiagnosticResult:
    """Breusch-Pagan LM test for heteroskedasticity."""
    u = np.asarray(fit.resid, dtype=float)
    X = fit.X.values
    n = len(u)
    g = u ** 2
    e = _ols_resid(X, g)
    sse = float(e @ e)
    sst = float(((g - g.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    lm = n * r2
    df = X.shape[1] - 1
    p = float(stats.chi2.sf(lm, df)) if df > 0 else float("nan")
    return DiagnosticResult(
        name_ar="اختبار ثبات التباين (Breusch-Pagan)",
        statistic=lm, pvalue=p, df=df, ok=(p > 0.05),
    )


def run_diagnostics(fit: ARDLFit, bg_lags: int = 1):
    """Run the standard residual diagnostic battery on a conditional fit."""
    return {
        "serial_correlation": breusch_godfrey(fit, nlags=bg_lags),
        "normality": jarque_bera(fit),
        "heteroskedasticity": breusch_pagan(fit),
    }
