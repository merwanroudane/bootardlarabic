"""English API for the bootstrap ARDL bounds test.

``bootstrap_ardl_test`` is a thin English alias over the same shared engine used
by the Arabic API. It returns an :class:`EnglishBootARDLResult` that wraps the
underlying Arabic result (computation is identical; only labels differ).
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .core.engine import run_bootstrap_ardl
from .results.english_result import EnglishBootARDLResult


def bootstrap_ardl_test(
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
) -> EnglishBootARDLResult:
    """Run the VECM-free conditional bootstrap ARDL bounds test.

    Parameters mirror :func:`اختبار_ARDL_بالبوتستراب`. Returns an
    ``EnglishBootARDLResult`` wrapping the shared Arabic result object.
    """
    ar = run_bootstrap_ardl(
        data=data,
        yvar=yvar,
        xvar=xvar,
        fixed_ardl_lags=fixed_ardl_lags,
        ardl_ic=ardl_ic,
        max_lag=max_lag,
        case=case,
        n_boot=n_boot,
        bootstrap_levels=bootstrap_levels,
        decision_level=decision_level,
        progress=progress,
        random_state=random_state,
    )
    return EnglishBootARDLResult(ar)
