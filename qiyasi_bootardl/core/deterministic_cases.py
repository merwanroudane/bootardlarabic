"""Deterministic case handling (PSS cases 1-5). الحالات الحتمية.

Encapsulates how the intercept and trend enter the conditional ARDL-ECM and
which coefficients are jointly restricted under each null hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseSpec:
    case: int
    has_intercept: bool          # an intercept column is present in the regression
    has_trend: bool              # a linear trend column is present in the regression
    intercept_restricted: bool   # intercept enters the F-overall restriction (case 2)
    trend_restricted: bool       # trend enters the F-overall restriction (case 4)
    t_test_applicable: bool      # PSS/SMG t & F-ind bounds defined (cases 1,3,5)


def get_case_spec(case: int) -> CaseSpec:
    """Return the :class:`CaseSpec` for a given PSS case (1-5)."""
    table = {
        1: CaseSpec(1, False, False, False, False, True),
        2: CaseSpec(2, True, False, True, False, False),
        3: CaseSpec(3, True, False, False, False, True),
        4: CaseSpec(4, True, True, False, True, False),
        5: CaseSpec(5, True, True, False, False, True),
    }
    return table[case]
