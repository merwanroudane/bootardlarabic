"""المحرك الحسابي لاختبار ARDL بالبوتستراب (بدون VECM).

Core computational engine for the conditional bootstrap ARDL bounds test.
"""
from __future__ import annotations

from .deterministic_cases import CaseSpec, get_case_spec
from .lag_selection import DesignInfo, build_design, select_diff_lags
from .ardl import ARDLFit, estimate_ardl, coefficients_frame
from .statistics import TestStatistics, compute_statistics, statistics_frame
from .bounds import (
    BoundsResult,
    SMGResult,
    pss_bounds,
    bounds_zone,
    smg_bounds,
    PSS_NOTE,
)
from .bootstrap import BootstrapResult, run_bootstrap
from .diagnostics import DiagnosticResult, run_diagnostics
from .validation import validate_and_prepare

__all__ = [
    "CaseSpec", "get_case_spec",
    "DesignInfo", "build_design", "select_diff_lags",
    "ARDLFit", "estimate_ardl", "coefficients_frame",
    "TestStatistics", "compute_statistics", "statistics_frame",
    "BoundsResult", "SMGResult", "pss_bounds", "bounds_zone", "smg_bounds", "PSS_NOTE",
    "BootstrapResult", "run_bootstrap",
    "DiagnosticResult", "run_diagnostics",
    "validate_and_prepare",
]
