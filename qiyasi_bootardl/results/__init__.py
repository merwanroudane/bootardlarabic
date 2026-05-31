"""كائنات النتائج ومحرّكات القرار والتفسير والتحذير.

Result objects and the decision / interpretation / methodology-warning engines.
"""
from __future__ import annotations

from .decisions import (
    Decision,
    decide,
    COINTEGRATION,
    NO_COINTEGRATION,
    INCONCLUSIVE,
    DEGENERATE,
)
from .interpretation import interpret
from .warnings import ArabicMethodologyAdvisor, MethodologyWarning
from .arabic_result import ArabicBootARDLResult
from .english_result import EnglishBootARDLResult
from . import tables

__all__ = [
    "Decision", "decide",
    "COINTEGRATION", "NO_COINTEGRATION", "INCONCLUSIVE", "DEGENERATE",
    "interpret",
    "ArabicMethodologyAdvisor", "MethodologyWarning",
    "ArabicBootARDLResult", "EnglishBootARDLResult",
    "tables",
]
