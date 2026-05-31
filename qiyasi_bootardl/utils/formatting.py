"""Formatting helpers for Arabic tabular output. أدوات التنسيق."""
from __future__ import annotations

from typing import Sequence


def fmt(value, digits: int = 3) -> str:
    """Format a number, returning a dash for None/NaN."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "—"
    return f"{f:.{digits}f}"


def pct(level: float) -> str:
    """Format a significance level as a percentage label, e.g. 0.05 -> '5%'."""
    return f"{level * 100:g}%"


def rule(width: int = 52, char: str = "-") -> str:
    return char * width


def center_block(title: str, width: int = 52) -> str:
    line = "=" * width
    return f"{line}\n{title}\n{line}"


def simple_table(headers: Sequence[str], rows: Sequence[Sequence], col_width: int = 14) -> str:
    """Render a fixed-width plain-text table (LTR layout; numbers read fine)."""
    head = "".join(str(h).ljust(col_width) for h in headers)
    body = "\n".join(
        "".join(str(c).ljust(col_width) for c in row) for row in rows
    )
    return f"{head}\n{body}"
