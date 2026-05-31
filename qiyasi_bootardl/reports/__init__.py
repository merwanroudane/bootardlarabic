"""مولّدات التقارير: HTML و Word و Excel و PDF (كلها بدعم RTL عربي).

Report generators. HTML is dependency-free; Word/Excel/PDF need optional extras.
"""
from __future__ import annotations

from .html_report import build_html_report

__all__ = [
    "build_html_report", "build_word_report", "build_excel_report", "build_pdf_report",
    "plot_bootstrap_distributions", "plot_decision", "plot_series",
    "plot_fake_cointegration",
]


def __getattr__(name):
    if name in {
        "plot_bootstrap_distributions", "plot_decision", "plot_series",
        "plot_fake_cointegration",
    }:
        from . import visualization
        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_word_report(result, path):
    from .word_report import build_word_report as _f
    return _f(result, path)


def build_excel_report(result, path):
    from .excel_report import build_excel_report as _f
    return _f(result, path)


def build_pdf_report(result, path):
    from .pdf_report import build_pdf_report as _f
    return _f(result, path)
