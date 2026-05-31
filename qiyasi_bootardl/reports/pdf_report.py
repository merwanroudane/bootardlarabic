"""PDF report generator (via HTML).

مولّد تقرير PDF عبر تحويل تقرير HTML. يتطلب weasyprint (اختياري).
"""
from __future__ import annotations

from .html_report import build_html_report


def build_pdf_report(result, path: str) -> str:
    """Render the RTL HTML report to a PDF file and return ``path``.

    يتطلب weasyprint. إن لم يكن متاحاً، احفظ تقرير HTML وحوّله يدوياً.
    """
    html_str = build_html_report(result, path=None)
    try:
        from weasyprint import HTML
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "تصدير PDF يتطلب weasyprint. ثبّته عبر: pip install weasyprint، "
            "أو استخدم تقرير_HTML ثم اطبع الصفحة إلى PDF."
        ) from exc
    HTML(string=html_str).write_pdf(path)
    return path
