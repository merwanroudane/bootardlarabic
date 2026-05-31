"""Exceptions for qiyasi_bootardl / استثناءات المكتبة."""
from __future__ import annotations


class QiyasiError(Exception):
    """Base exception. الاستثناء الأساسي للمكتبة."""


class DataValidationError(QiyasiError):
    """Raised on invalid input data. خطأ في بيانات الإدخال."""


class SpecificationError(QiyasiError):
    """Raised on an invalid model specification. خطأ في توصيف النموذج."""


class EstimationError(QiyasiError):
    """Raised when estimation cannot be performed. خطأ في تقدير النموذج."""
