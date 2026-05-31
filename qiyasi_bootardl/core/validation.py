"""Input validation. التحقق من صحة المدخلات.

Validates the data and specification and returns a clean numeric DataFrame
ordered as [y, x1, x2, ...]. Emits Arabic warnings when rows are dropped.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ..utils.exceptions import DataValidationError, SpecificationError


def validate_and_prepare(
    data: pd.DataFrame,
    yvar: Optional[str],
    xvar: Optional[Sequence[str]],
    case: int,
    max_lag: int,
    n_boot: int,
) -> Tuple[pd.DataFrame, str, List[str], int, int]:
    """Validate inputs and return (clean_data, yvar, xvar, case, n_boot).

    التحقق من البيانات وإرجاع إطار بيانات نظيف مرتب [y, x1, x2, ...].
    """
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as exc:  # noqa: BLE001
            raise DataValidationError(
                "تعذّر تحويل المدخلات إلى DataFrame. يرجى تمرير pandas.DataFrame."
            ) from exc

    names = list(data.columns)
    if yvar is None:
        yvar = names[0]
    if xvar is None:
        xvar = [c for c in names if c != yvar]

    xvar = list(xvar)

    if yvar not in names:
        raise DataValidationError(f"المتغير التابع '{yvar}' غير موجود في البيانات.")
    missing_x = [x for x in xvar if x not in names]
    if missing_x:
        raise DataValidationError(
            f"المتغيرات المستقلة التالية غير موجودة في البيانات: {missing_x}"
        )
    if yvar in xvar:
        raise SpecificationError(
            "يجب فصل المتغير التابع عن المتغيرات المستقلة (لا يمكن أن يكون نفسه)."
        )
    if len(xvar) < 1:
        raise SpecificationError("يجب توفير متغير مستقل واحد على الأقل.")

    if case not in (1, 2, 3, 4, 5):
        warnings.warn("حالة غير صالحة. سيتم استخدام الحالة الثالثة افتراضياً.", stacklevel=2)
        case = 3

    cols = [yvar] + xvar
    clean = data[cols].copy()

    # Coerce to numeric
    for c in cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    n_before = len(clean)
    clean = clean.dropna().reset_index(drop=True)
    n_after = len(clean)
    if n_after < n_before:
        warnings.warn(
            f"تم حذف {n_before - n_after} مشاهدة بسبب وجود قيم مفقودة.",
            stacklevel=2,
        )

    d = len(cols)
    min_needed = max_lag + 2 * d + 5
    if n_after <= min_needed:
        raise DataValidationError(
            f"عدد المشاهدات ({n_after}) صغير جداً مقارنة بأقصى إبطاء وعدد المتغيرات. "
            f"المطلوب أكثر من {min_needed} مشاهدة تقريباً."
        )

    if n_boot < 1:
        raise SpecificationError("عدد تكرارات البوتستراب يجب أن يكون موجباً.")

    return clean, yvar, xvar, case, int(n_boot)
