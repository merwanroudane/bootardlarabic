"""ARDL-ECM design-matrix construction and lag-order selection.

بناء مصفوفة الانحدار لنموذج ARDL-ECM واختيار رتب الإبطاء.

Design-matrix column convention (canonical order)::

    const | <y>.l1 <x1>.l1 ... (level block, d cols)
          | D_<v>.l1 ... (lagged differences)
          | D_<x1> ...    (contemporaneous diffs of X; conditional model only)
          | trend         (case >= 4)

The level block is the long-run part; the lagged/contemporaneous differences are
the short-run dynamics.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_VALID_ARDL_IC = ("AIC", "AICc", "BIC", "SC", "HQ", "adjR2")
_GRID_CAP = 4096


@dataclass
class DesignInfo:
    dep_name: str
    d: int
    const: Optional[str]
    trend: Optional[str]
    level_names: List[str]
    y_level: str
    x_levels: List[str] = field(default_factory=list)
    lagdiff_names: List[str] = field(default_factory=list)
    contemp_names: List[str] = field(default_factory=list)


def build_design(
    numdata: pd.DataFrame,
    diff_lags: Sequence[int],
    case: int,
    conditional: bool,
    trim: int,
) -> Tuple[pd.DataFrame, pd.Series, DesignInfo]:
    """Build the ARDL-ECM design matrix for the given difference-lag orders.

    ``diff_lags`` is a length-``d`` vector giving the number of lagged
    differences for [y, x1, x2, ...] respectively. ``trim`` initial rows are
    dropped so that competing specifications share a common sample.
    """
    cols = list(numdata.columns)
    d = len(cols)
    yname = cols[0]
    xcols = cols[1:]

    lev = numdata.shift(1)
    dif = numdata.diff(1)

    dep_name = f"D_{yname}"
    dep = dif[yname].rename(dep_name)

    parts: Dict[str, pd.Series] = {}
    level_names: List[str] = []
    for c in cols:
        nm = f"{c}.l1"
        parts[nm] = lev[c]
        level_names.append(nm)

    lagdiff_names: List[str] = []
    for i, c in enumerate(cols):
        k = int(diff_lags[i])
        for j in range(1, k + 1):
            nm = f"D_{c}.l{j}"
            parts[nm] = dif[c].shift(j)
            lagdiff_names.append(nm)

    contemp_names: List[str] = []
    if conditional:
        for c in xcols:
            nm = f"D_{c}"
            parts[nm] = dif[c]
            contemp_names.append(nm)

    trend_name: Optional[str] = None
    if case >= 4:
        trend_name = "trend"
        parts[trend_name] = pd.Series(
            np.arange(len(numdata), dtype=float), index=numdata.index
        )

    X = pd.DataFrame(parts)

    const_name: Optional[str] = None
    if case >= 2:
        const_name = "const"
        X[const_name] = 1.0

    ordered: List[str] = []
    if const_name:
        ordered.append(const_name)
    ordered += level_names + lagdiff_names + contemp_names
    if trend_name:
        ordered.append(trend_name)
    X = X[ordered]

    dep = dep.iloc[trim:]
    X = X.iloc[trim:]
    mask = dep.notna() & X.notna().all(axis=1)
    dep = dep[mask].reset_index(drop=True)
    X = X[mask].reset_index(drop=True)

    info = DesignInfo(
        dep_name=dep_name,
        d=d,
        const=const_name,
        trend=trend_name,
        level_names=level_names,
        y_level=level_names[0],
        x_levels=level_names[1:],
        lagdiff_names=lagdiff_names,
        contemp_names=contemp_names,
    )
    return X, dep, info


def _ols_ssr(X: np.ndarray, y: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def _information_criterion(ssr: float, k: int, n: int, crit: str, tss: float) -> float:
    """Return a value to be *minimised* for the given criterion."""
    ssr = max(ssr, 1e-300)
    ll = n * np.log(ssr / n)
    if crit == "AIC":
        return ll + 2 * k
    if crit == "AICc":
        denom = max(n - k - 1, 1)
        return ll + 2 * k + (2 * k * (k + 1)) / denom
    if crit in ("BIC", "SC"):
        return ll + k * np.log(n)
    if crit == "HQ":
        return ll + 2 * k * np.log(max(np.log(n), 1.0 + 1e-9))
    if crit == "adjR2":
        adj = 1.0 - (ssr / max(n - k, 1)) / (max(tss, 1e-300) / max(n - 1, 1))
        return -adj
    raise ValueError(f"Unknown criterion {crit}")


def select_diff_lags(
    numdata: pd.DataFrame,
    case: int,
    max_lag: int,
    criterion: str = "AIC",
) -> Tuple[List[int], float, str]:
    """Select per-variable lagged-difference orders by information criterion.

    Returns ``(diff_lags, best_ic, criterion)``. Selection uses the *conditional*
    model on a common (max_lag-trimmed) sample so candidates are comparable.
    """
    if criterion not in _VALID_ARDL_IC:
        criterion = "AIC"

    d = numdata.shape[1]
    trim = max_lag + 1
    grid_size = (max_lag + 1) ** d

    if grid_size <= _GRID_CAP:
        candidates = itertools.product(range(max_lag + 1), repeat=d)
    else:
        # Fall back to a common lag applied to every variable.
        candidates = ([k] * d for k in range(max_lag + 1))

    best_lags: Optional[List[int]] = None
    best_ic = np.inf
    for combo in candidates:
        diff_lags = list(combo)
        X, y, _ = build_design(numdata, diff_lags, case, conditional=True, trim=trim)
        if len(y) <= X.shape[1] + 1:
            continue
        Xv = X.values
        yv = y.values
        ssr = _ols_ssr(Xv, yv)
        tss = float(np.sum((yv - yv.mean()) ** 2))
        ic = _information_criterion(ssr, Xv.shape[1], len(yv), criterion, tss)
        if ic < best_ic:
            best_ic = ic
            best_lags = diff_lags

    if best_lags is None:
        best_lags = [0] * d
    return best_lags, float(best_ic), criterion
