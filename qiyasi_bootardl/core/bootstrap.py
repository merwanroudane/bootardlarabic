"""VECM-free bootstrap for the conditional ARDL bounds test.

البوتستراب أحادي المعادلة لاختبار حدود ARDL (بدون VECM).

Design (single-equation, conditional, X held fixed)
---------------------------------------------------
1. Estimate the *restricted* ARDL-ECM under the global null of **no
   cointegration** (drop the entire lagged-level block, i.e. both the lagged
   level of y and of every x).
2. Resample (and re-centre) the restricted residuals.
3. Regenerate Delta-y recursively from the restricted dynamics, holding the
   observed X fixed, and cumulate to levels y*.
4. Re-estimate the *unrestricted* ARDL on each bootstrap sample and recompute
   F-overall, t and F-independent -> empirical null distributions.

Because the global no-cointegration null nests a_yy = 0 and a_yx = 0, the same
generated data delivers the null distribution of all three statistics.

Fake / degenerate cointegration is flagged per significance level when the
*conditional* F-independent rejects but the *unconditional* one does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ardl import ARDLFit, estimate_ardl
from .lag_selection import build_design, DesignInfo
from .statistics import f_overall_terms
from ..utils.progress import ProgressBar


@dataclass
class BootstrapResult:
    levels: List[float]
    n_boot: int
    n_boot_effective: int
    crit_f_overall: Dict[float, float]
    crit_t_dep: Dict[float, float]
    crit_f_ind: Dict[float, float]
    pvalue_f_overall: float
    pvalue_t_dep: float
    pvalue_f_ind: float
    fake_cointegration: Dict[float, bool]
    f_ind_unconditional: float
    crit_f_ind_uncond: Dict[float, float]
    dist_f_overall: np.ndarray = field(repr=False, default=None)
    dist_t_dep: np.ndarray = field(repr=False, default=None)
    dist_f_ind: np.ndarray = field(repr=False, default=None)
    dist_f_ind_uncond: np.ndarray = field(repr=False, default=None)


def _ols(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(XtX)
    return beta, sigma2, XtX_inv


def _wald_F(beta: np.ndarray, sigma2: float, XtX_inv: np.ndarray, idx: Sequence[int]) -> float:
    idx = list(idx)
    q = len(idx)
    if q == 0:
        return float("nan")
    b = beta[idx]
    V = sigma2 * XtX_inv[np.ix_(idx, idx)]
    try:
        w = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        w = float(b @ np.linalg.pinv(V) @ b)
    return w / q


def _tstat(beta: np.ndarray, sigma2: float, XtX_inv: np.ndarray, j: int) -> float:
    return float(beta[j] / np.sqrt(sigma2 * XtX_inv[j, j]))


def _term_indices(info: DesignInfo, names: List[str], case: int) -> Dict[str, object]:
    pos = {n: i for i, n in enumerate(names)}
    fov = [pos[n] for n in f_overall_terms_from(info, case) if n in pos]
    find = [pos[n] for n in info.x_levels if n in pos]
    y_idx = pos[info.y_level]
    return {"fov": fov, "find": find, "t": y_idx}


def f_overall_terms_from(info: DesignInfo, case: int) -> List[str]:
    # mirrors statistics.f_overall_terms but works directly off DesignInfo
    from .deterministic_cases import get_case_spec

    spec = get_case_spec(case)
    terms = list(info.level_names)
    if spec.intercept_restricted and info.const:
        terms = [info.const] + terms
    if spec.trend_restricted and info.trend:
        terms = terms + [info.trend]
    return terms


def run_bootstrap(
    numdata: pd.DataFrame,
    diff_lags: Sequence[int],
    case: int,
    n_boot: int,
    levels: Sequence[float],
    cond_fit: ARDLFit,
    uncond_fit: ARDLFit,
    f_ind_obs_cond: float,
    f_ind_obs_uncond: float,
    f_overall_obs: float,
    t_obs: float,
    rng: np.random.Generator,
    progress: bool = True,
) -> BootstrapResult:
    """Run the fixed-X residual bootstrap and return critical values & p-values."""
    yname = numdata.columns[0]
    N = len(numdata)
    trim = 1 + int(max(diff_lags))

    # --- restricted model under H0 (drop the level block) ---
    Xc, dep, info = build_design(numdata, diff_lags, case, conditional=True, trim=trim)
    cond_names = list(Xc.columns)
    restricted_names = [n for n in cond_names if n not in info.level_names]
    Xr = Xc[restricted_names].values
    yv = dep.values
    beta_r, _, _ = _ols(Xr, yv)
    resid_r = yv - Xr @ beta_r
    resid_r = resid_r - resid_r.mean()

    # split restricted coefficients into y-lag (psi) and exogenous parts
    dy_lag_prefix = f"D_{yname}.l"
    psi_cols = [j for j, n in enumerate(restricted_names) if n.startswith(dy_lag_prefix)]
    psi_lags = [int(restricted_names[j].split(".l")[-1]) for j in psi_cols]
    psi_vals = beta_r[psi_cols]
    exog_cols = [j for j in range(len(restricted_names)) if j not in psi_cols]
    exog_part = Xr[:, exog_cols] @ beta_r[exog_cols]  # length n, aligned to rows trim..N-1
    n = len(yv)

    # observed series for seeding
    y_obs = numdata[yname].values.astype(float)
    dy_full = np.empty(N)
    dy_full[:] = np.nan
    dy_full[1:] = np.diff(y_obs)

    x_cols = list(numdata.columns[1:])
    x_obs = {c: numdata[c].values.astype(float) for c in x_cols}

    # --- index sets for statistic recomputation (built once) ---
    idx_cond = _term_indices(info, cond_names, case)
    Xu0, _, info_u = build_design(numdata, diff_lags, case, conditional=False, trim=trim)
    uncond_names = list(Xu0.columns)
    find_uncond_idx = [uncond_names.index(nm) for nm in info_u.x_levels if nm in uncond_names]

    dist_fov = np.full(n_boot, np.nan)
    dist_t = np.full(n_boot, np.nan)
    dist_find = np.full(n_boot, np.nan)
    dist_find_uc = np.full(n_boot, np.nan)

    bar = ProgressBar(n_boot, label="محاكاة البوتستراب", enabled=progress)
    p_max = max(psi_lags) if psi_lags else 0

    for b in range(n_boot):
        e_star = resid_r[rng.integers(0, n, size=n)]
        dystar = dy_full.copy()  # observed seed for r < trim
        for k in range(n):
            r = trim + k
            ar = 0.0
            for coef, lag in zip(psi_vals, psi_lags):
                ar += coef * dystar[r - lag]
            dystar[r] = exog_part[k] + ar + e_star[k]
        ystar = y_obs.copy()
        for r in range(trim, N):
            ystar[r] = ystar[r - 1] + dystar[r]

        numdata_star = pd.DataFrame({yname: ystar, **{c: x_obs[c] for c in x_cols}})

        # conditional fit on bootstrap data
        Xb, yb, _ = build_design(numdata_star, diff_lags, case, conditional=True, trim=trim)
        try:
            beta, s2, inv = _ols(Xb.values, yb.values)
        except np.linalg.LinAlgError:
            bar.update(b)
            continue
        dist_fov[b] = _wald_F(beta, s2, inv, idx_cond["fov"])
        dist_t[b] = _tstat(beta, s2, inv, idx_cond["t"])
        dist_find[b] = _wald_F(beta, s2, inv, idx_cond["find"])

        # unconditional fit on the same bootstrap data
        Xbu, ybu, _ = build_design(numdata_star, diff_lags, case, conditional=False, trim=trim)
        try:
            betau, s2u, invu = _ols(Xbu.values, ybu.values)
            dist_find_uc[b] = _wald_F(betau, s2u, invu, find_uncond_idx)
        except np.linalg.LinAlgError:
            pass
        bar.update(b)

    dist_fov = dist_fov[~np.isnan(dist_fov)]
    dist_t = dist_t[~np.isnan(dist_t)]
    dist_find = dist_find[~np.isnan(dist_find)]
    dist_find_uc = dist_find_uc[~np.isnan(dist_find_uc)]
    n_eff = len(dist_fov)

    crit_fov = {a: float(np.quantile(dist_fov, 1 - a)) for a in levels} if len(dist_fov) else {}
    crit_t = {a: float(np.quantile(dist_t, a)) for a in levels} if len(dist_t) else {}
    crit_find = {a: float(np.quantile(dist_find, 1 - a)) for a in levels} if len(dist_find) else {}
    crit_find_uc = {a: float(np.quantile(dist_find_uc, 1 - a)) for a in levels} if len(dist_find_uc) else {}

    p_fov = float(np.mean(dist_fov >= f_overall_obs)) if len(dist_fov) else float("nan")
    p_t = float(np.mean(dist_t <= t_obs)) if len(dist_t) else float("nan")
    p_find = float(np.mean(dist_find >= f_ind_obs_cond)) if len(dist_find) else float("nan")

    # fake / degenerate cointegration: conditional Find rejects but unconditional does not
    fake = {}
    for a in levels:
        cond_rej = (a in crit_find) and (f_ind_obs_cond > crit_find[a])
        uncond_norej = (a in crit_find_uc) and (f_ind_obs_uncond <= crit_find_uc[a])
        fake[a] = bool(cond_rej and uncond_norej)

    return BootstrapResult(
        levels=list(levels),
        n_boot=n_boot,
        n_boot_effective=n_eff,
        crit_f_overall=crit_fov,
        crit_t_dep=crit_t,
        crit_f_ind=crit_find,
        pvalue_f_overall=p_fov,
        pvalue_t_dep=p_t,
        pvalue_f_ind=p_find,
        fake_cointegration=fake,
        f_ind_unconditional=f_ind_obs_uncond,
        crit_f_ind_uncond=crit_find_uc,
        dist_f_overall=dist_fov,
        dist_t_dep=dist_t,
        dist_f_ind=dist_find,
        dist_f_ind_uncond=dist_find_uc,
    )
