"""Arabic-labelled visualizations for the bootstrap ARDL bounds test.

الرسوم البيانية العربية: توزيعات البوتستراب الصفرية، مقارنة القرار، والسلاسل
الزمنية — جميعها بعناوين عربية صحيحة الاتجاه (RTL) وألوان متناسقة.

يتطلب matplotlib. ويُفضّل arabic_reshaper و python-bidi لعرض الحروف العربية
متصلةً وبالاتجاه الصحيح؛ في غيابهما يُستخدم النص كما هو.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..utils.arabic_text import STAT_LABELS_AR, CASE_NAMES_AR
from ..utils.formatting import pct

# ---------------------------------------------------------------------------
# Arabic text shaping (reshape + bidi) with graceful fallback.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on optional libs
    import arabic_reshaper
    from bidi.algorithm import get_display

    def ع(text) -> str:
        """تهيئة نص عربي للعرض في matplotlib (تشكيل الحروف + ترتيب RTL)."""
        return get_display(arabic_reshaper.reshape(str(text)))

    _SHAPING = True
except Exception:  # pragma: no cover

    def ع(text) -> str:
        return str(text)

    _SHAPING = False


# ---------------------------------------------------------------------------
# Palette — لوحة ألوان متناسقة وأنيقة.
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#0b5394",      # أزرق عميق
    "secondary": "#134f5c",    # أزرق مخضر
    "hist": "#4a90d9",         # أزرق فاتح للمدرّج
    "hist_edge": "#2b6cb0",
    "observed": "#c0392b",     # أحمر للإحصائية المحسوبة
    "crit": "#e69138",         # برتقالي للقيم الحرجة
    "coint": "#5fa052",        # أخضر — تكامل مشترك
    "nocoint": "#c0392b",      # أحمر — لا تكامل
    "incon": "#e1b12c",        # أصفر — غير حاسم
    "degen": "#e08e0b",        # برتقالي — متدهور
    "grid": "#dfe6ee",
    "text": "#1a1a2e",
}

_DECISION_COLOR = {
    "تكامل_مشترك": PALETTE["coint"],
    "لا_تكامل_مشترك": PALETTE["nocoint"],
    "غير_حاسم": PALETTE["incon"],
    "تكامل_زائف_متدهور": PALETTE["degen"],
}

_PREFERRED_FONTS = ["Tahoma", "Arial", "Segoe UI", "Times New Roman", "DejaVu Sans"]


def _apply_style() -> None:
    """ضبط نمط رسومي أنيق وخط يدعم العربية."""
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in _PREFERRED_FONTS if f in available), None)
    if chosen:
        mpl.rcParams["font.family"] = chosen
    mpl.rcParams.update(
        {
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#b8c4d0",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "text.color": PALETTE["text"],
            "figure.dpi": 110,
        }
    )


def _save_or_return(fig, path: Optional[str]):
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# 1) Bootstrap null distributions — توزيعات البوتستراب الصفرية.
# ---------------------------------------------------------------------------
def plot_bootstrap_distributions(result, path: Optional[str] = None):
    """مدرّجات تكرارية للتوزيعات الصفرية الثلاثة مع الإحصائية المحسوبة والقيم الحرجة."""
    import matplotlib.pyplot as plt

    _apply_style()
    boot = result.bootstrap
    a = result.decision_level

    panels = [
        (STAT_LABELS_AR["f_overall"], boot.dist_f_overall,
         result.statistics.f_overall, boot.crit_f_overall.get(a), "right"),
        (STAT_LABELS_AR["t_dep"], boot.dist_t_dep,
         result.statistics.t_dep, boot.crit_t_dep.get(a), "left"),
        (STAT_LABELS_AR["f_ind"], boot.dist_f_ind,
         result.statistics.f_ind, boot.crit_f_ind.get(a), "right"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (label, dist, obs, crit, tail) in zip(axes, panels):
        dist = np.asarray(dist)
        dist = dist[np.isfinite(dist)]
        if dist.size:
            ax.hist(dist, bins=40, color=PALETTE["hist"], edgecolor=PALETTE["hist_edge"],
                    alpha=0.85, density=True, linewidth=0.6)
        if obs is not None and np.isfinite(obs):
            ax.axvline(obs, color=PALETTE["observed"], linewidth=2.4,
                       label=ع("الإحصائية المحسوبة"))
        if crit is not None and np.isfinite(crit):
            ax.axvline(crit, color=PALETTE["crit"], linewidth=2.0, linestyle="--",
                       label=ع(f"القيمة الحرجة ({pct(a)})"))
        ax.set_title(ع(label))
        ax.set_xlabel(ع("القيمة"))
        ax.set_ylabel(ع("الكثافة"))
        leg = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
        for txt in leg.get_texts():
            txt.set_fontsize(9)

    fig.suptitle(ع("توزيعات البوتستراب الصفرية لإحصائيات الاختبار"),
                 fontsize=15, fontweight="bold", color=PALETTE["primary"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_or_return(fig, path)


# ---------------------------------------------------------------------------
# 2) Decision comparison — مقارنة القرار.
# ---------------------------------------------------------------------------
def plot_decision(result, path: Optional[str] = None):
    """رسم أعمدة يقارن الإحصائيات المحسوبة بالقيم الحرجة بالبوتستراب عند مستوى القرار."""
    import matplotlib.pyplot as plt

    _apply_style()
    a = result.decision_level
    st = result.statistics
    boot = result.bootstrap

    # use |t| so all three bars are comparable on a positive axis
    labels = [STAT_LABELS_AR["f_overall"], STAT_LABELS_AR["t_dep"], STAT_LABELS_AR["f_ind"]]
    obs = [st.f_overall, abs(st.t_dep), st.f_ind]
    crit = [
        boot.crit_f_overall.get(a, np.nan),
        abs(boot.crit_t_dep.get(a, np.nan)),
        boot.crit_f_ind.get(a, np.nan),
    ]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    dec_color = _DECISION_COLOR.get(result.decision.code, PALETTE["primary"])

    b1 = ax.bar(x - w / 2, obs, w, color=dec_color, edgecolor="white",
                label=ع("الإحصائية المحسوبة"))
    b2 = ax.bar(x + w / 2, crit, w, color=PALETTE["crit"], edgecolor="white",
                label=ع(f"القيمة الحرجة ({pct(a)})"))

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ع(l) for l in labels], fontsize=10)
    ax.set_ylabel(ع("القيمة (|t| للإحصائية t)"))
    ax.set_title(ع(f"القرار: {result.decision.label_ar} — عند {pct(a)}"),
                 color=dec_color)
    leg = ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_fontsize(9)
    fig.tight_layout()
    return _save_or_return(fig, path)


# ---------------------------------------------------------------------------
# 3) Time-series of the variables — السلاسل الزمنية.
# ---------------------------------------------------------------------------
def plot_series(result, path: Optional[str] = None):
    """رسم السلاسل الزمنية للمتغيرات (التابع والمستقلة) إن توفّرت البيانات."""
    import matplotlib.pyplot as plt

    data = getattr(result, "source_data", None)
    if data is None or data.empty:
        raise ValueError("البيانات الأصلية غير متاحة لرسم السلاسل الزمنية.")

    _apply_style()
    cols = list(data.columns)
    colors = [PALETTE["primary"], PALETTE["coint"], PALETTE["degen"],
              PALETTE["secondary"], PALETTE["incon"], PALETTE["nocoint"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, c in enumerate(cols):
        style = "-" if i == 0 else "--"
        lw = 2.4 if i == 0 else 1.8
        ax.plot(data.index, data[c].values, style, linewidth=lw,
                color=colors[i % len(colors)], label=ع(str(c)))

    ax.set_title(ع("السلاسل الزمنية للمتغيرات"), color=PALETTE["primary"])
    ax.set_xlabel(ع("الزمن"))
    ax.set_ylabel(ع("القيمة"))
    leg = ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_fontsize(10)
    fig.tight_layout()
    return _save_or_return(fig, path)


# ---------------------------------------------------------------------------
# 4) Fake/degenerate comparison — مقارنة التكامل الزائف (مشروط مقابل غير مشروط).
# ---------------------------------------------------------------------------
def plot_fake_cointegration(result, path: Optional[str] = None):
    """مقارنة إحصائية F للمتغيرات المستقلة في النموذجين المشروط وغير المشروط."""
    import matplotlib.pyplot as plt

    _apply_style()
    a = result.decision_level
    boot = result.bootstrap

    labels = [ع("مشروط"), ع("غير مشروط")]
    obs = [result.statistics.f_ind, result.f_ind_uncond]
    crit = [boot.crit_f_ind.get(a, np.nan), boot.crit_f_ind_uncond.get(a, np.nan)]

    x = np.arange(2)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w / 2, obs, w, color=PALETTE["primary"], edgecolor="white",
           label=ع("الإحصائية المحسوبة"))
    ax.bar(x + w / 2, crit, w, color=PALETTE["crit"], edgecolor="white",
           label=ع(f"القيمة الحرجة ({pct(a)})"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ع("إحصائية F للمتغيرات المستقلة"))
    flag = result.bootstrap.fake_cointegration.get(a, False)
    note = "⚠ تكامل زائف محتمل" if flag else "لا مؤشر على تكامل زائف"
    ax.set_title(ع(f"فحص التكامل الزائف/المتدهور — {note}"),
                 color=(PALETTE["degen"] if flag else PALETTE["secondary"]))
    leg = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
    for txt in leg.get_texts():
        txt.set_fontsize(9)
    fig.tight_layout()
    return _save_or_return(fig, path)
