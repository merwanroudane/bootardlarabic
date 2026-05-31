"""RTL Arabic HTML report generator.

مولّد تقرير HTML عربي بترتيب من اليمين إلى اليسار.
"""
from __future__ import annotations

import html
from typing import Optional

import pandas as pd

from ..utils.arabic_text import CASE_NAMES_AR, TERMS_AR, IC_NAMES_AR, REFERENCES
from ..utils.formatting import fmt, pct

_CSS = """
body { font-family: 'Segoe UI', 'Tahoma', sans-serif; direction: rtl;
       text-align: right; background:#fafafa; color:#1a1a1a; margin:2em; }
h1 { color:#0b5394; border-bottom:3px solid #0b5394; padding-bottom:.3em; }
h2 { color:#134f5c; margin-top:1.5em; }
table { border-collapse: collapse; margin:1em 0; width:100%; background:#fff; }
th, td { border:1px solid #ccc; padding:.5em .8em; text-align:center; }
th { background:#0b5394; color:#fff; }
tr:nth-child(even){ background:#f0f6fb; }
.decision { padding:1em; border-radius:8px; font-size:1.1em; font-weight:bold; }
.coint { background:#d9ead3; border:2px solid #6aa84f; }
.nocoint { background:#f4cccc; border:2px solid #cc0000; }
.incon { background:#fff2cc; border:2px solid #f1c232; }
.degen { background:#fce5cd; border:2px solid #e69138; }
.warn { background:#fff8e1; border-right:4px solid #f1c232; padding:.6em; margin:.4em 0; }
.advice { color:#555; font-size:.92em; }
.note { font-size:.85em; color:#777; }
footer { margin-top:2em; font-size:.85em; color:#666; border-top:1px solid #ccc; padding-top:1em; }
"""

_DECISION_CLASS = {
    "تكامل_مشترك": "coint",
    "لا_تكامل_مشترك": "nocoint",
    "غير_حاسم": "incon",
    "تكامل_زائف_متدهور": "degen",
}


def _df_to_html(df: pd.DataFrame, digits: int = 3) -> str:
    fmtd = df.copy()
    for c in fmtd.columns:
        fmtd[c] = fmtd[c].map(lambda v: fmt(v, digits))
    return fmtd.to_html(border=0, escape=False)


def build_html_report(result, path: Optional[str] = None) -> str:
    """Build an RTL Arabic HTML report; write to ``path`` if given. Returns HTML."""
    r = result
    dec = r.decision
    cls = _DECISION_CLASS.get(dec.code, "incon")

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html dir="rtl" lang="ar"><head><meta charset="utf-8">')
    parts.append(f"<title>{html.escape(TERMS_AR['test_name'])}</title>")
    parts.append(f"<style>{_CSS}</style></head><body>")

    parts.append(f"<h1>{html.escape(TERMS_AR['test_name'])}</h1>")
    parts.append("<table>")
    parts.append(f"<tr><th>{TERMS_AR['dependent']}</th><td>{html.escape(str(r.yvar))}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['independent']}</th><td>{html.escape('، '.join(r.xvar))}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['selected_case']}</th><td>{CASE_NAMES_AR.get(r.case, r.case)}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['n_obs']}</th><td>{r.n_obs}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['ardl_ic']}</th><td>{IC_NAMES_AR.get(r.ardl_ic, r.ardl_ic)}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['selected_lags']}</th><td>{r.diff_lags}</td></tr>")
    parts.append(f"<tr><th>{TERMS_AR['n_boot']}</th><td>{r.n_boot} (فعّالة: {r.bootstrap.n_boot_effective})</td></tr>")
    parts.append("</table>")

    parts.append(f"<h2>{TERMS_AR['final_decision']} (عند {pct(r.decision_level)})</h2>")
    parts.append(f'<div class="decision {cls}">{html.escape(dec.label_ar)}</div>')
    parts.append(f"<p>{html.escape(dec.detail_ar)}</p>")

    parts.append(f"<h2>{TERMS_AR['interpretation']}</h2>")
    parts.append("<p>" + html.escape(r.تفسير()).replace("\n", "<br>") + "</p>")

    parts.append(f"<h2>{TERMS_AR['test_statistics']}</h2>")
    parts.append(_df_to_html(r.جدول_الإحصائيات()))

    parts.append(f"<h2>{TERMS_AR['boot_crit']}</h2>")
    parts.append(_df_to_html(r.جدول_البوتستراب()))

    parts.append(f"<h2>{TERMS_AR['boot_pval']}</h2>")
    parts.append(_df_to_html(r.جدول_القيم_الاحتمالية()))

    parts.append(f"<h2>{TERMS_AR['pss_bounds']}</h2>")
    parts.append('<p class="note">' + html.escape(r.bounds.note) + "</p>")
    parts.append(_df_to_html(r.جدول_حدود_PSS()))

    smg_df = r.جدول_حدود_SMG()
    if not smg_df.empty:
        parts.append(f"<h2>{TERMS_AR['smg']}</h2>")
        parts.append(_df_to_html(smg_df))

    parts.append(f"<h2>{TERMS_AR['test_statistics']} — {TERMS_AR['dependent']} (المعاملات)</h2>")
    parts.append(_df_to_html(r.جدول_المعاملات()))

    if r.warnings_list:
        parts.append(f"<h2>{TERMS_AR['warnings']}</h2>")
        for w in r.warnings_list:
            parts.append(
                f'<div class="warn"><b>[{html.escape(w.severity)}]</b> '
                f'{html.escape(w.message_ar)}<br>'
                f'<span class="advice">← {html.escape(w.advice_ar)}</span></div>'
            )

    parts.append("<footer><b>المراجع:</b><ol>")
    for ref in REFERENCES:
        parts.append(f"<li>{html.escape(ref)}</li>")
    parts.append("</ol>")
    parts.append("قياسي BootARDL — تأليف د. مروان رودان (Dr. Merwan Roudane)")
    parts.append("</footer></body></html>")

    out = "\n".join(parts)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
    return out
