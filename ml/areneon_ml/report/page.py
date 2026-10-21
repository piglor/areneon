"""The report page shell: one hashed style, one hashed script, a CSP meta first in ``<head>``.

It reuses the inspector's escaping and data-block helpers (ADR-017 section 4c). The one script is
our own code with no data in it: it reads the JSON data block and writes text with ``textContent``
only. The style block is the inspector's plus the report's chart and layout rules.
"""

# ruff: noqa: E501  (CSS and HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

from typing import Any

from areneon_ml.inspector import page as inspector_page
from areneon_ml.inspector.page import esc, json_block, sha256_source

REPORT_CSS = """
:root{--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;--mute:#8a8984}
@media (prefers-color-scheme:dark){:root{--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--mute:#9a998f}}
.banner{border:1px solid var(--line);border-left:4px solid var(--s2);background:var(--panel);border-radius:6px;padding:8px 12px;margin:12px 0;font-weight:600}
.claim{border-left:3px solid var(--line);padding:2px 12px;margin:8px 0;max-width:75ch}
.stat b{overflow-wrap:anywhere;font-size:1.1rem}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}
figure{margin:0}figcaption{font-size:13px;color:var(--muted);margin-top:2px}
figure h3{margin:0 0 2px}
.readout{min-height:1.4em;font-size:13px;color:var(--muted)}
svg.chart{max-width:none}
svg.chart .axis{stroke:var(--line)}svg.chart .ref{stroke:var(--muted);stroke-dasharray:3 3}
svg.chart .ln{fill:none;stroke-width:2}
svg.chart .ring{stroke:var(--bg);stroke-width:2}
svg.chart .wh{stroke-width:2}
svg.chart circle:focus,svg.chart rect:focus{outline:2px solid var(--fg);outline-offset:1px}
svg.chart .l1{stroke:var(--s1)}svg.chart .l2{stroke:var(--s2)}svg.chart .l3{stroke:var(--s3)}svg.chart .l4{stroke:var(--s4)}svg.chart .l5{stroke:var(--s5)}svg.chart .l6{stroke:var(--s6)}svg.chart .l7{stroke:var(--s7)}svg.chart .l8{stroke:var(--s8)}svg.chart .l9{stroke:var(--mute)}
.f1{fill:var(--s1)}.f2{fill:var(--s2)}.f3{fill:var(--s3)}.f4{fill:var(--s4)}.f5{fill:var(--s5)}.f6{fill:var(--s6)}.f7{fill:var(--s7)}.f8{fill:var(--s8)}.f9{fill:var(--mute)}
.sw1{background:var(--s1)}.sw2{background:var(--s2)}.sw3{background:var(--s3)}.sw4{background:var(--s4)}.sw5{background:var(--s5)}.sw6{background:var(--s6)}.sw7{background:var(--s7)}.sw8{background:var(--s8)}.sw9{background:var(--mute)}
.legend{flex-wrap:wrap}
.legend .sq{border-radius:0}.legend .ci{border-radius:50%}
td.txt{white-space:normal;min-width:12ch}
@media (max-width:600px){main,header.top{padding:0 12px}.grid{grid-template-columns:repeat(auto-fill,minmax(130px,1fr))}}
""".strip()

STYLE = inspector_page.STYLE + "\n" + REPORT_CSS

# Our own code, with no data in it: it reads the data block and writes text only.
SCRIPT = """
(function () {
  var block = document.getElementById("page-data");
  if (!block) { return; }
  var tips = JSON.parse(block.textContent).tips;
  function show(event) {
    var target = event.target;
    if (!target || !target.getAttribute) { return; }
    var key = target.getAttribute("data-t");
    if (key === null || tips[Number(key)] === undefined) { return; }
    var figure = target.closest("figure");
    var out = figure ? figure.querySelector(".readout") : null;
    if (out) { out.textContent = tips[Number(key)]; }
  }
  document.addEventListener("pointerover", show);
  document.addEventListener("focusin", show);
})();
""".strip()


def csp() -> str:
    return (
        f"default-src 'none'; script-src {sha256_source(SCRIPT)}; style-src {sha256_source(STYLE)}; "
        "img-src 'none'; base-uri 'none'; form-action 'none'"
    )


def document(title: str, body: str, tips: list[str], footer: str) -> str:
    data: dict[str, Any] = {"tips": tips}
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(csp())}">\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        f"<main>\n{body}\n</main>\n"
        f"<footer>{footer}</footer>\n{json_block(data)}\n<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )
