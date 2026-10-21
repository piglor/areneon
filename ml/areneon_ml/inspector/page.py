"""The page shell and its escaping rules (ADR-017 section 4c).

Every string that comes from data is escaped for the context it lands in:

* HTML text and double-quoted attribute values: ``&``, ``<``, ``>``, ``"`` and ``'`` (``esc``);
* a data block: JSON in ``<script type="application/json">`` with ``<``, ``>``, ``&``, U+2028 and
  U+2029 written as ``\\u003c`` and so on (``json_block``), read back by ``JSON.parse``;
* links are built only from ids that match ``[a-z0-9_-]+`` (``link_id``), never from data.

A page has one hashed ``<style>`` block and one hashed ``<script>`` of our own code with no data
spliced into it, and a ``<meta>`` Content-Security-Policy as the first child of ``<head>`` that
allows nothing else: ``default-src 'none'`` (so no fetch, image, frame or font), the two hashes,
``base-uri 'none'`` and ``form-action 'none'``. No attribute name, tag name or URL is built from data.
"""

# ruff: noqa: E501  (HTML, CSS and SVG template lines are kept whole so the markup reads as written)

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from typing import Any, NamedTuple

_LINK_ID = re.compile(r"[a-z0-9_-]+")

STYLE = """
:root{color-scheme:light dark;--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#dcdbd5;--panel:#f4f3ef;
--s1:#2a78d6;--s2:#eb6834;--link:#1c5cab;--bad:#d03b3b;--good:#0c7a0c}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#383835;--panel:#242422;
--s1:#3987e5;--s2:#d95926;--link:#86b6ef;--bad:#e66767;--good:#4fc24f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main,header.top{max-width:1100px;margin:0 auto;padding:0 16px}
header.top{padding-top:12px}
nav{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:14px;border-bottom:1px solid var(--line);padding-bottom:8px}
a{color:var(--link)}
h1{font-size:1.6rem;margin:16px 0 4px}h2{font-size:1.2rem;margin:28px 0 8px}h3{font-size:1rem;margin:16px 0 4px}
p.note,.muted{color:var(--muted)}p.note{margin:4px 0 12px;max-width:70ch}
code,.m{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;overflow-wrap:anywhere}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:4px 8px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--panel);position:sticky;top:0}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.stat{background:var(--panel);border-radius:6px;padding:8px 12px}.stat b{display:block;font-size:1.3rem}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0;background:var(--bg)}
.card[hidden]{display:none}.card h3{margin:0 0 4px}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}
.chip{border:1px solid var(--line);border-radius:10px;padding:0 8px;font-size:12px;color:var(--muted)}
.sets{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:6px 12px;margin:6px 0}
.set{font-size:13px;line-height:1.35;border-left:3px solid var(--line);padding-left:8px}
.set b{font-weight:600}.set.mega{border-left-color:var(--s2)}
.legend{display:flex;gap:16px;font-size:13px;margin:4px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.sw.s1,.b1{background:var(--s1);fill:var(--s1)}.sw.s2,.b2{background:var(--s2);fill:var(--s2)}
svg.chart{width:100%;height:auto;max-width:720px;display:block}
svg.chart text{fill:var(--fg);font:11px system-ui,sans-serif}svg.chart text.v{fill:var(--muted)}
svg.chart line{stroke:var(--line)}
details{margin:8px 0}summary{cursor:pointer;color:var(--link)}
input.filter{width:100%;max-width:420px;padding:6px 8px;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;margin:8px 0}
.win{color:var(--good)}.loss,.forfeit{color:var(--bad)}
footer{max-width:1100px;margin:32px auto;padding:0 16px;color:var(--muted);font-size:13px}
@media (forced-colors:active){.b1,.b2{fill:CanvasText}.b2{fill:Highlight}}
""".strip()

# Our own code, with no data in it. The data comes from the JSON block, through JSON.parse,
# and reaches the page through textContent and property writes only.
SCRIPT = """
(function () {
  var block = document.getElementById("page-data");
  var input = document.getElementById("filter");
  var count = document.getElementById("filter-count");
  if (!block || !input || !count) { return; }
  var data = JSON.parse(block.textContent);
  var nodes = data.ids.map(function (id) { return document.getElementById(id); });
  var texts = [];
  input.hidden = false;
  input.placeholder = "Filter " + data.noun + " (" + data.run + ")";
  input.addEventListener("input", function () {
    var q = input.value.toLowerCase().trim();
    var shown = 0;
    nodes.forEach(function (node, i) {
      if (!node) { return; }
      if (texts[i] === undefined) { texts[i] = node.textContent.toLowerCase(); }
      var show = q === "" || texts[i].indexOf(q) >= 0;
      node.hidden = !show;
      if (show) { shown += 1; }
    });
    count.textContent = shown + " of " + data.ids.length + " shown";
  });
})();
""".strip()


def esc(value: object) -> str:
    """Escape for HTML text and double-quoted attribute values."""
    return html.escape(str(value), quote=True)


def link_id(value: str) -> str:
    """An id that may be used in a link or an id attribute: only ``[a-z0-9_-]+``."""
    if _LINK_ID.fullmatch(value) is None:
        raise ValueError("not a link id")
    return value


def json_block(value: Any, block_id: str = "page-data") -> str:
    """A data block: JSON with ``<``, ``>``, ``&``, U+2028 and U+2029 escaped, never executed."""
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for char, code in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        (chr(0x2028), "\\u2028"),
        (chr(0x2029), "\\u2029"),
    ):
        body = body.replace(char, code)
    return f'<script type="application/json" id="{link_id(block_id)}">{body}</script>'


def sha256_source(text: str) -> str:
    return (
        "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"
    )


def csp(script: str = SCRIPT, style: str = STYLE) -> str:
    return (
        f"default-src 'none'; script-src {sha256_source(script)}; style-src {sha256_source(style)}; "
        "img-src 'none'; base-uri 'none'; form-action 'none'"
    )


class Shell(NamedTuple):
    """What a page carries besides its content: its one script, its one style block, its data id."""

    script: str = SCRIPT
    style: str = STYLE
    data_id: str = "page-data"


def document(  # noqa: PLR0913  (the shell is one extra, optional argument)
    title: str,
    nav: str,
    body: str,
    data: dict[str, Any] | None = None,
    footer: str = "",
    *,
    shell: Shell = Shell(),  # noqa: B008  (an immutable NamedTuple)
) -> str:
    """One page. ``nav``, ``body`` and ``footer`` are HTML the caller built with ``esc``.

    ``script`` and ``style`` are the page's one script and one style block (our code, no data in
    either); the CSP names exactly their hashes. A page with its own pair (the replay page) passes
    them here, so the shell, the escaping and the policy stay this module's.
    """
    script, style, data_id = shell
    for text, end in ((script, "</script"), (style, "</style")):
        if end in text.lower() or "<!--" in text:
            raise ValueError("a script or style block must not hold a closing tag or a comment")
    block = json_block(data, data_id) if data is not None else ""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(csp(script, style))}">\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)}</title>\n<style>{style}</style>\n</head>\n<body>\n"
        f'<header class="top"><nav>{nav}</nav></header>\n<main>\n{body}\n</main>\n'
        f"<footer>{footer}</footer>\n{block}\n<script>{script}</script>\n</body>\n</html>\n"
    )
