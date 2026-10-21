"""The pages: offline, escaped, content-secured (ADR-017 sections 4 and 4c)."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import html
import json
import re
import socket
from collections import Counter
from html.parser import HTMLParser
from typing import Any

import pytest
from areneon_ml.inspector import charts, page, render
from areneon_ml.inspector.model import InspectModel
from areneon_ml.inspector.pool import read_pool
from areneon_ml.inspector.readers import Forfeit, GameRow

from tests.ml.inspector.support import FIXTURE, fixture_model, fixture_pool

LS = chr(0x2028)
PS = chr(0x2029)
HOSTILE_NAME = "</script><img src=x onerror=alert(1)>"
HOSTILE_LOG = "a </script> b <!-- c " + LS + " d " + PS + " e \"q\" 's' & <b>"
HOSTILE_LABEL = 'lab"el<x>'
HOSTILE_RUN = "run\"'</script><svg onload=alert(2)>" + LS + PS
HOSTILE_ATTR = 'x" onmouseover="alert(3)" a=\'b\' autofocus c="'
ATTRIBUTES = {
    "aria-label", "charset", "class", "content", "height", "hidden", "href", "http-equiv", "id",
    "lang", "name", "role", "rx", "text-anchor", "type", "viewbox", "width", "x", "x1", "x2", "y",
    "y1", "y2", "title",
}  # fmt: skip
CANARIES = (HOSTILE_NAME, HOSTILE_LOG, HOSTILE_LABEL, HOSTILE_RUN)
TEMPLATE_TAGS = {
    "html",
    "head",
    "meta",
    "title",
    "style",
    "body",
    "header",
    "nav",
    "a",
    "span",
    "main",
    "h1",
    "h2",
    "h3",
    "p",
    "div",
    "b",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "code",
    "details",
    "summary",
    "svg",
    "line",
    "rect",
    "text",
    "i",
    "input",
    "article",
    "br",
    "script",
    "footer",
}


class Audit(HTMLParser):
    """Everything a browser would build from a page: tags, attributes and text, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str, str | None]] = []
        self.text: list[str] = []
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self._open: tuple[dict[str, str | None], list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            self.attrs.append((tag, name, value))
        if tag == "script":
            self._open = (dict(attrs), [])

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._open is not None:
            self.scripts.append((self._open[0], "".join(self._open[1])))
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._open[1].append(data)
        else:
            self.text.append(data)


def audit(raw: str) -> Audit:
    parser = Audit()
    parser.feed(raw)
    parser.close()
    return parser


def problems(raw: str, canaries: tuple[str, ...] = CANARIES) -> list[str]:
    """What is wrong with a page: the rules of ADR-017 section 4c, as a checker."""
    found = []
    a = audit(raw)
    if not set(a.tags) <= TEMPLATE_TAGS:
        found.append(f"elements the template does not have: {sorted(set(a.tags) - TEMPLATE_TAGS)}")
    if not {name for _, name, _ in a.attrs} <= ATTRIBUTES:
        found.append("an attribute the template does not have")
    if any(name.startswith("on") or name == "style" for _, name, _ in a.attrs):
        found.append("an event handler or style attribute")
    if any(name in ("src", "href") and not _local(value) for _, name, value in a.attrs):
        found.append("a src or href that is not a local page")
    if raw.count("</script") != a.tags.count("script") or a.tags.count("script") > 2:
        found.append("a script end or start outside the template's own two")
    if raw.count("<!--") != 0:
        found.append("a comment opener in the page")
    ours = [s for s in a.scripts if s[0].get("type") != "application/json"]
    if len(ours) != 1 or ours[0][1] != page.SCRIPT:
        found.append("the executable script is not exactly ours")
    data = [s for s in a.scripts if s[0].get("type") == "application/json"]
    for _, body in data:
        if any(c in body for c in "<>&" + LS + PS):
            found.append("an unescaped character in a data block")
    for canary in canaries:
        if canary in raw:
            found.append(f"a canary appears verbatim in the markup: {canary[:12]!r}")
    return found


def _local(value: str | None) -> bool:
    return (
        value is not None
        and re.fullmatch(r"(#[a-z0-9_-]+|[a-z0-9_-]+\.html(#[a-z0-9_-]+)?)", value) is not None
    )


def hostile_model() -> InspectModel:
    """The fixture run with hostile strings in every field the pages show."""
    raw = json.loads((FIXTURE / "pool.json").read_text())
    for team in raw["teams"][:5]:
        team["labels"] = [{"label_id": HOSTILE_LABEL, "value": HOSTILE_LOG}]
        team["split"] = HOSTILE_LABEL
        for item in team["sets"]:
            item["name"] = HOSTILE_NAME
            item["item"] = HOSTILE_LOG
            item["moves"] = [HOSTILE_NAME, HOSTILE_LOG]
            item["ability"] = HOSTILE_LABEL
            item["mega_stone"] = True
    pool = read_pool(raw)
    model = dataclasses.replace(
        fixture_model(), pool=pool, run_key=HOSTILE_RUN, lineage_id=HOSTILE_LABEL
    )
    # The fixture's teams keep their hashes, so the join holds; forfeits carry hostile text.
    gens = []
    for gen in model.generations:
        games = []
        for index, game in enumerate(gen.games):
            if index == 0:
                row = GameRow(
                    game.row.game,
                    game.row.seed,
                    game.row.opponent,
                    "forfeit",
                    None,
                    0,
                    Forfeit("p1", HOSTILE_LOG, 3, HOSTILE_ATTR),
                )
                game = dataclasses.replace(game, row=row)  # noqa: PLW2901
            games.append(game)
        gens.append(
            dataclasses.replace(
                gen,
                games=tuple(games),
                forfeit_causes={HOSTILE_LOG: 1},
                pair_policy_id=HOSTILE_NAME,
            )
        )
    return dataclasses.replace(model, generations=tuple(gens))


def test_the_fixture_run_builds_clean_pages() -> None:
    pages = render.build_pages(fixture_model())
    assert set(pages) >= {
        "index.html",
        "distributions.html",
        "games-g000001.html",
        "games-g000002.html",
        "drawn-g000001.html",
        "drawn-g000002.html",
        "teams-0001.html",
    }
    for name, raw in pages.items():
        assert problems(raw) == [], name


def test_every_page_has_the_csp_first_in_head_and_nothing_remote() -> None:
    for name, raw in render.build_pages(fixture_model()).items():
        assert not re.search(r"https?:|//[a-z]", raw.replace("<!doctype", "")), name
        head = raw[raw.index("<head>") + len("<head>") :].lstrip()
        assert head.startswith('<meta http-equiv="Content-Security-Policy"'), name
        a = audit(raw)
        assert [
            t
            for t in a.tags
            if t in ("style", "link", "base", "iframe", "form", "img", "object", "embed")
        ] == ["style"]
        content = re.search(r'content="([^"]+)"', raw)
        assert content is not None
        csp = {
            name: sources
            for name, sources in (
                part.strip().split(" ", 1) for part in content[1].replace("&#x27;", "'").split(";")
            )
        }
        assert csp["default-src"] == "'none'"
        assert csp["base-uri"] == "'none'"
        assert csp["form-action"] == "'none'"
        assert csp["img-src"] == "'none'"
        for directive, source in (("script-src", page.SCRIPT), ("style-src", page.STYLE)):
            digest = base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
            assert csp[directive] == f"'sha256-{digest}'"
        assert "unsafe-inline" not in raw
        assert "unsafe-eval" not in raw


def test_our_script_writes_only_through_safe_sinks() -> None:
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "setAttribute",
        "eval(",
        "Function(",
        "fetch(",
        "XMLHttpRequest",
        "import(",
    ):
        assert sink not in page.SCRIPT
    assert not re.search(r"\bon[a-z]+\s*=", page.STYLE)
    assert "http" not in page.SCRIPT + page.STYLE
    assert "url(" not in page.STYLE


def test_hostile_strings_stay_text() -> None:
    model = hostile_model()
    pages = render.build_pages(model)
    seen: set[str] = set()
    for name, raw in pages.items():
        assert problems(raw) == [], name
        a = audit(raw)
        text = "".join(a.text)
        seen |= {c for c in CANARIES if c in text}
        for _, body in (s for s in a.scripts if s[0].get("type") == "application/json"):
            seen |= {c for c in CANARIES if c in json.loads(body)["run"]}
    assert {HOSTILE_NAME, HOSTILE_LOG, HOSTILE_RUN} <= seen
    teams = audit(pages["teams-0001.html"])
    assert HOSTILE_LABEL in "".join(teams.text)


def test_the_checker_catches_what_it_checks_for() -> None:
    """Negative control: a page that splices a canary in unescaped fails the same checks."""
    raw = render.build_pages(fixture_model())["index.html"]
    assert problems(raw) == []
    spliced = raw.replace("<h1>", f"<h1>{HOSTILE_NAME}", 1)
    assert any("canary" in p for p in problems(spliced))
    assert any("elements" in p or "script" in p for p in problems(spliced))
    handler = raw.replace("<h1>", '<h1 onclick="x">', 1)
    assert any("handler" in p for p in problems(handler))
    remote = raw.replace('<a href="distributions.html">', '<a href="https://example.test/">', 1)
    assert any("not a local page" in p for p in problems(remote))
    comment = raw.replace("<h1>", "<h1><!-- x -->", 1)
    assert any("comment" in p for p in problems(comment))
    other = raw.replace(page.SCRIPT, page.SCRIPT + "alert(1)")
    assert any("not exactly ours" in p for p in problems(other))
    block = raw.replace("</main>", '</main><script type="application/json">{"x":"<"}</script>')
    assert any("data block" in p or "script" in p for p in problems(block))


def test_json_blocks_escape_what_can_end_a_block() -> None:
    block = page.json_block({"k": "</script><!-- & " + LS + PS + " >"})
    inner = block[block.index(">") + 1 : block.rindex("</script>")]
    assert not any(c in inner for c in "<>&" + LS + PS)
    assert json.loads(inner) == {"k": "</script><!-- & " + LS + PS + " >"}
    with pytest.raises(ValueError, match="link id"):
        page.link_id("a b")
    with pytest.raises(ValueError, match="link id"):
        page.json_block({}, block_id="x onload=1")
    assert page.esc("<a href='x'>&\"") == "&lt;a href=&#x27;x&#x27;&gt;&amp;&quot;"


def test_the_pages_are_deterministic_and_a_pure_function_of_the_files() -> None:
    assert render.build_pages(fixture_model()) == render.build_pages(fixture_model())


def test_pages_build_with_sockets_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building needs no network: any attempt to open a socket fails the build."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert render.build_pages(fixture_model())["index.html"]


def test_teams_are_chunked_in_hash_order_and_every_link_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(render, "CHUNK", 25)
    model = fixture_model()
    pages = render.build_pages(model)
    chunks = [n for n in pages if n.startswith("teams-")]
    assert len(chunks) == (len(model.pool.teams) + 24) // 25 == 4
    order = []
    for name in sorted(chunks):
        order += re.findall(r'<article class="card" id="t-([0-9a-f]{64})"', pages[name])
    assert order == [t.hash for t in model.pool.teams] == sorted(order)
    anchors = {(n, h) for n in chunks for h in re.findall(r'id="t-([0-9a-f]{64})"', pages[n])}
    ids = {n: set(re.findall(r'id="([a-z0-9_-]+)"', raw)) for n, raw in pages.items()}
    for name, raw in pages.items():
        for target, anchor in re.findall(r'href="([a-z0-9_-]+\.html)(?:#([a-z0-9_-]+))?"', raw):
            assert target in pages, (name, target)
            if anchor:
                assert anchor in ids[target], (name, target, anchor)
    assert len(anchors) == len(model.pool.teams)


def test_the_numbers_on_the_pages_are_the_models() -> None:
    model = fixture_model()
    pages = render.build_pages(model)
    games = pages["games-g000001.html"]
    gen = model.generations[0]
    assert games.count('<tr id="g') == len(gen.games) == 5
    assert "self-play" in games
    assert "random-legal@v1 (" in games
    assert "actor-rule@v1 (" in games
    assert "Forfeits: 0 (none)" in games
    first = pages["teams-0001.html"]
    drawn = max(model.use.values(), key=lambda u: u.draws)
    assert f"Drawn {drawn.draws} time" in "".join(pages.values())
    assert "Not drawn in any game of this run." in first
    assert f"<b>{model.pool.records_before}</b>" in pages["index.html"]
    assert f"<b>{len(model.pool.teams)}</b>" in pages["index.html"]
    assert f"<b>{len(model.cumulative_drawn)}</b>" in pages["index.html"]


def test_a_rate_is_shown_only_from_ten_games() -> None:
    model = fixture_model()
    team = next(h for h, u in model.use.items() if u.draws)
    tally = model.use[team].results
    key = next(iter(tally))
    tally[key]["games"] = 9
    assert "&ndash;</td></tr>" in render._results_table(model.use[team])
    tally[key].update(games=1, win=10)
    tally[key]["games"] = 10
    assert re.search(r">\d+%</td></tr>", render._results_table(model.use[team]))
    assert render._results_table(type(model.use[team])()) == ""


def test_forfeits_show_their_cause_or_say_not_recorded() -> None:
    model = fixture_model()
    gen = model.generations[0]
    game = gen.games[0]
    recorded = dataclasses.replace(
        game,
        row=GameRow(
            0,
            game.row.seed,
            game.row.opponent,
            "forfeit",
            None,
            0,
            Forfeit("p2", "latency", 4, "slow"),
        ),
    )
    older = dataclasses.replace(
        game, row=GameRow(0, game.row.seed, game.row.opponent, "forfeit", None, 0, None)
    )
    pages = {
        t: render._game_row({h.hash: 0 for h in model.pool.teams}, g)
        for t, g in (("a", recorded), ("b", older))
    }
    assert '<td>p2</td><td title="slow">latency</td>' in pages["a"]
    assert pages["b"].count("not recorded") == 2
    tie = render._game_row({h.hash: 0 for h in model.pool.teams}, game)
    assert "<td>tie</td>" in tie
    assert "(forfeit" not in tie


def test_a_chart_shows_shares_with_labels_and_a_table() -> None:
    model = fixture_model()
    pool = charts.distribution((t, 1) for t in model.pool.teams)
    drawn = charts.distribution(
        (t, model.cumulative_drawn[t.hash])
        for t in model.pool.teams
        if t.hash in model.cumulative_drawn
    )
    svg = charts.bar_chart("Species", pool, drawn, "species", "Drawn")
    assert svg.count("<rect") == 2 * min(charts.TOP, len(charts.ordered(pool, drawn, "species")))
    assert 'role="img"' in svg
    assert "<title>" in svg
    assert "%" in svg
    assert "style=" not in svg
    empty = charts.Distribution(0, {k: Counter() for k, _ in charts.CATEGORIES})
    assert "No values." in charts.bar_chart("x", empty, empty, "mega", "d")
    table = charts.value_table(pool, drawn, "species", "Drawn")
    assert table.count("<tr>") == 1 + len(
        set(pool.counts["species"]) | set(drawn.counts["species"])
    )
    assert charts._clip("x" * 40).endswith("…")
    assert len(charts._clip("x" * 40)) == charts._MAX_LABEL
    assert fixture_pool().teams


def test_the_script_filters_only_through_ids_the_page_lists() -> None:
    raw = render.build_pages(fixture_model())["games-g000001.html"]
    a = audit(raw)
    data = next(
        json.loads(body) for attrs, body in a.scripts if attrs.get("type") == "application/json"
    )
    assert data["ids"] == [f"g{i}" for i in range(5)]
    assert data["noun"] == "games"
    assert 'id="filter"' in raw
    assert "hidden" in raw


def test_hostile_text_stays_inside_an_attribute_value_and_a_data_block() -> None:
    """A quote or apostrophe cannot end an attribute; U+2028 and U+2029 stay escaped in a block."""
    pages = render.build_pages(hostile_model())
    games = pages["games-g000001.html"]
    a = audit(games)
    titles = [value for _, name, value in a.attrs if name == "title"]
    assert HOSTILE_ATTR in titles
    assert not any(name.startswith("on") for _, name, _ in a.attrs)
    assert 'title="x&quot; onmouseover=&quot;' in games
    block = next(body for attrs, body in a.scripts if attrs.get("type") == "application/json")
    assert LS not in block
    assert PS not in block
    assert "\\u2028" in games
    assert "\\u2029" in games
    assert json.loads(block)["run"] == HOSTILE_RUN


def test_negative_control_quotes_left_unescaped_break_the_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escaping text but not quotes (the quote=False mutation) fails the same page checks."""

    def no_quotes(value: object) -> str:
        return html.escape(str(value), quote=False)

    monkeypatch.setattr(render, "esc", no_quotes)
    mutated = render.build_pages(hostile_model())["games-g000001.html"]
    found = problems(mutated)
    assert any("attribute the template does not have" in p or "event handler" in p for p in found)
    assert "onmouseover" in {name for _, name, _ in audit(mutated).attrs}


def test_negative_control_a_data_block_without_the_separator_escapes_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def naive(value: Any, block_id: str = "page-data") -> str:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for char, code in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e")):
            body = body.replace(char, code)
        return f'<script type="application/json" id="{block_id}">{body}</script>'

    monkeypatch.setattr(page, "json_block", naive)
    mutated = render.build_pages(hostile_model())["games-g000001.html"]
    assert any("data block" in p for p in problems(mutated))
