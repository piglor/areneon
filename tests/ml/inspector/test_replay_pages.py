"""The replay page: offline, escaped, content-secured (ADR-017 sections 4b and 4c)."""

# ruff: noqa: E501  (page and command assertions are kept whole so they read as written)

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector import page, replay, replay_page
from areneon_ml.inspector.readers import InspectError

from tests.ml.inspector.support import FIXTURE, ROOT
from tests.ml.inspector.test_pages import LS, PS, audit

ENGINE_DATA = ROOT / "fixtures" / "showdown" / "gen9championsvgc2026regmc" / "engine-data.json"
NICK = "</script><img src=x onerror=alert(1)>"
LOG = "a </script> b <!-- c " + LS + " d " + PS + " e \"q\" 's' & <b>"
POLICY = "policy</script><svg onload=alert(3)>"
RUN = "run\"'</script><svg onload=alert(2)>" + LS + PS
CANARIES = (NICK, LOG, POLICY, RUN)
TAGS = {
    "html", "head", "meta", "title", "style", "body", "header", "nav", "a", "main", "h1", "h2", "p",
    "div", "noscript", "footer", "script",
}  # fmt: skip
ATTRS = {"charset", "class", "content", "href", "http-equiv", "id", "lang", "name", "type"}


def table() -> dict[str, Any]:
    return replay_page.read_engine_data(ENGINE_DATA, "gen9championsvgc2026regmc")


def game(index: int = 0) -> replay.GameReplay:
    return replay.read_games(FIXTURE, 1, [index])[1][0]


def build(g: replay.GameReplay | None = None) -> str:
    g = g or game()
    return replay_page.replay_page(g, replay_page.dex_for(table(), g))


def hostile() -> replay.GameReplay:
    g = game()
    views = []
    for view in g.views:
        lines = list(view.lines)
        lines.insert(len(lines) // 2, f"|-message|{LOG}")
        lines.insert(len(lines) // 2, f"|{NICK}|{LOG}")
        team = tuple({**s, "name": NICK, "item": LOG} for s in view.team)
        decisions = tuple(dataclasses.replace(d, policy=POLICY) for d in view.decisions)
        views.append(dataclasses.replace(view, lines=tuple(lines), team=team, decisions=decisions))
    return dataclasses.replace(g, run_key=RUN, views=tuple(views))


def problems(raw: str) -> list[str]:
    found = []
    a = audit(raw)
    if not set(a.tags) <= TAGS:
        found.append(f"elements the template does not have: {sorted(set(a.tags) - TAGS)}")
    if not {name for _, name, _ in a.attrs} <= ATTRS:
        found.append("an attribute the template does not have")
    if any(name.startswith("on") or name == "style" for _, name, _ in a.attrs):
        found.append("an event handler or style attribute")
    if any(
        name == "href" and re.fullmatch(r"(\.\./)*[a-z0-9_/.#-]+", value or "") is None
        for _, name, value in a.attrs
    ):
        found.append("a href that is not a local page")
    if raw.count("</script") != a.tags.count("script") or a.tags.count("script") != 2:
        found.append("a script end or start outside the template's own two")
    if raw.count("<!--") != 0:
        found.append("a comment opener in the page")
    ours = [s for s in a.scripts if s[0].get("type") != "application/json"]
    if len(ours) != 1 or ours[0][1] != replay_page.assets()[0]:
        found.append("the executable script is not exactly ours")
    data = [s for s in a.scripts if s[0].get("type") == "application/json"]
    if len(data) != 1 or data[0][0].get("id") != "replay-data":
        found.append("not exactly one data block")
    for _, body in data:
        if any(c in body for c in "<>&" + LS + PS):
            found.append("an unescaped character in a data block")
    found += [f"a canary appears verbatim in the markup: {c[:12]!r}" for c in CANARIES if c in raw]
    return found


def test_the_page_is_the_template_the_script_and_one_data_block() -> None:
    raw = build()
    assert problems(raw) == []
    a = audit(raw)
    assert a.tags.count("style") == 1
    script, style = replay_page.assets()
    # the script is the interpreter then the view, byte for byte
    web = ROOT / "packages" / "replay-view" / "web"
    assert script == (
        'var AR_MODE = "player";\n'
        + (web / "interpreter.cjs").read_text().rstrip("\n")
        + "\n"
        + (web / "view.cjs").read_text().rstrip("\n")
    )
    assert style.startswith(page.STYLE)
    assert (web / "replay.css").read_text().strip() in style


def test_the_data_block_holds_both_own_channels_and_nothing_else_of_the_game() -> None:
    g = game()
    raw = build(g)
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    data = json.loads(block)
    assert [v["player"] for v in data["views"]] == ["p1", "p2"]
    for sent, view in zip(data["views"], g.views, strict=True):
        assert sent["lines"] == list(view.lines)
        assert sent["team"] == list(view.team)
        assert [d["choice"] for d in sent["decisions"]] == [d.choice for d in view.decisions]
    assert set(data) == {"title", "views", "dex", "links", "notes"}
    # the table is cut to the species the game uses, and the terrains
    assert 0 < len(data["dex"]["species"]) < 60
    assert data["dex"]["terrains"] == [
        "electricterrain",
        "grassyterrain",
        "mistyterrain",
        "psychicterrain",
    ]
    names = {replay.to_id(s["species"]) for v in g.views for s in v.team}
    assert names <= set(data["dex"]["species"])


def test_the_csp_is_first_in_head_names_our_hashes_and_nothing_remote_is_in_the_page() -> None:
    raw = build()
    assert not re.search(r"https?:|//[a-z]", raw.replace("<!doctype", ""))
    head = raw[raw.index("<head>") + len("<head>") :].lstrip()
    assert head.startswith('<meta http-equiv="Content-Security-Policy"')
    content = re.search(r'content="([^"]+)"', raw)
    assert content is not None
    csp = dict(part.strip().split(" ", 1) for part in content[1].replace("&#x27;", "'").split(";"))
    assert csp["default-src"] == "'none'"
    assert csp["base-uri"] == "'none'"
    assert csp["form-action"] == "'none'"
    assert csp["img-src"] == "'none'"
    a = audit(raw)
    (ours,) = [s for attrs, s in a.scripts if attrs.get("type") != "application/json"]
    style = re.search(r"<style>(.*?)</style>", raw, re.S)
    assert style is not None
    for directive, source in (("script-src", ours), ("style-src", style[1])):
        digest = base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        assert csp[directive] == f"'sha256-{digest}'"
    assert "unsafe-inline" not in raw
    assert "unsafe-eval" not in raw
    assert [
        t for t in a.tags if t in ("link", "base", "iframe", "form", "img", "object", "embed")
    ] == []


def test_our_script_writes_only_through_safe_sinks() -> None:
    script, style = replay_page.assets()
    for sink in (
        "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "Function(",
        "fetch(", "XMLHttpRequest", "import(", "WebSocket", "sendBeacon",
    ):  # fmt: skip
        assert sink not in script
    assert not re.search(r"https?:|//[a-z]", script + style)
    assert not re.search(r"setAttribute\(\s*[\"'](style|on)", script, re.I)
    assert "url(" not in style
    assert "</script" not in script.lower()
    assert "<!--" not in script


def test_hostile_strings_stay_out_of_the_markup() -> None:
    raw = build(hostile())
    assert problems(raw) == []
    (block,) = [b for attrs, b in audit(raw).scripts if attrs.get("type") == "application/json"]
    data = json.loads(block)
    assert f"|-message|{LOG}" in data["views"][0]["lines"]
    assert f"|{NICK}|{LOG}" in data["views"][1]["lines"]
    assert data["views"][0]["team"][0]["name"] == NICK
    assert data["views"][0]["decisions"][0]["policy"] == POLICY
    assert RUN in "".join(audit(raw).text)  # the run key reaches the page only as escaped HTML text


def test_the_checker_catches_what_it_checks_for() -> None:
    """Negative controls: a page that splices a canary in unescaped, or changes our script, fails."""
    raw = build()
    assert problems(raw) == []
    assert any("canary" in p for p in problems(raw.replace("<h1>", f"<h1>{NICK}", 1)))
    assert any("handler" in p for p in problems(raw.replace("<h1>", '<h1 onclick="x">', 1)))
    assert any(
        "not a local page" in p
        for p in problems(
            raw.replace('<a href="../index.html"', '<a href="https://example.test/"', 1)
        )
    )
    assert any("comment" in p for p in problems(raw.replace("<h1>", "<h1><!-- x -->", 1)))
    assert any(
        "not exactly ours" in p
        for p in problems(raw.replace("</script>\n</body>", "alert(1)</script>\n</body>"))
    )
    assert any(
        "data block" in p or "script" in p
        for p in problems(
            raw.replace("</main>", '</main><script type="application/json">{"x":"<"}</script>')
        )
    )
    block = re.search(
        r'(<script type="application/json" id="replay-data">)(.*?)(</script>)', raw, re.S
    )
    assert block is not None
    leaked = raw.replace(block[0], block[1] + block[2].replace("a", "<", 1) + block[3])
    assert any("unescaped" in p for p in problems(leaked))


def test_a_page_is_a_pure_function_of_the_files() -> None:
    assert build() == build()
    assert build(game(0)) != build(game(1))


def test_the_page_builds_with_sockets_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = build()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    replay_page.assets.cache_clear()
    assert build() == expected


def test_the_links_are_relative_and_built_from_ids() -> None:
    raw = build()
    hrefs = re.findall(r'href="([^"]+)"', raw)
    assert hrefs == ["../../index.html", "../../games-g000001.html#g0", "../index.html"]
    assert replay_page.page_path(1, 12) == "replays/g000001/game-000012.html"
    with pytest.raises(ValueError, match="link id"):
        page.link_id("a b")
    index = replay_page.index_page({(1, 3), (1, 1), (2, 0)}, RUN)
    assert [m for m in re.findall(r'href="([^"]+)"', index)] == [
        "../index.html",
        "../games-g000001.html",
        "g000001/game-000001.html",
        "g000001/game-000003.html",
        "../games-g000002.html",
        "g000002/game-000000.html",
    ]
    assert replay_page.index_page(set()).count("None built yet") == 1
    assert RUN not in index


def test_the_engine_data_must_be_the_runs_export(tmp_path: Path) -> None:
    with pytest.raises(InspectError, match="another format"):
        replay_page.read_engine_data(ENGINE_DATA, "gen9other")
    bad = tmp_path / "e.json"
    for content in (
        "{not json",
        json.dumps({"format_id": "x", "species": [{"id": "a"}], "terrains": []}),
    ):
        bad.write_text(content)
        with pytest.raises(InspectError, match="not an engine-data export"):
            replay_page.read_engine_data(bad, "x")
    with pytest.raises(InspectError, match="not an engine-data export"):
        replay_page.read_engine_data(tmp_path / "missing.json", "x")


def test_an_asset_with_a_carriage_return_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("interpreter.cjs", "view.cjs", "replay.css"):
        (tmp_path / name).write_text("x\r\n" if name == "view.cjs" else "x\n", newline="")
    monkeypatch.setattr(replay_page, "ASSETS", tmp_path)
    replay_page.assets.cache_clear()
    try:
        with pytest.raises(InspectError, match="carriage return"):
            replay_page.assets()
    finally:
        monkeypatch.undo()
        replay_page.assets.cache_clear()


def test_a_shell_with_a_closing_tag_is_refused() -> None:
    with pytest.raises(ValueError, match="closing tag"):
        page.document("t", "", "", shell=page.Shell("a</script>b", "x", "page-data"))
    with pytest.raises(ValueError, match="closing tag"):
        page.document("t", "", "", shell=page.Shell("a", "x<!--y", "page-data"))
    assert "<h1>x</h1>" in page.document("t", "", "<h1>x</h1>")


RENDERER = (
    os.environ.get("ARENEON_RENDER_BIN")
    or shutil.which("google-chrome")
    or shutil.which("chromium")
)


@pytest.mark.skipif(
    os.environ.get("ARENEON_RENDER_TEST") != "1" or RENDERER is None,
    reason="the headless render is a manual check: ARENEON_RENDER_TEST=1 and a Chrome or Chromium",
)
def test_a_headless_render_draws_the_page_with_the_network_unreachable(tmp_path: Path) -> None:
    """Manual check (the ADR-017 section 4 headless render): the page runs under its CSP and draws
    the board with every host name unresolvable, so any request would fail the page."""
    assert RENDERER is not None
    target = tmp_path / "game.html"
    target.write_text(build(), encoding="utf-8")
    shown = {}
    for frame in (0, 60):
        run = subprocess.run(
            [
                RENDERER, "--headless=new", "--no-sandbox", "--disable-gpu", "--dump-dom",
                "--virtual-time-budget=4000", "--host-resolver-rules=MAP * ~NOTFOUND",
                f"{target.as_uri()}#f={frame}",
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )  # fmt: skip
        shown[frame] = run.stdout
    for dom in shown.values():
        assert dom.count('class="board"') == 2
        assert dom.count('<p class="flagnote"></p>') == 2  # no line was flagged in either view
        assert 'class="slot' in dom
    assert "Team Preview" in shown[0]
    assert "Turn " in shown[60]
    assert "ERR_" not in shown[60]
