"""The run dashboard: numbers equal the files, offline, escaped, content-secured (ADR-017)."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import html
import json
import os
import re
import shutil
import socket
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout
from areneon_ml.loop.__main__ import main
from areneon_ml.loop.records import FEATURES_FAMILY
from areneon_ml.progress.looks import LookError, looks_path, read_looks
from areneon_ml.progress.stats import (
    bootstrap_stream_key,
    match_pair_scores,
    pair_interval,
)
from areneon_ml.report import charts, page
from areneon_ml.report.command import inspector_links, report_run
from areneon_ml.report.model import (
    GenerationFacts,
    ReportError,
    ReportModel,
    Tally,
    read_report_model,
    read_train_report,
)
from areneon_ml.report.render import (
    BANNER,
    CLAIM,
    CONDITIONAL,
    CONDITIONAL_PAIRS,
    SEEN,
    Links,
    build_report,
)

from tests.ml.report.support import (
    FIXTURE,
    HOSTILE_CAUSE,
    TRAIN_REPORT,
    fixture_copy,
    make_run,
    progress_root,
)

HOSTILE = "</script><img src=x onerror=alert(1)>" + chr(0x2028) + "\"'&<!--"


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: Counter[str] = Counter()
        self.attrs: set[str] = set()
        self.values: list[str] = []
        self.text: list[str] = []
        self.first_in_head: str | None = None
        self.bodies: list[tuple[str, str]] = []  # (tag, text) of every script and style element
        self._open: str | None = None
        self._in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags[tag] += 1
        if tag in ("script", "style"):
            self._open = tag
            self.bodies.append((tag, ""))
        if tag == "head":
            self._in_head = True
        elif self._in_head and self.first_in_head is None:
            self.first_in_head = tag
        for name, value in attrs:
            self.attrs.add(name)
            self.values.append(value or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == self._open:
            self._open = None
        if tag == "head":
            self._in_head = False

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self.bodies[-1] = (self._open, self.bodies[-1][1] + data)
        self.text.append(data)


def parsed(text: str) -> Page:
    found = Page()
    found.feed(text)
    return found


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("r") / "run"
    make_run(path, generations=3)
    return path


@pytest.fixture(scope="module")
def root(run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return progress_root(run, tmp_path_factory.mktemp("p") / "progress")


def build(run_dir: Path, progress: Path | None = None, links: Links | None = None) -> str:
    return build_report(read_report_model(run_dir, progress), links)


# --- numbers equal a recomputation from the files ---


def test_fixture_run_numbers_equal_the_files(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path, train=True)
    html_text = build(run_dir)
    model = read_report_model(run_dir, None)
    assert [g.number for g in model.generations] == [0, 1, 2]
    for n in (1, 2):
        path = layout.generation_dir(run_dir, n)
        gen = json.loads((path / "generation.json").read_text())
        league = json.loads((path / "league-run.json").read_text())
        plan = json.loads((path / "plan.json").read_text())
        facts = model.generations[n]
        assert (facts.games, facts.played, facts.forfeits) == (
            len(gen["games"]),
            league["played"],
            league["forfeits"],
        )
        assert (facts.episodes, facts.examples) == (len(gen["episode_hashes"]), gen["examples"])
        selfs = [g for g, p in zip(gen["games"], plan, strict=True) if p["opponent"] == "self"]
        assert facts.self_play is not None
        assert facts.self_play.games == len(selfs)
        assert facts.self_play.ties == sum(g["winner"] is None for g in selfs)
        assert facts.self_play.turns == sum(g["turns"] for g in selfs)
        assert sorted(facts.baselines) == sorted(
            {p["opponent"]["policy_id"] for p in plan if p["opponent"] != "self"}
        )
        assert facts.train is not None
        assert facts.train.roles["battle"]["steps"] == TRAIN_REPORT["battle"]["steps"]
    assert "not recorded" in html_text  # wall-clock time
    assert "generations committed" in html_text


def test_progress_numbers_equal_the_records(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    assert sorted(model.progress) == [0, 1, 2]
    for g in (0, 1, 2):
        record = json.loads((root / f"g{g:06d}" / "progress.json").read_text())
        facts = model.progress[g]
        assert len(facts.matches) == len(record["matches"])
        for found, match in zip(facts.matches, record["matches"], strict=True):
            assert (found.wins, found.ties, found.losses) == (
                match["wins"],
                match["ties"],
                match["losses"],
            )
            interval = recomputed(record, match)
            assert found.interval_kind == "pairs"
            assert (found.score_ppm, found.low_ppm, found.high_ppm) == (
                (2 * match["wins"] + match["ties"]) * 1_000_000 // (2 * match["games"]),
                interval.low_ppm,
                interval.high_ppm,
            )
            assert found.turns_total == match["turns_total"]
            low = (interval.low_ppm // 1000) / 10
            assert f"{low:.1f}%" in text
    assert CLAIM in html_unescape(text)
    assert CONDITIONAL_PAIRS in html_unescape(text)
    assert CONDITIONAL not in html_unescape(text)  # no Wilson interval is on a both-seats page
    assert SEEN in html_unescape(text)
    assert BANNER in html_unescape(text)


def recomputed(record: dict[str, Any], match: dict[str, Any]) -> Any:
    """The pair-level interval of one match, from the record's own games."""
    key = bootstrap_stream_key(record["run_id"], record["generation"], f"match:{match['match_id']}")
    return pair_interval(match_pair_scores(record, match["match_id"]), key)


def html_unescape(text: str) -> str:
    return html.unescape(text)


def test_without_a_progress_root_the_page_says_not_evaluated(run: Path) -> None:
    text = build(run)
    assert "not evaluated yet" in text
    assert "Score vs" not in text
    assert BANNER in html_unescape(text)


def test_a_partly_evaluated_root_names_the_missing_generations(run: Path, tmp_path: Path) -> None:
    only = progress_root(run, tmp_path / "p", generations=(1,))
    text = html_unescape(build(run, only))
    assert "Evaluated generations: 1." in text
    assert "Not evaluated yet: 0, 2, 3." in text
    model = read_report_model(run, only)
    assert sorted(model.progress) == [1]


def test_an_empty_progress_root_degrades(run: Path, tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / ".lock").write_text("")
    text = build(run, tmp_path / "p")
    assert "holds no committed generation" in text


def test_the_fixture_run_without_train_reports_shows_none_not_zero(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path)
    model = read_report_model(run_dir, None)
    assert all(g.train is None for g in model.generations)
    text = build_report(model)
    assert "No train-report.json was recorded." in text


def test_a_forfeit_with_a_hostile_cause_is_text_only(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path, train=True, forfeit=True)
    model = read_report_model(run_dir, None)
    assert model.generations[1].forfeit_causes == {HOSTILE_CAUSE: 1}
    text = build_report(model)
    assert_inert(text, (HOSTILE_CAUSE,))


# --- strings from data are inert (ADR-017 4c) -----------------------------------------------

TEMPLATE_TAGS = {
    "html", "head", "meta", "title", "style", "body", "main", "footer", "h1", "h2", "h3", "p",
    "div", "b", "table", "caption", "thead", "tbody", "tr", "th", "td", "code", "span", "a",
    "svg", "line", "text", "polyline", "circle", "rect", "figure", "figcaption", "details",
    "summary", "i", "script",
}  # fmt: skip


def assert_inert(text: str, canaries: tuple[str, ...]) -> None:
    found = parsed(text)
    assert set(found.tags) <= TEMPLATE_TAGS, set(found.tags) - TEMPLATE_TAGS
    assert not {a for a in found.attrs if a.startswith("on")}
    assert "style" not in found.attrs
    assert text.count("</script") == 2  # the data block and the one script, nothing else
    assert "<!--" not in text
    for tag, body in found.bodies:
        assert tag in ("script", "style")
        assert chr(0x2028) not in body
        assert chr(0x2029) not in body
    for canary in canaries:
        assert canary not in text
        assert "<img" not in text
        assert "onerror" not in found.attrs


def hostile_model(run_dir: Path) -> Any:
    base = read_report_model(run_dir, None)
    g = base.generations[1]
    assert g.self_play is not None
    hostile = dataclasses.replace(
        g,
        forfeit_causes={HOSTILE: 1, "plain": 2},
        baselines={
            HOSTILE + "@v1": Tally(games=1, a=1, turns=3),
            "b@v1": Tally(games=1, b=1, turns=4),
        },
    )
    return dataclasses.replace(
        base,
        run_key=HOSTILE,
        lineage_id=HOSTILE,
        format_id=HOSTILE,
        generations=(base.generations[0], hostile),
        inputs={HOSTILE: "0" * 64},
    )


def test_hostile_strings_everywhere_are_inert(tmp_path: Path) -> None:
    model = hostile_model(fixture_copy(tmp_path, train=True))
    assert_inert(build_report(model), (HOSTILE,))


def test_canary_negative_control(tmp_path: Path) -> None:
    """The check does flag a page that does not escape: it is not vacuous."""
    model = hostile_model(fixture_copy(tmp_path, train=True))
    text = build_report(model)
    broken = text.replace("&lt;/script&gt;", "</script>", 1)
    with pytest.raises(AssertionError):
        assert_inert(broken, (HOSTILE,))
    with pytest.raises(AssertionError):
        assert_inert(text.replace("</main>", "<b onclick=x>y</b></main>", 1), ())


# --- offline and content-secured -------------------------------------------------------------


def all_pages(run: Path, root: Path) -> list[str]:
    return [build(run), build(run, root)]


def test_csp_is_first_and_matches_the_inline_blocks(run: Path, root: Path) -> None:
    for text in all_pages(run, root):
        found = parsed(text)
        assert found.first_in_head == "meta"
        policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', text)
        assert policy is not None
        content = policy[1].replace("&#x27;", "'")
        styles = [b for t, b in found.bodies if t == "style"]
        scripts = [b for t, b in found.bodies if t == "script"]
        assert len(styles) == 1
        assert len(scripts) == 2  # the data block and our one script
        for kind, body in (("style", styles[0]), ("script", scripts[1])):
            digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
            assert f"{kind}-src 'sha256-{digest}'" in content
        for part in (
            "default-src 'none'",
            "img-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        ):
            assert part in content
        assert text.count("<script>") == 1
        assert text.count("<style>") == 1
        assert text.count('<script type="application/json"') == 1
        assert re.search(r"https?:|//[a-z]|url\(|@import|<link|<img|<iframe", text) is None
        assert "style=" not in text


def test_the_script_uses_no_html_sinks() -> None:
    for sink in (
        "innerHTML",
        "insertAdjacentHTML",
        "document.write",
        "setAttribute",
        "eval(",
        "fetch",
    ):
        assert sink not in page.SCRIPT
    assert "textContent" in page.SCRIPT
    assert "<" not in page.SCRIPT.replace("<", "", 0) or "</" not in page.SCRIPT


def test_a_page_has_accessible_tables_and_charts(run: Path, root: Path) -> None:
    text = build(run, root)
    assert 'name="viewport"' in text
    assert "max-width:600px" in text
    assert "prefers-color-scheme:dark" in text
    assert re.search(r"minmax\(min\(100%,\s*320px\)", text) or "minmax(min(100%,320px)" in text
    for table in re.findall(r"<table>.*?</table>", text, re.S):
        assert 'scope="col"' in table
        assert 'scope="row"' in table or "<tbody></tbody>" in table
    svgs = re.findall(r"<svg.*?</svg>", text, re.S)
    assert svgs
    for svg in svgs:
        assert 'role="group"' in svg
        assert 'aria-label="' in svg
        assert "<title>" in svg
    for figure in re.findall(r"<figure>.*?</figure>", text, re.S):
        assert "<h3>" in figure


def test_the_build_needs_no_network(run: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert build(run, root)


def test_the_page_is_deterministic(run: Path, tmp_path: Path) -> None:
    root = progress_root(run, tmp_path / "progress")
    assert build(run, root) == build(run, root)
    out = tmp_path / "x" / "a.html"
    report_run(run, root, out)
    first = out.read_bytes()
    assert b"logged for this run: 1." in first
    # The same files give the same page; the second run is a look of its own, so the log differs.
    report_run(run, root, out)
    assert b"logged for this run: 2." in out.read_bytes()
    assert out.stat().st_mode & 0o777 == 0o644


def test_the_page_lists_every_input_hash(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    for rel, digest in model.inputs.items():
        assert digest == hashlib.sha256((_where(run, root, rel)).read_bytes()).hexdigest()
        assert rel in text
        assert digest in text
    assert "generations/g000001/train-report.json" in model.inputs
    assert "progress/g000001/progress.json" in model.inputs


def _where(run: Path, root: Path, rel: str) -> Path:
    if rel.startswith("fixtures/"):  # the committed move-class table and engine-data table
        return Path(__file__).resolve().parents[3] / rel
    return root / rel.removeprefix("progress/") if rel.startswith("progress/") else run / rel


# --- links to the inspector ---


def test_links_only_to_pages_that_exist(run: Path, tmp_path: Path) -> None:
    inspect = tmp_path / "inspect"
    (inspect / "replays" / "g000001").mkdir(parents=True)
    (inspect / "games-g000001.html").write_text("x")
    (inspect / "replays" / "g000001" / "game-3.html").write_text("x")
    (inspect / "replays" / "g000001" / "Bad Name.html").write_text("x")
    (inspect / "replays" / "g000001" / "notes.txt").write_text("x")
    (inspect / "replays" / "g000002").mkdir()
    out = tmp_path / "out" / "report.html"
    report_run(run, None, out, inspect)
    text = out.read_text()
    assert 'href="../inspect/games-g000001.html"' in text
    assert 'href="../inspect/replays/g000001/game-3.html"' in text
    assert "games-g000002" not in text
    assert "Bad%20Name" not in text
    assert "notes.txt" not in text
    assert text.count("<a href=") == 2


def test_inspector_links_edges(tmp_path: Path) -> None:
    assert inspector_links(None, tmp_path, [1]) == Links()
    with pytest.raises(ReportError, match="inspector directory"):
        inspector_links(tmp_path / "missing", tmp_path, [1])
    (tmp_path / "i" / "replays").mkdir(parents=True)
    (tmp_path / "i" / "games-g000001.html").mkdir()
    assert inspector_links(tmp_path / "i", tmp_path, [1]).games == {}


# --- refusals ---


def test_a_changed_file_is_refused_by_the_manifest(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path, train=True)
    path = layout.generation_dir(run_dir, 1) / "train-report.json"
    path.write_bytes(path.read_bytes().replace(b'"steps":100', b'"steps":101', 1))
    with pytest.raises(ReportError, match="differ from its manifest"):
        read_report_model(run_dir, None)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["battle"].update(extra=1), "does not know"),
        (lambda r: r["battle"].pop("entropy"), "lacks entropy"),
        (lambda r: r["battle"].update(entropy=float("nan")), "finite"),
        (lambda r: r["battle"].update(entropy=True), "finite"),
        (lambda r: r["battle"].update(steps=-1), "integer"),
        (lambda r: r.pop("preview"), "lacks preview"),
        (lambda r: r["battle"].update(epochs_run=1), "come together"),
        (lambda r: r["battle"].update(epochs_run=1, kl_stopped=1), "come together"),
        (lambda r: r["battle"].update(epochs_run=0, kl_stopped=True), "integer"),
        (lambda r: r["battle"].update(value_warmup=False), "value_warmup is true"),
        (lambda r: r["battle"].update(value_warmup=1), "value_warmup is true"),
        (lambda r: r["battle"].update(anchor_kl=0.1), "come together"),
        (lambda r: r["battle"].update(anchor_beta_micro=5), "come together"),
        (lambda r: r["battle"].update(anchor_kl=-0.1, anchor_beta_micro=5), "cannot be negative"),
        (lambda r: r["battle"].update(anchor_kl=True, anchor_beta_micro=5), "finite"),
        (lambda r: r["battle"].update(anchor_kl=0.1, anchor_beta_micro=-5), "integer"),
    ],
)
def test_the_train_report_is_read_strictly(change: Any, message: str) -> None:
    raw = json.loads(json.dumps(TRAIN_REPORT))
    change(raw)
    with pytest.raises(InspectError, match=message):
        read_train_report(raw)


def test_the_kl_stop_keys_are_optional_and_read_when_present() -> None:
    assert read_train_report(json.loads(json.dumps(TRAIN_REPORT))).kl_stops == {}
    raw = json.loads(json.dumps(TRAIN_REPORT))
    raw["battle"].update(epochs_run=2, kl_stopped=True)
    facts = read_train_report(raw)
    assert facts.kl_stops == {"battle": (2, True)}
    assert "epochs_run" not in facts.roles["battle"]


def test_the_value_warmup_key_is_optional_and_read_when_present() -> None:
    raw = json.loads(json.dumps(TRAIN_REPORT))
    raw["battle"].update(value_warmup=True)
    raw["preview"].update(value_warmup=True)
    facts = read_train_report(raw)
    assert "value_warmup" not in facts.roles["battle"]
    assert facts.kl_stops == {}


def test_the_anchor_keys_are_optional_and_read_when_present() -> None:
    plain = read_train_report(json.loads(json.dumps(TRAIN_REPORT)))
    assert plain.anchors == {}
    raw = json.loads(json.dumps(TRAIN_REPORT))
    raw["battle"].update(anchor_kl=0.125, anchor_beta_micro=50_000)
    raw["preview"].update(anchor_kl=0.0, anchor_beta_micro=0)
    facts = read_train_report(raw)
    assert facts.anchors == {"battle": (0.125, 50_000), "preview": (0.0, 0)}
    assert "anchor_kl" not in facts.roles["battle"]


def test_the_dashboard_charts_the_anchor_kl_only_when_a_run_has_one(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path, train=True)
    assert "Anchor KL" not in build(run_dir)
    for n in (1, 2):
        path = layout.generation_dir(run_dir, n)
        raw = json.loads(json.dumps(TRAIN_REPORT))
        raw["battle"].update(anchor_kl=0.125 * n, anchor_beta_micro=100_000)
        raw["preview"].update(anchor_kl=0.25 * n, anchor_beta_micro=100_000)
        layout.write_json(path / "train-report.json", raw)
        manifest = layout.read_json(path / layout.MANIFEST)
        manifest["files"] = layout.hash_tree(path)
        layout.write_json(path / layout.MANIFEST, manifest)
    text = build(run_dir)
    assert "Anchor KL" in text
    assert "anchor KL: 0.25 (beta 0.1)" in text
    assert text.index("Approximate KL") < text.index("Anchor KL")


def test_a_private_lineage_is_refused(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path)
    config = layout.read_json(run_dir / layout.CONFIG)
    config["lineage_id"] = "private-main"
    layout.write_json(run_dir / layout.CONFIG, config)
    with pytest.raises(InspectError, match="not a public one"):
        read_report_model(run_dir, None)


def test_a_non_synthetic_checkpoint_record_is_refused(run: Path, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    path = layout.generation_dir(copy, 1) / "records" / "battle.json"
    record = layout.read_json(path)
    record["data_class"] = "private"
    layout.write_json(path, record)
    manifest_path = layout.generation_dir(copy, 1) / layout.MANIFEST
    manifest = layout.read_json(manifest_path)
    manifest["files"] = layout.hash_tree(layout.generation_dir(copy, 1))
    layout.write_json(manifest_path, manifest)
    with pytest.raises(ReportError):
        read_report_model(copy, None)


def test_a_gap_in_the_generations_is_refused(run: Path, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(run, copy)
    shutil.rmtree(layout.generation_dir(copy, 1))
    with pytest.raises(InspectError, match="no gap"):
        read_report_model(copy, None)


def test_inconsistent_loop_files_are_refused(tmp_path: Path) -> None:
    def seed(gen: dict[str, Any]) -> None:
        gen["games"][0]["seed"] = "sodium," + "0" * 32

    def forfeits(gen: dict[str, Any]) -> None:
        gen["forfeits"] = 3

    def episodes(gen: dict[str, Any]) -> None:
        gen["episode_hashes"].pop()

    def short(gen: dict[str, Any]) -> None:
        gen["games"].pop()

    for change, message in (
        (seed, "seed differs"),
        (forfeits, "game counts"),
        (episodes, "Episode counts"),
        (short, "differ in games"),
    ):
        sub = tmp_path / change.__name__
        sub.mkdir()
        run_dir = fixture_copy(sub)
        path = layout.generation_dir(run_dir, 1)
        gen = layout.read_json(path / "generation.json")
        change(gen)
        layout.write_json(path / "generation.json", gen)
        manifest = layout.read_json(path / layout.MANIFEST)
        manifest["files"] = layout.hash_tree(path)
        layout.write_json(path / layout.MANIFEST, manifest)
        with pytest.raises(ReportError, match=message):
            read_report_model(run_dir, None)


def test_league_files_of_another_generation_are_refused(tmp_path: Path) -> None:
    run_dir = fixture_copy(tmp_path)
    one, two = (layout.generation_dir(run_dir, n) for n in (1, 2))
    shutil.copy(two / "league-run.json", one / "league-run.json")
    shutil.copy(two / "plan.json", one / "plan.json")
    manifest = layout.read_json(one / layout.MANIFEST)
    manifest["files"] = layout.hash_tree(one)
    layout.write_json(one / layout.MANIFEST, manifest)
    with pytest.raises(ReportError):
        read_report_model(run_dir, None)


def test_progress_root_refusals(run: Path, root: Path, tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="does not exist"):
        read_report_model(run, tmp_path / "none")
    stray = tmp_path / "p1"
    shutil.copytree(root, stray)
    (stray / "notes").mkdir()
    with pytest.raises(ReportError, match="not a generation"):
        read_report_model(run, stray)
    beyond = tmp_path / "p2"
    shutil.copytree(root, beyond)
    shutil.copytree(beyond / "g000001", beyond / "g000009")
    with pytest.raises(ReportError, match="no such generation"):
        read_report_model(run, beyond)
    tampered = tmp_path / "p3"
    shutil.copytree(root, tampered)
    file = tampered / "g000001" / "progress.json"
    file.write_bytes(file.read_bytes().replace(b'"turns":5', b'"turns":6', 1))
    with pytest.raises(ReportError, match="differ from its manifest"):
        read_report_model(run, tampered)


def test_a_progress_directory_of_another_generation_is_refused(
    run: Path, root: Path, tmp_path: Path
) -> None:
    swapped = tmp_path / "p"
    shutil.copytree(root, swapped)
    (swapped / "g000001").rename(swapped / "tmp")
    (swapped / "g000002").rename(swapped / "g000001")
    (swapped / "tmp").rename(swapped / "g000002")
    with pytest.raises(ReportError):
        read_report_model(run, swapped)


def test_a_record_of_a_changed_generation_is_refused(run: Path, root: Path, tmp_path: Path) -> None:
    copy_run = tmp_path / "run"
    shutil.copytree(run, copy_run)
    path = layout.generation_dir(copy_run, 1)
    manifest = layout.read_json(path / layout.MANIFEST)
    manifest["note"] = "changed"
    layout.write_json(path / layout.MANIFEST, manifest)
    with pytest.raises(ReportError, match="not of the run's committed generation"):
        read_report_model(copy_run, root)


# --- the command ---


def test_the_command_writes_one_file(run: Path, root: Path, tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "d"
    assert (
        main(["report", "--run-dir", str(run), "--progress-dir", str(root), "--out", str(out)]) == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["generations"] == 4
    assert printed["progress_generations"] == 3
    assert [p.name for p in out.iterdir()] == ["index.html"]
    default = run.with_name(run.name + ".report.html")
    try:
        assert main(["report", "--run-dir", str(run)]) == 0
        assert default.is_file()
    finally:
        default.unlink(missing_ok=True)


def test_the_command_refuses_with_exit_3(run: Path, tmp_path: Path, capsys: Any) -> None:
    assert main(["report", "--run-dir", str(run), "--out", str(run / "report.html")]) == 3
    assert "must not be written inside" in capsys.readouterr().err
    assert main(["report", "--run-dir", str(tmp_path / "none")]) == 3


def test_a_failed_write_leaves_no_partial(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk"):
        report_run(run, None, tmp_path / "o.html")
    assert list(tmp_path.glob(".o.html*")) == []


# --- charts ---


def test_chart_edges() -> None:
    tips = charts.Tips()
    assert charts.nice_ticks(3.0, 3.0)[0] <= 3.0 <= charts.nice_ticks(3.0, 3.0)[-1]
    assert charts.nice_ticks(0.0, 1.0)[-1] >= 1.0
    empty = charts.line_chart("t", "c", [1], [], tips)
    assert "Not recorded" in empty
    assert "<svg" not in empty
    one = charts.line_chart(
        "t", "c", [1], [charts.Series("s", 1, "circle", (charts.Pt(1, 2.0, "tip", "2"),))], tips
    )
    assert one.count("<circle") == 1
    assert "legend" not in one
    gap = charts.line_chart(
        "t",
        "c",
        [1, 2, 3, 4, 5],
        [
            charts.Series(
                "a",
                1,
                "circle",
                tuple(charts.Pt(x, float(x), "t", str(x)) for x in (1, 2, 4, 5)),
            ),
            charts.Series("b", 2, "square", (charts.Pt(3, 1.0, "t", "1", 0.5, 1.5),)),
        ],
        tips,
        ref=2.0,
    )
    assert gap.count("<polyline") == 2
    assert "<rect" in gap
    assert "legend" in gap
    zero = charts.stacked_chart(
        "t", "c", [1, 2], [charts.Stack("s", 1, {1: 0, 2: 0})], tips, "none"
    )
    assert "none" in zero
    assert "<svg" not in zero
    many = charts.stacked_chart(
        "t",
        "c",
        [1, 2],
        [charts.Stack("s", 1, {1: 7, 2: 1}), charts.Stack("u", 2, {1: 0, 2: 3})],
        tips,
        "x",
    )
    assert many.count("<rect") == 3
    assert len(tips.items) == tips.items.index(tips.items[-1]) + 1 or tips.items


def test_many_forfeit_causes_fold_into_other(tmp_path: Path) -> None:
    base = read_report_model(fixture_copy(tmp_path, train=True), None)
    causes = {f"cause-{i}": 1 for i in range(9)} | {"not recorded": 2}
    g = dataclasses.replace(base.generations[1], forfeits=11, forfeit_causes=causes)
    text = build_report(dataclasses.replace(base, generations=(base.generations[0], g)))
    assert "other causes (3)" in text
    assert "cause not recorded" in text
    assert text.count("cause-0") >= 1


def test_a_run_with_only_the_seeded_pair(tmp_path: Path) -> None:
    base = read_report_model(fixture_copy(tmp_path), None)
    only = dataclasses.replace(base, generations=(base.generations[0],))
    text = build_report(only)
    assert "no generation with games yet" in text.lower() or "No generation with games" in text
    assert isinstance(only.generations[0], GenerationFacts)


def test_the_committed_fixture_is_read_without_copying() -> None:
    model = read_report_model(FIXTURE, None)
    assert model.run_key == "inspector-fixture"


# --- the interval's order and the progress strings ---


def _low(ppm: int) -> str:
    tenths = ppm // 1000
    return f"{tenths // 10}.{tenths % 10}%"


def _high(ppm: int) -> str:
    tenths = (ppm + 999) // 1000
    return f"{tenths // 10}.{tenths % 10}%"


def test_the_pair_interval_reads_low_then_high(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    found = parsed(text)
    tips = json.loads(next(b for t, b in found.bodies if t == "script"))["tips"]
    for g, facts in model.progress.items():
        for m in facts.matches:
            record = json.loads((root / f"g{g:06d}" / "progress.json").read_text())
            match = next(x for x in record["matches"] if x["match_id"] == m.match_id)
            interval = recomputed(record, match)
            assert (m.low_ppm, m.high_ppm) == (interval.low_ppm, interval.high_ppm)
            low, high = _low(interval.low_ppm), _high(interval.high_ppm)
            assert interval.low_ppm <= m.score_ppm <= interval.high_ppm
            row = re.search(
                rf'<tr><th scope="row">{g}</th><td class="txt">{m.match_id}[^\n]*?</tr>', text
            )
            assert row is not None
            assert f"{low} to {high}" in row[0]
            tip = f"Generation {g}, {m.mode} vs {m.match_id.removesuffix('-' + m.mode)}: "
            assert any(
                t.startswith(tip)
                and f"(pair bootstrap 95%, descriptive {low} to {high}, conditional" in t
                for t in tips
            )
    first = re.search(r"(\d+\.\d)% to (\d+\.\d)%", text)
    assert first is not None
    assert float(first[1]) <= float(first[2])


def test_a_version_1_page_keeps_the_wilson_wording(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    old = {
        g: dataclasses.replace(
            facts,
            matches=tuple(
                dataclasses.replace(m, interval_kind="wilson", low_ppm=0, high_ppm=1_000_000)
                for m in facts.matches
            ),
        )
        for g, facts in model.progress.items()
    }
    text = html_unescape(build_report(dataclasses.replace(model, progress=old)))
    assert CONDITIONAL in text
    assert CONDITIONAL_PAIRS not in text
    assert "Wilson 95%, conditional on the plan" in text
    assert "(Wilson 95% 0.0% to 100.0%, conditional on the fixed validation plan)" in text


def test_the_score_by_seat_and_the_previous_cells(run: Path, root: Path) -> None:
    model = read_report_model(run, root)
    text = build_report(model)
    for g, facts in model.progress.items():
        record = json.loads((root / f"g{g:06d}" / "progress.json").read_text())
        for m in facts.matches:
            for seat, won, tied, lost in m.seats:
                rows = [r for r in record["games"] if r["match_id"] == m.match_id]
                mine = [r for r in rows if r["candidate_seat"] == seat]
                outcomes = [r["outcome"] for r in mine]
                assert won + tied + lost == len(mine)
                assert won == sum(
                    o.get("winner") == "candidate" or o.get("loser") == "opponent" for o in outcomes
                )
            assert sum(w + t + lo for _, w, t, lo in m.seats) == m.games
    assert "Score by seat (diagnostic)" in text
    assert "not applicable" in text  # generation 0 has no previous generation
    assert "Lines join looks, not a trend." in text


def hostile_progress(run: Path, root: Path) -> Any:
    model = read_report_model(run, root)
    facts = model.progress[1]
    hostile = dataclasses.replace(
        facts.matches[0],
        match_id=HOSTILE + "-sample",
        mode="sample",
        forfeits=((HOSTILE, HOSTILE, 1),),
        seen_label=HOSTILE,
    )
    previous = dataclasses.replace(
        facts.matches[1], match_id="previous-argmax", mode="argmax", seen_label=HOSTILE
    )
    changed = dataclasses.replace(facts, matches=(hostile, previous))
    return dataclasses.replace(model, progress={**model.progress, 1: changed})


def test_hostile_progress_strings_are_inert(run: Path, root: Path) -> None:
    assert_inert(build_report(hostile_progress(run, root)), (HOSTILE,))


def test_hostile_progress_negative_control(run: Path, root: Path) -> None:
    text = build_report(hostile_progress(run, root))
    for old in ("&lt;img src=x onerror=alert(1)&gt;", "&lt;/script&gt;"):
        broken = text.replace(old, old.replace("&lt;", "<").replace("&gt;", ">"), 1)
        with pytest.raises(AssertionError):
            assert_inert(broken, (HOSTILE,))


def test_a_symlinked_replays_directory_is_not_followed(run: Path, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere" / "g000001"
    real.mkdir(parents=True)
    (real / "game-1.html").write_text("x")
    inspect = tmp_path / "inspect"
    inspect.mkdir()
    (inspect / "replays").symlink_to(real.parent, target_is_directory=True)
    assert inspector_links(inspect, tmp_path, [1]).replays == {}


def test_the_report_logs_a_look_and_shows_the_count_and_the_log_hash(
    run: Path, tmp_path: Path
) -> None:
    root = progress_root(run, tmp_path / "progress")
    before = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    out = tmp_path / "o" / "r.html"
    report_run(run, root, out)
    log = looks_path(root)
    assert read_looks(log).count == 1
    text = out.read_text()
    assert "Looks at validation results logged for this run: 1." in text
    assert "(looks logged: 1)" in text
    assert "never held-out evidence" in text
    assert hashlib.sha256(log.read_bytes()).hexdigest() in text
    assert {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()} == before
    # No progress directory, or nothing evaluated: no result is shown, so no look is logged.
    report_run(run, None, tmp_path / "o" / "n.html")
    assert read_looks(log).count == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    report_run(run, empty, tmp_path / "o" / "e.html")
    assert not looks_path(empty).exists()


def test_a_tampered_log_stops_the_report_before_it_writes(run: Path, tmp_path: Path) -> None:
    root = progress_root(run, tmp_path / "progress")
    looks_path(root).write_bytes(b"x\n")
    out = tmp_path / "o.html"
    with pytest.raises(LookError):
        report_run(run, root, out)
    assert not out.exists()


def test_a_pair_that_reads_a_feature_set_is_named_by_its_family() -> None:
    # ADR-022 §6: the dashboard says which family a pair is in; a pair without a set is unchanged.
    plain = ReportModel("r", "l", "f", "0" * 64, (), {}, False)
    named = ReportModel("r", "l", "f", "0" * 64, (), {}, False, family=FEATURES_FAMILY)
    assert "Policy family" not in build_report(plain)
    page = build_report(named)
    assert "Policy family: " + html.escape(FEATURES_FAMILY) in page
    assert "latency includes the probe" in page


def test_the_entropy_weight_key_is_optional_and_an_integer_when_present() -> None:
    plain = read_train_report(json.loads(json.dumps(TRAIN_REPORT)))
    assert "entropy_coef_micro" not in plain.roles["battle"]
    raw = json.loads(json.dumps(TRAIN_REPORT))
    raw["battle"]["entropy_coef_micro"] = 5000
    raw["preview"]["entropy_coef_micro"] = 5000
    read_train_report(raw)
    raw["battle"]["entropy_coef_micro"] = 0.5
    with pytest.raises(InspectError, match="integer"):
        read_train_report(raw)
