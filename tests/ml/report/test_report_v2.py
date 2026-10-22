"""The run dashboard on version 2 progress records: anchors, Preview ablation, a mixed root (#951).

Every number is recomputed from the stored files with the progress package's own functions; the
page-security properties (hostile strings inert, CSP, sockets blocked) are the same as the other
report tests, here on the new sections.
"""

from __future__ import annotations

import dataclasses
import html
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.progress import run as run_module
from areneon_ml.progress.ablation import ablation_summary
from areneon_ml.progress.run import summary_of
from areneon_ml.progress.stats import PPM
from areneon_ml.report.model import read_report_model
from areneon_ml.report.render import (
    ABLATION_NOTE,
    ANCHOR_NOTE,
    CONDITIONAL,
    CONDITIONAL_PAIRS,
    SEAT_CAPTION,
    SEAT_CAPTION_PAIRS,
    SEEN,
    build_report,
    signed_pct,
)

from tests.ml.progress.support import FakeProgressPlayer, committed_manifest, downgrade_to_v1
from tests.ml.report.support import _config, make_run
from tests.ml.report.test_report import HOSTILE, assert_inert


def played(run: Path, root: Path, generation: int, **kwargs: Any) -> None:
    fake = FakeProgressPlayer(_config(run), committed_manifest(run, 0), previous=False)
    games = kwargs.pop("games", 2)
    anchors = kwargs.pop("anchors", ())
    run_module.progress_generation(run, generation, root, fake, games, 1, anchors, 4, **kwargs)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("r2") / "run"
    make_run(path, generations=3)
    return path


@pytest.fixture(scope="module")
def mixed(run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One root: generation 0 a version 1 record, generation 1 version 2 with an anchor,
    generation 2 a version 2 Preview ablation."""
    root = tmp_path_factory.mktemp("p2") / "progress"
    played(run, root, 0)
    downgrade_to_v1(root / "g000000")
    played(run, root, 1, anchors=(0,))
    played(run, root, 2, games=4, preview_ablation=True)
    return root


def text_of(run: Path, root: Path) -> str:
    return html.unescape(build_report(read_report_model(run, root)))


def record(root: Path, generation: int) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((root / f"g{generation:06d}" / "progress.json").read_text())
    return found


def test_a_mixed_root_reads_each_record_by_its_own_version(run: Path, mixed: Path) -> None:
    model = read_report_model(run, mixed)
    assert [record(mixed, g)["schema_version"] for g in (0, 1, 2)] == [1, 2, 2]
    assert {m.interval_kind for m in model.progress[0].matches} == {"wilson"}
    assert {m.interval_kind for m in model.progress[1].matches} == {"pairs"}
    # the version 1 rows are the Wilson numbers of the files, unchanged
    for found, summary in zip(model.progress[0].matches, summary_of(record(mixed, 0)), strict=True):
        assert (found.low_ppm, found.high_ppm) == (
            summary["wilson_low_ppm"],
            summary["wilson_high_ppm"],
        )
        assert found.diff is None
        assert found.pairs == 0
    page = html.unescape(build_report(model))
    assert CONDITIONAL in page
    assert CONDITIONAL_PAIRS in page
    assert SEAT_CAPTION not in page  # a mixed page says how both kinds are played
    assert SEAT_CAPTION_PAIRS in page


def test_anchor_numbers_equal_a_recomputation_from_the_files(run: Path, mixed: Path) -> None:
    model = read_report_model(run, mixed)
    rec = record(mixed, 1)
    assert rec["plan"]["anchor_games_per_match"] == 4
    assert model.progress[1].anchor_games == 4
    anchors = [m for m in model.progress[1].matches if m.anchor_generation is not None]
    assert sorted(m.match_id for m in anchors) == ["anchor-g0-argmax", "anchor-g0-sample"]
    by_id = {s["match_id"]: s for s in summary_of(rec)}
    for m in anchors:
        stored = by_id[m.match_id]
        assert m.anchor_generation == 0
        assert m.pairs == m.games // 2 == 2
        assert m.diff == (stored["diff_mean_ppm"], stored["diff_low_ppm"], stored["diff_high_ppm"])
        # the paired difference is the pair score minus one half
        assert m.diff[0] == stored["score_ppm"] - PPM // 2
        assert m.seen_label == SEEN
    text = text_of(run, mixed)
    assert "Anchor comparisons (paired differences)" in text
    assert ANCHOR_NOTE in text
    assert "games per anchor match: 4" in text
    for m in anchors:
        assert m.diff is not None
        assert f"{signed_pct(m.diff[0])}" in text
        assert f"{signed_pct(m.diff[1], down=True)} to {signed_pct(m.diff[2], up=True)}" in text
    # the anchor rows (matches table and anchor table) carry the same-lineage label
    assert text.count(f'anchor-g0-argmax <span class="muted">({SEEN})') == 2


def test_a_page_without_an_anchor_has_no_anchor_table(
    run: Path, mixed: Path, tmp_path: Path
) -> None:
    root = tmp_path / "progress"
    played(run, root, 1)
    text = text_of(run, root)
    assert "Anchor comparisons" not in text
    assert "games per anchor match" not in text
    assert "Team Preview ablation" not in text


def test_the_ablation_is_shown_from_the_records_own_summary(run: Path, mixed: Path) -> None:
    model = read_report_model(run, mixed)
    rec = record(mixed, 2)
    facts = model.progress[2]
    assert facts.ablation == ablation_summary(rec) != {}
    variants = {m.variant for m in facts.matches}
    assert variants == {"learned", "random", "g0"}
    text = text_of(run, mixed)
    assert "Team Preview ablation" in text
    assert ABLATION_NOTE in text
    rows = facts.ablation["baselines"]["actor-rule"]
    # the stand-in: a random Preview loses every game, generation 0's wins every game, the learned
    # one splits each pair; so the differences to the learned Preview are -1/2 and +1/2
    assert rows["random"]["minus_learned"]["mean_ppm"] == -500_000
    assert rows["g0"]["minus_learned"]["mean_ppm"] == 500_000
    assert rows["g0"]["minus_learned"]["better"] == 2
    assert "-50.0%" in text
    assert "+50.0%" in text
    assert "2 / 0 / 0" in text  # g0: both pairs better
    # ablation matches are not points of the score charts: no chart is named for them
    assert "Score vs preview-" not in text
    # and a baseline's own chart is not drawn for an ablation generation's variants
    assert "Score vs actor-rule" in text


def test_a_version_1_only_page_has_none_of_the_new_sections(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "progress"
    played(run, root, 0)
    downgrade_to_v1(root / "g000000")
    text = text_of(run, root)
    assert CONDITIONAL in text
    assert CONDITIONAL_PAIRS not in text
    assert SEAT_CAPTION in text
    for absent in ("Anchor comparisons", "Team Preview ablation", "games per anchor match"):
        assert absent not in text


def test_signed_percentages_round_outward_and_keep_the_sign() -> None:
    assert signed_pct(0) == "0.0%"
    assert signed_pct(500_000) == "+50.0%"
    assert signed_pct(-500_000) == "-50.0%"
    assert signed_pct(-12_345, down=True) == "-1.3%"  # outward: more negative
    assert signed_pct(-12_345, up=True) == "-1.2%"
    assert signed_pct(12_345, down=True) == "+1.2%"
    assert signed_pct(12_345, up=True) == "+1.3%"
    assert signed_pct(-400) == "0.0%"  # a value that rounds to zero carries no sign


def test_the_new_sections_keep_hostile_strings_inert(run: Path, mixed: Path) -> None:
    model = read_report_model(run, mixed)
    facts = model.progress[2]
    baselines = {
        HOSTILE + b: {HOSTILE + v: row for v, row in rows.items()}
        for b, rows in facts.ablation["baselines"].items()
    }
    ablation = {**facts.ablation, "baselines": baselines}
    anchors = tuple(
        dataclasses.replace(m, match_id=HOSTILE + m.match_id) for m in model.progress[1].matches
    )
    changed = {
        **model.progress,
        1: dataclasses.replace(model.progress[1], matches=anchors, anchor_games=4),
        2: dataclasses.replace(facts, ablation=ablation),
    }
    page = build_report(dataclasses.replace(model, progress=changed))
    assert_inert(page, (HOSTILE,))
    assert (
        html.escape(HOSTILE) in page
    )  # the negative control: the strings are in the page, escaped


def test_the_v2_page_builds_with_sockets_blocked(
    run: Path, mixed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the report built a page with a network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert "Anchor comparisons" in text_of(run, mixed)
    assert layout.read_json(mixed / "g000001" / layout.MANIFEST)["files"]
