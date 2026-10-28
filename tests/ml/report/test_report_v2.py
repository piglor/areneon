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
from areneon_ml.progress.eval_dataset import EvalDataset
from areneon_ml.progress.records import CONTROL_LABEL, SEARCH_LABEL, SEARCH_PRIOR_LABEL
from areneon_ml.progress.run import CROSS_RUN_LABEL, VersusSpec, summary_of
from areneon_ml.progress.stats import PPM
from areneon_ml.report.command import report_run
from areneon_ml.report.model import read_report_model
from areneon_ml.report.render import (
    ABLATION_NOTE,
    ANCHOR_NOTE,
    CONDITIONAL,
    CONDITIONAL_PAIRS,
    OTHER_DATASET_LABEL,
    SEAT_CAPTION,
    SEAT_CAPTION_PAIRS,
    SEEN,
    VERSUS_NOTE,
    build_report,
    signed_pct,
)

from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    downgrade_to_v1,
)
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


def test_a_cross_run_match_is_a_labelled_head_to_head_with_its_pair_interval(
    run: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    make_run(other, generations=2, config=dataset_config("other-run", (21, 22)))
    root = tmp_path / "progress"
    played(run, root, 1, versus=VersusSpec(other, 1, 4, "enc-other"))
    model = read_report_model(run, root)
    rec = record(root, 1)
    by_id = {s["match_id"]: s for s in summary_of(rec)}
    found = [m for m in model.progress[1].matches if m.versus is not None]
    assert sorted(m.match_id for m in found) == ["run-argmax", "run-sample"]
    for m in found:
        stored = by_id[m.match_id]
        assert m.versus == ("other-run", 1)
        assert m.pairs == m.games // 2 == 2
        assert m.seen_label == CROSS_RUN_LABEL
        assert m.diff == (stored["diff_mean_ppm"], stored["diff_low_ppm"], stored["diff_high_ppm"])
        assert (m.low_ppm, m.high_ppm) == (stored["interval_low_ppm"], stored["interval_high_ppm"])
    text = text_of(run, root)
    assert "Head to head against another run (cross-run comparison)" in text
    assert VERSUS_NOTE in text
    assert "other-run g1" in text
    assert f'run-sample <span class="muted">({CROSS_RUN_LABEL})' in text
    assert SEEN not in text.split("Head to head against another run")[1]
    assert "Score vs run (cross-run comparison)" in text


def test_a_record_with_damage_aware_and_cross_run_matches_shows_both_sections(
    run: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    make_run(other, generations=2, config=dataset_config("other-run", (21, 22)))
    root = tmp_path / "progress"
    played(
        run, root, 1, versus=VersusSpec(other, 1, 4, "enc-other"), damage_aware_budget_minutes=30
    )
    model = read_report_model(run, root)
    kinds = {
        m.match_id: (m.baseline is not None, m.versus is not None)
        for m in model.progress[1].matches
    }
    assert kinds["damage-aware-sample"] == (True, False)
    assert kinds["run-sample"] == (False, True)
    text = text_of(run, root)
    assert "Head to head against another run (cross-run comparison)" in text
    assert "damage-aware" in text
    assert f'run-sample <span class="muted">({CROSS_RUN_LABEL})' in text


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
    for absent in (
        "Anchor comparisons",
        "Team Preview ablation",
        "games per anchor match",
        "Head to head",
    ):
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


def test_the_damage_aware_baseline_is_shown_with_its_counts_and_label(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "progress"
    played(run, root, 1, damage_aware_budget_minutes=30)
    model = read_report_model(run, root)
    found = {m.match_id: m for m in model.progress[1].matches}
    sums = {s["match_id"]: s for s in summary_of(record(root, 1))}
    for mode in ("argmax", "sample"):
        match = found[f"damage-aware-{mode}"]
        assert match.baseline == sums[match.match_id]["baseline"]
        assert match.seen_label == run_module.DAMAGE_AWARE_LABEL
    assert found["actor-rule-sample"].baseline is None
    text = text_of(run, root)
    assert "Damage-aware baseline counts" in text
    assert f"Score vs damage-aware ({run_module.DAMAGE_AWARE_LABEL})" in text
    assert f'damage-aware-sample <span class="muted">({run_module.DAMAGE_AWARE_LABEL})' in text
    counts = found["damage-aware-argmax"].baseline["counts"]  # type: ignore[index]
    assert f'<td class="n">{counts["queries"]}</td>' in text


def test_a_page_without_the_baseline_has_no_baseline_table(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "progress"
    played(run, root, 1)
    assert "Damage-aware baseline counts" not in text_of(run, root)


def test_the_dashboard_labels_an_evaluation_dataset(run: Path, tmp_path: Path) -> None:
    """The eval-dataset option (ADR-020 E-17.1): a generation on another dataset says so."""
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps({"splits": [{"split": "validation", "teams": 6}]})
    )
    plain, labelled = tmp_path / "plain", tmp_path / "labelled"
    played(run, plain, 1)
    played(run, labelled, 1, eval_dataset=EvalDataset(other, "7" * 64))
    assert OTHER_DATASET_LABEL not in text_of(run, plain)
    assert read_report_model(run, plain).progress[1].evaluation_dataset is None
    page = text_of(run, labelled)
    assert OTHER_DATASET_LABEL in page
    assert f"generation 1: dataset {'7' * 64}" in page
    assert read_report_model(run, labelled).progress[1].evaluation_dataset == "7" * 64


def test_the_dashboard_labels_an_evaluation_key(run: Path, tmp_path: Path) -> None:
    """The evaluation-key option (#1177): a generation keyed by it says so."""
    plain, keyed = tmp_path / "plain", tmp_path / "keyed"
    played(run, plain, 1)
    played(run, keyed, 1, evaluation_key="paired-eval-1")
    assert "Evaluation key:" not in text_of(run, plain)
    assert read_report_model(run, plain).progress[1].evaluation_key is None
    page = text_of(run, keyed)
    assert "Evaluation key:" in page
    assert "generation 1: paired-eval-1" in page
    assert read_report_model(run, keyed).progress[1].evaluation_key == "paired-eval-1"
    # The report command logs a look that names the key, per match.
    out = tmp_path / "report.html"
    report_run(run, keyed, out)
    first = json.loads((tmp_path / "keyed.looks.jsonl").read_text().splitlines()[0])
    assert {m["evaluation_key"] for m in first["matches"]} == {"paired-eval-1"}


def test_the_decision_time_search_is_labelled_and_its_counts_and_the_controls_are_shown(
    run: Path, tmp_path: Path
) -> None:
    """ADR-026 slice P: the search matches say what they are, never held-out evidence, and the
    control's counts (searched decisions 0) sit beside the search's."""
    root = tmp_path / "progress"
    played(run, root, 1, search_mode="argmax")
    model = read_report_model(run, root)
    found = {m.match_id: m for m in model.progress[1].matches}
    assert sorted(found) == [
        "search-actor-rule-argmax",
        "search-random-legal-argmax",
        "search-self-no-search-argmax",
    ]
    control = found["search-self-no-search-argmax"]
    assert control.opponent_search is not None
    assert control.opponent_search["counts"]["searched_decisions"] == 0
    assert found["search-actor-rule-argmax"].opponent_search is None
    text = text_of(run, root)
    assert "Decision-time search counts" in text
    assert SEARCH_LABEL in text
    assert SEARCH_PRIOR_LABEL in text
    assert "Search results are never held-out evidence" in text
    assert f'<td class="n">{control.search["counts"]["leaf_queries"]}</td>' in text  # type: ignore[index]
    assert CONTROL_LABEL in text


def test_a_page_without_search_has_no_search_table(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "progress"
    played(run, root, 1)
    assert "Decision-time search counts" not in text_of(run, root)
