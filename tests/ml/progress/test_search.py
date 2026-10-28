"""The search candidate as a progress-evaluation candidate (ADR-026 section 5, slice P)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.progress.records import (
    CONTROL_LABEL,
    SEARCH_EVIDENCE_NOTE,
    SEARCH_LABEL,
    SEARCH_PRIOR_LABEL,
    ProgressRecordError,
    read_game_record,
    read_progress_record,
    sum_search_counts,
)
from areneon_ml.progress.run import ProgressError, verify_progress_dir
from areneon_ml.progress.run import progress_generation as _progress_generation

from tests.ml.progress.support import (
    SEARCH_IDENTITY,
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
    sample,
    search_counts,
)

V2 = "progress-evaluation-record-2"
GAME = "progress-game-record"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def player(run: Path) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))


def evaluate(run: Path, root: Path, fake: Any, **kwargs: Any) -> dict[str, Any]:
    found: dict[str, Any] = _progress_generation(run, 1, root, fake, 2, 1, (), 4, **kwargs)
    return found


def test_a_search_evaluation_plays_one_mode_with_the_control_and_says_what_it_is(
    run: Path, tmp_path: Path
) -> None:
    fake = player(run)
    done = evaluate(run, tmp_path / "p", fake, search_mode="argmax")
    assert fake.requests[0].search_mode == "argmax"
    by_id = {m["match_id"]: m for m in done["matches"]}
    assert sorted(by_id) == [
        "search-actor-rule-argmax",
        "search-random-legal-argmax",
        "search-self-no-search-argmax",
    ]
    for match in by_id.values():
        assert match["search"]["label"] == SEARCH_LABEL
        assert match["search"]["prior_label"] == SEARCH_PRIOR_LABEL
        assert match["search"]["note"] == SEARCH_EVIDENCE_NOTE
        assert match["search"]["identity_hash"] == SEARCH_IDENTITY
    control = by_id["search-self-no-search-argmax"]
    assert control["comparison"] == CONTROL_LABEL
    assert control["opponent_search"]["counts"]["searched_decisions"] == 0
    assert "diff_mean_ppm" in control  # a head to head: the pair score minus one half
    assert "opponent_search" not in by_id["search-actor-rule-argmax"]
    # Rerun is a verified skip with the same summary; the record reads back.
    again = evaluate(run, tmp_path / "p", fake, search_mode="argmax")
    assert again["skipped"] is True
    assert again["matches"] == done["matches"]
    record = read_progress_record(layout.read_json(tmp_path / "p" / "g000001" / "progress.json"))
    assert {m["match_id"] for m in record["matches"]} == set(by_id)


def test_the_labels_are_the_committed_search_policy_and_prior_labels() -> None:
    config = json.loads(
        (Path(__file__).resolve().parents[3] / "config/search/rerank-1.v1.json").read_text()
    )
    assert config["worlds"]["prior"]["label"] == SEARCH_PRIOR_LABEL
    assert SEARCH_LABEL == "learned pair with decision-time search (one-ply re-rank)"


def test_the_search_baseline_and_head_to_head_matches_use_the_one_mode(
    run: Path, tmp_path: Path
) -> None:
    done = evaluate(
        run, tmp_path / "p", player(run), search_mode="sample", damage_aware_budget_minutes=1
    )
    ids = sorted(m["match_id"] for m in done["matches"])
    assert "search-damage-aware-sample" in ids
    assert all(i.endswith("-sample") for i in ids)
    assert done["damage_aware"] == "played"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"search_mode": "greedy"}, "argmax or sample"),
        ({"search_mode": "argmax", "anchors": (0,)}, "evaluated alone"),
        ({"search_mode": "argmax", "previous_matches": True}, "evaluated alone"),
        ({"search_mode": "argmax", "preview_ablation": True}, "evaluated alone"),
    ],
)
def test_a_search_request_that_cannot_be_played_is_refused(
    run: Path, tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ProgressError, match=message):
        _progress_generation(
            run, 1, tmp_path / "p", player(run), 2, 1, kwargs.pop("anchors", ()), 4, **kwargs
        )
    assert not (tmp_path / "p" / "g000001").exists()


def test_a_directory_holds_one_kind_of_record(run: Path, tmp_path: Path) -> None:
    fake = player(run)
    evaluate(run, tmp_path / "plain", fake)
    with pytest.raises(ProgressError, match="already holds a normal record"):
        evaluate(run, tmp_path / "plain", fake, search_mode="argmax")
    evaluate(run, tmp_path / "searched", fake, search_mode="argmax")
    with pytest.raises(ProgressError, match="search record of another kind"):
        evaluate(run, tmp_path / "searched", fake)
    with pytest.raises(ProgressError, match="search record of another kind"):
        evaluate(run, tmp_path / "searched", fake, search_mode="sample")


def test_the_files_the_simulator_side_wrote_are_checked_against_the_request(
    run: Path, tmp_path: Path
) -> None:
    # A simulator side that ignores the request writes a normal record: refused, nothing committed.
    class Ignoring(FakeProgressPlayer):
        def progress(self, request: Any) -> dict[str, Any]:
            return super().progress(dataclasses.replace(request, search_mode=None))

    fake = Ignoring(dataset_config(), committed_manifest(run, 0))
    with pytest.raises(ProgressError, match="no search matches"):
        evaluate(run, tmp_path / "p", fake, search_mode="argmax")
    assert not (tmp_path / "p" / "g000001").exists()


def test_the_stored_games_carry_both_seats_counts_and_a_tampered_one_is_refused(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), search_mode="argmax")
    target = root / "g000001"
    config = dataset_config()
    verify_progress_dir(target, config, run, 1)
    name = "search-self-no-search-argmax-0.json"
    stored = layout.read_json(target / "games" / name)
    assert stored["search"]["seat"] == "p1"
    assert stored["opponent_search"]["seat"] == "p2"
    for change, message in (
        (lambda g: g["search"]["counts"].__setitem__("turns", 1), "match's identity"),
        (lambda g: g.pop("opponent_search"), "match's identity"),
    ):
        broken = copy.deepcopy(stored)
        change(broken)
        # the stored game's own rules pass; the tie to the row and match is the verify step's
        read_game_record(broken)
        layout.write_json(target / "games" / name, broken)
        manifest = layout.read_json(target / layout.MANIFEST)
        manifest["files"] = layout.hash_tree(target)
        layout.write_json(target / layout.MANIFEST, manifest)
        with pytest.raises(ProgressError, match=message):
            verify_progress_dir(target, config, run, 1)
    layout.write_json(target / "games" / name, stored)


def test_the_candidates_policy_id_and_the_controls_are_tied_to_the_run(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), search_mode="argmax")
    raw = layout.read_json(root / "g000001" / "progress.json")
    config = dataset_config()
    read_progress_record(raw, config)  # ids derived from the pair, mode and the search identity
    wrong_id = copy.deepcopy(raw)
    wrong_id["matches"][0]["search"]["identity_hash"] = "d" * 64
    with pytest.raises(ProgressRecordError):
        read_progress_record(wrong_id, config)
    other_control = copy.deepcopy(raw)
    control = next(m for m in other_control["matches"] if m["opponent"]["kind"] == "self-no-search")
    control["opponent"]["policy_id"] = "policy:" + "0" * 64
    with pytest.raises(ProgressRecordError, match="without search"):
        read_progress_record(other_control, config)


def test_the_corpus_samples_read_and_the_reader_invalid_ones_do_not() -> None:
    for name in ("search-match", "search-self-no-search-match"):
        read_progress_record(sample(f"{V2}/valid/{name}"))
    for name in ("search", "search-control"):
        read_game_record(sample(f"{GAME}/valid/{name}"))
    for name in (
        "search-counts-disagree",
        "search-row-missing",
        "search-misnamed",
        "search-on-plain-match",
        "search-mixed-with-plain",
        "search-more-fallbacks-than-decisions",
        "search-unsorted-table",
        "control-without-search",
        "control-row-missing",
        "control-searches",
        "control-other-identity",
        "control-mode-differs",
        "control-counts-on-baseline",
    ):
        with pytest.raises(ProgressRecordError):
            read_progress_record(sample(f"{V2}/reader-invalid/{name}"))
    for name in (
        "search-decisions-differ",
        "search-control-same-seat",
        "search-control-other-identity",
        "search-control-without-search",
    ):
        with pytest.raises(ProgressRecordError):
            read_game_record(sample(f"{GAME}/reader-invalid/{name}"))


def test_summing_counts_merges_the_keyed_tables_by_key() -> None:
    a = search_counts(5)
    b = search_counts(7)
    b["refusals"] = [{"key": "a", "count": 1}, {"key": "state_unwritten", "count": 3}]
    total = sum_search_counts([a, b])
    assert total["decisions"] == 12
    assert total["refusals"] == [{"key": "a", "count": 1}, {"key": "state_unwritten", "count": 5}]
    assert total["fallbacks"]["team_preview"] == 2
