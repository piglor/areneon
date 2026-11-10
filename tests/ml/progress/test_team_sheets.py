"""Open-sheet matches of the progress evaluation (ADR-027 section 5a and 8, ADR-017 erratum E4)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.loop import layout
from areneon_ml.progress.records import read_progress_record, record_hash
from areneon_ml.progress.run import (
    ProgressError,
    ProgressRequest,
    summary_of,
    verify_progress_dir,
)
from areneon_ml.progress.run import progress_generation as _progress_generation
from areneon_ml.progress.team_sheets import (
    TeamSheetsError,
    check_team_sheets,
    closed_view,
    team_sheets_view,
    twin_of,
)

from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

OPEN_FORMAT = "gen9championsvgc2026regmcbo3"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("run") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def add_open(out: Path, request: ProgressRequest, *, wrong_manifest: bool = False) -> None:
    """What the simulator side writes with ``--team-sheets open``: the open twin of every baseline
    and cross-run match, its games on the open manifest, and both manifests in the record."""
    path = out / "progress.json"
    record = layout.read_json(path)
    closed = record["version_manifest"]
    opened = {**closed, "format_id": OPEN_FORMAT, "ruleset_hash": "e" * 64}
    opened["format_rules_hash"] = "f" * 64
    record["game_manifests"] = sorted([closed, opened], key=jcs)
    matches, rows = [], []
    for match in record["matches"]:
        if match["opponent"]["kind"] not in ("baseline", "run"):
            continue
        twin = copy.deepcopy(match)
        twin["match_id"] = twin_of(match)
        twin["team_sheets"] = "open"
        hashes = []
        for game in range(min(request.record_games, match["games"])):
            stored = layout.read_json(out / "games" / f"{match['match_id']}-{game}.json")
            stored["match_id"] = twin["match_id"]
            stored["version_manifest"] = closed if wrong_manifest else opened
            layout.write_json(out / "games" / f"{twin['match_id']}-{game}.json", stored)
            hashes.append(record_hash(stored))
        twin["game_record_hashes"] = sorted(hashes)
        matches.append(twin)
        rows += [
            {**copy.deepcopy(r), "match_id": twin["match_id"]}
            for r in record["games"]
            if r["match_id"] == match["match_id"]
        ]
    record["matches"] = sorted(record["matches"] + matches, key=lambda m: m["match_id"])
    record["games"] = sorted(record["games"] + rows, key=lambda r: (r["match_id"], r["game"]))
    layout.write_json(path, record)


def player(run: Path, *, open_twins: bool = True, wrong_manifest: bool = False) -> Any:
    def change(out: Path, request: ProgressRequest) -> None:
        if open_twins:
            add_open(out, request, wrong_manifest=wrong_manifest)

    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0), change)


def evaluate(run: Path, root: Path, fake: Any, **kwargs: Any) -> dict[str, Any]:
    found: dict[str, Any] = _progress_generation(run, 1, root, fake, 2, 1, (), 4, **kwargs)
    return found


def test_the_open_twins_are_asked_for_played_and_shown_beside_the_closed_matches(
    run: Path, tmp_path: Path
) -> None:
    fake = player(run)
    done = evaluate(
        run, tmp_path / "p", fake, damage_aware_budget_minutes=30, team_sheets_open=True
    )
    assert fake.requests[0].team_sheets_open is True
    by_id = {m["match_id"]: m for m in done["matches"]}
    assert "open-damage-aware-argmax" in by_id
    assert by_id["open-damage-aware-argmax"]["team_sheets"] == "open"
    assert "team_sheets" not in by_id["damage-aware-argmax"]
    view = done["team_sheets"]
    assert [(r["opponent"], r["mode"]) for r in view["rows"]] == [
        ("damage-aware@v1", "argmax"),
        ("damage-aware@v1", "sample"),
    ]
    row = view["rows"][0]
    assert set(row) == {"opponent", "mode", "pairs", "closed", "open"}
    assert set(row["closed"]) == {"score_ppm", "low_ppm", "high_ppm"}
    record = read_progress_record(layout.read_json(tmp_path / "p" / "g000001" / "progress.json"))
    assert len(record["game_manifests"]) == 2
    assert team_sheets_view(record) == view


def test_a_committed_root_is_kept_only_for_the_same_kind(run: Path, tmp_path: Path) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), team_sheets_open=True)
    again = evaluate(run, root, player(run, open_twins=False), team_sheets_open=True)
    assert again["skipped"] is True
    with pytest.raises(ProgressError, match="already holds a record with open-sheet matches"):
        evaluate(run, root, player(run, open_twins=False))
    plain = tmp_path / "q"
    evaluate(run, plain, player(run, open_twins=False))
    with pytest.raises(ProgressError, match="already holds a record without open-sheet matches"):
        evaluate(run, plain, player(run, open_twins=False), team_sheets_open=True)
    assert verify_progress_dir(root / "g000001", dataset_config(), run, 1)["game_manifests"]


def test_open_matches_not_asked_for_or_not_played_are_refused(run: Path, tmp_path: Path) -> None:
    with pytest.raises(ProgressError, match="asks for no open-sheet matches"):
        evaluate(run, tmp_path / "a", player(run))
    with pytest.raises(ProgressError, match="asked for are missing"):
        evaluate(run, tmp_path / "b", player(run, open_twins=False), team_sheets_open=True)
    assert not (tmp_path / "a" / "g000001").exists()


def test_a_game_of_an_open_match_must_use_the_open_manifest(run: Path, tmp_path: Path) -> None:
    with pytest.raises(ProgressError, match="is not the game it is named for"):
        evaluate(run, tmp_path / "p", player(run, wrong_manifest=True), team_sheets_open=True)


def test_the_open_format_is_played_alone_of_the_other_kinds_of_candidate(
    run: Path, tmp_path: Path
) -> None:
    for kwargs in (
        {"preview_ablation": True},
        {"anchors": (0,)},
        {"previous_matches": True},
        {"search_mode": "argmax"},
        {"preview_solve": "argmax"},
    ):
        extra: dict[str, Any] = dict(kwargs)
        with pytest.raises(ProgressError, match="--team-sheets open plays the scripted"):
            _progress_generation(
                run,
                1,
                tmp_path / "p",
                player(run),
                2,
                1,
                extra.pop("anchors", ()),
                4,
                team_sheets_open=True,
                **extra,
            )


def test_the_request_names_the_flag_to_the_simulator_side(run: Path) -> None:
    from areneon_ml.progress.run import NodeProgressPlayer  # noqa: PLC0415

    assert NodeProgressPlayer.progress is not None
    request = ProgressRequest(
        generation=1,
        preview="a",
        battle="b",
        policy_id_sample="s",
        policy_id_argmax="m",
        manifest_sha256="c",
        team_set=run,
        games=2,
        record_games=1,
        out_dir=run,
        team_sheets_open=True,
    )
    assert request.team_sheets_open


def sample_record(run: Path, tmp_path: Path) -> dict[str, Any]:
    evaluate(
        run, tmp_path / "p", player(run), damage_aware_budget_minutes=30, team_sheets_open=True
    )
    return read_progress_record(layout.read_json(tmp_path / "p" / "g000001" / "progress.json"))


def test_the_check_names_each_way_a_record_can_fail_the_request(run: Path, tmp_path: Path) -> None:
    record = sample_record(run, tmp_path)
    check_team_sheets(record, True, legacy=False)
    check_team_sheets(record, False, legacy=True)
    check_team_sheets(closed_view(record), False, legacy=False)
    check_team_sheets(closed_view(record), False, legacy=True)

    short = copy.deepcopy(record)
    short["matches"] = [m for m in short["matches"] if m["match_id"] != "open-actor-rule-argmax"]
    with pytest.raises(TeamSheetsError, match="open twins of the baseline"):
        check_team_sheets(short, True, legacy=False)
    other = copy.deepcopy(record)
    twin = next(m for m in other["matches"] if m["match_id"] == "open-actor-rule-argmax")
    twin["games"] = 4
    with pytest.raises(TeamSheetsError, match="not played like its closed match"):
        check_team_sheets(other, True, legacy=False)
    third = copy.deepcopy(record)
    twin = next(m for m in third["matches"] if m["match_id"] == "open-damage-aware-argmax")
    twin["baseline"]["identity_hash"] = "8" * 64
    with pytest.raises(TeamSheetsError, match="another baseline identity"):
        check_team_sheets(third, True, legacy=False)


def test_summary_of_marks_open_matches_and_a_view_needs_a_twin(run: Path, tmp_path: Path) -> None:
    record = sample_record(run, tmp_path)
    marked = {m["match_id"]: m for m in summary_of(record)}
    assert marked["open-actor-rule-argmax"]["team_sheets"] == "open"
    assert "open-sheet format" in marked["open-actor-rule-argmax"]["format"]
    lone = copy.deepcopy(record)
    lone["matches"] = [m for m in lone["matches"] if m["match_id"] != "open-damage-aware-sample"]
    lone["games"] = [g for g in lone["games"] if g["match_id"] != "open-damage-aware-sample"]
    assert [r["mode"] for r in team_sheets_view(lone)["rows"]] == ["argmax"]
    assert team_sheets_view(closed_view(record))["rows"] == []
