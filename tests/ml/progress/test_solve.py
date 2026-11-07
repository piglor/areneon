"""The Team Preview solver as a progress-evaluation candidate (ADR-029 section 8, #1260)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.loop.records import policy_id
from areneon_ml.progress.records import (
    SOLVE_CONTROL_LABEL,
    SOLVE_EVIDENCE_NOTE,
    SOLVE_LABEL,
    SOLVE_PRIOR_LABEL,
    ProgressRecordError,
    read_game_record,
    read_progress_record,
    sum_solve_counts,
)
from areneon_ml.progress.run import ProgressError, summary_of, verify_progress_dir
from areneon_ml.progress.run import progress_generation as _progress_generation

from tests.ml.progress.support import (
    SOLVE_IDENTITY,
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
    sample,
    solve_counts,
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


def test_a_solver_evaluation_plays_one_mode_with_the_control_and_says_what_it_is(
    run: Path, tmp_path: Path
) -> None:
    fake = player(run)
    done = evaluate(run, tmp_path / "p", fake, preview_solve="argmax")
    assert fake.requests[0].preview_solve_mode == "argmax"
    by_id = {m["match_id"]: m for m in done["matches"]}
    assert sorted(by_id) == [
        "psolve-actor-rule-argmax",
        "psolve-random-legal-argmax",
        "psolve-self-no-solve-argmax",
    ]
    for match in by_id.values():
        assert match["solve"]["label"] == SOLVE_LABEL
        assert match["solve"]["prior_label"] == SOLVE_PRIOR_LABEL
        assert match["solve"]["note"] == SOLVE_EVIDENCE_NOTE
        assert match["solve"]["identity_hash"] == SOLVE_IDENTITY
    control = by_id["psolve-self-no-solve-argmax"]
    assert control["comparison"] == SOLVE_CONTROL_LABEL
    assert control["opponent_solve"]["counts"]["solved"] == 0
    assert "diff_mean_ppm" in control  # a head to head: the pair score minus one half
    assert "opponent_solve" not in by_id["psolve-actor-rule-argmax"]
    again = evaluate(run, tmp_path / "p", fake, preview_solve="argmax")
    assert again["skipped"] is True
    assert again["matches"] == done["matches"]
    record = read_progress_record(layout.read_json(tmp_path / "p" / "g000001" / "progress.json"))
    assert {m["match_id"] for m in record["matches"]} == set(by_id)
    assert record["plan"]["preview_solve_budget_ms"] == 30000


def test_the_labels_are_the_committed_solver_configs_and_the_ticket_s() -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "config/preview/gen9championsvgc2026regmc/preview-solve-1.v1.json"
        ).read_text()
    )
    assert config["worlds"]["prior"]["label"] == SOLVE_PRIOR_LABEL
    assert SOLVE_LABEL == "learned pair with a Team Preview solver (decision-time, one-shot)"
    assert SOLVE_EVIDENCE_NOTE == "solver results are never held-out evidence"
    assert SOLVE_CONTROL_LABEL == "control: the same frozen pair without the solver"


def test_the_solver_baseline_and_head_to_head_matches_use_the_one_mode(
    run: Path, tmp_path: Path
) -> None:
    done = evaluate(
        run, tmp_path / "p", player(run), preview_solve="sample", damage_aware_budget_minutes=1
    )
    ids = sorted(m["match_id"] for m in done["matches"])
    assert "psolve-damage-aware-sample" in ids
    assert all(i.endswith("-sample") for i in ids)
    assert done["damage_aware"] == "played"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"preview_solve": "greedy"}, "argmax or sample"),
        ({"preview_solve": "argmax", "anchors": (0,)}, "evaluated alone"),
        ({"preview_solve": "argmax", "previous_matches": True}, "evaluated alone"),
        ({"preview_solve": "argmax", "preview_ablation": True}, "evaluated alone"),
        ({"preview_solve": "argmax", "search_mode": "argmax"}, "evaluated alone"),
    ],
)
def test_a_solver_request_that_cannot_be_played_is_refused(
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
        evaluate(run, tmp_path / "plain", fake, preview_solve="argmax")
    evaluate(run, tmp_path / "solved", fake, preview_solve="argmax")
    with pytest.raises(ProgressError, match="solver record of another kind"):
        evaluate(run, tmp_path / "solved", fake)
    with pytest.raises(ProgressError, match="solver record of another kind"):
        evaluate(run, tmp_path / "solved", fake, preview_solve="sample")


def test_the_files_the_simulator_side_wrote_are_checked_against_the_request(
    run: Path, tmp_path: Path
) -> None:
    class Ignoring(FakeProgressPlayer):
        def progress(self, request: Any) -> dict[str, Any]:
            return super().progress(dataclasses.replace(request, preview_solve_mode=None))

    fake = Ignoring(dataset_config(), committed_manifest(run, 0))
    with pytest.raises(ProgressError, match="no solver matches"):
        evaluate(run, tmp_path / "p", fake, preview_solve="argmax")
    assert not (tmp_path / "p" / "g000001").exists()


def test_the_stored_games_carry_both_seats_counts_and_a_tampered_one_is_refused(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), preview_solve="argmax")
    target = root / "g000001"
    config = dataset_config()
    verify_progress_dir(target, config, run, 1)
    name = "psolve-self-no-solve-argmax-0.json"
    stored = layout.read_json(target / "games" / name)
    assert stored["solve"]["seat"] == "p1"
    assert stored["opponent_solve"]["seat"] == "p2"

    def other_worlds(g: dict[str, Any]) -> None:
        g["solve"]["counts"]["worlds_asked"] = 9

    def no_control(g: dict[str, Any]) -> None:
        g.pop("opponent_solve")

    for change in (other_worlds, no_control):
        broken = copy.deepcopy(stored)
        change(broken)
        read_game_record(
            broken
        )  # its own rules pass; the tie to the row and match is the verify step's
        layout.write_json(target / "games" / name, broken)
        manifest = layout.read_json(target / layout.MANIFEST)
        manifest["files"] = layout.hash_tree(target)
        layout.write_json(target / layout.MANIFEST, manifest)
        with pytest.raises(ProgressError, match="match's identity"):
            verify_progress_dir(target, config, run, 1)
    layout.write_json(target / "games" / name, stored)


def test_the_candidates_policy_id_and_the_controls_are_tied_to_the_run(
    run: Path, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    evaluate(run, root, player(run), preview_solve="argmax")
    raw = layout.read_json(root / "g000001" / "progress.json")
    config = dataset_config()
    read_progress_record(raw, config)  # ids derived from the pair, mode and the solver identity
    pair = raw["checkpoints"]
    assert raw["matches"][0]["policy_id"] != policy_id(pair["preview"], pair["battle"], "argmax")
    wrong_id = copy.deepcopy(raw)
    for match in wrong_id["matches"]:
        match["solve"]["identity_hash"] = "d" * 64
        if "opponent_solve" in match:
            match["opponent_solve"]["identity_hash"] = "d" * 64
    with pytest.raises(ProgressRecordError, match="policy_id is not the frozen pair's"):
        read_progress_record(wrong_id, config)
    other_control = copy.deepcopy(raw)
    control = next(m for m in other_control["matches"] if m["opponent"]["kind"] == "self-no-solve")
    control["opponent"]["policy_id"] = "policy:" + "0" * 64
    with pytest.raises(ProgressRecordError, match="without the solver"):
        read_progress_record(other_control, config)


def test_the_corpus_samples_read_and_the_reader_invalid_ones_do_not() -> None:
    for name in ("psolve-match", "psolve-self-no-solve-match"):
        read_progress_record(sample(f"{V2}/valid/{name}"))
    for name in ("psolve", "psolve-control"):
        read_game_record(sample(f"{GAME}/valid/{name}"))
    for name in (
        "psolve-counts-disagree",
        "psolve-row-missing",
        "psolve-misnamed",
        "psolve-on-plain-match",
        "psolve-mixed-with-plain",
        "psolve-no-budget",
        "budget-on-plain-record",
        "psolve-with-search",
        "psolve-solved-over-previews",
        "psolve-unsorted-refusals",
        "psolve-refusals-over-fallbacks",
        "psolve-control-without-solve",
        "psolve-control-row-missing",
        "psolve-control-solves",
        "psolve-control-other-identity",
        "psolve-control-mode-differs",
        "psolve-control-counts-on-baseline",
    ):
        with pytest.raises(ProgressRecordError):
            read_progress_record(sample(f"{V2}/reader-invalid/{name}"))
    for name in (
        "psolve-decisions-differ",
        "psolve-control-same-seat",
        "psolve-control-other-identity",
        "psolve-control-without-solve",
        "psolve-control-solves",
    ):
        with pytest.raises(ProgressRecordError):
            read_game_record(sample(f"{GAME}/reader-invalid/{name}"))


def test_a_game_cannot_carry_both_search_and_solver_counts() -> None:
    both = sample(f"{GAME}/valid/psolve")
    both["search"] = sample(f"{GAME}/valid/search")["search"]
    with pytest.raises(ProgressRecordError, match="never both"):
        read_game_record(both)
    two = sample(f"{V2}/valid/psolve-match")
    two["matches"][0]["search"] = sample(f"{V2}/valid/search-match")["matches"][0]["search"]
    with pytest.raises(ProgressRecordError, match="never both"):
        read_progress_record(two)


def test_a_record_with_two_solver_identities_and_a_summary_of_one_are_refused_or_read() -> None:
    two = sample(f"{V2}/valid/psolve-match")
    match = copy.deepcopy(two["matches"][0])
    match["match_id"] = "psolve-random-legal-argmax"
    match["solve"]["identity_hash"] = "d" * 64
    match["game_record_hashes"] = two["matches"][0]["game_record_hashes"]
    two["matches"].append(match)
    two["games"].extend(
        {**copy.deepcopy(g), "match_id": "psolve-random-legal-argmax"} for g in two["games"]
    )
    with pytest.raises(ProgressRecordError, match="one solver identity hash"):
        read_progress_record(two)
    summary = summary_of(read_progress_record(sample(f"{V2}/valid/psolve-match")))[0]
    assert summary["solve"]["label"] == SOLVE_LABEL


def test_summing_counts_merges_the_belief_refusals_by_key() -> None:
    a = solve_counts(5)
    b = solve_counts(7)
    a["fallbacks"]["belief_refusal"] = 1
    a["solved"] = 0
    a["belief_refusals"] = [{"key": "species_not_in_prior", "count": 1}]
    b["fallbacks"]["belief_refusal"] = 2
    b["solved"] = 0
    b["belief_refusals"] = [
        {"key": "a", "count": 1},
        {"key": "species_not_in_prior", "count": 1},
    ]
    total = sum_solve_counts([a, b])
    assert total["decisions"] == 12
    assert total["belief_refusals"] == [
        {"key": "a", "count": 1},
        {"key": "species_not_in_prior", "count": 2},
    ]
    assert total["fallbacks"]["belief_refusal"] == 3
