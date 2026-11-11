"""candidate_features on the worker side (ADR-022 §3): the handshake statement, the reader rules."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.training import StreamChecker, TrainingStreamError
from areneon_ml.checkpoint.files import checkpoint_id, write_checkpoint
from areneon_ml.model.network import ModelConfig
from areneon_ml.model.options import CANDIDATE_FEATURES, read_options
from areneon_ml.model.scorer import ScorerError
from areneon_ml.worker.__main__ import build
from areneon_ml.worker.features import (
    FOE_SPREAD_TARGET_TYPES,
    MAX_OUTCOME_BRANCHES,
    coverage_problem,
    damaging_moves,
    rows_problem,
)
from areneon_ml.worker.protocol import QueryError, check
from areneon_ml.worker.serve import EXIT_OK, FeatureSupport, Worker, read_query

from tests.ml.helpers import (
    ENGINE_DATA,
    MAX_LINE,
    battle_query,
    game,
    hello,
    line,
    make_run_dir,
    make_worker,
    sample,
    session,
    tiny_checkpoint,
)
from tests.ml.model.support import TINY_ENCODER, TINY_MODEL

REPO = Path(__file__).resolve().parents[2]
DAMAGING = damaging_moves(game())


def featured() -> dict[str, Any]:
    return sample("policy-query/valid/battle-features.json")


def stated(query: dict[str, Any]) -> dict[str, str]:
    features = query["candidate_features"]
    keys = ("feature_set", "feature_set_hash")
    return {key: features[key] for key in keys}


def support(query: dict[str, Any]) -> FeatureSupport:
    return FeatureSupport(stated(query), DAMAGING)


def feature_model(feature_set: dict[str, str]) -> ModelConfig:
    """The tiny model whose encoder reads the given feature set."""
    options = read_options("encoder", {CANDIDATE_FEATURES: feature_set})
    return replace(TINY_MODEL, encoder=replace(TINY_ENCODER, options=options))


def feature_worker(tmp_path: Path, feature_set: dict[str, str] | None) -> Worker:
    plain = make_worker(
        tmp_path, model=TINY_MODEL if feature_set is None else feature_model(feature_set)
    )
    return Worker(
        plain._scorer,
        plain._ready["message"]["engine_data_hash"],
        plain._ready["message"]["checkpoints"],
        MAX_LINE,
        None if feature_set is None else FeatureSupport(feature_set, DAMAGING),
    )


def test_the_ready_message_states_the_feature_set_only_when_given(tmp_path: Path) -> None:
    query = featured()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    code, frames = session(feature_worker(tmp_path / "a", stated(query)), line(hello()))
    assert code == EXIT_OK
    assert frames[0]["message"]["feature_set"] == stated(query)
    check("policy-handshake", frames[0])
    _, plain = session(make_worker(tmp_path / "b"), line(hello()))
    assert "feature_set" not in plain[0]["message"]
    committed = sample("policy-handshake/valid/ready-features.json")
    check("policy-handshake", committed)


def test_the_valid_samples_are_accepted_under_their_set() -> None:
    query = featured()
    assert read_query(query, support(query)) == query
    preview = sample("policy-query/valid/preview-features.json")
    assert read_query(preview, support(preview))["candidate_features"]["rows"] == []


def test_features_go_with_a_stated_set_only() -> None:
    query = featured()
    with pytest.raises(QueryError, match="stated no set"):
        read_query(query)
    with pytest.raises(QueryError, match="has no candidate_features"):
        read_query(battle_query(), support(query))
    other = {**stated(query), "feature_set_hash": "0" * 64}
    with pytest.raises(QueryError, match="another set"):
        read_query(query, FeatureSupport(other, DAMAGING))
    # no feature set anywhere: today's query reads as before
    assert read_query(battle_query())["query_id"] == 1


@pytest.mark.parametrize("name", ["features-unsorted", "features-duplicate-row"])
def test_rows_must_be_sorted_and_unique(name: str) -> None:
    query = sample(f"policy-query/reader-invalid/{name}.json")
    check("policy-query", query)  # schema-valid
    assert rows_problem(query["candidate_features"]) is not None
    with pytest.raises(QueryError, match="sorted and unique"):
        read_query(query, support(query))


def test_a_missing_row_is_an_invalid_query(tmp_path: Path) -> None:
    query = sample("policy-query/reader-invalid/features-missing-row.json")
    check("policy-query", query)
    with pytest.raises(QueryError, match="no candidate_features row"):
        read_query(query, support(query))
    code, frames = session(feature_worker(tmp_path, stated(query)), line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "error"
    assert frames[1]["reply"]["error"] == "invalid_query"
    assert "no candidate_features row" in frames[1]["reply"]["detail"]


def test_a_query_with_all_its_rows_is_answered(tmp_path: Path) -> None:
    query = featured()
    code, frames = session(feature_worker(tmp_path, stated(query)), line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"


def request_query(moves: list[dict[str, str]], choice: str) -> dict[str, Any]:
    request = "|request|" + json.dumps({"active": [{"moves": moves}]})
    return {"observation": {"log": [request]}, "candidates": [{"choice": choice}]}


def test_the_coverage_rule_is_the_typescript_one() -> None:
    moves = [
        {"id": "earthquake", "target": "allAdjacent"},
        {"id": "tackle", "target": "normal"},
        {"id": "protect", "target": "self"},
    ]

    def rows(*keys: tuple[int, int, int, bool]) -> dict[str, Any]:
        return {"rows": [{"slot": s, "move": m, "target": t, "mega": g} for s, m, t, g in keys]}

    def problem(choice: str, feature_rows: dict[str, Any]) -> str | None:
        return coverage_problem(request_query(moves, f"{choice}, pass"), feature_rows, DAMAGING)

    assert problem("move 2 1", rows()) is not None
    assert problem("move 2 1", rows((0, 2, 1, False))) is None
    # a fainted foe's location has no row of its own
    assert problem("move 2 1", rows((0, 2, 2, False))) is None
    assert problem("move 2 1", rows((1, 2, 1, False), (0, 1, 1, False))) is not None
    assert problem("move 2 1 mega", rows((0, 2, 1, False))) is not None
    assert problem("move 2 1 mega", rows((0, 2, 1, True))) is None
    assert problem("move 1", rows()) is not None
    assert problem("move 1", rows((0, 1, 2, False))) is None
    for free in ("move 3", "move 2 -2", "switch 3", "pass"):
        assert problem(free, rows()) is None, free
    off_request = {"observation": {"log": ["|turn|2"]}, "candidates": []}
    assert coverage_problem(off_request, rows(), DAMAGING) is not None
    team_preview = {"observation": {"log": ['|request|{"teamPreview":true}']}, "candidates": []}
    assert coverage_problem(team_preview, rows(), DAMAGING) is None


def test_an_unknown_move_id_counts_as_damaging() -> None:
    assert DAMAGING("nosuchmove") is True
    assert DAMAGING("protect") is False
    assert DAMAGING("earthquake") is True


def test_the_spread_target_types_are_typescript_s() -> None:
    assert sorted(FOE_SPREAD_TARGET_TYPES) == ["allAdjacent", "allAdjacentFoes", "randomNormal"]


def stream_line(example: dict[str, Any]) -> str:
    return json.dumps(example)


def checker_for(example: dict[str, Any]) -> StreamChecker:
    return StreamChecker(
        {
            "schema_version": 1,
            "data_class": "synthetic",
            "episode_hashes": [example["episode_hash"]],
            "evaluation_manifest_hashes": [],
            "examples": 1,
            "stream_sha256": "0" * 64,
        }
    )


def test_the_stream_reader_validates_example_rows() -> None:
    example = sample("training-example/valid/battle-features.json")
    checker = checker_for(example)
    raw, _ = checker.line(1, stream_line(example))
    assert raw["candidate_features"]["rows"]
    bad = sample("training-example/reader-invalid/features-unsorted.json")
    with pytest.raises(TrainingStreamError, match="sorted and unique"):
        checker_for(bad).line(1, stream_line(bad))
    plain = sample("training-example/valid/battle.json")
    assert "candidate_features" not in plain


def test_a_stream_carries_one_feature_set_or_none() -> None:
    with_rows = sample("training-example/valid/battle-features.json")
    without = sample("training-example/valid/battle.json")
    other = copy.deepcopy(with_rows)
    other["candidate_features"]["feature_set_hash"] = "1" * 64
    for pair in ((with_rows, without), (with_rows, other)):
        checker = checker_for(with_rows)
        for number, example in enumerate(pair, start=1):
            checker.line(number, stream_line(example))
        with pytest.raises(TrainingStreamError, match="same candidate feature set"):
            checker.finish()
    same = checker_for(with_rows)
    same.line(1, stream_line(with_rows))
    same.line(2, stream_line(with_rows))
    same.finish()


# -- the statement comes from the checkpoints' own metadata ---------------------------------------


def cli(run_dir: Path, preview: str, battle: str) -> list[str]:
    return [
        "--run-dir", str(run_dir), "--preview", preview, "--battle", battle,
        "--engine-data", str(ENGINE_DATA), "--max-line-bytes", str(MAX_LINE),
    ]  # fmt: skip


def test_the_worker_states_the_feature_set_its_checkpoints_were_made_for(tmp_path: Path) -> None:
    query = featured()
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    run_dir, preview, battle = make_run_dir(tmp_path / "on", feature_model(stated(query)))
    worker = build(cli(run_dir, preview, battle))
    assert worker._ready["message"]["feature_set"] == stated(query)
    # ... and it answers the query that carries the rows, refusing one that does not.
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[0]["message"]["feature_set"] == stated(query)
    assert frames[1]["reply"]["kind"] == "recommendation"
    code, frames = session(worker, line(hello()), line(battle_query()))
    assert frames[1]["reply"]["error"] == "invalid_query"
    # A pair made without the option states nothing, and takes no rows.
    run_dir, preview, battle = make_run_dir(tmp_path / "off")
    assert "feature_set" not in build(cli(run_dir, preview, battle))._ready["message"]


def test_a_pair_made_with_different_feature_sets_is_refused(tmp_path: Path) -> None:
    query = featured()
    run_dir, preview, _ = make_run_dir(tmp_path, TINY_MODEL)  # a Preview without the option
    data = tiny_checkpoint("battle", feature_model(stated(query)))
    write_checkpoint(run_dir, data)
    with pytest.raises(ScorerError, match="different encoders"):
        build(cli(run_dir, preview, checkpoint_id(data)))


# ---- probe-damage/2 (ADR-022 erratum E-22.3.2)


def featured_two() -> dict[str, Any]:
    return sample("policy-query/valid/battle-features-2.json")


ARRAYS = ("threats", "speeds", "own_priorities", "foe_priorities")


def test_the_valid_probe_damage_2_samples_are_accepted_under_their_set() -> None:
    query = featured_two()
    assert all(query["candidate_features"][name] for name in ARRAYS)
    assert read_query(query, support(query)) == query
    preview = sample("policy-query/valid/preview-features-2.json")
    assert read_query(preview, support(preview))["candidate_features"]["speeds"] == []
    check("policy-handshake", sample("policy-handshake/valid/ready-features-2.json"))
    check("training-example", sample("training-example/valid/battle-features-2.json"))
    # the two limits only the order arrays carry
    shown = json.dumps([query["candidate_features"][n] for n in ("speeds", "own_priorities")])
    assert "unqueueable" in shown
    assert "fractional_priority" in shown


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("array-under-1", "present exactly under probe-damage/2"),
        ("missing-array", "present exactly under probe-damage/2"),
        ("threats-unsorted", "threats must be sorted and unique"),
        ("speeds-duplicate", "speeds must be sorted and unique"),
        ("priorities-unsorted", "own_priorities must be sorted and unique"),
        ("foe-priorities-unsorted", "foe_priorities must be sorted and unique"),
    ],
)
def test_probe_damage_2_structure_is_a_reader_rule(name: str, message: str) -> None:
    query = sample(f"policy-query/reader-invalid/features-2-{name}.json")
    check("policy-query", query)  # schema-valid
    assert rows_problem(query["candidate_features"]) is not None
    with pytest.raises(QueryError, match=message):
        read_query(query, support(query))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("missing-speed-row", "speeds must hold exactly one row"),
        ("missing-own-priority", "own_priorities must hold one row"),
        ("missing-foe-priority", "foe_priorities must hold a row for the foe move"),
        ("foreign-own", "threats name an own Pokemon"),
    ],
)
def test_probe_damage_2_arrays_are_judged_against_the_request(name: str, message: str) -> None:
    query = sample(f"policy-query/reader-invalid/features-2-{name}.json")
    check("policy-query", query)
    assert rows_problem(query["candidate_features"]) is None
    with pytest.raises(QueryError, match=message):
        read_query(query, support(query))


def test_a_missing_probe_damage_2_row_is_an_invalid_query_reply(tmp_path: Path) -> None:
    # No encoder reads /2 yet, so no checkpoint states it: the statement is given directly.
    query = sample("policy-query/reader-invalid/features-2-missing-speed-row.json")
    plain = make_worker(tmp_path)
    worker = Worker(
        plain._scorer,
        plain._ready["message"]["engine_data_hash"],
        plain._ready["message"]["checkpoints"],
        MAX_LINE,
        support(query),
    )
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[0]["message"]["feature_set"] == stated(query)
    assert frames[1]["reply"]["error"] == "invalid_query"
    assert "speeds must hold exactly one row" in frames[1]["reply"]["detail"]


def test_the_probe_damage_2_coverage_rule_is_the_typescript_one() -> None:
    side = {"pokemon": [{"condition": c} for c in ("100/100", "100/100", "100/100", "0 fnt")]}
    protect = {"id": "protect", "target": "self"}

    def features(**given: list[dict[str, Any]]) -> dict[str, Any]:
        empty: dict[str, list[dict[str, Any]]] = {name: [] for name in ARRAYS}
        return {"feature_set": "probe-damage/2", "rows": [], **empty, **given}

    def speed(own: int, mega: bool, foe: int) -> dict[str, Any]:
        return {"kind": "computed", "own": own, "mega": mega, "foe": foe, "relation": "tie"}

    def prio(slot: int, move: int, mega: bool) -> dict[str, Any]:
        return {"kind": "computed", "slot": slot, "move": move, "mega": mega, "priority": 0}

    def problem(request: dict[str, Any], choices: list[str], given: dict[str, Any]) -> str | None:
        query = {
            "observation": {"log": ["|request|" + json.dumps(request)]},
            "candidates": [{"choice": c} for c in choices],
        }
        return coverage_problem(query, given, DAMAGING)

    move_request = {"active": [{"moves": [protect]}, {"moves": [protect]}], "side": side}
    both = [prio(0, 1, False), prio(1, 1, False)]
    base = [speed(0, False, 1), speed(1, False, 1)]
    plain = ["move 1, move 1"]
    assert problem(move_request, plain, features(speeds=base, own_priorities=both)) is None
    with_bench = ["move 1, move 1", "switch 3, move 1"]
    bench = [*base, speed(2, False, 1)]
    assert problem(move_request, with_bench, features(speeds=bench, own_priorities=both)) is None
    assert "speeds must hold" in (
        problem(move_request, plain, features(speeds=bench, own_priorities=both)) or ""
    )
    # a fainted bench Pokemon has no row; one the request does not list is refused
    fainted = features(speeds=base, own_priorities=[prio(1, 1, False)])
    assert problem(move_request, ["switch 4, move 1"], fainted) is None
    assert "does not list" in (
        problem(move_request, ["switch 7, move 1"], features(speeds=base, own_priorities=both))
        or ""
    )
    # the Mega variant
    mega_speeds = [speed(0, False, 1), speed(0, True, 1), speed(1, False, 1)]
    mega = ["move 1 mega, move 1"]
    good = features(speeds=mega_speeds, own_priorities=[prio(0, 1, True), prio(1, 1, False)])
    assert problem(move_request, mega, good) is None
    extra = features(
        speeds=mega_speeds, own_priorities=[prio(0, 1, False), *good["own_priorities"]]
    )
    assert "own_priorities must hold" in (problem(move_request, mega, extra) or "")
    assert "speeds must hold" in (
        problem(move_request, plain, features(speeds=mega_speeds, own_priorities=both)) or ""
    )
    # a forced switch and Team Preview
    forced = {"forceSwitch": [True, False], "side": side}
    assert problem(forced, ["switch 3, pass"], features(speeds=[speed(2, False, 1)])) is None
    threat = {
        "kind": "computed", "own": 2, "mega": False, "foe": 1, "move": "protect",
        "mean_ppm": 1, "max_ppm": 1, "min_ppm": 1, "ohko_rolls": 0,
    }  # fmt: skip
    # the builder's forced-switch shape: threats for the switch-in, no foe priorities
    assert (
        problem(forced, ["switch 3, pass"], features(speeds=[speed(2, False, 1)], threats=[threat]))
        is None
    )
    foe = {"kind": "computed", "foe": 1, "move": "protect", "priority": 4}
    assert "foe_priorities are empty" in (
        problem(
            forced, ["switch 3, pass"], features(speeds=[speed(2, False, 1)], foe_priorities=[foe])
        )
        or ""
    )
    preview = {"teamPreview": True, "side": side}
    assert problem(preview, ["team 1234"], features()) is None
    assert "speeds must hold" in (
        problem(preview, ["team 1234"], features(speeds=[speed(0, False, 1)])) or ""
    )
    threat = {
        "kind": "computed", "own": 2, "mega": False, "foe": 2, "move": "tackle",
        "mean_ppm": 1, "max_ppm": 1, "min_ppm": 1, "ohko_rolls": 0,
    }  # fmt: skip
    only = features(speeds=[speed(2, False, 1)], threats=[threat])
    assert "speeds must hold" in (problem(forced, ["switch 3, pass"], only) or "")


def test_the_stream_reader_validates_probe_damage_2_arrays() -> None:
    example = sample("training-example/valid/battle-features-2.json")
    raw, _ = checker_for(example).line(1, stream_line(example))
    assert raw["candidate_features"]["speeds"]
    for name, message in (
        ("array-under-1", "present exactly under probe-damage/2"),
        ("missing-array", "present exactly under probe-damage/2"),
        ("speeds-unsorted", "speeds must be sorted and unique"),
    ):
        bad = sample(f"training-example/reader-invalid/features-2-{name}.json")
        with pytest.raises(TrainingStreamError, match=message):
            checker_for(bad).line(1, stream_line(bad))
    # unsorted arrays of the other three kinds
    for array in ("threats", "own_priorities", "foe_priorities"):
        broken = copy.deepcopy(example)
        broken["candidate_features"][array].reverse()
        with pytest.raises(TrainingStreamError, match=f"{array} must be sorted and unique"):
            checker_for(broken).line(1, stream_line(broken))
    # a /2 and a /1 example in one stream are two feature sets
    other = copy.deepcopy(sample("training-example/valid/battle-features.json"))
    checker = checker_for(example)
    checker.line(1, stream_line(example))
    checker.line(2, stream_line(other))
    with pytest.raises(TrainingStreamError, match="same candidate feature set"):
        checker.finish()


def test_the_worker_states_the_second_set_and_answers_a_query_with_its_arrays(
    tmp_path: Path,
) -> None:
    query = sample("policy-query/valid/battle-features-2.json")
    run_dir, preview, battle = make_run_dir(tmp_path, feature_model(stated(query)))
    worker = build(cli(run_dir, preview, battle))
    assert worker._ready["message"]["feature_set"] == stated(query)
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    # a query missing an array its checkpoint needs is an invalid query, not a guess
    short = copy.deepcopy(query)
    del short["candidate_features"]["speeds"]
    _, frames = session(worker, line(hello()), line(short))
    assert frames[1]["reply"]["error"] == "invalid_query"
    # and one that lacks a row the arrays must cover is refused by the encoder as invalid too
    thin = copy.deepcopy(query)
    thin["candidate_features"]["own_priorities"] = thin["candidate_features"]["own_priorities"][1:]
    _, frames = session(worker, line(hello()), line(thin))
    assert frames[1]["reply"]["error"] == "invalid_query"


# ---- probe-damage/3 (ADR-022, belief_threats)


def featured_three() -> dict[str, Any]:
    return sample("policy-query/valid/battle-features-3.json")


def test_the_valid_probe_damage_3_samples_are_accepted_under_their_set() -> None:
    query = featured_three()
    assert query["candidate_features"]["belief_threats"]
    assert read_query(query, support(query)) == query
    preview = sample("policy-query/valid/preview-features-3.json")
    assert read_query(preview, support(preview))["candidate_features"]["belief_threats"] == []
    check("policy-handshake", sample("policy-handshake/valid/ready-features-3.json"))
    check("training-example", sample("training-example/valid/battle-features-3.json"))


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("belief-under-2", "present exactly under probe-damage/2"),
        ("missing-belief-threats", "present exactly under probe-damage/2"),
        ("belief-unsorted", "belief_threats must be sorted and unique"),
        ("belief-repeats-revealed", "repeats a move already in threats"),
    ],
)
def test_probe_damage_3_structure_is_a_reader_rule(name: str, message: str) -> None:
    query = sample(f"policy-query/reader-invalid/features-3-{name}.json")
    check("policy-query", query)  # schema-valid
    assert rows_problem(query["candidate_features"]) is not None
    with pytest.raises(QueryError, match=message):
        read_query(query, support(query))


def test_a_belief_threat_of_an_own_pokemon_the_request_does_not_imply_is_refused() -> None:
    query = sample("policy-query/reader-invalid/features-3-belief-foreign-own.json")
    check("policy-query", query)
    assert rows_problem(query["candidate_features"]) is None
    with pytest.raises(QueryError, match="belief_threats name an own Pokemon"):
        read_query(query, support(query))


def test_the_stream_reader_validates_probe_damage_3_arrays() -> None:
    example = sample("training-example/valid/battle-features-3.json")
    raw, _ = checker_for(example).line(1, stream_line(example))
    assert raw["candidate_features"]["belief_threats"]
    for name, message in (
        ("belief-under-2", "present exactly under probe-damage/2"),
        ("belief-unsorted", "belief_threats must be sorted and unique"),
    ):
        bad = sample(f"training-example/reader-invalid/features-3-{name}.json")
        with pytest.raises(TrainingStreamError, match=message):
            checker_for(bad).line(1, stream_line(bad))
    repeated = copy.deepcopy(example)
    first = repeated["candidate_features"]["threats"][0]
    repeated["candidate_features"]["belief_threats"][0].update(
        foe=first["foe"], move=first["move"], own=first["own"], mega=first["mega"]
    )
    with pytest.raises(TrainingStreamError, match="repeats a move already in threats"):
        checker_for(repeated).line(1, stream_line(repeated))


def test_the_worker_states_the_third_set_and_answers_a_query_with_its_belief_threats(
    tmp_path: Path,
) -> None:
    query = featured_three()
    run_dir, preview, battle = make_run_dir(tmp_path, feature_model(stated(query)))
    worker = build(cli(run_dir, preview, battle))
    assert worker._ready["message"]["feature_set"] == stated(query)
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    short = copy.deepcopy(query)
    del short["candidate_features"]["belief_threats"]
    _, frames = session(worker, line(hello()), line(short))
    assert frames[1]["reply"]["error"] == "invalid_query"


def test_a_refused_belief_is_a_stated_field_with_no_rows() -> None:
    query = sample("policy-query/valid/battle-features-3-refused.json")
    check("policy-query", query)
    assert read_query(query, support(query)) == query
    check("training-example", sample("training-example/valid/battle-features-3-refused.json"))
    for name in ("refused-with-rows", "refused-under-2"):
        bad = sample(f"policy-query/reader-invalid/features-3-{name}.json")
        check("policy-query", bad)
        with pytest.raises(QueryError, match="belief_threats_refused"):
            read_query(bad, support(bad))
    raw = sample("training-example/valid/battle-features-3-refused.json")
    broken = copy.deepcopy(raw)
    broken["candidate_features"]["belief_threats"] = sample(
        "training-example/valid/battle-features-3.json"
    )["candidate_features"]["belief_threats"]
    with pytest.raises(TrainingStreamError, match="belief_threats_refused"):
        checker_for(broken).line(1, stream_line(broken))


# ---- probe-outcome/1 (ADR-022 erratum E-22.5.3)

OUTCOME_ARRAYS = (*ARRAYS, "outcomes")


def featured_outcome() -> dict[str, Any]:
    return sample("policy-query/valid/battle-features-outcome.json")


def test_the_valid_probe_outcome_samples_are_accepted_under_their_set() -> None:
    query = featured_outcome()
    assert all(query["candidate_features"][name] for name in OUTCOME_ARRAYS)
    assert read_query(query, support(query)) == query
    preview = sample("policy-query/valid/preview-features-outcome.json")
    assert read_query(preview, support(preview))["candidate_features"]["outcomes"] == []
    check("policy-handshake", sample("policy-handshake/valid/ready-features-outcome.json"))
    check("training-example", sample("training-example/valid/battle-features-outcome.json"))
    outcomes = query["candidate_features"]["outcomes"]
    assert {r.get("precondition") for r in outcomes} == {None, "passes", "chance", "fails"}
    assert any(r["assumed"] for r in outcomes)
    assert any(r.get("reason") == "history_unknown" for r in outcomes)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("under-2", "present exactly under probe-damage/2"),
        ("missing-array", "with outcomes under probe-outcome/1"),
        ("missing-speeds", "present exactly under probe-damage/2"),
        ("unsorted", "outcomes must be sorted and unique"),
        ("duplicate", "outcomes must be sorted and unique"),
        ("missing-switch-branch", "starts with its switch branch"),
        ("missing-move-branch", "a switch branch and one to four move branches"),
        ("assumed-with-revealed", "an assumed branch is the only move branch"),
        ("switch-with-foe-move", "a switch branch has no foe move"),
        ("success-mismatch", "success_ppm must be floor"),
        ("precondition-mismatch", "the precondition does not fit its ppm"),
        ("history-unknown-under-2", "history_unknown is a probe-outcome/1 value"),
    ],
)
def test_probe_outcome_structure_is_a_reader_rule(name: str, message: str) -> None:
    query = sample(f"policy-query/reader-invalid/features-outcome-{name}.json")
    check("policy-query", query)  # schema-valid
    assert rows_problem(query["candidate_features"]) is not None
    with pytest.raises(QueryError, match=message):
        read_query(query, support(query))


@pytest.mark.parametrize("name", ["missing-part", "foreign-part"])
def test_probe_outcome_blocks_are_judged_against_the_request(name: str) -> None:
    query = sample(f"policy-query/reader-invalid/features-outcome-{name}.json")
    check("policy-query", query)
    assert rows_problem(query["candidate_features"]) is None
    with pytest.raises(QueryError, match="outcomes must hold the rows of exactly the own move"):
        read_query(query, support(query))


def test_probe_outcome_value_rules() -> None:
    features = featured_outcome()["candidate_features"]

    def changed(index: int, **change: Any) -> dict[str, Any]:
        copied: dict[str, Any] = copy.deepcopy(features)
        copied["outcomes"][index].update(change)
        return copied

    # row 1 is part 0's `protect` branch, row 2 its `thunderbolt` branch (passes, 800000)
    wrong = (
        changed(2, precondition_ppm=500000, success_ppm=400000),
        changed(2, precondition="fails"),
        changed(2, precondition="chance", precondition_ppm=0, success_ppm=0),
    )
    for broken in wrong:
        assert "the precondition does not fit its ppm" in (rows_problem(broken) or "")
    assert "a move branch names its foe move" in (rows_problem(changed(1, foe_move=None)) or "")
    assert "an assumed branch is the only move branch" in (
        rows_problem(changed(2, assumed=True)) or ""
    )
    extra = copy.deepcopy(features)
    rows = extra["outcomes"]
    more = [{**rows[2], "foe_move": move} for move in ("uturn", "xscissor", "zenheadbutt")]
    five = [*rows[:3], *more, *rows[3:]]
    assert "one to four move branches" in (rows_problem({**extra, "outcomes": five}) or "")
    four = [*rows[:3], *more[:2], *rows[3:]]
    assert rows_problem({**extra, "outcomes": four}) is None
    assert "starts with its switch branch" in (rows_problem({**extra, "outcomes": rows[1:]}) or "")


def test_history_unknown_is_this_sets_reason_on_any_array() -> None:
    base = sample("policy-query/valid/battle-features-2.json")["candidate_features"]
    unknown = {"kind": "unsupported", "reason": "history_unknown"}
    threat = {**unknown, "own": 0, "mega": False, "foe": 1, "move": "protect"}
    row = {**unknown, "slot": 0, "move": 1, "target": 1, "mega": False}
    for broken in ({**base, "threats": [threat]}, {**base, "rows": [row]}):
        assert "history_unknown is a probe-outcome/1 value" in (rows_problem(broken) or "")
    mine = featured_outcome()["candidate_features"]
    assert rows_problem({**mine, "threats": [threat]}) is None


def request_for(moves: list[dict[str, str]], conditions: list[str]) -> dict[str, Any]:
    return {
        "active": [{"moves": moves}, {"moves": moves}],
        "side": {"pokemon": [{"condition": c} for c in conditions]},
    }


def test_probe_outcome_blocks_follow_the_candidates_foes_and_living_actives() -> None:
    protect = {"id": "protect"}

    def outcome_part(slot: int, move: int, target: int, mega: bool = False) -> list[dict[str, Any]]:
        key = {"slot": slot, "move": move, "target": target, "mega": mega}
        done = {"precondition": "passes", "precondition_ppm": 10**6, "hit_ppm": 10**6}
        return [
            {"kind": "computed", **key, "action": "switch", "foe_move": None, "assumed": False}
            | done
            | {"success_ppm": 10**6},
            {**key, "kind": "unsupported", "action": "move", "foe_move": "protect"}
            | {"assumed": True, "reason": "history_unknown"},
        ]

    def speed(own: int, foe: int, mega: bool = False) -> dict[str, Any]:
        return {"kind": "computed", "own": own, "mega": mega, "foe": foe, "relation": "tie"}

    def prio(slot: int, move: int, mega: bool = False) -> dict[str, Any]:
        return {"kind": "computed", "slot": slot, "move": move, "mega": mega, "priority": 0}

    def problem(request: dict[str, Any], choices: list[str], **given: Any) -> str | None:
        empty: dict[str, list[Any]] = {name: [] for name in OUTCOME_ARRAYS}
        features = {"feature_set": "probe-outcome/1", "rows": [], **empty, **given}
        query = {
            "observation": {"log": ["|request|" + json.dumps(request)]},
            "candidates": [{"choice": c} for c in choices],
        }
        return coverage_problem(query, features, DAMAGING)

    live = ["100/100", "100/100", "100/100"]
    request = request_for([protect], live)
    base = {"speeds": [speed(0, 1), speed(1, 1)], "own_priorities": [prio(0, 1), prio(1, 1)]}
    untargeted = ["move 1, move 1 -1"]
    both = [*outcome_part(0, 1, 0), *outcome_part(1, 1, -1)]
    assert problem(request, untargeted, **base, outcomes=both) is None
    assert "outcomes must hold" in (problem(request, untargeted, **base) or "")
    # the second foe is down (the arrays show a foe at 1 only): no block for location 2
    aimed = ["move 1 1, move 1 2"]
    assert problem(request, aimed, **base, outcomes=outcome_part(0, 1, 1)) is None
    two_blocks = [*outcome_part(0, 1, 1), *outcome_part(1, 1, 2)]
    assert "outcomes must hold" in (problem(request, aimed, **base, outcomes=two_blocks) or "")
    foes = {"speeds": [speed(0, 1), speed(0, 2), speed(1, 1), speed(1, 2)]}
    assert (
        problem(request, aimed, **foes, own_priorities=base["own_priorities"], outcomes=two_blocks)
        is None
    )
    # a fainted active has no block, a forced switch and Team Preview have none
    fainted = request_for([protect], ["100/100", "0 fnt", "100/100"])
    one = {"speeds": [speed(0, 1)], "own_priorities": [prio(0, 1)]}
    assert problem(fainted, ["move 1 1, pass"], **one, outcomes=outcome_part(0, 1, 1)) is None
    extra = [*outcome_part(0, 1, 1), *outcome_part(1, 1, 1)]
    assert "outcomes must hold" in (
        problem(fainted, ["move 1 1, pass"], **one, outcomes=extra) or ""
    )
    forced = {"forceSwitch": [True, False], "side": request["side"]}
    assert problem(forced, ["switch 3, pass"], speeds=[speed(2, 1)]) is None
    stray = problem(
        forced, ["switch 3, pass"], speeds=[speed(2, 1)], outcomes=outcome_part(0, 1, 1)
    )
    assert "outcomes must hold" in (stray or "")
    # the Mega variant is a part of its own
    mega = {
        "speeds": [speed(0, 1), speed(0, 1, True), speed(1, 1)],
        "own_priorities": [prio(0, 1, True), prio(1, 1)],
    }
    blocks = [*outcome_part(0, 1, 1, True), *outcome_part(1, 1, 1)]
    assert problem(request, ["move 1 1 mega, move 1 1"], **mega, outcomes=blocks) is None


def test_the_stream_reader_validates_probe_outcome_arrays() -> None:
    example = sample("training-example/valid/battle-features-outcome.json")
    raw, _ = checker_for(example).line(1, stream_line(example))
    assert raw["candidate_features"]["outcomes"]
    for name, message in (
        ("under-2", "present exactly under probe-damage/2"),
        ("missing-array", "present exactly under probe-damage/2"),
        ("unsorted", "outcomes must be sorted and unique"),
        ("missing-switch-branch", "starts with its switch branch"),
        ("success-mismatch", "success_ppm must be floor"),
        ("history-unknown-under-2", "history_unknown is a probe-outcome/1 value"),
    ):
        bad = sample(f"training-example/reader-invalid/features-outcome-{name}.json")
        with pytest.raises(TrainingStreamError, match=message):
            checker_for(bad).line(1, stream_line(bad))
    # the value rules of the stream reader
    for change, message in (
        ({"precondition_ppm": 500000, "success_ppm": 400000}, "precondition does not fit"),
        ({"precondition": "fails"}, "precondition does not fit"),
        ({"precondition": "chance", "precondition_ppm": 0, "success_ppm": 0}, "does not fit"),
    ):
        broken = copy.deepcopy(example)
        broken["candidate_features"]["outcomes"][2].update(change)
        with pytest.raises(TrainingStreamError, match=message):
            checker_for(broken).line(1, stream_line(broken))
    for index, edit, message in (
        (1, {"foe_move": None}, "a move branch names its foe move"),
        (2, {"assumed": True}, "an assumed branch is the only move branch"),
    ):
        broken = copy.deepcopy(example)
        broken["candidate_features"]["outcomes"][index].update(edit)
        with pytest.raises(TrainingStreamError, match=message):
            checker_for(broken).line(1, stream_line(broken))
    rows = example["candidate_features"]["outcomes"]
    more = [{**rows[2], "foe_move": move} for move in ("uturn", "xscissor", "zenheadbutt")]
    broken = copy.deepcopy(example)
    broken["candidate_features"]["outcomes"] = [*rows[:3], *more, *rows[3:]]
    with pytest.raises(TrainingStreamError, match="one to four move branches"):
        checker_for(broken).line(1, stream_line(broken))
    broken["candidate_features"]["outcomes"] = rows[1:]
    with pytest.raises(TrainingStreamError, match="starts with its switch branch"):
        checker_for(broken).line(1, stream_line(broken))
    switch = copy.deepcopy(example)
    switch["candidate_features"]["outcomes"][0]["foe_move"] = "protect"
    with pytest.raises(TrainingStreamError, match="a switch branch has no foe move"):
        checker_for(switch).line(1, stream_line(switch))
    # history_unknown under /2, on every array kind
    two = sample("training-example/valid/battle-features-2.json")
    two["candidate_features"]["threats"][0] = {
        "kind": "unsupported",
        "own": two["candidate_features"]["threats"][0]["own"],
        "mega": False,
        "foe": 1,
        "move": two["candidate_features"]["threats"][0]["move"],
        "reason": "history_unknown",
    }
    with pytest.raises(TrainingStreamError, match="history_unknown is a probe-outcome/1 value"):
        checker_for(two).line(1, stream_line(two))


def test_the_worker_states_the_outcome_set_and_answers_a_query_with_its_outcomes(
    tmp_path: Path,
) -> None:
    query = sample("policy-query/valid/battle-features-outcome.json")
    run_dir, preview, battle = make_run_dir(tmp_path, feature_model(stated(query)))
    worker = build(cli(run_dir, preview, battle))
    assert worker._ready["message"]["feature_set"] == stated(query)
    code, frames = session(worker, line(hello()), line(query))
    assert code == EXIT_OK
    assert frames[1]["reply"]["kind"] == "recommendation"
    # a query missing the outcomes its checkpoint needs is an invalid query, not a guess
    short = copy.deepcopy(query)
    del short["candidate_features"]["outcomes"]
    _, frames = session(worker, line(hello()), line(short))
    assert frames[1]["reply"]["error"] == "invalid_query"
    # and one whose outcomes miss a block the candidates need is refused as invalid too
    thin = copy.deepcopy(query)
    thin["candidate_features"]["outcomes"] = [
        r for r in thin["candidate_features"]["outcomes"] if (r["slot"], r["move"]) != (0, 1)
    ]
    _, frames = session(worker, line(hello()), line(thin))
    assert frames[1]["reply"]["error"] == "invalid_query"


def test_the_branch_ceiling_is_the_one_outcome_config_allows() -> None:
    source = (REPO / "packages/showdown-bridge/src/outcome-config.ts").read_text()
    found = re.search(r"MAX_BRANCHES_LIMIT = (\d+);", source)
    assert found is not None
    assert int(found.group(1)) == MAX_OUTCOME_BRANCHES


def test_a_malformed_request_is_a_coverage_problem_not_a_crash() -> None:
    protect = {"id": "protect"}
    # two active slots but one Pokemon listed: the readers refuse, they do not raise
    request = request_for([protect], ["100/100"])
    for choices in (["move 1 1, move 1 1"], ["move 1 1, pass"], ["switch 5, pass"]):
        query = {
            "observation": {"log": ["|request|" + json.dumps(request)]},
            "candidates": [{"choice": c} for c in choices],
        }
        for name in ("probe-damage/2", "probe-outcome/1"):
            empty: dict[str, list[Any]] = {n: [] for n in OUTCOME_ARRAYS}
            features = {"feature_set": name, "rows": [], **empty}
            assert coverage_problem(query, features, DAMAGING) is not None


def test_a_candidate_beyond_the_requests_actives_or_moves_is_a_problem_not_a_crash() -> None:
    request = request_for([{"id": "protect"}], ["100/100"] * 2)
    request["active"] = request["active"][:1]  # one active entry
    for choice in ("move 1 1, move 1 1", "move 2 1, pass"):
        query = {
            "observation": {"log": ["|request|" + json.dumps(request)]},
            "candidates": [{"choice": choice}],
        }
        features = {"feature_set": "probe-damage/1", "rows": []}
        problem = coverage_problem(query, features, DAMAGING)
        assert problem is not None
        assert "does not list" in problem
