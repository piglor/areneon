"""Anchor calibration and the per-generation ladder record (ADR-021 sections 4 and 8, slice 7)."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import ladder_write as lw
from areneon_ml.loop.ladder import read_ladder_record
from areneon_ml.loop.league import calibration_game_seed, league_game_seed
from areneon_ml.loop.rating import SCALE

from tests.ml.loop.test_population import ANCHORS, population_config

HASHES = ["a" * 64, "b" * 64]


def pair_rows(
    pair: int,
    kind: str,
    member: str,
    opponent: str,
    results: tuple[str | None, str | None],
) -> list[dict[str, Any]]:
    """The two game rows of a team pair; a result is who won each half (or None for a tie)."""
    rows = []
    for half, who in enumerate(results):
        seats = {"p1": member, "p2": opponent} if half == 0 else {"p1": opponent, "p2": member}
        winner = None
        if who is not None:
            name = member if who == "member" else opponent
            winner = "p1" if seats["p1"] == name else "p2"
        rows.append(
            {
                "game": 2 * pair + half,
                "half": half,
                "kind": kind,
                "member": member,
                "opponent": opponent,
                "outcome": "played",
                "pair": pair,
                "seats": seats,
                "seed": league_game_seed("pop-tiny", 1, pair),
                "turns": 5,
                "winner": winner,
            }
        )
    return rows


def generation_rows(
    ladder: Sequence[str | None] = ("member", "member"),
    anchor: Sequence[str | None] = ("member", "member"),
) -> list[dict[str, Any]]:
    """One generation: m0 and m1 meet on the ladder and each plays both anchors once."""
    rows = pair_rows(0, "ladder", "m0", "m1", (ladder[0], ladder[1]))
    pair = 1
    for member in ("m0", "m1"):
        for baseline in ANCHORS:
            rows += pair_rows(
                pair, "anchor", member, str(baseline["policy_id"]), (anchor[0], anchor[1])
            )
            pair += 1
    rows += pair_rows(pair, "self", "m0", "m0", (None, None))
    return rows


ANCHOR_STRENGTHS = {"actor-rule@v1": SCALE, "random-legal@v1": 700_000_000}


# -- the anchors -------------------------------------------------------------------------------


def test_the_anchors_are_the_configured_baselines_with_a_known_id_and_actor() -> None:
    assert lw.anchor_ids(population_config()) == ["actor-rule@v1", "random-legal@v1"]

    def named(*baselines: tuple[str, str, int]) -> RunConfig:
        found = [{"actor": a, "policy_id": p, "games": g} for a, p, g in baselines]
        return population_config(edit=lambda raw: raw["generation"].update(baseline_games=found))

    three = named(
        ("actor-rule", "actor-rule@v1", 2),
        ("damage-aware", "damage-aware@v1", 2),
        ("random-legal", "random-legal@tiny", 2),
    )
    assert lw.anchor_ids(three) == ["actor-rule@v1", "damage-aware@v1"]
    # A known id with the wrong actor kind is no anchor.
    wrong = named(("actor-rule", "actor-rule@v1", 2), ("damage-aware", "random-legal@v1", 2))
    assert lw.anchor_ids(wrong) == ["actor-rule@v1"]


@pytest.mark.parametrize(
    "baselines",
    [
        [],
        [{"actor": "random-legal", "policy_id": "random-legal@v1", "games": 2}],
        [{"actor": "actor-rule", "policy_id": "actor-rule@v1", "games": 1}],
    ],
)
def test_a_run_without_the_reference_anchor_is_refused(baselines: list[dict[str, Any]]) -> None:
    config = population_config(edit=lambda raw: raw["generation"].update(baseline_games=baselines))
    with pytest.raises(lw.LadderBuildError, match="actor-rule@v1, the ladder's reference anchor"):
        lw.anchor_ids(config)


# -- pair scores ---------------------------------------------------------------------------------


def test_a_pair_score_is_the_sum_of_two_half_scores_and_a_forfeit_is_a_loss() -> None:
    cases = {
        ("member", "member"): 4,
        ("member", None): 3,
        ("member", "opponent"): 2,
        (None, None): 2,
        ("opponent", None): 1,
        ("opponent", "opponent"): 0,
    }
    for results, bucket in cases.items():
        rows = pair_rows(0, "ladder", "m0", "m1", results)
        assert lw.pair_buckets(rows, lambda r: r["member"])[0][1] == bucket
    # A forfeit has a winner (the other side), so it scores as the loss it is.
    rows = pair_rows(0, "ladder", "m0", "m1", ("opponent", "member"))
    rows[0].update(outcome="forfeit")
    assert lw.pair_buckets(rows, lambda r: r["member"])[0][1] == 2


def test_a_pair_needs_exactly_its_two_games() -> None:
    rows = pair_rows(0, "ladder", "m0", "m1", ("member", "member"))
    with pytest.raises(lw.LadderBuildError, match="exactly its two games"):
        lw.pair_buckets(rows[:1], lambda r: r["member"])
    with pytest.raises(lw.LadderBuildError, match="different subjects"):
        lw.pair_buckets(rows, lambda r: r["game"])
    selfish = pair_rows(0, "self", "m0", "m0", ("member", "member"))
    with pytest.raises(lw.LadderBuildError, match="against itself"):
        lw.pair_buckets(selfish, lambda r: r["member"])


# -- calibration -----------------------------------------------------------------------------------


def calibration(config: RunConfig, winners: dict[int, str | None] | None = None) -> dict[str, Any]:
    ids = lw.anchor_ids(config)
    pairs = config.raw["population"]["rating"]["calibration_pairs"]
    games = []
    index = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            for _ in range(pairs):
                pair = index // 2
                for half in (0, 1):
                    seats = {"p1": a, "p2": b} if half == 0 else {"p1": b, "p2": a}
                    games.append(
                        {
                            "a": a,
                            "b": b,
                            "game": index,
                            "half": half,
                            "outcome": "played",
                            "pair": pair,
                            "seats": seats,
                            "seed": calibration_game_seed(config.run_key, pair),
                            "turns": 4,
                            # b (random-legal) loses every game unless a winner is forced
                            "winner": (winners or {}).get(index, "p1" if half == 0 else "p2"),
                        }
                    )
                    index += 1
    return {
        "schema_version": 1,
        "format": "population-calibration/1",
        "run_key": config.run_key,
        "run_config_hash": config.hash,
        "calibration_pairs": pairs,
        "anchors": ids,
        "games": games,
        "manifest": {},
        "team_hashes": HASHES,
    }


def test_a_calibration_is_fitted_against_the_reference_and_frozen_in_anchors_json() -> None:
    config = population_config()
    read = lw.read_calibration(calibration(config), config)
    assert [(c.player, c.opponent, c.scores) for c in read.cells] == [
        ("actor-rule@v1", "random-legal@v1", (0, 0, 0, 0, 3))
    ]
    assert read.team_hashes == tuple(HASHES)
    strengths = lw.fit_anchors(read)
    assert strengths["actor-rule@v1"] == SCALE
    assert strengths["random-legal@v1"] < SCALE  # it lost every pair to the reference
    document = lw.anchors_document(config, read, strengths)
    assert lw.read_anchors(document, config) == strengths
    assert document["cells"][0]["pairs"] == 3
    assert document["cells"][0]["score_sum_ppm"] == 3_000_000


def test_a_calibration_with_the_reference_alone_has_nothing_to_fit() -> None:
    config = population_config(
        edit=lambda raw: raw["generation"].update(baseline_games=[dict(ANCHORS[1])])
    )
    raw = calibration(config)
    assert raw["games"] == []
    read = lw.read_calibration(raw, config)
    assert lw.fit_anchors(read) == {"actor-rule@v1": SCALE}


def refused(change: Callable[[dict[str, Any]], object], match: str) -> None:
    config = population_config()
    raw = calibration(config)
    change(raw)
    with pytest.raises(lw.LadderBuildError, match=match):
        lw.read_calibration(raw, config)


def test_a_calibration_that_is_not_the_plan_is_refused() -> None:
    refused(lambda r: r.update(schema_version=2), "not a version 1")
    refused(lambda r: r.update(run_key="other"), "another run or config")
    refused(lambda r: r.update(run_config_hash="0" * 64), "another run or config")
    refused(lambda r: r.update(calibration_pairs=9), "other anchors or another number")
    refused(lambda r: r.update(anchors=["actor-rule@v1"]), "other anchors or another number")
    refused(lambda r: r["games"].pop(), "another number of games")
    refused(lambda r: r["games"][2].update(seed=league_game_seed("pop-tiny", 1, 1)), "game 2")
    refused(lambda r: r["games"][0].update(seats={"p1": "a", "p2": "b"}), "game 0")
    refused(lambda r: r["games"][0].update(winner="p3"), "game 0")
    refused(lambda r: r["games"][3].update(outcome="forfeit", winner=None), "game 3")
    refused(lambda r: r.update(team_hashes=["nothex"]), "team_hashes")
    refused(lambda r: r.update(team_hashes=HASHES[::-1]), "team_hashes")
    with pytest.raises(lw.LadderBuildError, match="not an object"):
        lw.read_calibration([], population_config())


def test_anchors_json_is_this_runs_and_names_the_anchors() -> None:
    config = population_config()
    read = lw.read_calibration(calibration(config), config)
    document = lw.anchors_document(config, read, lw.fit_anchors(read))
    cases: list[tuple[Callable[[dict[str, Any]], object], str]] = [
        (lambda d: d.update(schema_version=2), "not a version 1"),
        (lambda d: d.update(run_id="other"), "another run"),
        (lambda d: d["anchors"].pop(), "other anchors"),
        (lambda d: d["anchors"][0].update(strength=5), "reference anchor"),
    ]
    for change, match in cases:
        broken = copy.deepcopy(document)
        change(broken)
        with pytest.raises(lw.LadderBuildError, match=match):
            lw.read_anchors(broken, config)
    with pytest.raises(lw.LadderBuildError, match="not a version 1"):
        lw.read_anchors("x", config)
    with pytest.raises(lw.LadderBuildError, match="frozen at generation 0"):
        lw.check_anchors({"anchors": [{"anchor_id": "actor-rule@v1", "strength": SCALE}]}, {})


# -- the window and the copy rule ------------------------------------------------------------


def history(copied: dict[int, tuple[str, ...]] | None = None) -> list[lw.GenerationRows]:
    return [lw.GenerationRows(g, generation_rows(), (copied or {}).get(g, ())) for g in (1, 2, 3)]


def tallied(results: list[lw.WindowResult], member: str, kind: str, opponent: str) -> int:
    return sum(1 for r in results if (r.member, r.kind, r.opponent) == (member, kind, opponent))


def test_the_window_holds_every_pair_result_of_its_generations() -> None:
    results = lw.window_results(history(), 3, 3, list(ANCHOR_STRENGTHS), ["m0", "m1"])
    assert tallied(results, "m0", "member", "m1") == 3
    assert tallied(results, "m1", "anchor", "random-legal@v1") == 3
    assert not any(
        r.opponent == "m0" and r.member == "m0" for r in results
    )  # self-play is not rated
    table = lw.result_table(results)
    assert table[("m0", "member", "m1")] == [0, 0, 0, 0, 3]
    assert table[("m1", "member", "m0")] == [3, 0, 0, 0, 0]  # the other direction is the reverse


def test_a_window_of_two_ends_at_the_current_generation() -> None:
    older = lw.window_results(history()[1:], 3, 2, ["actor-rule@v1"], ["m0", "m1"])
    assert {r.generation for r in older} == {2, 3}
    with pytest.raises(lw.LadderBuildError, match="each generation of the window once"):
        lw.window_results(history(), 3, 2, ["actor-rule@v1"], ["m0", "m1"])
    with pytest.raises(lw.LadderBuildError, match="each generation of the window once"):
        lw.window_results([history()[0], history()[0]], 3, 3, [], ["m0", "m1"])


def test_results_from_before_a_copy_are_dropped_from_the_window() -> None:
    # A copy decided in generation 2 replaced slot m1: everything m1 did in generations 1 and 2, and
    # everyone's results against m1 in those generations, is of weights that no longer exist.
    results = lw.window_results(history({2: ("m1",)}), 3, 3, list(ANCHOR_STRENGTHS), ["m0", "m1"])
    assert {r.generation for r in results if r.member == "m1"} == {3}
    assert tallied(results, "m0", "member", "m1") == 1  # only generation 3's pair
    assert tallied(results, "m1", "member", "m0") == 0 or all(
        r.generation == 3 for r in results if r.member == "m1"
    )
    # The slot that was not copied keeps its anchor results of every generation.
    assert {r.generation for r in results if r.member == "m0" and r.kind == "anchor"} == {1, 2, 3}
    # A copy decided in the current window's last-but-one generation does not reach generation 3.
    late = lw.window_results(history({1: ("m0",)}), 3, 3, list(ANCHOR_STRENGTHS), ["m0", "m1"])
    assert {r.generation for r in late if r.member == "m0"} == {2, 3}
    # The donor is untouched by the copy.
    donor = lw.window_results(history({2: ("m1",)}), 3, 3, list(ANCHOR_STRENGTHS), ["m0", "m1"])
    assert {r.generation for r in donor if r.member == "m0" and r.opponent == "m1"} == {3}


def test_unknown_members_and_ladder_opponents_are_refused() -> None:
    rows = pair_rows(0, "ladder", "m0", "m7", ("member", "member"))
    with pytest.raises(lw.LadderBuildError, match="names m7 as opponent"):
        lw.window_results([lw.GenerationRows(1, rows)], 1, 3, [], ["m0", "m1"])
    rows = pair_rows(0, "ladder", "m5", "m1", ("member", "member"))
    with pytest.raises(lw.LadderBuildError, match="m5 is not a member"):
        lw.window_results([lw.GenerationRows(1, rows)], 1, 3, [], ["m0", "m1"])


# -- the record -------------------------------------------------------------------------------


def test_a_ladder_record_passes_its_reader_and_the_previous_check() -> None:
    config = population_config()
    first = lw.build_ladder_record(config, 1, history()[:1], ANCHOR_STRENGTHS)
    second = lw.build_ladder_record(config, 2, history()[:2], ANCHOR_STRENGTHS, first)
    third = lw.build_ladder_record(config, 3, history(), ANCHOR_STRENGTHS, second)
    for g, record in enumerate((first, second, third), start=1):
        assert read_ladder_record(record, None, config).generation == g
        assert record["anchors"] == [
            {"anchor_id": "actor-rule@v1", "strength": SCALE},
            {"anchor_id": "random-legal@v1", "strength": 700_000_000},
        ]
        assert record["selections"] == [{"kind": "none", "failed": "no_ready_member"}]
        assert record["snapshots"] == []
        assert [m["root"] for m in record["members"]] == ["m0", "m1"]
    assert third["window"] == {"size": 3, "first_generation": 1, "last_generation": 3}
    assert second["window"]["first_generation"] == 1
    # m0 beat m1 and both anchors in every pair: its strength is the larger.
    assert third["members"][0]["strength"]["strength"] > third["members"][1]["strength"]["strength"]
    assert third["cyclic_triples"] == 0
    cell = next(c for c in third["cells"] if (c["member"], c["opponent"]["id"]) == ("m0", "m1"))
    assert (cell["scores"], cell["pairs"], cell["score_sum_ppm"]) == ([0, 0, 0, 0, 3], 3, 3_000_000)
    # The table has both directions of the member pair and one cell per anchor per member.
    assert len(third["cells"]) == 2 + 2 * 2


def test_a_ladder_record_is_a_pure_function_of_its_inputs() -> None:
    config = population_config()
    one = lw.build_ladder_record(config, 2, history()[:2], ANCHOR_STRENGTHS)
    two = lw.build_ladder_record(config, 2, list(reversed(history()[:2])), ANCHOR_STRENGTHS)
    assert one == two
    other = lw.build_ladder_record(config, 3, history(), ANCHOR_STRENGTHS)
    assert (
        other["members"][0]["strength"] != one["members"][0]["strength"]
    )  # another seed and window


def test_a_member_without_a_result_against_an_anchor_is_refused() -> None:
    config = population_config()
    rows = [r for r in generation_rows() if r["opponent"] != "random-legal@v1"]
    with pytest.raises(lw.LadderBuildError, match="no result against the anchor random-legal@v1"):
        lw.build_ladder_record(config, 1, [lw.GenerationRows(1, rows)], ANCHOR_STRENGTHS)


def test_a_copy_in_the_history_removes_the_losers_old_cells_from_the_record() -> None:
    config = population_config()
    record = lw.build_ladder_record(config, 3, history({2: ("m1",)}), ANCHOR_STRENGTHS)
    by_cell = {(c["member"], c["opponent"]["id"]): c for c in record["cells"]}
    assert by_cell[("m1", "actor-rule@v1")]["pairs"] == 1
    assert by_cell[("m0", "actor-rule@v1")]["pairs"] == 3
    assert by_cell[("m0", "m1")]["pairs"] == 1
    assert by_cell[("m1", "m0")]["pairs"] == 1


def test_member_settings_are_the_effective_ppo_values() -> None:
    def edit(raw: dict[str, Any]) -> None:
        raw["ppo"]["target_kl_micro"] = 20_000
        raw["population"]["explore"]["bounds"]["target_kl_micro"] = {"min": 10_000, "max": 40_000}
        raw["population"]["member_ppo"] = [{"member": 1, "epochs": 1, "target_kl_micro": 30_000}]

    config = population_config(edit=edit)
    zero, one = lw.member_settings(config, 0), lw.member_settings(config, 1)
    assert zero["epochs"] == config.raw["ppo"]["epochs"]
    assert zero["target_kl_micro"] == 20_000
    assert one["epochs"] == 1
    assert one["target_kl_micro"] == 30_000
    assert zero["shares"] == config.raw["population"]["shares"]
    assert "target_kl_micro" not in lw.member_settings(population_config(), 0)
