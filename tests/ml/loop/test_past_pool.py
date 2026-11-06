"""The frozen past pool (ADR-021 sections 3 and 4, slice 10): eviction, PFSP weights, the pool file,
frozen-rated snapshots in the ladder record and a tiny loop that plays past pairs."""

from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_ml.learner.config import RunConfig
from areneon_ml.loop import ladder_write as lw
from areneon_ml.loop import layout
from areneon_ml.loop import past_pool as pp
from areneon_ml.loop.driver import LoopError
from areneon_ml.loop.ladder import LadderRecordError, read_ladder_record
from areneon_ml.loop.play import PopulationPlayRequest

from tests.ml.loop.support import Killed, killer
from tests.ml.loop.test_ladder_write import ANCHOR_STRENGTHS, generation_rows, pair_rows
from tests.ml.loop.test_population import (
    FakePopulationPlayer,
    gen_path,
    make_population,
    population_config,
    tree_bytes,
)

SETTINGS = {
    "snapshot_every": 5,
    "max_size": 32,
    "max_past_per_member": 4,
    "pfsp_power": 2,
    "floor_ppm": 50_000,
}


def policy(digit: str) -> str:
    return "policy:" + digit * 64


def snap(member: str, generation: int, digit: str) -> pp.Snapshot:
    return pp.Snapshot(member, generation, policy(digit), digit * 64, digit * 64)


def stand_in(**settings: int) -> Any:
    """A config with only what the pool functions read (the vectors below were computed with it)."""
    block = {"members": 2, "past_pool": {**SETTINGS, **settings}}
    return types.SimpleNamespace(run_key="tiny-run", hash="h", raw={"population": block})


# -- membership and eviction ------------------------------------------------------------------


def test_snapshots_are_taken_every_snapshot_every_generations_generation_zero_included() -> None:
    assert pp.pool_keys(2, 5, 32, 0) == [(0, 0), (0, 1)]
    assert pp.pool_keys(2, 5, 32, 4) == [(0, 0), (0, 1)]
    assert pp.pool_keys(2, 5, 32, 10) == [(0, 0), (0, 1), (5, 0), (5, 1), (10, 0), (10, 1)]


def test_the_oldest_snapshots_are_evicted_first_once_the_cap_is_passed() -> None:
    assert pp.pool_keys(2, 5, 3, 5) == [(0, 0), (0, 1), (5, 1)]
    assert pp.pool_keys(2, 5, 3, 10) == [(0, 0), (0, 1), (10, 1)]
    assert pp.pool_keys(2, 5, 4, 15) == [(0, 0), (0, 1), (15, 0), (15, 1)]
    assert pp.pool_keys(1, 2, 3, 8) == [(0, 0), (6, 0), (8, 0)]


def test_generation_zero_pairs_are_never_evicted_so_the_pool_may_pass_the_cap_by_them() -> None:
    assert pp.pool_keys(3, 5, 2, 0) == [(0, 0), (0, 1), (0, 2)]
    assert pp.pool_keys(3, 5, 2, 25) == [(0, 0), (0, 1), (0, 2)]
    assert pp.pool_keys(2, 1, 1, 6) == [(0, 0), (0, 1)]


def test_the_pool_after_a_generation_is_the_previous_pool_less_evictions_plus_new_snapshots() -> (
    None
):
    config = stand_in(snapshot_every=5, max_size=3)
    zero = pp.next_snapshots(config, (), 0, {"m0": snap("m0", 0, "1"), "m1": snap("m1", 0, "2")})
    assert [s.id for s in zero] == ["m0@g0", "m1@g0"]
    # No snapshot is due in generation 3: the pool is as it was.
    assert pp.next_snapshots(config, zero, 3, {}) == zero
    five = pp.next_snapshots(config, zero, 5, {"m0": snap("m0", 5, "3"), "m1": snap("m1", 5, "4")})
    assert [s.id for s in five] == ["m0@g0", "m1@g0", "m1@g5"]  # m0@g5 is the oldest non-zero one
    ten = pp.next_snapshots(
        config, five, 10, {"m0": snap("m0", 10, "5"), "m1": snap("m1", 10, "6")}
    )
    assert [s.id for s in ten] == ["m0@g0", "m1@g0", "m1@g10"]


def test_a_snapshot_that_is_not_the_pair_being_sealed_or_is_missing_is_refused() -> None:
    config = stand_in(snapshot_every=5)
    with pytest.raises(pp.PastPoolError, match="at the generation being sealed"):
        pp.next_snapshots(config, (), 0, {"m0": snap("m0", 5, "1"), "m1": snap("m1", 0, "2")})
    with pytest.raises(pp.PastPoolError, match="lacks the committed snapshot m0@g0"):
        pp.next_snapshots(config, (), 5, {"m0": snap("m0", 5, "1"), "m1": snap("m1", 5, "2")})


# -- PFSP -------------------------------------------------------------------------------------


def test_the_f_hard_weight_is_the_integer_rule_from_the_mean_pair_score() -> None:
    def w(scores: list[int] | None, power: int = 2, floor: int = 50_000) -> int:
        return pp.pfsp_weight(floor, power, scores)

    assert w([0, 0, 2, 0, 0]) == 250_000  # s = 500,000: 500,000^2 / 10^6
    assert w([0, 1, 0, 1, 0]) == 250_000
    assert w([0, 0, 1, 0, 1]) == 62_500  # s = 750,000: 250,000^2 / 10^6, above the floor
    assert w([0, 2, 1, 0, 0]) == 444_444  # s = 333,333: 666,667^2 / 10^6 = 444,444.88 floored
    assert w([4, 0, 0, 0, 0]) == 1_000_000
    assert w([0, 0, 2, 0, 0], power=3) == 125_000
    assert w([0, 0, 2, 0, 0], power=1) == 500_000
    assert w([0, 0, 2, 0, 0], power=16, floor=0) == 15  # 500,000^16 / 10^90 = 15.2...


def test_the_floor_and_the_no_games_maximum() -> None:
    assert pp.pfsp_weight(50_000, 2, [0, 0, 0, 0, 4]) == 50_000
    assert pp.pfsp_weight(50_000, 2, [0, 0, 0, 1, 9]) == 50_000
    assert pp.pfsp_weight(0, 2, [0, 0, 0, 0, 4]) == 0
    assert pp.pfsp_weight(50_000, 2, None) == 1_000_000
    assert pp.pfsp_weight(50_000, 2, [0, 0, 0, 0, 0]) == 1_000_000


# The fixture of the TypeScript twin (packages/showdown-bridge/test/past-pool.test.ts): the same
# vectors on both sides.
SNAPSHOTS = (snap("m0", 0, "1"), snap("m1", 0, "2"), snap("m0", 5, "3"), snap("m1", 5, "4"))
CELLS = {
    ("m0", "m0@g0"): (0, 0, 2, 0, 0),
    ("m0", "m1@g0"): (0, 0, 0, 0, 4),
    ("m1", "m0@g0"): (1, 0, 0, 0, 0),
    ("m1", "m1@g0"): (0, 0, 1, 0, 1),
}
POOL = pp.PastPool(5, SNAPSHOTS, CELLS)


def test_the_weights_read_each_members_own_cell() -> None:
    config = stand_in()
    assert pp.pfsp_weights(config, POOL, "m0", SNAPSHOTS) == [250_000, 50_000, 1_000_000, 1_000_000]
    assert pp.pfsp_weights(config, POOL, "m1", SNAPSHOTS) == [
        1_000_000,
        62_500,
        1_000_000,
        1_000_000,
    ]


def ids(member: str, pairs: int, current: str, generation: int = 6, **settings: int) -> list[str]:
    drawn = pp.draw_past(stand_in(**settings), generation, member, pairs, policy(current), POOL)
    return [s.id for s in drawn]


def test_the_draws_are_the_vectors_the_typescript_twin_pins() -> None:
    assert ids("m0", 1, "3") == ["m1@g5"]
    assert ids("m0", 3, "3") == ["m1@g5", "m1@g5", "m0@g0"]
    assert ids("m0", 6, "3") == ["m1@g5", "m1@g5", "m0@g0", "m1@g5", "m1@g5", "m1@g0"]
    assert ids("m1", 1, "4") == ["m0@g0"]
    assert ids("m1", 3, "4") == ["m0@g0", "m0@g0", "m0@g0"]
    assert ids("m1", 6, "4") == ["m0@g0", "m0@g0", "m0@g0", "m0@g5", "m0@g0", "m0@g0"]
    assert ids("m0", 3, "3", 7) == ["m1@g5", "m1@g5", "m1@g0"]
    assert ids("m1", 6, "4", 7) == ["m0@g0", "m0@g5", "m0@g5", "m0@g5", "m0@g5", "m0@g5"]


def test_a_member_never_meets_its_own_current_pair_and_the_per_member_cap_holds() -> None:
    assert all(s != "m0@g5" for s in ids("m0", 30, "3"))
    for generation in range(6, 14):
        assert len(set(ids("m0", 40, "3", generation, max_past_per_member=2))) <= 2
        assert len(set(ids("m0", 1, "3", generation))) == 1  # never more than it has pairs
    assert pp.draw_past(stand_in(), 6, "m0", 4, policy("3"), pp.PastPool(5, (), {})) == []
    assert ids("m0", 0, "3") == []


# -- the file ---------------------------------------------------------------------------------


def file_config() -> RunConfig:
    def past(raw: dict[str, Any]) -> None:
        raw["population"]["shares"] = {
            "ladder_ppm": 500_000,
            "self_play_ppm": 250_000,
            "past_pool_ppm": 250_000,
        }
        bounds = raw["population"]["explore"]["bounds"]
        bounds["self_play_ppm"] = {"min": 200_000, "max": 300_000}
        bounds["past_pool_ppm"] = {"min": 200_000, "max": 300_000}
        raw["population"]["past_pool"] = {**SETTINGS, "max_past_per_member": 2}

    return population_config(edit=past)


def test_a_past_pool_file_round_trips_and_is_checked_against_the_config() -> None:
    config = file_config()
    assert pp.enabled(config)
    assert not pp.enabled(population_config())
    raw = pp.build_file(config, 5, SNAPSHOTS, CELLS)
    assert pp.read_file(copy.deepcopy(raw), config) == pp.PastPool(5, SNAPSHOTS, CELLS)
    # Cells for snapshots outside the pool, or without pairs, are not written.
    assert pp.build_file(config, 5, SNAPSHOTS[:2], CELLS)["cells"] == [
        c for c in raw["cells"] if c["snapshot"].endswith("g0")
    ]

    def refused(change: Any, match: str) -> None:
        bad = copy.deepcopy(raw)
        change(bad)
        with pytest.raises(pp.PastPoolError, match=match):
            pp.read_file(bad, config)

    refused(lambda r: r.update(run_id="other"), "another run or config")
    refused(lambda r: r.update(extra=1), "not a past-pool file")
    refused(lambda r: r.update(format="past-pool/2"), "version 1")
    refused(lambda r: r.update(generation=-1), "non-negative")
    refused(lambda r: r["snapshots"].pop(0), "not the pool the config gives")
    refused(lambda r: r["snapshots"][0].update(id="m0@g1"), "id must be its member")
    refused(lambda r: r["cells"].reverse(), "must be sorted")
    refused(lambda r: r["cells"][0].update(snapshot="m0@g10"), "unknown player")
    refused(lambda r: r["cells"][0].update(scores=[0, 0, 0, 0, 0]), "at least one pair")
    with pytest.raises(pp.PastPoolError, match="not a past-pool file"):
        pp.read_file([], config)


# -- snapshots in the ladder record -----------------------------------------------------------


def rated_config() -> RunConfig:
    config = file_config()
    config.raw["population"]["selection"]["n_min"] = 2
    return config


def past_rows(first_pair: int, plan: list[tuple[str, str, tuple[str | None, str | None]]]) -> Any:
    rows: list[dict[str, Any]] = []
    for offset, (member, snapshot, result) in enumerate(plan):
        rows += pair_rows(first_pair + offset, "past", member, snapshot, result)
    return rows


WIN = ("member", "member")
LOSS = ("opponent", "opponent")
POOL_G0 = (snap("m0", 0, "1"), snap("m1", 0, "2"))


def gen1() -> list[lw.GenerationRows]:
    # m0 plays m1@g0 three times (2 wins and a split) and m1 plays m0@g0 once.
    past = past_rows(
        6,
        [
            ("m0", "m1@g0", WIN),
            ("m0", "m1@g0", WIN),
            ("m0", "m1@g0", ("member", "opponent")),
            ("m1", "m0@g0", LOSS),
        ],
    )
    return [lw.GenerationRows(1, generation_rows() + past)]


def test_a_snapshot_with_n_min_pairs_is_rated_once_and_the_rest_stay_out_of_the_record() -> None:
    config = rated_config()
    record = lw.build_ladder_record(config, 1, gen1(), ANCHOR_STRENGTHS, None, POOL_G0)
    assert [s["snapshot_id"] for s in record["snapshots"]] == ["m1@g0"]  # m0@g0 has one pair
    entry = record["snapshots"][0]
    assert (entry["member"], entry["generation"]) == ("m1", 0)
    assert entry["strength"]["strength"] > 0
    snapshot_cells = [c for c in record["cells"] if c["opponent"]["kind"] == "snapshot"]
    assert [(c["member"], c["opponent"]["id"], c["pairs"]) for c in snapshot_cells] == [
        ("m0", "m1@g0", 3)
    ]
    assert read_ladder_record(record, None, config).generation == 1
    # The matchmaking sees every pool snapshot's counts, rated or not.
    cells = lw.snapshot_cells(config, 1, gen1(), ANCHOR_STRENGTHS, ["m0@g0", "m1@g0"])
    assert cells == {("m0", "m1@g0"): [0, 0, 1, 0, 2], ("m1", "m0@g0"): [1, 0, 0, 0, 0]}
    # A pure function of its inputs.
    assert record == lw.build_ladder_record(config, 1, gen1(), ANCHOR_STRENGTHS, None, POOL_G0)


def test_a_rated_snapshot_is_frozen_and_a_fixed_reference_from_the_next_record_on() -> None:
    config = rated_config()
    first = lw.build_ladder_record(config, 1, gen1(), ANCHOR_STRENGTHS, None, POOL_G0)
    more = past_rows(6, [("m0", "m1@g0", LOSS), ("m1", "m0@g0", LOSS), ("m1", "m0@g0", WIN)])
    history = [gen1()[0], lw.GenerationRows(2, generation_rows() + more)]
    second = lw.build_ladder_record(config, 2, history, ANCHOR_STRENGTHS, first, POOL_G0)
    by_id = {s["snapshot_id"]: s for s in second["snapshots"]}
    # m1@g0 keeps the strength and bounds it was frozen with, although its results changed.
    assert by_id["m1@g0"] == first["snapshots"][0]
    # m0@g0 now has 3 pairs in the window (one before, two now): rated at entry.
    assert set(by_id) == {"m0@g0", "m1@g0"}
    assert read_ladder_record(second, first, config).generation == 2
    # The record keeps its frozen strength: a changed one is refused by the previous check.
    tampered = copy.deepcopy(second)
    tampered["snapshots"][1]["strength"]["strength"] += 1
    with pytest.raises(LadderRecordError, match="frozen at entry"):
        read_ladder_record(tampered, first, config)
    # The members' fit used the frozen snapshot as a reference: the same record without it differs.
    without = lw.build_ladder_record(config, 2, history, ANCHOR_STRENGTHS, None, POOL_G0[:1])
    assert [m["strength"] for m in second["members"]] != [m["strength"] for m in without["members"]]


def test_an_evicted_snapshot_leaves_the_record_and_its_old_results_are_dropped() -> None:
    config = rated_config()
    first = lw.build_ladder_record(config, 1, gen1(), ANCHOR_STRENGTHS, None, POOL_G0)
    history = [gen1()[0], lw.GenerationRows(2, generation_rows())]
    second = lw.build_ladder_record(config, 2, history, ANCHOR_STRENGTHS, first, POOL_G0[:1])
    assert second["snapshots"] == []
    assert not [c for c in second["cells"] if c["opponent"]["kind"] == "snapshot"]
    assert read_ladder_record(second, first, config).generation == 2


# -- a tiny loop that plays past pairs --------------------------------------------------------

GENERATIONS = 3


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("past-pool")
    make_population(run_dir, loop_config_edit()).run(GENERATIONS)
    return run_dir


def loop_config_edit() -> RunConfig:
    def edit(raw: dict[str, Any]) -> None:
        raw["population"]["shares"] = {
            "ladder_ppm": 500_000,
            "self_play_ppm": 250_000,
            "past_pool_ppm": 250_000,
        }
        raw["population"]["seats_per_member"] = 4
        bounds = raw["population"]["explore"]["bounds"]
        bounds["self_play_ppm"] = {"min": 200_000, "max": 300_000}
        bounds["past_pool_ppm"] = {"min": 200_000, "max": 300_000}
        raw["population"]["past_pool"] = {
            **SETTINGS,
            "snapshot_every": 1,
            "max_size": 6,
            "max_past_per_member": 2,
        }
        raw["population"]["selection"]["n_min"] = 2

    return population_config(edit=edit)


def pool_of(run_dir: Path, generation: int) -> dict[str, Any]:
    found: dict[str, Any] = layout.read_json(gen_path(run_dir, generation) / pp.FILE)
    return found


def test_every_generation_commits_the_pool_the_next_one_plays(finished: Path) -> None:
    for generation in range(GENERATIONS + 1):
        found = pool_of(finished, generation)
        assert found["generation"] == generation
        manifest = layout.read_json(gen_path(finished, generation) / "manifest.json")
        assert pp.FILE in manifest["files"]
        assert manifest["past_pool_hash"] == lw.json_hash(found)
    assert [s["id"] for s in pool_of(finished, 0)["snapshots"]] == ["m0@g0", "m1@g0"]
    assert pool_of(finished, 0)["cells"] == []
    # A snapshot every generation, capped at 6: g2 holds all 6; g3 evicts the generation-1 ones.
    assert len(pool_of(finished, 2)["snapshots"]) == 6
    assert [s["id"] for s in pool_of(finished, 3)["snapshots"]] == [
        "m0@g0",
        "m1@g0",
        "m0@g2",
        "m1@g2",
        "m0@g3",
        "m1@g3",
    ]
    # A snapshot's pair is the pair frozen in its generation.
    manifest1 = layout.read_json(gen_path(finished, 1) / "manifest.json")["members"]
    for s in pool_of(finished, 1)["snapshots"]:
        if s["generation"] == 1:
            assert (s["preview"], s["battle"]) == (
                manifest1[s["member"]]["checkpoints"]["preview"],
                manifest1[s["member"]]["checkpoints"]["battle"],
            )


def test_the_plan_meets_snapshots_of_the_pool_and_the_ladder_rates_them(finished: Path) -> None:
    plan = layout.read_json(gen_path(finished, 2) / "plan.json")
    past = [e for e in plan if e["kind"] == "past" and e["half"] == 0]
    assert sorted(e["member"] for e in past) == ["m0", "m1"]
    pool = {s["id"] for s in pool_of(finished, 1)["snapshots"]}
    assert all(e["seats"]["p2"]["snapshot"] in pool for e in past)
    rows = layout.read_json(gen_path(finished, 2) / "generation.json")["games"]
    assert {r["opponent"] for r in rows if r["kind"] == "past"} <= pool
    ladder = layout.read_json(gen_path(finished, 3) / "ladder.json")
    assert (
        read_ladder_record(
            ladder, layout.read_json(gen_path(finished, 2) / "ladder.json")
        ).generation
        == 3
    )
    assert ladder["cells"] == sorted(
        ladder["cells"], key=lambda c: (c["member"], c["opponent"]["kind"], c["opponent"]["id"])
    )


def test_an_independent_rerun_is_byte_identical(finished: Path, tmp_path: Path) -> None:
    make_population(tmp_path, loop_config_edit()).run(GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)


@pytest.mark.parametrize(("generation", "stage"), [(2, "played"), (2, "rated"), (3, "written")])
def test_a_killed_run_resumes_and_replays_the_same_plan(
    finished: Path, tmp_path: Path, generation: int, stage: str
) -> None:
    config = loop_config_edit()
    with pytest.raises(Killed):
        make_population(tmp_path, config, on_stage=killer(generation, stage)).run(GENERATIONS)
    player = FakePopulationPlayer(config)
    make_population(tmp_path, config, player).run(GENERATIONS)
    assert tree_bytes(tmp_path) == tree_bytes(finished)
    assert player.calibrations == []
    assert all(r.past_pool is not None for r in player.requests)
    assert layout.sha256_file(gen_path(tmp_path, generation) / "plan.json") == layout.sha256_file(
        gen_path(finished, generation) / "plan.json"
    )


def test_the_driver_hands_the_play_side_the_committed_pool_of_the_generation_before(
    tmp_path: Path,
) -> None:
    config = loop_config_edit()
    player = FakePopulationPlayer(config)
    make_population(tmp_path, config, player).run(2)
    assert [r.generation for r in player.requests] == [1, 2]
    assert [r.past_pool for r in player.requests] == [
        gen_path(tmp_path, 0) / pp.FILE,
        gen_path(tmp_path, 1) / pp.FILE,
    ]


def test_a_plan_that_is_not_the_pfsp_draw_of_the_committed_pool_is_refused(tmp_path: Path) -> None:
    config = loop_config_edit()

    class Swapping(FakePopulationPlayer):
        def play_population(self, request: PopulationPlayRequest) -> None:
            super().play_population(request)
            path = request.out_dir / "plan.json"
            plan = layout.read_json(path)
            for entry in plan:
                if entry["kind"] == "past" and entry["member"] == "m0":
                    seat = entry["seats"]["p2" if entry["half"] == 0 else "p1"]
                    other = {"m0@g0": "m1@g0", "m1@g0": "m0@g0"}[seat["snapshot"]]
                    pool = layout.read_json(request.past_pool or path)["snapshots"]
                    swap = next(s for s in pool if s["id"] == other)
                    seat.update(snapshot=other, policy_id=swap["policy_id"])
            layout.write_json(path, plan)
            record_path = request.out_dir / "league-run.json"
            record = layout.read_json(record_path)
            record["plan_hash"] = sha256_hex(jcs(plan))
            layout.write_json(record_path, record)

    with pytest.raises(LoopError, match="not the PFSP draw"):
        make_population(tmp_path, config, Swapping(config)).run(1)


def test_a_past_pool_file_that_is_not_the_one_the_manifest_names_is_refused(tmp_path: Path) -> None:
    config = loop_config_edit()
    make_population(tmp_path, config).run(1)
    raw = pool_of(tmp_path, 1)
    raw["cells"] = [{"member": "m0", "scores": [1, 0, 0, 0, 0], "snapshot": "m1@g0"}]
    layout.write_json(gen_path(tmp_path, 1) / pp.FILE, raw)
    # The files of the generation are resealed, as a forger who also fixes the file list would: only
    # the hash the manifest names the pool by still disagrees.
    manifest = layout.read_json(gen_path(tmp_path, 1) / "manifest.json")
    manifest["files"] = layout.hash_tree(gen_path(tmp_path, 1))
    layout.write_json(gen_path(tmp_path, 1) / "manifest.json", manifest)
    with pytest.raises(LoopError, match="is not the one its manifest names"):
        make_population(tmp_path, config).run(2)


def test_a_run_without_a_past_share_writes_no_pool_and_is_unchanged(tmp_path: Path) -> None:
    config = population_config()
    make_population(tmp_path, config).run(1)
    assert not (gen_path(tmp_path, 0) / pp.FILE).exists()
    assert "past_pool_hash" not in layout.read_json(gen_path(tmp_path, 1) / "manifest.json")


# -- review follow-ups ------------------------------------------------------------------------


def test_the_configured_past_count_is_the_seat_split_of_the_shares() -> None:
    config = file_config()
    # 2 seats at 500,000 / 250,000 / 250,000: floors 1 / 0 / 0, the leftover seat goes to the first
    # of the equal largest remainders in the order ladder, self-play, past: self-play.
    assert pp.past_pairs(config) == 0
    config.raw["population"]["seats_per_member"] = 512
    assert pp.past_pairs(config) == 128
    config.raw["population"]["seats_per_member"] = 5
    config.raw["population"]["shares"] = {
        "ladder_ppm": 333_334,
        "self_play_ppm": 333_333,
        "past_pool_ppm": 333_333,
    }
    assert pp.past_pairs(config) == 1


def test_a_plan_that_swaps_past_pairs_for_self_play_or_drops_them_is_refused(
    tmp_path: Path,
) -> None:
    config = loop_config_edit()

    class Dropping(FakePopulationPlayer):
        def past_snapshots(
            self, request: PopulationPlayRequest, member: str, generation: int, policy: str
        ) -> list[pp.Snapshot]:
            return []  # the member's past pair is not scheduled at all

    with pytest.raises(LoopError, match="has 0 past pairs, the config schedules 1"):
        make_population(tmp_path, config, Dropping(config)).run(1)


def test_a_member_with_nothing_to_meet_plays_no_past_pair(tmp_path: Path) -> None:
    config = loop_config_edit()
    config.raw["population"]["members"] = 1
    config.raw["population"]["selection"]["min_roots"] = 1
    pool = pp.PastPool(0, (snap("m0", 0, "1"),), {})
    loop = make_population(tmp_path, config)
    loop._check_past_plan([], 1, (types.SimpleNamespace(member="m0", policy_id=policy("1")),), pool)  # type: ignore[arg-type]


# -- the reader fails closed ------------------------------------------------------------------


def test_every_malformed_pool_file_is_a_past_pool_error() -> None:
    config = file_config()
    good = pp.build_file(config, 5, SNAPSHOTS, CELLS)

    def bad(change: Any) -> None:
        raw = copy.deepcopy(good)
        change(raw)
        with pytest.raises(pp.PastPoolError):
            pp.read_file(raw, config)

    bad(lambda r: r.update(snapshots=[1]))
    bad(lambda r: r.update(snapshots=["x"]))
    bad(lambda r: r["snapshots"].__setitem__(0, {}))
    bad(lambda r: r["snapshots"][0].pop("member"))
    bad(lambda r: r["snapshots"][0].update(member=7))
    bad(lambda r: r["snapshots"][0].update(member="x0"))
    bad(lambda r: r["snapshots"][0].update(generation="0"))
    bad(lambda r: r["snapshots"][0].update(generation=True))
    bad(lambda r: r["snapshots"][0].update(generation=-1))
    bad(lambda r: r["snapshots"][0].update(policy_id=3))
    bad(lambda r: r["snapshots"][0].update(policy_id="policy:xyz"))
    bad(lambda r: r["snapshots"][0].update(policy_id="a" * 64))
    bad(lambda r: r["snapshots"][0].update(preview="g" * 64))
    bad(lambda r: r["snapshots"][0].update(battle="a" * 63))
    bad(lambda r: r["snapshots"][0].update(preview=None))
    bad(lambda r: r["snapshots"][0].update(id="m0@g9"))
    bad(lambda r: r.update(generation=True))
    bad(lambda r: r.update(generation="5"))
    bad(lambda r: r.update(snapshots={}))
    bad(lambda r: r.update(cells={}))
    bad(lambda r: r.update(cells=[1]))
    bad(lambda r: r["cells"].__setitem__(0, {}))
    bad(lambda r: r["cells"][0].update(member=3))
    bad(lambda r: r["cells"][0].update(member="m9"))
    bad(lambda r: r["cells"][0].update(snapshot=[]))
    bad(lambda r: r["cells"][0].update(scores="abcde"))
    bad(lambda r: r["cells"][0].update(scores=[1, 0, 0, 0]))
    bad(lambda r: r["cells"][0].update(scores=[1.0, 0, 0, 0, 0]))
    bad(lambda r: r["cells"][0].update(scores=[True, 0, 0, 0, 0]))
    bad(lambda r: r["cells"][0].update(scores=[None, 0, 0, 0, 0]))
    bad(lambda r: r["cells"].append(copy.deepcopy(r["cells"][-1])))
