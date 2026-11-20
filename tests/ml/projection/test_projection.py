"""The reference packer and the player-view projection of the battle-state spec (ADR-030 G4, #1464).

On the committed projection points (``fixtures/showdown/<format>/projection/points.json``, captured
from replays of committed battles on the pinned engine by
``packages/showdown-bridge/scripts/capture-projection-points.ts``): the engine state at a request
plus both seats' channels is packed into the spec's tensors, projected for the seat through the
visibility classes alone, rebuilt as the reducer-state view the encoder reads and encoded; the
tensors must be bit-equal to ``Encoder.encode`` of the seat's query under the default encoder
options, the lineage run's options and ``exact_volatiles``. The privacy canary plants values in
every field the seat may not read (in the engine state and in the packed tensors) and requires the
seat's view unchanged, with negative controls: a projection that reads a field through a raw path,
and a record doctored to class a hidden field public. The whole committed corpus (both seats, every
request) and the invariance test run in ``battle-state-projection.python.test.ts``.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.model.gamedata import GameData
from areneon_ml.projection import (
    CONFIGS,
    PackError,
    PackSession,
    ProjectionError,
    SpecIndex,
    compare_encodings,
    encoders_for,
    pack,
    plant_engine,
    plant_unreadable,
    project,
    query_of,
    reduced_view,
    view_digest,
)
from areneon_ml.projection.packer import (
    pack_reveal,
    pack_volatiles,
    unpack_reveal,
    unpack_volatiles,
)

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "gen9championsvgc2026regmc"
FORMAT_DIR = ROOT / "fixtures" / "showdown" / FORMAT
SPEC_DIR = ROOT / "config" / "engine-state" / FORMAT


@cache
def table() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FORMAT_DIR / "engine-data-2.json").read_text())
    return loaded


@cache
def game() -> GameData:
    return GameData.from_table(table())


def record(version: str = "v2") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (SPEC_DIR / f"battle-state-spec.{version}.json").read_text()
    )
    return loaded


@cache
def index() -> SpecIndex:
    return SpecIndex(record(), table())


@cache
def fixture() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FORMAT_DIR / "projection" / "points.json").read_text())
    return loaded


def decisions() -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """(battle, point, decision) of every committed decision."""
    return [
        (battle, point, decision)
        for battle in fixture()["battles"]
        for point in battle["points"]
        for decision in point["decisions"]
    ]


def point_of(battle: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_id": FORMAT,
        "state": point["state"],
        "channels": {seat: battle["channels"][seat][: point["cut"][seat]] for seat in ("p1", "p2")},
        "teams": battle["teams"],
    }


# ---- the record ------------------------------------------------------------------------------


def test_the_v1_record_is_refused_naming_the_fields_it_lacks() -> None:
    """The four gaps of v1: unbrought own Pokemon, own base species over a Mega Evolution, raw
    numbered volatiles and a fifth revealed move. The projection needs v2."""
    with pytest.raises(ProjectionError, match=r"seat_roster\.own_stats") as caught:
        SpecIndex(record("v1"), table())
    for name in ("seat_active.volatile_numbered", "seat_roster.revealed_extra"):
        assert name in str(caught.value)


def test_the_index_reads_dimensions_and_vocabularies_from_the_record_and_the_table() -> None:
    spec = index()
    assert spec.size("SEAT") == 2
    assert spec.size("ROSTER") == 6
    assert spec.shape("seat_roster.own_moves") == (2, 2, 6, 4)
    assert spec.shape("battle.turn") == ()
    assert spec.encode_id("seat_roster.own_moves", None) == 0
    move = spec.dim_or_vocab("seat_roster.own_moves")[0]
    assert (
        spec.decode_id("seat_roster.own_moves", spec.encode_id("seat_roster.own_moves", move))
        == move
    )
    assert spec.encode_id("seat_roster.own_moves", "notamove") == 1
    with pytest.raises(ProjectionError, match="not a field"):
        spec.shape("roster.nothing")


# ---- the codecs of the seat bookkeeping -------------------------------------------------------


def test_raw_volatile_ids_round_trip_with_their_numbered_stat_and_protocol_forms() -> None:
    raw = {
        "taunt",
        "stockpile1",
        "stockpile2",
        "protosynthesisatk",
        "perish3",
        "perish2",
        "typechange",
    }
    codes = pack_volatiles(index(), game().vocabulary, raw)
    assert unpack_volatiles(index(), codes) == raw
    assert pack_volatiles(index(), game().vocabulary, set()) == pack_volatiles(
        index(), game().vocabulary, set()
    )
    with pytest.raises(PackError, match="volatile"):
        pack_volatiles(index(), game().vocabulary, {"zzznotavolatile"})
    with pytest.raises(PackError, match="volatile"):
        pack_volatiles(index(), game().vocabulary, {"stockpile99"})


def test_a_fifth_revealed_move_round_trips_at_its_position() -> None:
    moves = sorted(index().dim_or_vocab("seat_roster.revealed_moves"))[:5]
    for n in range(6):
        assert unpack_reveal(index(), *pack_reveal(index(), moves[:n])) == moves[:n]
    with pytest.raises(PackError, match="reveal"):
        pack_reveal(index(), [*moves, moves[0]])


# ---- bit-equality on the committed points -------------------------------------------------------


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_the_projection_encodes_bit_equal_to_the_encoder_at_every_committed_decision(
    config: str,
) -> None:
    encoder = encoders_for(game())[config]
    seen = 0
    for battle, point, decision in decisions():
        packed = pack(index(), game().vocabulary, point_of(battle, point))
        seat = decision["seat"]
        view = project(index(), packed, seat)
        query = query_of(battle, point, decision)
        state = reduced_view(index(), view, battle["teams"][seat])
        assert compare_encodings(encoder, query, state) is None, (
            battle["battle_id"],
            point["step"],
            seat,
        )
        seen += 1
    assert seen >= 20
    kinds = {d["kind"] for _, _, d in decisions()}
    assert {"preview", "battle"} <= kinds


def test_a_doctored_reduced_view_is_named_by_its_tensor_and_column() -> None:
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    state = reduced_view(
        index(),
        project(index(), pack(index(), game().vocabulary, point_of(battle, point)), seat),
        battle["teams"][seat],
    )
    state.own.roster[0].boosts["atk"] = 5
    difference = compare_encodings(
        encoders_for(game())["base"], query_of(battle, point, decision), state
    )
    assert difference is not None
    assert difference["tensor"] == "mon_num"
    assert difference["column"] == "boost_atk"


def test_the_incremental_session_packs_the_same_bytes_as_a_fresh_pack() -> None:
    session = PackSession(index(), game().vocabulary)
    for battle, point, _ in decisions():
        fresh = pack(index(), game().vocabulary, point_of(battle, point))
        assert session.pack(point_of(battle, point)) == fresh


def test_the_packer_refuses_a_channel_the_reducer_refuses_and_a_state_of_another_format() -> None:
    battle, point, _ = decisions()[0]
    broken = point_of(battle, point)
    broken["channels"] = {**broken["channels"], "p1": [*broken["channels"]["p1"], "|split|p2"]}
    with pytest.raises(PackError) as caught:
        pack(index(), game().vocabulary, broken)
    assert caught.value.code == "reducer_refused"
    with pytest.raises(PackError) as other:
        pack(index(), game().vocabulary, {**point_of(battle, point), "format_id": "gen9ou"})
    assert other.value.code == "format_mismatch"


# ---- privacy: the canary and its negative controls ----------------------------------------------


def test_values_planted_in_every_field_a_seat_may_not_read_never_change_its_view() -> None:
    """Engine planting: the other side's hidden and own-exact engine values, the PRNG, the Quick
    Claw roll and every duration. Tensor planting: every value of every field the class does not
    let the seat read (hidden fields, the other side's own-exact fields, the other seat's
    bookkeeping)."""
    for battle, point, decision in decisions():
        seat = decision["seat"]
        real = point_of(battle, point)
        before = view_digest(project(index(), pack(index(), game().vocabulary, real), seat))
        planted = {**real, "state": plant_engine(real["state"], seat)}
        assert planted["state"] != real["state"]
        after_engine = view_digest(
            project(index(), pack(index(), game().vocabulary, planted), seat)
        )
        assert after_engine == before, (battle["battle_id"], point["step"], seat)
        tensors = pack(index(), game().vocabulary, real)
        doctored = plant_unreadable(index(), tensors, seat)
        assert doctored != tensors
        assert view_digest(project(index(), doctored, seat)) == before


def test_a_projection_that_reads_a_field_through_a_raw_path_is_flagged() -> None:
    """Negative control: a projection that copies the other side's exact hit points out of the
    tensors (not through the class accessor) changes with the planted values."""
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    real = point_of(battle, point)

    def leaky(packed: dict[str, Any]) -> str:
        view = project(index(), packed, seat)
        other = 1 if seat == "p1" else 0
        view["fields"]["raw.hp"] = packed["fields"]["roster.hp"][other]
        return view_digest(view)

    tensors = pack(index(), game().vocabulary, real)
    assert leaky(tensors) != leaky(plant_unreadable(index(), tensors, seat))
    planted = {**real, "state": plant_engine(real["state"], seat)}
    assert leaky(tensors) != leaky(pack(index(), game().vocabulary, planted))


def test_a_record_doctored_to_class_a_hidden_field_public_is_flagged() -> None:
    """Negative control: with ``roster.move_pp`` and ``battle.prng`` reclassified public, the
    planted values reach the other seat's view."""
    doctored = copy.deepcopy(record())
    for field in doctored["fields"]:
        if field["id"] in ("roster.move_pp", "battle.prng"):
            field["visibility"] = "public"
            field.pop("projection", None)
    leaky = SpecIndex(doctored, table())
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    real = point_of(battle, point)
    planted = {**real, "state": plant_engine(real["state"], seat)}
    before = view_digest(project(leaky, pack(leaky, game().vocabulary, real), seat))
    after = view_digest(project(leaky, pack(leaky, game().vocabulary, planted), seat))
    assert before != after


def test_a_view_holds_no_hidden_field_no_other_seat_row_and_no_other_side_own_exact_field() -> None:
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    view = project(
        index(), pack(index(), game().vocabulary, point_of(battle, point)), decision["seat"]
    )
    fields = index().fields
    for key in view["fields"]:
        field_id, _, half = key.partition("@")
        field = fields[field_id]
        assert field["visibility"] != "hidden", key
        if field["visibility"] == "own-exact":
            assert half == "own", key
    assert view["seat"] == decision["seat"]
    assert view["spec_hash"] == index().spec_hash
    with pytest.raises(ProjectionError, match="seat"):
        project(index(), pack(index(), game().vocabulary, point_of(battle, point)), "p3")
