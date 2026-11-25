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
request) runs in the four shards ``battle-state-projection-corpus-{0..3}.python.test.ts`` and the
invariance test in ``battle-state-projection.python.test.ts``.
"""

from __future__ import annotations

import copy
import io
import json
from functools import cache
from itertools import cycle
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
from areneon_ml.projection.__main__ import Server, main
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


@pytest.mark.parametrize(
    "field_id", ["roster.move_pp", "roster.item", "roster.ability", "roster.hp", "battle.prng"]
)
def test_a_record_doctored_to_class_a_field_public_is_flagged(field_id: str) -> None:
    """Negative control, one field at a time: with it reclassified public, the value planted in
    the engine state reaches the other seat's view."""
    doctored = copy.deepcopy(record())
    for field in doctored["fields"]:
        if field["id"] == field_id:
            field["visibility"] = "public"
            field.pop("projection", None)
    leaky = SpecIndex(doctored, table())
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    real = point_of(battle, point)
    planted = {**real, "state": plant_engine(real["state"], seat)}
    before = project(leaky, pack(leaky, game().vocabulary, real), seat)
    after = project(leaky, pack(leaky, game().vocabulary, planted), seat)
    assert view_digest(before) != view_digest(after)
    differing = {k for k in before["fields"] if before["fields"][k] != after["fields"][k]}
    assert differing, field_id
    # ... and with the real record the same planting changes nothing.
    clean_before = project(index(), pack(index(), game().vocabulary, real), seat)
    clean_after = project(index(), pack(index(), game().vocabulary, planted), seat)
    assert clean_before == clean_after


def test_planted_move_ids_in_the_other_sides_slots_never_change_a_view() -> None:
    """``plant_engine(moves=...)``: the other side's four move slots hold other moves."""
    ids = iter(cycle(sorted(table()["moves"], key=lambda m: m["id"])))
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    real = point_of(battle, point)
    planted = {**real, "state": plant_engine(real["state"], seat, lambda: str(next(ids)["id"]))}
    moved = sum(
        a["id"] != b["id"]
        for side_a, side_b in zip(real["state"]["sides"], planted["state"]["sides"], strict=True)
        for mon_a, mon_b in zip(side_a["pokemon"], side_b["pokemon"], strict=True)
        for a, b in zip(mon_a["moveSlots"], mon_b["moveSlots"], strict=True)
    )
    assert moved > 0
    before = view_digest(project(index(), pack(index(), game().vocabulary, real), seat))
    after = view_digest(project(index(), pack(index(), game().vocabulary, planted), seat))
    assert before == after


IDENTITY_KEYS = ("set", "details", "species", "baseSpecies", "gender")
ROSTER_LEAKS = ("roster.is_active", "roster.fainted", "roster.boosts", "roster.transformed")


def illusion_point() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    """The committed Illusion battle at a request where a Zoroark is disguised in play: the
    battle, its point, p1's side of the engine state, the disguised Pokemon's engine index and the
    engine index of the entry it is shown as."""
    for battle in fixture()["battles"]:
        for point in battle["points"]:
            side = next(s for s in point["state"]["sides"] if s["id"] == "p1")
            for i, mon in enumerate(side["pokemon"]):
                if mon.get("illusion") and mon.get("isActive"):
                    target = ord(str(mon["illusion"]).rstrip("]")[-1]) - 97
                    if battle["battle_id"].startswith("illusion") and point["decisions"]:
                        return battle, point, side, i, target
    raise AssertionError("no disguised Pokemon in play in the committed Illusion battle")


def undisguised(point: dict[str, Any], disguised: int, target: int) -> dict[str, Any]:
    """The same engine state with the real Pokemon of the shown entry in the slot instead of the
    Illusion user: identities swapped between the two engine entries, every battle value (stages,
    volatiles, in play, ...) left where it is, the disguise gone. What a seat is shown is equal."""
    state = copy.deepcopy(point["state"])
    side = next(s for s in state["sides"] if s["id"] == "p1")
    user, shown = side["pokemon"][disguised], side["pokemon"][target]
    for key in IDENTITY_KEYS:
        user[key], shown[key] = shown.get(key), user.get(key)
    user.pop("illusion", None)
    return {**point, "state": state}


def test_an_illusion_user_does_not_show_through_the_entry_it_is_shown_as() -> None:
    """R-9: equal player channels, the opponent shows entry S either as the real Pokemon in play
    or as an Illusion user disguised as S (the entry S itself stays benched in the engine): the
    observing seat's view is byte-identical. The roster values held at the real index (in play,
    fainted, stages, transformed) are never read by the other seat."""
    battle, point, side, disguised, target = illusion_point()
    assert side["pokemon"][target]["isActive"] is False  # the engine entry S is benched
    real = point_of(battle, point)
    shown = point_of(battle, undisguised(point, disguised, target))
    a = project(index(), pack(index(), game().vocabulary, real), "p2")
    b = project(index(), pack(index(), game().vocabulary, shown), "p2")
    assert a == b
    assert view_digest(a) == view_digest(b)
    for field_id in ROSTER_LEAKS:
        assert f"{field_id}@opponent" not in a["fields"], field_id
    # The seat sees the shown entry in play through the slot, and the entry is not marked fainted.
    assert target in a["fields"]["seat_active.shown_roster_index@opponent"]


def test_a_projection_reading_the_roster_by_real_index_is_flagged_under_an_illusion() -> None:
    """Negative controls: the record before this fix (the roster values classed public) and a raw
    read of the real-index tensor each tell the two states apart."""
    battle, point, _, disguised, target = illusion_point()
    real = point_of(battle, point)
    shown = point_of(battle, undisguised(point, disguised, target))
    doctored = copy.deepcopy(record())
    for field in doctored["fields"]:
        if field["id"] in ROSTER_LEAKS:
            field["visibility"] = "public"
            field.pop("projection", None)
    leaky = SpecIndex(doctored, table())
    one = project(leaky, pack(leaky, game().vocabulary, real), "p2")
    two = project(leaky, pack(leaky, game().vocabulary, shown), "p2")
    assert view_digest(one) != view_digest(two)
    for field_id in ROSTER_LEAKS:
        assert f"{field_id}@opponent" in one["fields"], field_id
    # A raw read of the real-index tensor (outside the class accessor) differs: the shown entry is
    # benched in one engine state and in play in the other.
    raw_real = pack(index(), game().vocabulary, real)["fields"]["roster.is_active"][0]
    raw_shown = pack(index(), game().vocabulary, shown)["fields"]["roster.is_active"][0]
    assert raw_real[target] == 0
    assert raw_shown[target] == 1


def test_values_planted_at_the_real_roster_index_never_change_the_other_seats_view() -> None:
    """The canary on the Illusion battle: every roster value held at the real index of the other
    side (in play, fainted, transformed, a benched Pokemon's stages) is planted."""
    battle, point, _, _, _ = illusion_point()
    real = point_of(battle, point)
    planted = {**real, "state": plant_engine(real["state"], "p2")}
    before = view_digest(project(index(), pack(index(), game().vocabulary, real), "p2"))
    after = view_digest(project(index(), pack(index(), game().vocabulary, planted), "p2"))
    assert before == after


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


# ---- the child the TS side drives ---------------------------------------------------------


def test_the_server_answers_check_view_leak_and_refused_frames(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    battle, point, decision = next(d for d in decisions() if d[2]["kind"] == "battle")
    seat = decision["seat"]
    frame = {
        "kind": "check",
        "id": f"{battle['battle_id']}/{point['step']}/{seat}",
        "point": point_of(battle, point),
        "seat": seat,
        "query": query_of(battle, point, decision),
    }
    server = Server(index(), game())
    reply = server.answer(frame)
    assert reply["status"] == "equal", reply["differences"]
    assert reply["canary"] == {"engine": True, "tensors": True}
    view = server.answer({**frame, "kind": "view"})
    assert view["view_sha256"] == reply["view_sha256"]
    leak = server.answer({**frame, "kind": "view", "leak": "roster.item"})
    assert leak["view_sha256"] != reply["view_sha256"]
    other = {**frame, "point": {**frame["point"], "format_id": "gen9ou"}}
    refused = {"id": frame["id"], "status": "refused", "code": "format_mismatch", "detail": ""}
    assert server.answer(other) == refused
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**frame, "kind": "view"}) + "\n\n"))
    engine_data = str(FORMAT_DIR / "engine-data-2.json")
    spec_path = str(SPEC_DIR / "battle-state-spec.v2.json")
    assert main(["serve", "--engine-data", engine_data, "--spec", spec_path]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["view_sha256"] for p in printed] == [reply["view_sha256"]]


def test_the_projection_imports_no_runner_trace_or_omniscient_log_reader() -> None:
    """Gate 4 of #1464 on the Python side: the packer and the projection read a point (state,
    channels, teams) and nothing that records or replays a battle (the TS trace reader,
    ``projection-points.ts``, is the one module of the path that touches the recorder)."""
    package = ROOT / "ml" / "areneon_ml" / "projection"
    for path in sorted(package.glob("*.py")):
        text = path.read_text()
        for name in ("inspector", "resimulate", "omniscient", "areneon_ml.loop", "subprocess"):
            assert name not in text, (path.name, name)


def test_a_frame_the_server_cannot_answer_is_an_error_reply_not_a_dead_child() -> None:
    reply = Server(index(), game()).answer({"kind": "view", "id": "g/1/p1", "seat": "p1"})
    assert reply["id"] == "g/1/p1"
    assert reply["status"] == "error"
    assert reply["code"] == "KeyError"
