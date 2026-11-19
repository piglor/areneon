"""The battle-state spec reader (ADR-030 step 5 groundwork; #1463).

The Python reader refuses exactly what the TS one does: every reader-invalid corpus sample and the
schema-invalid ones. The committed record reads against the engine-data table it names, its hash is
the one the TS test pins, and the projection rules give each visibility class its view.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.battle_state_spec import (
    BattleStateSpecError,
    battle_state_spec_hash,
    path_keys,
    read_battle_state_spec,
    seat_view,
    state_item_ids,
    vocabulary_ids,
)
from areneon_contracts.contracts import jcs, sha256_hex

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "fixtures/contracts/v1/battle-state-spec"
FORMAT = "gen9championsvgc2026regmc"
COMMITTED = ROOT / f"config/engine-state/{FORMAT}/battle-state-spec.v1.json"
TABLE = json.loads((ROOT / f"fixtures/showdown/{FORMAT}/engine-data-2.json").read_text())
# Pinned by the TS test too (`battle-state-spec.test.ts`): a change of the record is a new record.
HASH = "22735303afd5a636ff68ad5941749da2a7dbf938c991a24329603d77e4720d41"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def committed() -> Any:
    return load(COMMITTED)


READER_INVALID = {
    "encoder-reads-hidden": "battle.prng is hidden",
    "encoder-unknown-field": "battle.turns is not a field",
    "group-mismatch": "battle.turn: the id does not start with its group",
    "groups-out-of-order": "groups: every group exactly once",
    "omitted-but-read": "battle.turn is read by a field",
    "opponent-reads-unprojectable": "roster.stats is not projectable to another seat",
    "projection-on-public": "a projection exactly for an own-exact field",
    "range-overflows-dtype": "the range does not fit int8",
    "range-reversed": "min exceeds max",
    "reveal-missing": "a reveal mask exactly for a revealed-only field",
    "reveal-wrong-mask": "its mask must be a 0/1 seat field",
    "seat-engine-source": "a seat field is a derivation",
    "seat-not-public": "its class is public",
    "unknown-dim": "MOVES is not a dimension",
    "unsorted-fields": "is repeated or out of order",
    "unsorted-out-of-scope": "out_of_scope",
    "unsorted-reducer": "reducer:",
    "unused-dim": "ZZZ is not used",
    "world-item-missing": "every state item of the world writers exactly once",
    "world-item-unknown-field": "roster.nothing is not a field",
    "wrong-engine-root": "cannot read the engine root battle",
}


def test_the_committed_record_reads_against_its_table_and_hash() -> None:
    spec = read_battle_state_spec(committed(), TABLE)
    assert battle_state_spec_hash(spec) == HASH
    assert spec == load(SAMPLES / "valid/v1.json")
    assert [w["id"] for w in spec["world_items"]] == state_item_ids()


def test_every_reader_invalid_sample_is_refused_for_its_rule() -> None:
    names = sorted(p.stem for p in (SAMPLES / "reader-invalid").glob("*.json"))
    assert names == sorted(READER_INVALID)
    for name, message in READER_INVALID.items():
        with pytest.raises(BattleStateSpecError, match=message.replace(".", r"\.")):
            read_battle_state_spec(load(SAMPLES / f"reader-invalid/{name}.json"))


def test_every_schema_invalid_sample_is_refused() -> None:
    for path in sorted((SAMPLES / "invalid").glob("*.json")):
        with pytest.raises(BattleStateSpecError):
            read_battle_state_spec(load(path))


def test_the_table_binds_the_vocabulary_sizes() -> None:
    with pytest.raises(BattleStateSpecError, match="engine_data_hash is not the table's"):
        read_battle_state_spec(committed(), {**TABLE, "species": []})
    narrow = committed()
    next(f for f in narrow["fields"] if f["id"] == "roster.base_species")["dtype"] = "int8"
    with pytest.raises(BattleStateSpecError, match="401 indices do not fit int8"):
        read_battle_state_spec(narrow, TABLE)
    table = {**TABLE, "types": []}
    empty = committed()
    empty["engine_data_hash"] = sha256_hex(jcs(table))
    with pytest.raises(BattleStateSpecError, match="TYPE has no id"):
        read_battle_state_spec(empty, table)
    with pytest.raises(BattleStateSpecError, match="no list nosuchlist"):
        vocabulary_ids(TABLE, ["nosuchlist"])
    assert len(vocabulary_ids(TABLE, ["abilities", "conditions", "items", "moves"])) > 1000


def test_other_refusals_the_corpus_does_not_hold() -> None:
    lists = committed()
    next(f for f in lists["fields"] if f["id"] == "battle.last_move")["value"]["lists"] = [
        "moves",
        "abilities",
    ]
    with pytest.raises(BattleStateSpecError, match=r"battle\.last_move: lists"):
        read_battle_state_spec(lists)
    dims = committed()
    next(d for d in dims["dims"] if d["symbol"] == "VOLATILE")["lists"].reverse()
    with pytest.raises(BattleStateSpecError, match="dims: VOLATILE"):
        read_battle_state_spec(dims)
    mask = committed()
    next(f for f in mask["fields"] if f["id"] == "roster.item")["reveal"] = "seat_roster.nothing"
    with pytest.raises(BattleStateSpecError, match="is not a field"):
        read_battle_state_spec(mask)
    held = committed()
    next(w for w in held["world_items"] if w["kind"] == "held" and len(w["fields"]) > 1)[
        "fields"
    ].reverse()
    with pytest.raises(BattleStateSpecError, match="world_items:"):
        read_battle_state_spec(held)
    unsorted_omitted = committed()
    unsorted_omitted["omitted"].reverse()
    with pytest.raises(BattleStateSpecError, match="omitted:"):
        read_battle_state_spec(unsorted_omitted)


def test_the_projection_rules() -> None:
    def field(visibility: str, projection: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"visibility": visibility}
        if projection is not None:
            out["projection"] = projection
        return out

    assert [seat_view(field("public"), o) for o in (True, False)] == ["exact", "exact"]
    assert [seat_view(field("hidden"), o) for o in (True, False)] == ["none", "none"]
    assert [seat_view(field("revealed-only"), o) for o in (True, False)] == ["masked", "masked"]
    pct = field("own-exact", "hp_percent")
    assert [seat_view(pct, o) for o in (True, False)] == ["exact", "percent"]
    none = field("own-exact", "none")
    assert [seat_view(none, o) for o in (True, False)] == ["exact", "none"]


def test_engine_paths_name_their_levels() -> None:
    assert path_keys("active.volatiles{}.duration") == [
        ("pokemon", "volatiles"),
        ("effect", "duration"),
    ]
    assert path_keys("pokemon.moveSlots[].pp") == [("pokemon", "moveSlots"), ("move_slot", "pp")]
    assert path_keys("pokemon.statusState.stage") == [
        ("pokemon", "statusState"),
        ("effect", "stage"),
    ]
    assert path_keys("battle.lastMove.move") == [("battle", "lastMove")]


def test_a_mutation_is_refused_and_the_input_is_not_changed() -> None:
    spec = committed()
    before = copy.deepcopy(spec)
    read_battle_state_spec(spec, TABLE)
    assert spec == before
