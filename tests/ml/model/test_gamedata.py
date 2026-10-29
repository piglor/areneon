"""Game vocabularies come from the hashed engine-data export, never from model code."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from areneon_contracts.contracts import ContractError
from areneon_ml.model.gamedata import NONE, RESERVED, UNKNOWN, GameData, GameDataError

from .support import FORMAT_DIR, game, table


def test_sizes_are_the_export_s_counts_plus_the_two_reserved_indices() -> None:
    summary = json.loads((FORMAT_DIR / "engine-data.summary.json").read_text())
    counts = summary["counts"]
    data = game()
    assert data.engine_data_hash == summary["engine_data_hash"]
    assert data.showdown_commit == summary["showdown_commit"]
    for name, vocab in [
        ("types", data.types), ("species", data.species), ("moves", data.moves),
        ("items", data.items), ("abilities", data.abilities), ("statuses", data.statuses),
        ("weathers", data.weathers), ("terrains", data.terrains),
    ]:  # fmt: skip
        assert len(vocab) == counts[name] + RESERVED, name
    assert data.species_attributes.shape[0] == len(data.species)
    assert data.move_attributes.shape[0] == len(data.moves)
    assert data.item_attributes.shape[0] == len(data.items)
    assert len(data.move_base_pp) == len(data.moves)


def test_a_vocabulary_grows_with_the_table_not_with_the_code() -> None:
    changed = copy.deepcopy(table())
    extra = copy.deepcopy(changed["items"][-1])
    extra.update(id="zzzitem", name="Zzz Item", num=9999)
    changed["items"].append(extra)
    assert len(GameData.from_table(changed).items) == len(game().items) + 1
    assert GameData.from_table(changed).engine_data_hash != game().engine_data_hash


def test_ids_index_from_the_first_unreserved_slot_in_the_export_s_order() -> None:
    data = game()
    first = table()["species"][0]["id"]
    assert data.species.lookup(first) == RESERVED
    assert data.species.ids[0] == first
    assert data.moves.lookup(table()["moves"][-1]["id"]) == len(data.moves) - 1
    assert NONE == 0
    assert UNKNOWN == 1
    assert data.vocabulary.species("Venusaur") == "venusaur"


def test_an_id_outside_the_table_is_refused_or_unknown_where_the_engine_owns_the_name() -> None:
    data = game()
    with pytest.raises(GameDataError, match="not a species"):
        data.species.lookup("missingno")
    assert data.moves.lookup_or_unknown("recharge") == UNKNOWN


def test_attributes_come_from_the_table_and_are_scaled_by_the_data() -> None:
    data = game()
    for tensor in (data.species_attributes, data.move_attributes, data.item_attributes):
        assert tensor.dtype == torch.float32
        assert float(tensor.abs().max()) <= 1.0
        assert float(tensor[NONE].abs().sum()) == 0.0
        assert float(tensor[UNKNOWN].abs().sum()) == 0.0
    flamethrower = data.move_attributes[data.moves.lookup("flamethrower")]
    fire = data.types.lookup("fire") - RESERVED
    assert flamethrower[fire] == 1.0
    assert data.move_base_pp[data.moves.lookup("flamethrower")] == next(
        m["pp"] for m in table()["moves"] if m["id"] == "flamethrower"
    )
    venusaur = data.species_attributes[data.species.lookup("venusaur")]
    assert venusaur[data.types.lookup("grass") - RESERVED] == 1.0
    berry = data.item_attributes[data.items.lookup("aspearberry")]
    assert berry[0] == 1.0
    assert data.item_attributes[data.items.lookup("leftovers")][0] == 0.0


def test_a_table_that_breaks_its_contract_is_refused() -> None:
    broken = copy.deepcopy(table())
    broken["species"][0]["types"] = ["a", "b", "c"]
    with pytest.raises(ContractError):
        GameData.from_table(broken)


def test_a_version_2_table_loads_beside_version_1_and_changes_nothing_else() -> None:
    """ADR-010 erratum A4 item 7: `conditions` is the only addition to what version 1 gives."""
    v2 = json.loads((FORMAT_DIR / "engine-data-2.json").read_text())
    summary = json.loads((FORMAT_DIR / "engine-data-2.summary.json").read_text())
    loaded, v1 = GameData.from_table(v2), game()
    assert loaded.engine_data_hash == summary["engine_data_hash"] != v1.engine_data_hash
    assert loaded.conditions is not None
    assert v1.conditions is None
    assert len(loaded.conditions) == summary["counts"]["conditions"] + RESERVED
    assert loaded.attribute_digest == v1.attribute_digest
    assert loaded.species.ids == v1.species.ids
    assert loaded.moves.ids == v1.moves.ids
    assert loaded.type_chart == v1.type_chart
    with pytest.raises(ContractError):  # version 2 must carry its condition list
        GameData.from_table({k: v for k, v in v2.items() if k != "conditions"})
    with pytest.raises(ContractError):  # and version 1 must not
        GameData.from_table({**table(), "conditions": []})
