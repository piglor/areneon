"""The battle-state spec against the live encoder and reducer (#1463, ADR-030 step 5 groundwork).

Gate items 2 and 3 of the ticket: every state column of the encoder's ``Layout`` (the base
layout, the lineage run's options ``candidate_matchup``, ``recent_history`` and
``history_facts``, and ``exact_volatiles``) and every key the reducer's ``BattleState.to_dict``
holds at every request of the committed channels (both seats) maps to spec fields; and no
opponent mapping reads a field another seat cannot see, cross-checked with the reducer's own
opponent-secret rule (an opponent Pokemon holds none of the exact own-side values), with
negative controls.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.battle_state_spec import (
    BattleStateSpecError,
    read_battle_state_spec,
    seat_view,
)
from areneon_ml.model.encoder import EncoderConfig, Layout
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.options import read_options
from areneon_ml.reducer import Vocabulary
from areneon_ml.reducer.reducer import ChannelReducer

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "gen9championsvgc2026regmc"
FORMAT_DIR = ROOT / "fixtures" / "showdown" / FORMAT
SPEC_PATH = ROOT / "config" / "engine-state" / FORMAT / "battle-state-spec.v1.json"
TABLE: dict[str, Any] = json.loads((FORMAT_DIR / "engine-data-2.json").read_text())
STATE_GROUPS = ("mon_cat", "mon_num", "field_cat", "field_num", "cand_cat", "cand_tok")
# The option sets the gate names: the base layout, the lineage run's options, exact_volatiles.
CONFIGS: dict[str, dict[str, bool]] = {
    "base": {},
    "lineage": {"candidate_matchup": True, "recent_history": True, "history_facts": True},
    "exact": {"exact_volatiles": True},
}
# The reducer's id-keyed maps: their keys are ids, so a path reads them as `*`.
ID_MAPS = {
    "boosts",
    "conditions",
    "condition_turns",
    "field_conditions",
    "pp",
    "slots",
    "stats",
    "active",
    "volatile_started",
    "pending_slot",
}
EMPTY: tuple[Any, ...] = (None, {}, [], False, 0)


def spec() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SPEC_PATH.read_text())
    return loaded


@cache
def game() -> GameData:
    return GameData.from_table(TABLE)


def layout(options: dict[str, bool]) -> Layout:
    config = EncoderConfig(
        volatile_buckets=64,
        condition_buckets=32,
        max_moves=8,
        max_team=6,
        max_picks=4,
        stat_scale=500.0,
        turn_scale=30.0,
        options=read_options("encoder", options),
    )
    return Layout.of(config, game())


def pattern(column: str) -> re.Pattern[str]:
    escaped = re.escape(column).replace("\\#", "#").replace("\\*", "*")
    return re.compile("^" + escaped.replace("#", r"\d+").replace("*", "[a-z0-9]+") + "$")


def columns(lay: Layout) -> Iterator[tuple[str, str]]:
    for group in STATE_GROUPS:
        for entry in getattr(lay, group):
            yield group, entry[0] if isinstance(entry, tuple) else entry


def unmapped(record: dict[str, Any]) -> list[str]:
    """Every state column of the gate's layouts that no entry (of an option the layout has) maps."""
    out = []
    entries = [(e, pattern(e["column"])) for e in record["encoder"]]
    for name, options in CONFIGS.items():
        allowed = {"base", *options}
        for group, column in columns(layout(options)):
            hits = [e for e, p in entries if e["group"] == group and p.match(column)]
            if len(hits) != 1 or hits[0]["option"] not in allowed:
                out.append(f"{name} {group} {column}: {len(hits)} entries")
    return out


def test_every_state_column_of_the_gate_layouts_maps_to_exactly_one_entry() -> None:
    record = read_battle_state_spec(spec(), TABLE)
    assert unmapped(record) == []
    # Nothing outside the state groups is laid out here (probe rows, sheet: out of scope).
    for options in CONFIGS.values():
        lay = layout(options)
        assert (lay.cand_num, lay.thr_cat, lay.thr_num, lay.foe_cat, lay.foe_num) == ((),) * 5
    scoped = {o["option"] for o in record["out_of_scope"]}
    assert {"candidate_features", "candidate_target", "opponent_sheet"} <= scoped


def test_no_entry_is_stale() -> None:
    record = spec()
    seen = {(g, c) for options in CONFIGS.values() for g, c in columns(layout(options))}
    stale = [
        e["column"]
        for e in record["encoder"]
        if not any(g == e["group"] and pattern(e["column"]).match(c) for g, c in seen)
    ]
    assert stale == []


def test_a_dropped_entry_leaves_its_columns_unmapped() -> None:
    record = spec()
    record["encoder"] = [e for e in record["encoder"] if e["column"] != "hist_used_#"]
    assert any("hist_used_" in line for line in unmapped(record))
    wrong = spec()
    next(e for e in wrong["encoder"] if e["column"] == "turn")["option"] = "history_facts"
    assert any("field_num turn" in line for line in unmapped(wrong))


# ---- the reducer's keys at every request of the committed channels ----


def battles() -> Iterator[dict[str, Any]]:
    for path in sorted((FORMAT_DIR / "random").glob("b*.json")):
        yield json.loads(path.read_text())
    for path in sorted((FORMAT_DIR / "unavailable").glob("b*.json")):
        yield json.loads(path.read_text())
    for path in sorted((FORMAT_DIR / "search-fidelity").glob("b*.json")):
        yield json.loads(path.read_text())
    for name in (
        "illusion",
        "baton-pass",
        "baton-pass-volatiles",
        "fling",
        "side-guard",
        "transform",
        "revival",
    ):
        yield json.loads((FORMAT_DIR / name / "battle.json").read_text())


def flatten(value: Any, path: str, out: dict[str, list[Any]], ids: bool = False) -> None:
    """``to_dict`` paths: sides as SIDE, list entries of objects as [], id-keyed map keys as *.

    An empty map holds no key, so it adds nothing."""
    if value == {}:  # an empty map holds no key
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            name = "*" if ids else key
            child_ids = key in ID_MAPS or (ids and path.endswith("pending_slot"))
            if path in ("", "history") and key in ("own", "opponent"):
                name = "SIDE"
            if path.endswith("history.SIDE") and key == "roster":
                child_ids = True
            flatten(inner, f"{path}.{name}" if path else name, out, child_ids)
        return
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        for entry in value:
            flatten(entry, f"{path}[]", out)
        return
    out.setdefault(path, []).append(value)


@cache
def observed() -> dict[str, dict[str, list[Any]]]:
    """Per seat side (own, opponent), the normalised keys and their values at every request."""
    vocab = Vocabulary.from_engine_data(TABLE)
    halves: dict[str, dict[str, list[Any]]] = {"own": {}, "opponent": {}, "shared": {}}
    for battle in battles():
        for player in ("p1", "p2"):
            reducer = ChannelReducer(player, battle["teams"][player], vocab)
            for line in battle["channels"][player]["lines"]:
                reducer.feed(line)
                if not line.startswith("|request|"):
                    continue
                state = reducer.state.to_dict()
                flat: dict[str, list[Any]] = {}
                flatten(state, "", flat)
                for key, values in flat.items():
                    if not key.startswith(("SIDE.", "history.SIDE.")):
                        halves["shared"].setdefault(key, []).extend(values)
                history = state.get("history")
                for side in ("own", "opponent"):
                    part: dict[str, list[Any]] = {}
                    flatten(state[side], "SIDE", part)
                    if isinstance(history, dict) and side in history:
                        flatten(history[side], "history.SIDE", part)
                    for key, values in part.items():
                        halves[side].setdefault(key, []).extend(values)
    return halves


def test_every_reducer_key_at_every_request_maps_to_spec_fields() -> None:
    record = spec()
    mapped = {r["key"] for r in record["reducer"]}
    keys = set(observed()["own"]) | set(observed()["opponent"]) | set(observed()["shared"])
    assert len(keys) > 80
    assert sorted(keys - mapped) == []


def test_no_opponent_value_of_an_unprojectable_key() -> None:
    """The reducer's opponent-secret rule and the spec agree: where the spec maps the opponent half
    to nothing although the own half reads fields another seat does not see exactly, the reducer's
    opponent half holds nothing either."""
    record = read_battle_state_spec(spec(), TABLE)
    fields = {f["id"]: f for f in record["fields"]}
    secret = [
        r["key"]
        for r in record["reducer"]
        if not r["opponent"]
        and any(seat_view(fields[ref], owner=False, seat=True) != "exact" for ref in r["own"])
    ]
    # the exact own-side values the reducer's canary (`leaks`) refuses on an opponent are among them
    for key in (
        "SIDE.roster[].stats",
        "SIDE.roster[].moves_own",
        "SIDE.roster[].pp.*",
        "SIDE.roster[].can_mega_evo",
    ):
        assert key in secret, key
    values = observed()["opponent"]
    full = [key for key in secret if any(v not in EMPTY for v in values.get(key, []))]
    assert full == []
    # Negative control: an opponent Pokemon holding exact stats is flagged by the same scan.
    doctored = {**values, "SIDE.roster[].stats.*": [123]}
    assert [k for k in secret if any(v not in EMPTY for v in doctored.get(k, []))] == [
        "SIDE.roster[].stats.*"
    ]


def test_the_reader_refuses_an_opponent_mapping_of_exact_values() -> None:
    leak = spec()
    next(r for r in leak["reducer"] if r["key"] == "SIDE.roster[].pp.*")["opponent"] = [
        "roster.move_pp"
    ]
    with pytest.raises(BattleStateSpecError, match=r"roster\.move_pp is not projectable"):
        read_battle_state_spec(leak)
    hp = spec()
    next(r for r in hp["reducer"] if r["key"] == "SIDE.roster[].hp_den")["opponent"] = [
        "roster.max_hp"
    ]
    with pytest.raises(BattleStateSpecError, match=r"roster\.max_hp is not projectable"):
        read_battle_state_spec(copy.deepcopy(hp))
