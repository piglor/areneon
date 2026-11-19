"""Reader for the battle-state tensor schema (``BattleStateSpec``, ADR-030 step 5 groundwork).

The schema is closed and integer-only; this reader adds what the contract profile cannot say, as
the TS reader (``packages/showdown-bridge/src/battle-state-spec.ts``) does: sorted and unique
rows, every group exactly once, field ids that start with their group, dimensions that exist and
are used, ranges that fit their dtype, a seat field that is of class seat, a projection
exactly for an own-exact field and a reveal mask exactly for a revealed-only one (a 0/1 seat
field of the same entity), no omitted engine key that a field reads, mappings that read no
hidden field and, on the opponent side, nothing another seat cannot see, and every world-writer
state item exactly once. Given the engine-data table, it also checks ``engine_data_hash`` and
that every vocabulary index fits its dtype (the sizes are the table's, never the record's). The
engine-bound checks (the count dimensions, the engine paths on real states) are the TS side's.
``battle_state_spec_hash`` is SHA-256 of the JCS bytes.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from areneon_contracts.contracts import ContractError, jcs, schema, sha256_hex, validate

GROUPS = (
    "active",
    "battle",
    "field",
    "roster",
    "seat_active",
    "seat_field",
    "seat_roster",
    "seat_side",
    "side",
)
DTYPE_RANGE = {"int8": (-128, 127), "int16": (-32768, 32767), "int32": (-2147483648, 2147483647)}
ROOTS: dict[str, tuple[str, ...]] = {
    "battle": ("battle",),
    "side": ("side",),
    "roster": ("pokemon",),
    "active": ("active", "side"),
    "field": ("field",),
}
_SEGMENT = re.compile(r"^[a-z]+|\.[A-Za-z]+|\[\]|\{\}")
EFFECT_MAPS = ("sideConditions", "volatiles", "pseudoWeather", "slotConditions")


class BattleStateSpecError(ValueError):
    """A battle-state spec the reader refuses."""


def state_item_ids() -> list[str]:
    """The world writers' state items, from that contract's closed list."""
    return sorted(schema("world-writers")["$defs"]["Item"]["properties"]["id"]["enum"])


def path_keys(path: str) -> list[tuple[str, str]]:
    """The (level, key) pairs of ``Battle.toJSON()`` an engine path reads (TS ``pathKeys``)."""
    segments = _SEGMENT.findall(path)
    root = segments[0]
    level: str | None = "pokemon" if root == "active" else root
    out: list[tuple[str, str]] = []
    for segment in segments[1:]:
        if segment in ("[]", "{}"):
            continue
        if level is None:
            break
        key = segment[1:]
        out.append((level, key))
        if key in ("moveSlots", "baseMoveSlots"):
            level = "move_slot"
        elif key.endswith("State") or key in EFFECT_MAPS:
            level = "effect"
        else:
            level = None
    return out


def vocabulary_ids(table: Mapping[str, Any], lists: Sequence[str]) -> list[str]:
    """Every id of the union of the named lists of the engine-data table, sorted."""
    ids: set[str] = set()
    for name in lists:
        entries = table.get(name)
        if not isinstance(entries, list):
            raise BattleStateSpecError(f"the engine-data table has no list {name}")
        ids.update(str(entry["id"]) for entry in entries)
    return sorted(ids)


def seat_view(field: Mapping[str, Any], owner: bool, seat: bool) -> str:
    """What a seat sees of a field (the projection rules): exact, masked or none.

    ``owner``: the field's side is the reader's own; ``seat``: a seat field is read at the reader's
    own seat index (a seat field is exact only there)."""
    visibility = field["visibility"]
    if visibility == "seat":
        return "exact" if seat else "none"
    if visibility == "hidden":
        return "none"
    if visibility == "public":
        return "exact"
    if visibility == "revealed-only":
        return "masked"
    return "exact" if owner else "none"


def _is_flag(field: Mapping[str, Any]) -> bool:
    value = field["value"]
    return bool(
        field["visibility"] == "seat"
        and value["kind"] == "range"
        and (value["min"], value["max"]) == (0, 1)
    )


def _ordered(rows: Sequence[Any], key: Callable[[Any], str], what: str) -> None:
    for before, after in itertools.pairwise(rows):
        if not key(before) < key(after):
            raise BattleStateSpecError(f"{what}: {key(after)} is repeated or out of order")


def read_battle_state_spec(
    value: object,
    engine_data: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate a record and add what the profile cannot say; with a table, its binding."""
    try:
        validate("battle-state-spec", value)
    except ContractError as exc:
        raise BattleStateSpecError(str(exc)) from exc
    assert isinstance(value, dict)
    spec: dict[str, Any] = value

    _ordered(spec["dims"], lambda d: d["symbol"], "dims")
    for dim in spec["dims"]:
        if dim["kind"] == "vocabulary":
            _ordered(dim["lists"], lambda x: x, f"dims: {dim['symbol']}")
    dims = {d["symbol"] for d in spec["dims"]}
    used: set[str] = set()

    def shape(symbols: Sequence[str], what: str) -> None:
        for symbol in symbols:
            if symbol not in dims:
                raise BattleStateSpecError(f"{what}: {symbol} is not a dimension")
            used.add(symbol)

    if [g["id"] for g in spec["groups"]] != list(GROUPS):
        raise BattleStateSpecError("groups: every group exactly once, in order")
    for group in spec["groups"]:
        shape(group["shape"], f"groups: {group['id']}")

    _ordered(spec["fields"], lambda f: f["id"], "fields")
    fields = {f["id"]: f for f in spec["fields"]}
    for field in spec["fields"]:
        _check_field(field, shape)
    for field in spec["fields"]:
        _check_mask(field, fields)
    _check_gates(spec, fields)
    for dim in spec["dims"]:
        if dim["symbol"] not in used:
            raise BattleStateSpecError(f"dims: {dim['symbol']} is not used")

    _ordered(spec["omitted"], lambda o: f"{o['level']}.{o['key']}", "omitted")
    covered = {
        f"{level}.{key}"
        for field in spec["fields"]
        if field["source"]["kind"] == "engine"
        for level, key in path_keys(field["source"]["path"])
    }
    for row in spec["omitted"]:
        if f"{row['level']}.{row['key']}" in covered:
            raise BattleStateSpecError(f"omitted: {row['level']}.{row['key']} is read by a field")

    _check_maps(spec, fields)
    _check_bindings(spec, engine_data, counts)
    return spec


def _check_bindings(
    spec: Mapping[str, Any],
    engine_data: Mapping[str, Any] | None,
    counts: Mapping[str, int] | None,
) -> None:
    if engine_data is not None:
        _check_table(spec, engine_data)
    if counts is not None:
        _check_counts(spec, counts)


def _check_counts(spec: Mapping[str, Any], counts: Mapping[str, int]) -> None:
    """Each count dimension's size against the binding's count of its source (TS ``bindingFor``)."""
    for dim in spec["dims"]:
        if dim["kind"] == "count" and counts.get(dim["source"]) != dim["size"]:
            raise BattleStateSpecError(
                f"dims: {dim['symbol']} has size {dim['size']}, the {dim['source']} is "
                f"{counts.get(dim['source'])}"
            )


def _check_maps(spec: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    _ordered(spec["encoder"], lambda c: f"{c['group']}:{c['column']}", "encoder")
    for column in spec["encoder"]:
        _check_refs(column, fields, f"encoder: {column['group']} {column['column']}")
    _ordered(spec["reducer"], lambda r: r["key"], "reducer")
    for row in spec["reducer"]:
        _check_refs(row, fields, f"reducer: {row['key']}")
    _ordered(spec["out_of_scope"], lambda o: o["option"], "out_of_scope")

    if [w["id"] for w in spec["world_items"]] != state_item_ids():
        raise BattleStateSpecError(
            "world_items: every state item of the world writers exactly once, in order"
        )
    for item in spec["world_items"]:
        if item["kind"] == "held":
            _ordered(item["fields"], lambda x: x, f"world_items: {item['id']}")
            for ref in item["fields"]:
                if ref not in fields:
                    raise BattleStateSpecError(f"world_items: {item['id']}: {ref} is not a field")


def _check_field(field: Mapping[str, Any], shape: Callable[[Sequence[str], str], None]) -> None:
    where = f"fields: {field['id']}"
    group = field["group"]
    if field["id"].split(".")[0] != group:
        raise BattleStateSpecError(f"{where}: the id does not start with its group")
    shape(field["extra"], where)
    low, high = DTYPE_RANGE[field["dtype"]]
    value = field["value"]
    if value["kind"] == "range":
        if value["min"] > value["max"]:
            raise BattleStateSpecError(f"{where}: min exceeds max")
        if value["min"] < low or value["max"] > high:
            raise BattleStateSpecError(f"{where}: the range does not fit {field['dtype']}")
    else:
        _ordered(value["lists"], lambda x: x, f"{where}: lists")
    seat = group.startswith("seat_")
    source = field["source"]
    if source["kind"] == "engine":
        if seat:
            raise BattleStateSpecError(
                f"{where}: a seat field is a derivation; the engine's state holds nothing "
                "a seat was shown"
            )
        root = re.split(r"[.\[{]", source["path"])[0]
        if root not in ROOTS[group]:
            raise BattleStateSpecError(
                f"{where}: a {group} field cannot read the engine root {root}"
            )
    if seat != (field["visibility"] == "seat"):
        raise BattleStateSpecError(
            f"{where}: the class seat is exactly for a seat field (its own seat's derivation)"
        )
    if "masks" in field and not seat:
        raise BattleStateSpecError(f"{where}: only a seat field masks another")
    if ("projection" in field) != (field["visibility"] == "own-exact"):
        raise BattleStateSpecError(f"{where}: a projection exactly for an own-exact field")
    if ("reveal" in field) != (field["visibility"] == "revealed-only"):
        raise BattleStateSpecError(f"{where}: a reveal mask exactly for a revealed-only field")


def _check_mask(field: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    if "reveal" not in field:
        return
    mask = fields.get(field["reveal"])
    if mask is None:
        raise BattleStateSpecError(
            f"fields: {field['id']}: its mask {field['reveal']} is not a field"
        )
    if (
        not _is_flag(mask)
        or mask["group"] != f"seat_{field['group']}"
        or mask["extra"] != field["extra"]
    ):
        raise BattleStateSpecError(
            f"fields: {field['id']}: its mask must be a 0/1 seat field of the same entity and "
            "extra dimensions"
        )
    if mask.get("masks") != field["id"]:
        raise BattleStateSpecError(
            f"fields: {field['id']}: its mask {mask['id']} masks "
            f"{mask.get('masks', 'nothing')}, not this field"
        )


def _check_gates(spec: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]]) -> None:
    for group in spec["groups"]:
        if "gate" not in group:
            continue
        gate = fields.get(group["gate"])
        if (
            group["id"].startswith("seat_")
            or gate is None
            or not _is_flag(gate)
            or gate["group"] != f"seat_{group['id']}"
            or gate["extra"]
        ):
            raise BattleStateSpecError(
                f"groups: {group['id']}: its gate must be a 0/1 seat field of the group's entity"
            )


def _check_refs(row: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]], what: str) -> None:
    for side, opponent in (("own", False), ("opponent", True), ("shared", True)):
        refs = row[side]
        _ordered(refs, lambda x: x, f"{what} {side}")
        for ref in refs:
            field = fields.get(ref)
            if field is None:
                raise BattleStateSpecError(f"{what} {side}: {ref} is not a field")
            if field["visibility"] == "hidden":
                raise BattleStateSpecError(f"{what} {side}: {ref} is hidden: no seat may read it")
            if seat_view(field, owner=not opponent, seat=True) == "none":
                raise BattleStateSpecError(
                    f"{what} {side}: {ref} is not projectable to another seat"
                )


def _check_table(spec: Mapping[str, Any], table: Mapping[str, Any]) -> None:
    if sha256_hex(jcs(table)) != spec["engine_data_hash"]:
        raise BattleStateSpecError("engine_data_hash is not the table's")
    for dim in spec["dims"]:
        if dim["kind"] == "vocabulary" and not vocabulary_ids(table, dim["lists"]):
            raise BattleStateSpecError(f"dims: {dim['symbol']} has no id")
    for field in spec["fields"]:
        value = field["value"]
        if value["kind"] != "vocabulary":
            continue
        size = 2 + len(vocabulary_ids(table, value["lists"]))
        if size - 1 > DTYPE_RANGE[field["dtype"]][1]:
            raise BattleStateSpecError(
                f"fields: {field['id']}: {size} indices do not fit {field['dtype']}"
            )


def battle_state_spec_hash(spec: Mapping[str, Any]) -> str:
    """SHA-256 of the record's JCS bytes."""
    return sha256_hex(jcs(spec))
