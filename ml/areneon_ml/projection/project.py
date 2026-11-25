"""The player-view projection of the battle-state spec (ADR-030 G4, #1464).

``project(index, packed, seat)`` is what one seat may read of the packed tensors, and it reads a
field ONLY through its visibility class (``seat_view`` of the contracts reader, the one accessor;
the seat is always explicit): ``hidden`` has no accessor; ``own-exact`` with projection ``none``
reaches the owning side's seat only; ``revealed-only`` is the value where the seat's own reveal
mask is set and the unknown marker (null) elsewhere; ``seat`` is the seat's own row and never
another's; a gated group (``roster``, gate ``seat_roster.seen``) is the unknown entry (null) for
another seat wherever the gate is not set. A field of a side is keyed ``<id>@own`` or
``<id>@opponent`` from the seat's point of view, so a view never names a player. The view is
``{"seat", "spec_hash", "fields"}``: the shape the differential harness's ``projection`` kind
compares (``battle-projection.ts``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from areneon_contracts.battle_state_spec import seat_view
from areneon_contracts.contracts import jcs, sha256_hex

from .spec import SEATS, ProjectionError, SpecIndex

FIRST_ID = 2  # the index of a vocabulary's first id (0 none, 1 unknown)


def _seat_index(seat: str) -> int:
    if seat not in SEATS:
        raise ProjectionError(f"no seat {seat!r}")
    return SEATS.index(seat)


def _masked(values: Any, mask: Any) -> Any:
    """``values`` where ``mask`` is set, null elsewhere (same shape)."""
    if isinstance(values, list):
        return [_masked(v, m) for v, m in zip(values, mask, strict=True)]
    return values if mask else None


def project(index: SpecIndex, packed: Mapping[str, Any], seat: str) -> dict[str, Any]:  # noqa: PLR0912
    """The fields ``seat`` may read of ``packed`` (see the module text)."""
    si = _seat_index(seat)
    if packed.get("spec_hash") != index.spec_hash:
        raise ProjectionError("the tensors were packed with another record")
    tensors = packed["fields"]
    out: dict[str, Any] = {}
    for field_id, field in index.fields.items():
        group = field["group"]
        data = tensors[field_id]
        if group in ("battle", "field"):  # nobody's: read as another seat's
            if seat_view(field, owner=False, seat=True) == "exact":
                out[field_id] = copy.deepcopy(data)
            continue
        if group.startswith("seat_"):
            if seat_view(field, owner=True, seat=True) != "exact":
                continue
            row = data[si]  # this seat's own bookkeeping, never another seat's row
            if group == "seat_field":
                out[field_id] = copy.deepcopy(row)
            else:
                for d, half in ((si, "own"), (1 - si, "opponent")):
                    out[f"{field_id}@{half}"] = copy.deepcopy(row[d])
            continue
        gate = index.groups[group].get("gate")
        for d, half in ((si, "own"), (1 - si, "opponent")):
            view = seat_view(field, owner=half == "own", seat=True)
            if view == "none":
                continue
            values = copy.deepcopy(data[d])
            if view == "masked":
                values = _masked(values, tensors[field["reveal"]][si][d])
            if gate is not None and half == "opponent":
                seen = tensors[gate][si][d]
                values = [v if s else None for v, s in zip(values, seen, strict=True)]
            out[f"{field_id}@{half}"] = values
    return {"seat": seat, "spec_hash": index.spec_hash, "fields": out}


def view_digest(view: Mapping[str, Any]) -> str:
    """SHA-256 of a view's JCS bytes (the invariance and canary tests compare views by it)."""
    return sha256_hex(jcs(view))


def _changed(value: Any, rule: Mapping[str, Any]) -> Any:
    """A different value of the same field (within its range, or another vocabulary index)."""
    if isinstance(value, list):
        return [_changed(v, rule) for v in value]
    if rule["kind"] == "vocabulary":
        return FIRST_ID + 1 if value == FIRST_ID else FIRST_ID
    return value + 1 if value + 1 <= rule["max"] else value - 1


def plant_unreadable(index: SpecIndex, packed: Mapping[str, Any], seat: str) -> dict[str, Any]:
    """A copy of ``packed`` with every value ``seat`` may not read changed: hidden fields
    everywhere, the other side's own-exact fields, the other seat's bookkeeping, the other side's
    gated entries the seat has not seen and every masked value whose mask is not set. Computed
    from the record's classes directly (not through ``project``), so it is the canary's own rule."""
    si = _seat_index(seat)
    out = copy.deepcopy(dict(packed))
    tensors = out["fields"]
    original = packed["fields"]
    for field_id, field in index.fields.items():
        rule = field["value"]
        group = field["group"]
        visibility = field["visibility"]
        if visibility == "hidden":
            tensors[field_id] = _changed(original[field_id], rule)
        elif group.startswith("seat_"):
            tensors[field_id][1 - si] = _changed(original[field_id][1 - si], rule)
        elif group in ("battle", "field"):
            continue
        else:
            other = 1 - si
            if visibility == "own-exact":
                tensors[field_id][other] = _changed(original[field_id][other], rule)
            if visibility == "revealed-only":
                for d in (si, other):
                    tensors[field_id][d] = _where(
                        original[field_id][d], original[field["reveal"]][si][d], rule
                    )
            gate = index.groups[group].get("gate")
            if gate is not None and visibility != "own-exact":
                seen = original[gate][si][other]
                current = tensors[field_id][other]
                tensors[field_id][other] = [
                    v if s else _changed(v, rule) for v, s in zip(current, seen, strict=True)
                ]
    return out


def _where(values: Any, mask: Any, rule: Mapping[str, Any]) -> Any:
    """``values`` changed where ``mask`` is not set."""
    if isinstance(values, list):
        return [_where(v, m, rule) for v, m in zip(values, mask, strict=True)]
    return values if mask else _changed(values, rule)
