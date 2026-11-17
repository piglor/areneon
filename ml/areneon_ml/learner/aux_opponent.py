"""Opponent next-action, item and ability labels and loss (ADR-010 erratum A14.2, #1434).

``model.options.aux_opponent_heads`` gives the Battle network three training-only heads that
predict, for each opponent Pokemon, what it will do next (a move or SWITCH), and which item and
ability it will reveal; ``ppo.aux_opponent`` trains them. Every label is a line of the player's
OWN channel that the player saw LATER in the same game: public, never an input, never written
into an observation, a stream or a contract. Nothing here reads a team, the omniscient log or a
verdict: the only inputs are the reduced states of the decisions (``snapshot``) and the lines of
the last observation's log (a channel the player holds).

Next action (A14.2.2). A decision is a unit when its request is a move request; each living
non-phantom opponent entry in an active position ``p`` is a target. The window is the log between
that decision's request and the next Battle decision's request (cut at the first ``|turn|`` line);
the last decision has no later request in the stream and so no window (declared: the stream carries
no channel tail). The first event of ``p`` in the window decides: a switch line before any move or
``cant`` line of the window is SWITCH; a move line with no ``[from]`` tag (not Struggle) is that
move; a ``[from]lockedmove`` line, a ``cant`` line, a faint, a drag, a later switch, no event and
any ``replace`` line of the position in the game are masked; a move line with another ``[from]``
tag (a call) is skipped. Targets and the Pokemon switched in are not predicted.

Item and ability (A14.2.3). A decision's non-phantom opponent entry whose item (ability) the reducer
does not show is a target. Its label is the item (ability) the channel first attributes to it AS ITS
OWN later in the game; a transfer, copy or change of it first, a Mega Evolution (ability), a reveal
while Transformed (ability), an Illusion ``replace`` line anywhere (both heads) or no later reveal
masks it. No negative is taken from absence.

The loss is a softmax over the head's vocabulary with the reserved indices masked, cross-entropy
against the label, the mean over the unmasked targets of a minibatch, times the head's coefficient.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from areneon_ml.model.gamedata import RESERVED, GameData
from areneon_ml.reducer.state import BattleState
from areneon_ml.reducer.vocab import to_id

HEADS = ("action", "item", "ability")

#: The engine's id of Struggle: a forced move, never a label (A14.2.2).
STRUGGLE = "struggle"
#: Protocol effect ids whose lines change which Pokemon holds which ability (A14.2.3). The ids are
#: held to the engine data by a test; the rest of the ability and item masks read only the shape of
#: the lines (``[from]``, ``[of]``, ``[silent]``).
SKILL_SWAP = "skillswap"
NEUTRALIZING_GAS = "neutralizinggas"
#: ``[from]`` text of the engine's Frisk: it shows the TARGET's own item (#1171), not a transfer.
FRISK = "ability: Frisk"

#: Lines carry a name after the Pokemon: a named effect has at least two arguments.
NAMED = 2
MIN_PARTS = 3

_POSITION = re.compile(r"^(p[1-4][a-z]?): (.*)$")


@dataclass(frozen=True)
class Unit:
    """One non-phantom opponent entry at one decision."""

    key: int  # its place among the non-phantom entries: the preview order, stable over a game
    row: int  # its token row in that decision's encoded query
    name: str | None  # the nickname the channel shows
    item_shown: bool
    ability_shown: bool


@dataclass(frozen=True)
class Snap:
    """The opponent half of the reduced state at one Battle decision."""

    log_len: int  # lines of the player's channel up to and including the request
    move_request: bool
    units: tuple[Unit, ...]
    active: tuple[tuple[str, int], ...]  # (position letter, key) of the living active entries


@dataclass(frozen=True)
class OppLabels:
    """The unmasked targets of one decision: (token row, label index) per head, and for each head
    the entries that were units but have no label."""

    action: tuple[tuple[int, int], ...] = ()
    item: tuple[tuple[int, int], ...] = ()
    ability: tuple[tuple[int, int], ...] = ()
    masked: tuple[int, int, int] = (0, 0, 0)  # action, item, ability

    def of(self, head: str) -> tuple[tuple[int, int], ...]:
        found: tuple[tuple[int, int], ...] = getattr(self, head)
        return found


@dataclass(frozen=True)
class Lookup:
    """Name to vocabulary index (None when the name is no id of that vocabulary)."""

    move: Callable[[str], int | None]
    item: Callable[[str], int | None]
    ability: Callable[[str], int | None]
    switch: int  # the SWITCH index of the action head


def lookup_of(game: GameData) -> Lookup:
    """The lookups of a game's vocabularies; SWITCH is one past the move vocabulary."""

    def of(vocab: Mapping[str, int]) -> Callable[[str], int | None]:
        return lambda name: vocab.get(to_id(name))

    return Lookup(
        of(game.moves.index), of(game.items.index), of(game.abilities.index), len(game.moves)
    )


def snapshot(state: BattleState, n_own: int, log_len: int) -> Snap:
    """The opponent entries of ``state`` as the decision's query rows them (``n_own`` own rows
    come first), at a channel ``log_len`` lines long."""
    units, active = [], []
    key = 0
    for index, mon in enumerate(state.opponent.roster):
        if mon.phantom:
            continue
        units.append(Unit(key, n_own + index, mon.name, mon.item_known, mon.ability is not None))
        if mon.active_slot is not None and not mon.fainted:
            active.append((mon.active_slot, key))
        key += 1
    return Snap(log_len, state.request_kind == "move", tuple(units), tuple(active))


# -- lines ----------------------------------------------------------------------------------------


def _split(raw: str) -> tuple[str, list[str]] | None:
    parts = raw.split("|")
    if len(parts) < MIN_PARTS or parts[0] != "":
        return None
    return parts[1], parts[2:]


def _tag(args: Sequence[str], name: str) -> str | None:
    """The text after ``[name]`` (stripped), or None when the line has no such tag."""
    prefix = f"[{name}]"
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :].strip()
    return None


def _ref(arg: str | None, side: str) -> tuple[str, str] | None:
    """(position, nickname) of an opponent reference ``p2a: Nick``; None for anything else."""
    found = _POSITION.match(arg or "")
    if found is None or not found.group(1).startswith(side):
        return None
    return found.group(1), found.group(2)


def _window_label(window: Sequence[str], position: str, lookup: Lookup) -> int | None:
    """The action label of ``position`` in ``window``: the first event of the position decides."""
    side = position[:2]
    acted = False
    for raw in window:
        split = _split(raw)
        if split is None:
            continue
        kind, args = split
        if kind == "turn":
            break
        ref = _ref(args[0], side)
        mine = ref is not None and ref[0] == position
        if kind in ("move", "cant"):
            if mine:
                found = _move_label(kind, args, lookup)
                if found is not _SKIP:
                    return found
            acted = True
        elif mine and kind == "switch":
            return None if acted else lookup.switch
        elif mine and kind in ("drag", "faint", "replace"):
            return None
    return None


#: A move line of the position that is a call (another ``[from]`` tag): not its action, scan on.
_SKIP: Any = object()


def _move_label(kind: str, args: Sequence[str], lookup: Lookup) -> int | None:
    if kind == "cant" or len(args) < NAMED:
        return None
    cause = _tag(args, "from")
    if cause is not None:
        # A forced continuation is masked; any other cause is a call made by another effect.
        return None if cause.replace(" ", "") == "lockedmove" else _SKIP
    if to_id(args[1]) == STRUGGLE:
        return None
    return lookup.move(args[1])


@dataclass
class _Event:
    index: int  # the line the event is on
    value: int | None  # the revealed vocabulary index; None for a mask


class _Scanner:
    """The first item and ability event of each opponent entry, over a whole channel."""

    def __init__(
        self, side: str, names: Mapping[str, int], keys: Sequence[int], lookup: Lookup
    ) -> None:
        self.item: dict[int, _Event] = {}
        self.ability: dict[int, _Event] = {}
        self._side, self._names, self._keys, self._lookup = side, names, keys, lookup
        self._occupant: dict[str, int] = {}
        self._transformed: set[int] = set()
        # The line being read.
        self._index = 0
        self._args: list[str] = []
        self._pos: str | None = None
        self._key: int | None = None
        self._other: int | None = None
        self._cause: str | None = None

    def _who(self, arg: str | None) -> tuple[str | None, int | None]:
        ref = _ref(arg, self._side)
        return (None, None) if ref is None else (ref[0], self._names.get(ref[1]))

    def _put(self, table: dict[int, _Event], key: int | None, value: int | None) -> None:
        if key is not None and key not in table:
            table[key] = _Event(self._index, value)

    def feed(self, index: int, raw: str) -> None:
        split = _split(raw)
        if split is None:
            return
        kind, self._args = split
        self._index = index
        self._pos, self._key = self._who(self._args[0])
        self._cause = _tag(self._args, "from")
        of = _tag(self._args, "of")
        self._other = self._who(of)[1] if of is not None else None
        handler = self._HANDLERS.get(kind)
        if handler is not None:
            handler(self)
        elif self._pos is not None and self._cause is not None and of is None:
            self._own_effect()
        if any(NEUTRALIZING_GAS in to_id(a) for a in self._args):
            for key in self._keys:  # every Pokemon's ability may be suppressed from here on
                self._put(self.ability, key, None)

    def _switch(self) -> None:
        if self._pos is None:
            return
        self._transformed.discard(self._occupant.pop(self._pos, -1))
        if self._key is not None:
            self._occupant[self._pos] = self._key

    def _faint(self) -> None:
        if self._pos is not None:
            self._transformed.discard(self._occupant.pop(self._pos, -1))

    def _transform(self) -> None:
        if self._key is not None:
            self._transformed.add(self._key)

    def _enditem(self) -> None:
        args = self._args
        if len(args) < NAMED or not args[1] or args[1].startswith("["):
            return  # an item the engine no longer names (Fling of a gone item) shows nothing
        # A silent end is a transfer (Thief, Pickpocket): the item is no longer the Pokemon's own.
        silent = _tag(args, "silent") is not None
        self._put(self.item, self._key, None if silent else self._lookup.item(args[1]))

    def _item(self) -> None:
        if len(self._args) < NAMED:
            return
        if self._cause is None:
            self._put(self.item, self._key, self._lookup.item(self._args[1]))
        elif self._cause == FRISK:  # Frisk shows the target's own item and the frisker's ability
            self._put(self.item, self._key, self._lookup.item(self._args[1]))
            self._put(self.ability, self._other, self._lookup.ability("Frisk"))
        else:  # Trick, Bestow, Recycle, Pickpocket and the like: a transfer
            self._put(self.item, self._key, None)

    def _mega(self) -> None:
        self._put(self.ability, self._key, None)  # the label would be the Mega forme's
        if len(self._args) > NAMED:
            self._put(self.item, self._key, self._lookup.item(self._args[2]))

    def _ability(self) -> None:
        if len(self._args) < NAMED:
            return
        if self._cause is not None or _tag(self._args, "of") is not None:
            self._put(self.ability, self._key, None)
            self._put(self.ability, self._other, None)
        elif self._key in self._transformed:
            self._put(self.ability, self._key, None)  # a reveal while Transformed is the copy's
        else:
            self._put(self.ability, self._key, self._lookup.ability(self._args[1]))

    def _end_ability(self) -> None:
        self._put(self.ability, self._key, None)

    def _activate(self) -> None:
        if len(self._args) > 1 and to_id(self._args[1]).endswith(SKILL_SWAP):
            self._put(self.ability, self._key, None)
            self._put(self.ability, self._other, None)
        elif self._pos is not None and self._cause is not None and self._other is None:
            self._own_effect()

    def _own_effect(self) -> None:
        """A line of the Pokemon itself caused by its own item or ability (no ``[of]``)."""
        assert self._cause is not None
        kind, _, name = self._cause.partition(":")
        if kind == "item" and name.strip():
            self._put(self.item, self._key, self._lookup.item(name))
        elif kind == "ability" and name.strip():
            copied = self._key in self._transformed
            self._put(self.ability, self._key, None if copied else self._lookup.ability(name))

    _HANDLERS: Mapping[str, Callable[[_Scanner], None]] = {
        "switch": _switch,
        "drag": _switch,
        "faint": _faint,
        "-transform": _transform,
        "-enditem": _enditem,
        "-item": _item,
        "-mega": _mega,
        "-ability": _ability,
        "-endability": _end_ability,
        "-activate": _activate,
    }


def _names(snap: Snap) -> dict[str, int]:
    """Nickname to key over the last decision's entries; a nickname two entries share is left out
    (a line naming it is attributable to neither)."""
    counts: dict[str, int] = {}
    for unit in snap.units:
        if unit.name is not None:
            counts[unit.name] = counts.get(unit.name, 0) + 1
    return {u.name: u.key for u in snap.units if u.name is not None and counts[u.name] == 1}


def _replaced(log: Sequence[str], side: str) -> set[str]:
    """The opponent positions that have a ``replace`` line anywhere in the channel."""
    found = set()
    for raw in log:
        split = _split(raw)
        if split is not None and split[0] == "replace":
            ref = _ref(split[1][0], side)
            if ref is not None:
                found.add(ref[0])
    return found


def labels(
    snaps: Sequence[Snap],
    log: Sequence[str],
    side: str,
    heads: Sequence[str],
    lookup: Lookup,
) -> list[OppLabels]:
    """The unmasked targets of each decision of one trajectory (``snaps`` in decision order).

    ``log`` is the longest observation log of the trajectory (every decision's log is a prefix of
    it); ``side`` is the opponent's side id (``p1`` or ``p2``)."""
    if not snaps:
        return []
    replaced = _replaced(log, side)
    timelines = None
    if "item" in heads or "ability" in heads:
        timelines = _Scanner(side, _names(snaps[-1]), [u.key for u in snaps[-1].units], lookup)
        for index, raw in enumerate(log):
            timelines.feed(index, raw)
    out: list[OppLabels] = []
    for t, snap in enumerate(snaps):
        rows = {unit.key: unit.row for unit in snap.units}
        found: dict[str, list[tuple[int, int]]] = {head: [] for head in HEADS}
        masked = {head: 0 for head in HEADS}
        if "action" in heads and snap.move_request:
            end = snaps[t + 1].log_len if t + 1 < len(snaps) else None
            for letter, key in snap.active:
                position = f"{side}{letter}"
                label = None
                if end is not None and position not in replaced:
                    label = _window_label(log[snap.log_len : end], position, lookup)
                _put(found["action"], masked, "action", rows[key], label)
        for unit in snap.units:
            if timelines is None:
                break
            shown = {"item": unit.item_shown, "ability": unit.ability_shown}
            for head, table in (("item", timelines.item), ("ability", timelines.ability)):
                if head not in heads or shown[head]:
                    continue
                event = table.get(unit.key)
                label = None
                if event is not None and event.index >= snap.log_len and not replaced:
                    label = event.value
                _put(found[head], masked, head, unit.row, label)
        out.append(
            OppLabels(
                tuple(found["action"]),
                tuple(found["item"]),
                tuple(found["ability"]),
                (masked["action"], masked["item"], masked["ability"]),
            )
        )
    return out


def _put(
    found: list[tuple[int, int]], masked: dict[str, int], head: str, row: int, label: int | None
) -> None:
    if label is None:
        masked[head] += 1
    else:
        found.append((row, label))


# -- the loss -------------------------------------------------------------------------------------


def targets_tensors(
    steps: Sequence[Sequence[tuple[int, int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(batch index, token row, label) of a minibatch's targets of one head; K may be 0."""
    batch, rows, target = [], [], []
    for b, targets in enumerate(steps):
        for row, label in targets:
            batch.append(b)
            rows.append(row)
            target.append(label)
    return (
        torch.tensor(batch, dtype=torch.int64),
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(target, dtype=torch.int64),
    )


def opponent_terms(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cross-entropy per target [K], top-1 hit per target [K]) of ``logits`` [K, V].

    The reserved indices are masked out of the softmax; the hit is whether the highest unmasked
    logit is the label (first highest on a tie)."""
    masked = logits.clone()
    masked[:, :RESERVED] = torch.finfo(logits.dtype).min
    log_probs = torch.log_softmax(masked, dim=-1)
    ce = -log_probs.gather(1, target[:, None]).squeeze(1)
    return ce, masked.argmax(-1) == target
