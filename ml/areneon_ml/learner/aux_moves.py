"""Move-reveal labels and loss of the auxiliary head (ADR-010 erratum A12.2, #1384).

``model.options.aux_move_head`` gives the Battle network a head that predicts, for each opponent
Pokemon, which move it will reveal NEXT; ``ppo.aux_moves`` trains it. The labels come from the
player's OWN channel only: for a Battle trajectory the reducer state at each decision lists the
moves each opponent entry has revealed (``moves_revealed``), and the label of an entry at decision
``t`` is the set of moves first revealed at the first LATER decision where that set grew. That is a
``|move|`` line the player itself saw later in the same game: public, never an input, and never
written into an observation, a stream or a contract.

Masked (no loss), declared:

* no later decision reveals a move of the entry (including moves revealed only after the
  trajectory's last decision): no negative is ever taken from absence;
* the entry is Transformed at any decision from ``t`` to the first later change (the reducer
  replaces the list with the copied one) and, as a guard for a Transform or a disguise that ended
  between two decisions, the revealed set at ``t`` is no longer contained in a later one;
* Struggle is no slot of a set: it is neither a label nor a known move;
* phantom entries (an Illusion twin) have no label.

The loss is a softmax over the move vocabulary with the already revealed moves (and the two
reserved indices) masked, cross-entropy against the uniform distribution over the label set, the
mean over the unmasked (decision, entry) targets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from areneon_ml.model.gamedata import RESERVED
from areneon_ml.reducer.state import BattleState

#: The engine's id of Struggle, which no set holds as a slot (ADR-010 erratum A12.2: masked).
STRUGGLE = "struggle"


@dataclass(frozen=True)
class AuxTarget:
    """One unmasked target: the token row of an opponent entry, the move vocabulary indices it has
    already revealed (masked in the softmax) and those it reveals next (the label)."""

    row: int
    known: tuple[int, ...]
    next: tuple[int, ...]


@dataclass(frozen=True)
class Seen:
    """One non-phantom opponent entry at one decision."""

    key: int  # its place among the non-phantom entries: the preview order, stable over a game
    row: int  # its token row in that decision's encoded query
    moves: frozenset[str]  # ``moves_revealed`` without Struggle
    transformed: bool


def snapshot(state: BattleState, n_own: int) -> tuple[Seen, ...]:
    """The opponent entries of ``state`` as the decision's query rows them (``n_own`` own rows
    come first)."""
    out = []
    key = 0
    for index, mon in enumerate(state.opponent.roster):
        if mon.phantom:
            continue
        out.append(
            Seen(
                key,
                n_own + index,
                frozenset(mon.moves_revealed) - {STRUGGLE},
                mon.moves_base is not None,
            )
        )
        key += 1
    return tuple(out)


def labels(
    seen: Sequence[Sequence[Seen]], index: Callable[[str], int]
) -> list[tuple[AuxTarget, ...]]:
    """The unmasked targets of each decision of one trajectory (``seen`` in decision order).

    ``index`` maps a move id to its vocabulary index."""
    out: list[tuple[AuxTarget, ...]] = []
    for t, now in enumerate(seen):
        targets = []
        for entry in now:
            found = _first_reveal(seen, t, entry)
            if found:
                targets.append(
                    AuxTarget(
                        entry.row,
                        tuple(sorted(index(m) for m in entry.moves)),
                        tuple(sorted(index(m) for m in found)),
                    )
                )
        out.append(tuple(targets))
    return out


def _first_reveal(seen: Sequence[Sequence[Seen]], t: int, entry: Seen) -> frozenset[str]:
    """The moves ``entry`` first reveals after decision ``t``; empty when masked or never."""
    if entry.transformed:
        return frozenset()
    for later in seen[t + 1 :]:
        match = next((e for e in later if e.key == entry.key), None)
        if match is None or match.transformed or not entry.moves <= match.moves:
            return frozenset()
        if match.moves != entry.moves:
            return match.moves - entry.moves
    return frozenset()


def targets_tensors(
    steps: Sequence[tuple[AuxTarget, ...]], width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(batch index, token row, known mask, target distribution) of a minibatch's targets.

    ``known`` and the distribution are [K, width] (the move vocabulary); K may be 0."""
    batch, rows, known, target = [], [], [], []
    for b, targets in enumerate(steps):
        for item in targets:
            batch.append(b)
            rows.append(item.row)
            mask = torch.zeros(width, dtype=torch.bool)
            mask[list(item.known)] = True
            known.append(mask)
            dist = torch.zeros(width)
            dist[list(item.next)] = 1.0 / len(item.next)
            target.append(dist)
    count = len(batch)
    return (
        torch.tensor(batch, dtype=torch.int64),
        torch.tensor(rows, dtype=torch.int64),
        torch.stack(known) if count else torch.zeros(0, width, dtype=torch.bool),
        torch.stack(target) if count else torch.zeros(0, width),
    )


def aux_terms(
    logits: torch.Tensor, known: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cross-entropy per target [K], top-1 hit per target [K]) of ``logits`` [K, V].

    The already revealed moves and the reserved indices are masked out of the softmax; the hit is
    whether the highest unmasked logit is in the label set (first highest on a tie)."""
    mask = known.clone()
    mask[:, :RESERVED] = True
    floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(mask, floor)
    log_probs = torch.log_softmax(masked, dim=-1)
    ce = -(target * log_probs.masked_fill(mask, 0.0)).sum(-1)
    top = masked.argmax(-1)
    hit = target.gather(1, top[:, None]).squeeze(1) > 0
    return ce, hit
