"""The bit-equality check of the projection against the encoder (ADR-030 G4, #1464).

``compare_encodings(encoder, query, state)`` encodes the seat's real query twice: as the encoder
always does (``Encoder.encode``: the log reduced) and against the rebuilt reducer-state view
(``Encoder.encode_reduced``). Every tensor must be bit-equal (a float compared by its 32 bits, so
-0.0 and 0.0, or two NaNs, are told apart); the first difference is named by tensor, row and
column. The option sets are the gate's: the default options, the lineage run's
(``candidate_matchup``, ``recent_history``, ``history_facts``) and ``exact_volatiles``, at the sizes
of the spec's own layout test.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from areneon_ml.model.encoder import Encoder, EncoderConfig, EncoderSession
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.options import read_options
from areneon_ml.reducer.state import BattleState

CONFIGS: dict[str, dict[str, bool]] = {
    "base": {},
    "lineage": {"candidate_matchup": True, "recent_history": True, "history_facts": True},
    "exact": {"exact_volatiles": True},
}


def encoders_for(game: GameData) -> dict[str, Encoder]:
    """One encoder per option set of the gate."""
    return {
        name: Encoder(
            game,
            EncoderConfig(
                volatile_buckets=64,
                condition_buckets=32,
                max_moves=8,
                max_team=6,
                max_picks=4,
                stat_scale=500.0,
                turn_scale=30.0,
                options=read_options("encoder", options),
            ),
        )
        for name, options in sorted(CONFIGS.items())
    }


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


def _columns(encoder: Encoder, name: str) -> list[str]:
    entries = getattr(encoder.layout, name, ())
    return [entry[0] if isinstance(entry, tuple) else str(entry) for entry in entries]


def compare_encodings(
    encoder: Encoder,
    query: Mapping[str, Any],
    state: BattleState,
    session: EncoderSession | None = None,
) -> dict[str, Any] | None:
    """The first difference between ``encode(query)`` and ``encode_reduced(query, state)``, or None.

    ``session`` (an ``EncoderSession`` of ``encoder``) encodes the real query incrementally: the
    same tensors as ``encode``."""
    expected = (session or encoder).encode(query).tensors()
    got = encoder.encode_reduced(query, state).tensors()
    for name in sorted(set(expected) | set(got)):
        if name not in expected or name not in got:
            return {"tensor": name, "row": None, "column": None, "index": None}
        a, b = _bits(expected[name]), _bits(got[name])
        if a.shape != b.shape:
            return {"tensor": name, "row": None, "column": "shape", "index": None}
        unequal = (a != b).nonzero()
        if unequal.numel() == 0:
            continue
        where = [int(i) for i in unequal[0].tolist()]
        columns = _columns(encoder, name)
        column = columns[where[-1]] if columns and where[-1] < len(columns) else None
        return {
            "tensor": name,
            "row": where[0] if len(where) > 1 else None,
            "column": column,
            "index": where,
        }
    return None


def query_of(
    battle: Mapping[str, Any], point: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    """The query of a committed fixture decision: the seat's observation as far as the encoder reads
    it (player, kind, the channel up to the request, the team) and its candidates."""
    seat = decision["seat"]
    return {
        "observation": {
            "player": seat,
            "kind": decision["kind"],
            "log": list(battle["channels"][seat][: point["cut"][seat]]),
            "team": {"team": battle["teams"][seat]},
        },
        "candidates": decision["candidates"],
    }
