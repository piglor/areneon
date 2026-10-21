"""A battle with Revival Blessing (the `reviving` request) through the reducer and the encoder.

`fixtures/showdown/<format>/revival/` holds a seeded random-legal battle in which p1's Pawmot
revives a Pokemon (`packages/showdown-bridge/scripts/capture-revival-battle.ts`; CI re-plays it)
and the engine-judged legal set of each reviving request.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from areneon_ml.model.candidates import parse_choice
from areneon_ml.reducer import Vocabulary, reduce_channel

from .support import FORMAT_DIR, encoder, request_points, table

BATTLE = json.loads((FORMAT_DIR / "revival" / "battle.json").read_text())
LEGAL = json.loads((FORMAT_DIR / "revival" / "legal.json").read_text())


def _reviving(body: dict[str, Any]) -> bool:
    return any(mon.get("reviving") for mon in body["side"]["pokemon"])


def test_the_reducer_follows_the_whole_battle_including_the_revival() -> None:
    for player in ("p1", "p2"):
        lines = BATTLE["channels"][player]["lines"]
        state = reduce_channel(
            player, lines, BATTLE["teams"][player], Vocabulary.from_engine_data(table())
        )
        assert state.turn > 1
    assert "[from] move: Revival Blessing" in "\n".join(BATTLE["channels"]["p1"]["lines"])


def test_every_reviving_request_encodes_with_the_engines_own_candidates() -> None:
    lines = BATTLE["channels"]["p1"]["lines"]
    points = [(i, b) for i, b in request_points(lines) if _reviving(b)]
    assert len(points) == len(LEGAL) >= 1
    for (index, body), legal in zip(points, LEGAL, strict=True):
        assert body["forceSwitch"] == [True, False]
        candidates = [{"choice": c, "pending": False} for c in legal["choices"]]
        assert candidates
        for candidate in candidates:
            parsed = parse_choice(candidate["choice"])
            assert parsed.format() == candidate["choice"]
        query = {
            "observation": {
                "kind": "battle",
                "player": "p1",
                "log": lines[: index + 1],
                "team": {"team": BATTLE["teams"]["p1"]},
            },
            "candidates": candidates,
        }
        encoded = encoder().encode(query)
        rows = encoded.mon_cat.shape[0]
        assert encoded.cand_tok.max() < rows
        assert torch.isfinite(encoded.mon_num).all()
        # A revival pick names a fainted Pokemon of the request, never a living one.
        for choice in legal["choices"]:
            first = choice.split(", ")[0]
            if first.startswith("switch "):
                mon = body["side"]["pokemon"][int(first.split()[1]) - 1]
                assert mon["condition"].endswith(" fnt")


def test_every_other_request_point_of_the_battle_encodes_too() -> None:
    for player in ("p1", "p2"):
        lines = BATTLE["channels"][player]["lines"]
        for index, body in request_points(lines):
            if _reviving(body):
                continue
            kind = "preview" if body.get("teamPreview") else "battle"
            if kind == "preview":
                choices = ["team 1234"]
            elif body.get("forceSwitch"):
                choices = ["pass, pass"]
            else:
                choices = ["move 1, move 1"]
            encoder().encode(
                {
                    "observation": {
                        "kind": kind,
                        "player": player,
                        "log": lines[: index + 1],
                        "team": {"team": BATTLE["teams"][player]},
                    },
                    "candidates": [{"choice": c, "pending": False} for c in choices],
                }
            )
