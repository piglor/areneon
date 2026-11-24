"""``python -m areneon_ml.projection serve --engine-data FILE --spec FILE``: one JSON frame per line
in, one per line out, in order (the TS side's projection gate and the harness's reference views).

Frames: ``{"kind": "check", "id", "point", "seat", "query"}`` packs the point (one ``PackSession``
per battle id prefix, so a battle's reducers are fed incrementally), projects it for the seat,
rebuilds the reducer-state view and compares its encoding with ``Encoder.encode`` of the query
under every option set of the gate, and runs the canary; it answers ``{"id", "status": "equal" |
"different" | "refused", "code"?, "differences", "view_sha256", "canary": {"engine", "tensors"}}``.
``{"kind": "view", "id", "point", "seat", "leak"?}`` answers the seat's view and its digest; with
``leak`` naming a field, the other side's raw values of it are added outside the class accessor
(the invariance test's negative control). A packer refusal is a ``refused`` answer with its code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from areneon_ml.model.encoder import EncoderSession
from areneon_ml.model.gamedata import GameData

from .check import compare_encodings, encoders_for
from .packer import PackError, PackSession, plant_engine
from .project import plant_unreadable, project, view_digest
from .spec import SEATS, SpecIndex
from .view import reduced_view


class Server:
    def __init__(self, index: SpecIndex, game: GameData) -> None:
        self.index, self.game = index, game
        self.encoders = encoders_for(game)
        self._battle: str | None = None
        self._session = PackSession(index, game.vocabulary)
        self._encoding: dict[tuple[str, str], EncoderSession] = {}

    def _for(self, frame_id: str) -> None:
        battle = frame_id.rsplit("/", 2)[0]
        if battle != self._battle:
            self._battle = battle
            self._session = PackSession(self.index, self.game.vocabulary)
            self._encoding = {}

    def answer(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        frame_id = str(frame["id"])
        point, seat = frame["point"], str(frame["seat"])
        self._for(frame_id)
        try:
            packed = self._session.pack(point)
        except PackError as error:
            return {"id": frame_id, "status": "refused", "code": error.code, "detail": error.detail}
        view = project(self.index, packed, seat)
        if frame["kind"] == "view":
            leak = frame.get("leak")
            if leak is not None:
                view["fields"][f"raw.{leak}"] = packed["fields"][str(leak)][1 - SEATS.index(seat)]
            return {"id": frame_id, "status": "ok", "view": view, "view_sha256": view_digest(view)}
        team = point["teams"][seat]
        differences: dict[str, Any] = {}
        for name, encoder in self.encoders.items():
            state = reduced_view(self.index, view, team)
            session = self._encoding.setdefault((seat, name), encoder.session())
            difference = compare_encodings(encoder, frame["query"], state, session)
            if difference is not None:
                differences[name] = difference
        digest = view_digest(view)
        planted = PackSession(self.index, self.game.vocabulary).pack(
            {**point, "state": plant_engine(point["state"], seat)}
        )
        canary = {
            "engine": view_digest(project(self.index, planted, seat)) == digest,
            "tensors": view_digest(
                project(self.index, plant_unreadable(self.index, packed, seat), seat)
            )
            == digest,
        }
        return {
            "id": frame_id,
            "status": "different" if differences else "equal",
            "differences": differences,
            "view_sha256": digest,
            "canary": canary,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m areneon_ml.projection")
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--engine-data", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args(argv)
    table = json.loads(args.engine_data.read_text())
    game = GameData.from_table(table)
    server = Server(SpecIndex(json.loads(args.spec.read_text()), table), game)
    for line in sys.stdin:
        if line.strip():
            sys.stdout.write(json.dumps(server.answer(json.loads(line))) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
