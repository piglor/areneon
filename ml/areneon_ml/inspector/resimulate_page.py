"""The engine re-simulation page (ADR-017 sections 4a, 4b and 4c).

The same board and controls as the progress-game replay: the clean-room interpreter and the view
in omniscient mode, one hashed script, one hashed style block and one JSON data block. The data is
the engine's own omniscient log (both sides' exact values), both teams, and each seat's choices with
the legal set its request gave (no verdict of a judge, no hidden state beyond the log itself). The
page says what it is: an engine re-simulation, checked against both recorded channels.
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

import re
from typing import Any

from areneon_ml.inspector import page
from areneon_ml.inspector.page import esc, link_id
from areneon_ml.inspector.progress_replay import decision_points
from areneon_ml.inspector.readers import SEATS
from areneon_ml.inspector.replay import SeatView, to_id
from areneon_ml.inspector.replay_page import assets, dex_for_ids
from areneon_ml.inspector.resimulate import Resimulated

DIR = "resimulations"
LABEL = "engine re-simulation, checked against both recorded channels"
WARNING = (
    "Omniscient view: both sides' hidden information. An evaluation artifact, never policy input; "
    "local use only. Do not share or commit this page."
)
_WORD = re.compile(r"[^A-Za-z0-9]+")


def page_path(generation: int, game: int) -> str:
    """The page's path under the output directory: ``resimulations/g<NNNNNN>/game-<NNNNNN>.html``."""
    return f"{DIR}/g{generation:06d}/{link_id(f'game-{game:06d}')}.html"


def species_in(done: Resimulated) -> set[str]:
    """Every word of the log and of both teams that could name a species (the table keeps real ids)."""
    found: set[str] = set()
    for line in done.omniscient_log:
        found.update(to_id(word) for word in _WORD.split(line) if word)
        found.update(to_id(part) for part in re.split(r"[|,]", line) if part)
    for view in done.game.views:
        found.update(to_id(str(s.get("species", ""))) for s in view.team)
    found.discard("")
    return found


def _played_by(done: Resimulated, view: SeatView) -> str:
    pair = done.game.pair_policy_id
    return "the frozen pair" if view.played_by == pair else view.played_by


def _result(done: Resimulated) -> str:
    row = done.game.row
    if row.winner is None:
        return f"tie after {row.turns} turns"
    return f"{row.winner} won after {row.turns} turns"


def payload(done: Resimulated, dex: dict[str, Any]) -> dict[str, Any]:
    """The page's data block. Every field is data the reader trusts nothing from."""
    game = done.game
    policies = {
        view.seat: view.decisions[0].policy if view.decisions else "" for view in game.views
    }
    record = {
        "omniscient_log": list(done.omniscient_log),
        "policies": policies,
        "decisions": [
            {
                "player": seat,
                "request_seq": d["request_seq"],
                "chosen": d["choice"],
                "refused": d.get("refused"),
                "candidates": d["candidates"],
            }
            for seat in SEATS
            for d in done.decisions[seat]
        ],
    }
    view = {
        "player": "p1",
        "label": f"Omniscient log: both sides, exact values ({LABEL})",
        "lines": list(done.omniscient_log),
        "team": [],
        "teams": {v.seat: list(v.team) for v in game.views},
        "decisions": [],
        "seats": [
            {"seat": v.seat, "label": f"{v.seat}: played by {_played_by(done, v)}"}
            for v in game.views
        ],
        "decision_points": list(decision_points(record)),
        "result": _result(done),
    }
    return {
        "title": f"Generation {game.generation}, game {game.row.game}",
        "views": [view],
        "dex": dex,
        "links": [],
        "notes": [
            WARNING,
            f"Re-run on the pinned engine from the game's seed, both teams and both seats' recorded choices; {LABEL}: both channels equal the Episodes' byte for byte, and the winner and turns equal generation.json. The omniscient log is the engine's, not stored by the run.",
        ],
    }


def dex_for(table: dict[str, Any], done: Resimulated) -> dict[str, Any]:
    return dex_for_ids(table, species_in(done))


def resimulation_page(done: Resimulated, dex: dict[str, Any], *, inspector: bool = False) -> str:
    """One re-simulated game's page."""
    script, style = assets("omniscient")
    game = done.game
    title = f"Re-simulation: generation {game.generation}, game {game.row.game}"
    seed = game.row.seed.removeprefix("sodium,")[:12]
    body = (
        f"<h1>{esc(title)}</h1>"
        f'<p class="warn"><strong>{esc(WARNING)}</strong></p>'
        f'<p class="note"><strong>{esc(LABEL[0].upper() + LABEL[1:])}.</strong> Run {esc(game.run_key)}, seed {esc(seed)}. '
        "Synthetic, public-lineage training game. Arrow keys step, the space bar plays.</p>"
        '<div id="replay-root"></div>'
        "<noscript><p>The replay needs JavaScript. It loads nothing from a network.</p></noscript>"
    )
    nav = '<a href="../../index.html">Run</a>' if inspector else ""
    return page.document(
        title,
        nav,
        body,
        payload(done, dex),
        esc("Omniscient view for local use. Built from the run's own files; no art is shipped."),
        shell=page.Shell(script, style, "replay-data"),
    )
