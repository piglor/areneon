"""The replay page and its index (ADR-017 sections 4, 4b and 4c).

One page per game: the clean-room interpreter and the board view (``packages/replay-view/web``,
inlined byte for byte as the page's ONE script, with the board's CSS after the inspector's as its
ONE style block), and one JSON data block read by ``JSON.parse``. Everything that comes from data
(names, log lines, decisions, policy ids) is in that block, escaped by ``page.json_block``, or is
escaped HTML text by ``page.esc``. No data is spliced into the script, and no art ships.
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from areneon_ml.inspector import page
from areneon_ml.inspector.page import esc, link_id
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay import GAME_NAME, GameReplay, SeatView, to_id

#: The viewer's own files: plain JavaScript and CSS, read as they are (no bundler, no build step).
ASSETS = Path(__file__).resolve().parents[3] / "packages" / "replay-view" / "web"
_TOKEN = re.compile(r"[^A-Za-z0-9]+")


@cache
def assets() -> tuple[str, str]:
    """(script, style) of a replay page. The script is the interpreter then the view."""
    parts = {name: (ASSETS / name).read_bytes().decode("utf-8") for name in (
        "interpreter.js", "view.js", "replay.css")}  # fmt: skip
    if any("\r" in text for text in parts.values()):
        raise InspectError("a replay asset has a carriage return: the page bytes must be fixed")
    script = parts["interpreter.js"].rstrip("\n") + "\n" + parts["view.js"].rstrip("\n")
    style = page.STYLE + "\n" + parts["replay.css"].strip()
    return script, style


def replay_dir(generation: int) -> str:
    return f"g{generation:06d}"


def page_name(game: int) -> str:
    return link_id(GAME_NAME.format(game=game)) + ".html"


def page_path(generation: int, game: int) -> str:
    """The page's path under the output directory: ``replays/g<NNNNNN>/game-<NNNNNN>.html``."""
    return f"replays/{replay_dir(generation)}/{page_name(game)}"


def species_in(game: GameReplay) -> set[str]:
    """Every word of every line, team and request that could name a species: the table lookup
    keeps only real ids, so the page embeds the species the game uses and nothing else."""
    found: set[str] = set()
    for view in game.views:
        for line in view.lines:
            found.update(to_id(word) for word in _TOKEN.split(line) if word)
            found.update(to_id(part) for part in re.split(r"[|,]", line) if part)
        for item in view.team:
            found.add(to_id(str(item.get("species", ""))))
    found.discard("")
    return found


def dex_for(table: Mapping[str, Any], game: GameReplay) -> dict[str, Any]:
    """The species table for this game's page: name, base species and types, for the species it
    uses (the type colours come from here), and the terrain ids (they tell a terrain from a field
    condition). Read from the hashed engine-data export; no list of game content is written here."""
    ids = species_in(game)
    species = {
        row["id"]: {"name": row["name"], "base": row["base_species"], "types": row["types"]}
        for row in table["species"]
        if row["id"] in ids
    }
    return {"species": species, "terrains": [row["id"] for row in table["terrains"]]}


def read_engine_data(path: Path, format_id: str) -> dict[str, Any]:
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
        species, terrains, fmt = table["species"], table["terrains"], table["format_id"]
        for row in species:
            row["id"], row["name"], row["base_species"], row["types"]
        for row in terrains:
            row["id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InspectError(f"{path.name} is not an engine-data export this viewer reads") from exc
    if fmt != format_id:
        raise InspectError("the engine-data export is for another format than the run's")
    return dict(table)


def _outcome(view: SeatView) -> str:
    verdict = "tie" if view.winner is None else ("won" if view.winner == view.seat else "lost")
    return f"{view.seat} {verdict} after {view.turns} turns"


def payload(game: GameReplay, dex: dict[str, Any]) -> dict[str, Any]:
    """The page's data block. Every field is data the reader trusts nothing from."""
    played = "the frozen pair (self-play)" if game.plan.opponent == "self" else None
    views = []
    for view in game.views:
        who = played or (
            "the frozen pair" if view.played_by == game.pair_policy_id else view.played_by
        )
        views.append(
            {
                "player": view.seat,
                "label": f"{view.seat}'s own channel, played by {who}",
                "lines": list(view.lines),
                "team": list(view.team),
                "decisions": [
                    {"choice": d.choice, "request_seq": d.request_seq, "policy": d.policy}
                    for d in view.decisions
                ],
                "result": _outcome(view),
            }
        )
    notes = [
        "Each panel is one player's own channel, exactly what that player saw. The other half is never "
        "filled in: an opponent shows only what its side's log revealed, and unrevealed Pokémon are "
        "blank tokens.",
    ]
    if game.plan.opponent != "self":
        notes.append(
            f"A scripted baseline ({game.plan.opponent}) played the other seat; its decisions are "
            "recorded in its own Episode."
        )
    return {
        "title": f"Generation {game.generation}, game {game.row.game}",
        "views": views,
        "dex": dex,
        "links": [],
        "notes": notes,
    }


def _nav(game: GameReplay) -> str:
    gen = f"{game.generation:06d}"
    return (
        '<a href="../../index.html">Run</a>'
        f'<a href="../../games-g{gen}.html#g{game.row.game}">Games of generation {game.generation}</a>'
        '<a href="../index.html">Replays</a>'
    )


def replay_page(game: GameReplay, dex: dict[str, Any]) -> str:
    """One game's page."""
    script, style = assets()
    title = f"Replay: generation {game.generation}, game {game.row.game}"
    seed = game.row.seed.removeprefix("sodium,")[:12]
    body = (
        f"<h1>{esc(title)}</h1>"
        f'<p class="note">Run {esc(game.run_key)}, seed {esc(seed)}. Synthetic, public-lineage game, '
        "read from the two players' Episodes. Arrow keys step, the space bar plays.</p>"
        '<div id="replay-root"></div>'
        "<noscript><p>The replay needs JavaScript. It loads nothing from a network.</p></noscript>"
    )
    return page.document(
        title,
        _nav(game),
        body,
        payload(game, dex),
        esc("Built from the run's own files; no art is shipped."),
        shell=page.Shell(script, style, "replay-data"),
    )


def index_page(found: Iterable[tuple[int, int]], run_key: str | None = None) -> str:
    """The list of replay pages that exist, by generation and game (links are relative)."""
    by_gen: dict[int, list[int]] = {}
    for generation, game in sorted(found):
        by_gen.setdefault(generation, []).append(game)
    sections = []
    for generation, games in by_gen.items():
        links = " · ".join(
            f'<a href="{replay_dir(generation)}/{page_name(g)}">game {g}</a>' for g in games
        )
        sections.append(
            f'<h2>Generation {generation}</h2><p><a href="../games-g{generation:06d}.html">Games table</a>'
            f"</p><p>{links}</p>"
        )
    body = (
        "<h1>Training-game replays</h1>"
        '<p class="note">Replay pages built so far. Each is one game seen through the two players\' own '
        "channels. They are built on demand, so a game without a page has not been built.</p>"
        + ("".join(sections) if sections else "<p>None built yet.</p>")
    )
    nav = '<a href="../index.html">Run</a>'
    return page.document("Replays", nav, body, footer=esc(run_key or ""))
