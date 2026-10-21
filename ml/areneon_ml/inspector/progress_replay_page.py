"""The progress-game replay page and its index (ADR-017 sections 4, 4b and 4c).

The same board and controls as the training-game replay (``replay_page``: the clean-room
interpreter and the view, one hashed script, one hashed style block, one JSON data block), fed with
ONE omniscient view: both sides' exact values, both seats' candidate lists and choices. The page
says so at the top: it holds both sides' hidden information and is for the owner's local use only.
"""

# ruff: noqa: E501  (HTML template lines are kept whole so the markup reads as written)

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from areneon_ml.inspector import page
from areneon_ml.inspector.page import esc, link_id
from areneon_ml.inspector.progress_replay import SEATS, ProgressGame
from areneon_ml.inspector.replay import to_id
from areneon_ml.inspector.replay_page import assets, dex_for_ids

PAGES = "progress-replays"
WARNING = (
    "Omniscient view: both sides' hidden information. An evaluation artifact, never policy input; "
    "local use only. Do not share or commit this page."
)
_WORD = re.compile(r"[^A-Za-z0-9]+")
#: A page file: ``<match>-<game>.html`` (ids only, ADR-017 4c).
GAME_PAGE = re.compile(r"([a-z0-9][a-z0-9_-]{0,63})-(\d+)\.html")


def replay_dir(generation: int) -> str:
    return f"g{generation:06d}"


def page_name(match: str, game: int) -> str:
    return link_id(f"{match}-{game}") + ".html"


def page_path(generation: int, match: str, game: int) -> str:
    """The page's path under the output directory: ``progress-replays/g<NNNNNN>/<match>-<k>.html``."""
    return f"{PAGES}/{replay_dir(generation)}/{page_name(match, game)}"


def species_in(game: ProgressGame) -> set[str]:
    """Every word of the log and of both teams that could name a species (the table keeps real ids)."""
    found: set[str] = set()
    for line in game.record["omniscient_log"]:
        found.update(to_id(word) for word in _WORD.split(line) if word)
        found.update(to_id(part) for part in re.split(r"[|,]", line) if part)
    for seat in SEATS:
        found.update(to_id(str(s.get("species", ""))) for s in game.record["teams"][seat]["team"])
    found.discard("")
    return found


def _seat_label(game: ProgressGame, seat: str) -> str:
    policy = game.record["policies"][seat]
    if seat == game.candidate_seat:
        return f"{seat}: the candidate, the frozen pair in {game.mode} mode ({policy})"
    who = (
        "the previous generation's frozen pair"
        if game.opponent_kind == "previous"
        else "the scripted baseline"
    )
    return f"{seat}: the opponent, {who} ({policy})"


def _result(game: ProgressGame) -> str:
    outcome = game.record["outcome"]
    if outcome["kind"] == "forfeit":
        loser = outcome["loser"]
        return f"{loser} forfeited ({outcome['cause']}) at its request {outcome['request_seq']} after {outcome['turns']} turns"
    if outcome["winner"] == "tie":
        return f"tie after {outcome['turns']} turns"
    return f"{outcome['winner']} won after {outcome['turns']} turns"


def payload(game: ProgressGame, dex: dict[str, Any]) -> dict[str, Any]:
    """The page's data block. Every field is data the reader trusts nothing from."""
    record = game.record
    view = {
        "player": "p1",
        "label": "Omniscient log: both sides, exact values",
        "lines": list(record["omniscient_log"]),
        "team": [],
        "teams": {seat: list(record["teams"][seat]["team"]) for seat in SEATS},
        "decisions": [],
        "seats": [{"seat": seat, "label": _seat_label(game, seat)} for seat in SEATS],
        "decision_points": list(game.points),
        "result": _result(game),
    }
    return {
        "title": f"Generation {game.generation}, {game.match_id}, game {game.game}",
        "views": [view],
        "dex": dex,
        "links": [],
        "notes": [
            WARNING,
            "The omniscient log has no request lines, so each decision is placed by the log's own "
            "structure: Team Preview before the start, a move decision at its turn, a forced switch "
            "at the replacement after a faint (else at the end of its turn). Both seats' candidate "
            "lists are the ones each seat's own request gave.",
        ],
    }


def _nav(game: ProgressGame, *, inspector: bool) -> str:
    links = '<a href="../index.html">Progress replays</a>'
    if inspector:
        links = '<a href="../../index.html">Run</a>' + links
    return links


def replay_page(game: ProgressGame, dex: dict[str, Any], *, inspector: bool = False) -> str:
    """One game's page."""
    script, style = assets("omniscient")
    title = f"Progress replay: generation {game.generation}, {game.match_id}, game {game.game}"
    seed = game.record["seed"].removeprefix("sodium,")[:12]
    body = (
        f"<h1>{esc(title)}</h1>"
        f'<p class="warn"><strong>{esc(WARNING)}</strong></p>'
        f'<p class="note">Run {esc(game.run_id)}, seed {esc(seed)}, {esc(game.mode)} mode against '
        f"{esc(game.opponent_kind)}. Synthetic, public-lineage game from a progress evaluation (a "
        "progress view, not the Training gate). Arrow keys step, the space bar plays.</p>"
        '<div id="replay-root"></div>'
        "<noscript><p>The replay needs JavaScript. It loads nothing from a network.</p></noscript>"
    )
    return page.document(
        title,
        _nav(game, inspector=inspector),
        body,
        payload(game, dex),
        esc(
            "Omniscient view for local use. Built from the progress run's own files; no art is shipped."
        ),
        shell=page.Shell(script, style, "replay-data"),
    )


def dex_for(table: Mapping[str, Any], game: ProgressGame) -> dict[str, Any]:
    return dex_for_ids(table, species_in(game))


def index_page(
    found: Iterable[tuple[int, str, int]], run_id: str | None = None, *, inspector: bool = False
) -> str:
    """The progress replay pages that exist, by generation, match and game (links are relative)."""
    by_gen: dict[int, list[tuple[str, int]]] = {}
    for generation, match, game in sorted(found):
        by_gen.setdefault(generation, []).append((match, game))
    sections = []
    for generation, games in by_gen.items():
        links = " · ".join(
            f'<a href="{replay_dir(generation)}/{page_name(m, g)}">{esc(m)} game {g}</a>'
            for m, g in games
        )
        sections.append(f"<h2>Generation {generation}</h2><p>{links}</p>")
    body = (
        "<h1>Progress-game replays</h1>"
        f'<p class="note"><strong>{esc(WARNING)}</strong></p>'
        '<p class="note">Replay pages built so far, each one stored game of a progress evaluation. '
        "They are built on demand, so a game without a page has not been built.</p>"
        + ("".join(sections) if sections else "<p>None built yet.</p>")
    )
    nav = '<a href="../index.html">Run</a>' if inspector else ""
    return page.document("Progress replays", nav, body, footer=esc(run_id or ""))
