"""Engine re-simulation of one played training game, accepted only when it matches (ADR-017 4a).

The two seats' Episodes of a played game hold everything the pinned engine needs to run the battle
again: the seed, both teams and both seats' recorded choices (a choice the engine refused as
``[Unavailable choice]`` included). ``packages/showdown-bridge/scripts/resimulate.ts`` re-runs it
with ``runBattle`` and ``replayActor`` and returns both players' channels and the omniscient log.
The result is accepted only if

* both re-simulated channels equal the two stored Episode channels line for line;
* the winner and the turn count equal ``generation.json``'s row; and
* the choices the engine took, refusals included, are the recorded ones.

Any difference is an :class:`InspectError` and nothing is shown. A forfeited game has no Episodes
and is refused before the engine runs. The log holds both sides' hidden information: it is never
written into the run directory. Refusal messages name a seat and a line number, never a line.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areneon_ml.inspector.progress_replay import log_result
from areneon_ml.inspector.readers import SEATS, InspectError, closed
from areneon_ml.inspector.replay import GameReplay, read_games

SCRIPT = "packages/showdown-bridge/scripts/resimulate.ts"
TIMEOUT_SECONDS = 600
RESULT_KEYS: tuple[set[str], set[str]] = (
    {"winner", "turns", "omniscient_log", "channels", "team_hashes", "decisions"},
    set(),
)
DECISION_KEYS: tuple[set[str], set[str]] = (
    {"player", "request_seq", "choice", "candidates"},
    {"refused"},
)

#: Runs the engine on a request (a JSON object) and returns the result's JSON text.
Runner = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Resimulated:
    """An accepted re-simulation: the game as read from the Episodes and the engine's own log."""

    game: GameReplay
    omniscient_log: tuple[str, ...]
    #: Per seat, the engine's decisions as it took them: (request_seq, choice, refused, candidates).
    decisions: dict[str, tuple[dict[str, Any], ...]]


def node_runner(repo: Path, node: str) -> Runner:
    """The runner that starts the Node script (stdin: the request, stdout: one JSON line)."""

    def run(request: dict[str, Any]) -> str:
        try:
            done = subprocess.run(
                [node, str(repo / SCRIPT)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
                timeout=TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InspectError(f"cannot run {SCRIPT} with {node}: {type(exc).__name__}") from exc
        if done.returncode != 0:
            first = (done.stderr.splitlines() or [""])[0][:200]  # one line: never a log fragment
            raise InspectError(f"{SCRIPT} failed (exit {done.returncode}): {first}")
        return done.stdout

    return run


def request_for(game: GameReplay, format_id: str, max_turns: int) -> dict[str, Any]:
    """The Node request: the seed, both Episodes' own teams and both seats' recorded choices."""
    teams = {view.seat: list(view.team) for view in game.views}
    choices = [
        {"player": view.seat, "request_seq": d.request_seq, "choice": d.choice}
        for view in game.views
        for d in view.decisions
    ]
    return {
        "format_id": format_id,
        "seed": game.row.seed,
        "max_turns": max_turns,
        "teams": teams,
        "choices": choices,
    }


def _lines(value: object, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise InspectError(f"the re-simulation's {what} is not a list of lines")
    return tuple(value)


def _first_difference(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def _read_result(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InspectError("the re-simulation did not print JSON") from exc
    body = closed(raw, RESULT_KEYS, "the re-simulation")
    channels = closed(body["channels"], (set(SEATS), set()), "the re-simulation's channels")
    if not (isinstance(body["turns"], int) and not isinstance(body["turns"], bool)):
        raise InspectError("the re-simulation's turn count is not an integer")
    if body["winner"] not in (*SEATS, None):
        raise InspectError("the re-simulation's winner is not a seat or none")
    if not isinstance(body["decisions"], list):
        raise InspectError("the re-simulation's decisions are not a list")
    return {
        "winner": body["winner"],
        "turns": body["turns"],
        "omniscient_log": _lines(body["omniscient_log"], "omniscient log"),
        "channels": {seat: _lines(channels[seat], f"{seat} channel") for seat in SEATS},
        "decisions": [
            closed(item, DECISION_KEYS, "a re-simulated decision") for item in body["decisions"]
        ],
    }


def _check_decisions(game: GameReplay, found: list[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[dict[str, Any], ...]] = {}
    for view in game.views:
        own = sorted((d for d in found if d["player"] == view.seat), key=lambda d: d["request_seq"])
        theirs = [(d.request_seq, d.choice, d.refused) for d in view.decisions]
        ours = [(d["request_seq"], d["choice"], d.get("refused")) for d in own]
        if theirs != ours:
            raise InspectError(
                f"{view.seat}: the choices the engine took (refusals included) differ from the "
                "recorded decisions"
            )
        out[view.seat] = tuple(own)
    return out


def accept(game: GameReplay, text: str) -> Resimulated:
    """The result, or an :class:`InspectError` on any difference from what was played."""
    result = _read_result(text)
    for view in game.views:
        replayed = result["channels"][view.seat]
        if replayed != view.lines:
            at = _first_difference(replayed, view.lines)
            raise InspectError(
                f"{view.seat}: the re-simulated channel differs from the recorded Episode's "
                f"(first difference at line {at}); nothing is shown"
            )
    if result["winner"] != game.row.winner or result["turns"] != game.row.turns:
        raise InspectError("the re-simulated winner or turn count differs from generation.json")
    winner, _ = log_result(list(result["omniscient_log"]))
    if winner != (game.row.winner or "tie"):
        raise InspectError("the re-simulated omniscient log ends in another result than the game's")
    return Resimulated(game, result["omniscient_log"], _check_decisions(game, result["decisions"]))


def resimulate(
    run_dir: Path, generation: int, index: int, *, runner: Runner
) -> tuple[Resimulated, str]:
    """Re-simulate game ``index`` of ``generation`` and accept it, or refuse. Reads the run only.

    Returns the accepted game and the run's format id (for the engine-data table).
    """
    config, games = read_games(run_dir, generation, [index])
    (game,) = games
    request = request_for(game, config.format_id, config.generation.max_turns)
    return accept(game, runner(request)), config.format_id
