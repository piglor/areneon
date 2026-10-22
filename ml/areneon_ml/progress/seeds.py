"""``progress-seed/1`` and the committed-plan check (ADR-017 section 2).

Everything a progress game is built from is a pure function of the run key and the game
index: SHA-256 over the JCS of ``{run_key, game, purpose: "progress-game"}``; the first 16
bytes are the battle seed (``sodium,<32 hex>``) and bytes 16 to 19, big endian, seed a
scripted baseline's stream. The TS side derives them and plays; this module derives them
again, independently, so the command refuses a plan whose seed equals a training seed and
the reader refuses a stored game whose seed is not the one its index gives.
"""

from __future__ import annotations

from pathlib import Path

from areneon_contracts.contracts import jcs, sha256_hex

from areneon_ml.loop import layout

SEED_DERIVATION = "progress-seed/1"
PAIRING = "progress-pairing/1"


def _digest(run_key: str, game: int) -> str:
    if game < 0:
        raise ValueError("a game counts from 0")
    return sha256_hex(jcs({"run_key": run_key, "game": game, "purpose": "progress-game"}))


def progress_game_seed(run_key: str, game: int) -> str:
    """The battle seed of progress game ``game`` (``progress-seed/1``)."""
    return "sodium," + _digest(run_key, game)[:32]


def progress_baseline_seed(run_key: str, game: int) -> int:
    """The seed of a scripted baseline's stream: digest bytes 16 to 19 as a big-endian uint32."""
    return int(_digest(run_key, game)[32:40], 16)


def committed_plan_seeds(run_dir: Path) -> set[str]:
    """Every battle seed of every committed ``plan.json`` of the run (generations 1 and up)."""
    seeds: set[str] = set()
    base = layout.generations_dir(run_dir)
    for path in sorted(base.glob("g[0-9][0-9][0-9][0-9][0-9][0-9]/plan.json")):
        plan = layout.read_json(path)
        seeds.update(entry["seed"] for entry in plan)
    return seeds


def check_not_training_seeds(run_dir: Path, run_key: str, games: int) -> None:
    """Refuse when a progress seed of indices 0..games-1 equals a seed a training game ran with.

    The index is the game (version 1) or the pair (version 2, one seed per pair).
    """
    training = committed_plan_seeds(run_dir)
    for game in range(games):
        if progress_game_seed(run_key, game) in training:
            raise ValueError(f"the progress seed of game {game} equals a seed of a committed plan")
