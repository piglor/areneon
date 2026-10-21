"""``progress-seed/1`` and the committed-plan check (ADR-017 section 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from areneon_ml.loop import layout
from areneon_ml.loop.league import league_game_seed
from areneon_ml.progress.seeds import (
    check_not_training_seeds,
    committed_plan_seeds,
    progress_baseline_seed,
    progress_game_seed,
)

from tests.ml.loop.support import reseal
from tests.ml.progress.support import make_run

# The ADR-017 test vector for run key public-main-001, computed with hashlib here and in the ADR.
VECTOR = [
    (0, "sodium,f1f6d174ab06d8da49d0d29c894c1efa", 2351535428),
    (1, "sodium,4fb0138fce45349660355601f54d606f", 1369276180),
    (2, "sodium,331b23b6b714a4feb87b0db05b6204fd", 2554743786),
]


def test_the_adr_vector_is_reproduced() -> None:
    for game, seed, baseline in VECTOR:
        assert progress_game_seed("public-main-001", game) == seed
        assert progress_baseline_seed("public-main-001", game) == baseline


def test_the_derivation_is_the_documented_byte_layout() -> None:
    """Re-derived without the module: SHA-256 of the JCS (sorted keys, no spaces) of the object."""
    body = json.dumps(
        {"game": 7, "purpose": "progress-game", "run_key": "r"},
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(body.encode()).digest()
    assert progress_game_seed("r", 7) == "sodium," + digest[:16].hex()
    assert progress_baseline_seed("r", 7) == int.from_bytes(digest[16:20], "big")


def test_a_progress_seed_is_never_a_league_seed() -> None:
    league = {league_game_seed("run", g, n) for g in range(1, 6) for n in range(200)}
    assert not league & {progress_game_seed("run", n) for n in range(200)}


def test_a_negative_game_is_refused() -> None:
    with pytest.raises(ValueError, match="from 0"):
        progress_game_seed("r", -1)


def test_the_check_reads_every_committed_plan_and_refuses_a_collision(tmp_path: Path) -> None:
    config = make_run(tmp_path / "run", generations=2)
    run_dir = tmp_path / "run"
    seeds = committed_plan_seeds(run_dir)
    assert seeds == {league_game_seed(config.run_key, g, n) for g in (1, 2) for n in range(2)}
    check_not_training_seeds(run_dir, config.run_key, 50)
    plan_file = layout.generation_dir(run_dir, 1) / "plan.json"
    plan = layout.read_json(plan_file)
    plan[0]["seed"] = progress_game_seed(config.run_key, 3)
    layout.write_json(plan_file, plan)
    reseal(run_dir, 1)
    check_not_training_seeds(run_dir, config.run_key, 3)  # game 3 is not in play yet
    with pytest.raises(ValueError, match="game 3"):
        check_not_training_seeds(run_dir, config.run_key, 4)
