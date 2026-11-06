"""The progress evaluation of a population run: the ladder top or a named member (ADR-021 §10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import read_run_config
from areneon_ml.loop import __main__ as entry
from areneon_ml.loop import ladder_write, layout
from areneon_ml.progress.run import ProgressError, ladder_top, progress_dir_of, verify_progress_dir
from areneon_ml.progress.run import progress_generation as evaluate

from tests.ml.loop.test_population import gen_path, make_population
from tests.ml.loop.test_population_copy import Dominant, config_with_copies
from tests.ml.progress.support import DATASET, FakeProgressPlayer, make_run


def with_dataset(raw: dict[str, Any]) -> None:
    raw["team_dataset"] = {"path": str(DATASET), "dataset_hash": "5" * 64}


def copying_config() -> Any:
    base = config_with_copies()
    raw = dict(base.raw)
    with_dataset(raw)
    return read_run_config(raw)


@pytest.fixture(scope="module")
def population(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    run_dir = tmp_path_factory.mktemp("population-progress")
    config = copying_config()
    make_population(run_dir, config, Dominant(config)).run(2)
    return run_dir, config


def player(run_dir: Path, config: Any, generation: int, member: str) -> FakeProgressPlayer:
    path = gen_path(run_dir, generation) / "members" / member / "records" / "battle.json"
    return FakeProgressPlayer(config, layout.read_json(path)["manifest"])


def run(
    population: tuple[Path, Any], root: Path, generation: int, member: str | None = None, **kw: Any
) -> dict[str, Any]:
    run_dir, config = population
    fake = player(run_dir, config, generation, member if member in ("m0", "m1") else "m0")
    return evaluate(run_dir, generation, root, fake, 2, 1, member=member, **kw)


def members(run_dir: Path, generation: int) -> dict[str, Any]:
    found: dict[str, Any] = layout.read_json(gen_path(run_dir, generation) / "manifest.json")[
        "members"
    ]
    return found


def test_the_default_is_the_ladder_top_and_the_record_names_its_pair(
    population: tuple[Path, Any], tmp_path: Path
) -> None:
    run_dir, _ = population
    ladder = layout.read_json(gen_path(run_dir, 2) / "ladder.json")
    strongest = max(ladder["members"], key=lambda m: m["strength"]["strength"])["member"]
    assert strongest == "m0"  # the Dominant stand-in lets m0 win everything
    result = run(population, tmp_path / "p", 2)
    assert result["member"] == "m0"
    record = layout.read_json(progress_dir_of(tmp_path / "p", 2) / "progress.json")
    assert record["checkpoints"] == members(run_dir, 2)["m0"]["checkpoints"]
    assert record["run_id"] == population[1].run_key
    # A second call verifies the committed directory and names the same member.
    again = run(population, tmp_path / "p", 2)
    assert again["skipped"] is True
    assert again["member"] == "m0"
    verify_progress_dir(progress_dir_of(tmp_path / "p", 2), population[1], run_dir, 2)


def test_a_named_member_is_evaluated_and_a_root_keeps_one_member(
    population: tuple[Path, Any], tmp_path: Path
) -> None:
    run_dir, _ = population
    result = run(population, tmp_path / "p", 2, "m1")
    assert result["member"] == "m1"
    record = layout.read_json(progress_dir_of(tmp_path / "p", 2) / "progress.json")
    assert record["checkpoints"] == members(run_dir, 2)["m1"]["checkpoints"]
    with pytest.raises(ProgressError, match="already holds m1's pair"):
        run(population, tmp_path / "p", 2, "m0")


def test_generation_zero_needs_a_named_member(population: tuple[Path, Any], tmp_path: Path) -> None:
    run_dir, config = population
    fake = player(run_dir, config, 0, "m0")
    with pytest.raises(ProgressError, match="generation 0 has no ladder"):
        evaluate(run_dir, 0, tmp_path / "a", fake, 2, 1)
    assert run(population, tmp_path / "b", 0, "m0")["member"] == "m0"


def test_the_refusals(population: tuple[Path, Any], tmp_path: Path) -> None:
    with pytest.raises(ProgressError, match="has no member m7"):
        run(population, tmp_path / "a", 2, "m7")
    with pytest.raises(ProgressError, match="baselines only"):
        run(population, tmp_path / "b", 2, "m0", anchors=(0,))
    with pytest.raises(ProgressError, match="baselines only"):
        run(population, tmp_path / "c", 2, "m0", previous_matches=True)
    other = tmp_path / "single"
    config = make_run(other, 1)
    fake = FakeProgressPlayer(config, {})
    with pytest.raises(ProgressError, match="names a member of a population run"):
        evaluate(other, 1, tmp_path / "d", fake, 2, 1, member="m0")


def test_a_tie_goes_to_the_lower_member_index(tmp_path: Path) -> None:
    def member(name: str, strength: int) -> dict[str, Any]:
        return {"member": name, "strength": {"strength": strength, "lower": 1, "upper": 2}}

    ladder = {"members": [member("m2", 5), member("m10", 5), member("m1", 5), member("m3", 4)]}
    layout.write_json(tmp_path / "ladder.json", ladder)
    facts = {"ladder_hash": ladder_write.json_hash(ladder)}
    assert ladder_top(tmp_path, facts, 3) == "m1"
    with pytest.raises(ProgressError, match="differs from its manifest"):
        ladder_top(tmp_path, {"ladder_hash": "0" * 64}, 3)
    layout.write_json(tmp_path / "ladder.json", {"members": []})
    empty = {"ladder_hash": ladder_write.json_hash({"members": []})}
    with pytest.raises(ProgressError, match="lists no members"):
        ladder_top(tmp_path, empty, 3)


def test_the_progress_command_no_longer_refuses_a_population_run() -> None:
    assert "progress" not in entry._SINGLE_LINEAGE_ONLY
