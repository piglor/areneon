"""The head to head between runs when either one is a population run (ADR-021, #1442)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.records import policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.run import ProgressError, VersusSpec
from areneon_ml.progress.run import progress_generation as evaluate

from tests.ml.loop.test_population import gen_path, make_population
from tests.ml.loop.test_population_copy import Dominant, config_with_copies
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)
from tests.ml.progress.test_population import with_dataset

ENCODER = "enc-other"


def pop_config(key: str, seeds: tuple[int, int] | None) -> RunConfig:
    raw = dict(config_with_copies().raw)
    raw["seeds"] = dict(raw["seeds"], run_key=key)
    if seeds is not None:
        raw["seeds"]["init_preview"], raw["seeds"]["init_battle"] = seeds
    with_dataset(raw)
    return read_run_config(raw)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, RunConfig]]:
    """Two populations (other keys and initial weights) and a single run, one dataset."""
    base = tmp_path_factory.mktemp("versus-population")
    found: dict[str, tuple[Path, RunConfig]] = {}
    for name, key, seeds in (("a", "pop-a", None), ("b", "pop-b", (31, 32))):
        config = pop_config(key, seeds)
        make_population(base / name, config, Dominant(config)).run(2)
        found[name] = (base / name, config)
    single = dataset_config("single-run", (21, 22))
    make_run(base / "s", generations=2, config=single)
    found["s"] = (base / "s", single)
    return found


def fake(run_dir: Path, config: RunConfig, member: str = "m0") -> FakeProgressPlayer:
    if config.population:
        path = gen_path(run_dir, 0) / "members" / member / "records" / "battle.json"
        return FakeProgressPlayer(config, layout.read_json(path)["manifest"])
    return FakeProgressPlayer(config, committed_manifest(run_dir, 0))


def go(  # noqa: PLR0913
    runs: dict[str, tuple[Path, RunConfig]],
    mine: str,
    other: str,
    root: Path,
    *,
    member: str | None = None,
    versus_member: str | None = None,
    generation: int = 2,
    versus_generation: int = 2,
) -> tuple[dict[str, Any], FakeProgressPlayer]:
    run_dir, config = runs[mine]
    player = fake(run_dir, config)
    versus = VersusSpec(runs[other][0], versus_generation, 4, ENCODER, None, versus_member)
    done = evaluate(run_dir, generation, root, player, 2, 1, versus=versus, member=member)
    return done, player


def members(run_dir: Path, generation: int) -> dict[str, Any]:
    found: dict[str, Any] = layout.read_json(gen_path(run_dir, generation) / "manifest.json")[
        "members"
    ]
    return found


def opponent_of(root: Path, generation: int = 2) -> dict[str, Any]:
    record = layout.read_json(run_module.progress_dir_of(root, generation) / "progress.json")
    found = next(m for m in record["matches"] if m["match_id"] == "run-sample")
    opponent: dict[str, Any] = found["opponent"]
    return opponent


def tree(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_the_ladder_top_of_each_side_is_the_default_and_the_record_names_the_member(
    runs: dict[str, tuple[Path, RunConfig]], tmp_path: Path
) -> None:
    done, player = go(runs, "a", "b", tmp_path / "p")
    other = player.requests[0].versus
    assert other is not None
    top = members(runs["b"][0], 2)["m0"]["checkpoints"]  # the Dominant stand-in: m0 wins all
    assert (other.preview, other.battle, other.member) == (top["preview"], top["battle"], "m0")
    assert done["member"] == "m0"
    assert opponent_of(tmp_path / "p")["member"] == "m0"
    assert opponent_of(tmp_path / "p")["policy_id"] == policy_id(
        other.preview, other.battle, "sample"
    )
    assert done["head_to_head"]["lineage"]["label"] == "independent lineage"


def test_members_on_both_sides_in_both_directions(
    runs: dict[str, tuple[Path, RunConfig]], tmp_path: Path
) -> None:
    for mine, other, member, versus_member in (("a", "b", "m1", "m0"), ("b", "a", "m0", "m1")):
        root = tmp_path / f"{mine}{other}"
        done, player = go(runs, mine, other, root, member=member, versus_member=versus_member)
        assert done["member"] == member
        record = layout.read_json(run_module.progress_dir_of(root, 2) / "progress.json")
        assert record["checkpoints"] == members(runs[mine][0], 2)[member]["checkpoints"]
        opponent = opponent_of(root)
        assert opponent["member"] == versus_member
        assert opponent["checkpoints"] == members(runs[other][0], 2)[versus_member]["checkpoints"]
        assert player.requests[0].versus is not None
        assert player.requests[0].versus.member == versus_member


def test_a_population_against_a_single_run_and_back(
    runs: dict[str, tuple[Path, RunConfig]], tmp_path: Path
) -> None:
    go(runs, "a", "s", tmp_path / "x", member="m1")
    assert "member" not in opponent_of(tmp_path / "x")  # the single run's entry is unchanged
    go(runs, "s", "b", tmp_path / "y", versus_member="m1")
    assert opponent_of(tmp_path / "y")["member"] == "m1"


def test_a_committed_directory_is_verified_and_a_rerun_is_byte_identical(
    runs: dict[str, tuple[Path, RunConfig]], tmp_path: Path
) -> None:
    for name in ("x", "y"):
        go(runs, "a", "b", tmp_path / name, member="m1", versus_member="m1")
    assert tree(tmp_path / "x") == tree(tmp_path / "y")
    again, _ = go(runs, "a", "b", tmp_path / "x", member="m1", versus_member="m1")
    assert again["skipped"] is True
    assert again["head_to_head"]["lineage"]["label"] == "independent lineage"
    with pytest.raises(ProgressError):  # another member than the committed record names
        go(runs, "a", "b", tmp_path / "x", member="m1", versus_member="m0")


def test_the_refusals(runs: dict[str, tuple[Path, RunConfig]], tmp_path: Path) -> None:
    with pytest.raises(ProgressError, match="has no member m7"):
        go(runs, "a", "b", tmp_path / "a", versus_member="m7")
    with pytest.raises(ProgressError, match="names a member of a population run"):
        go(runs, "a", "s", tmp_path / "b", versus_member="m0")
    with pytest.raises(ProgressError, match="has no member m7"):
        go(runs, "a", "b", tmp_path / "c", member="m7")
    with pytest.raises(ProgressError, match="is not committed"):
        go(runs, "a", "b", tmp_path / "d", versus_generation=9)
    run_dir, config = runs["a"]
    with pytest.raises(ProgressError, match="baselines and --versus-run-dir only"):
        evaluate(run_dir, 2, tmp_path / "e", fake(run_dir, config), 2, 1, anchors=(0,))
