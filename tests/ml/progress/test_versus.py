"""The head-to-head match between two runs' frozen pairs (ADR-010 erratum A4 item 6, #1003)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.records import policy_id
from areneon_ml.progress import run as run_module
from areneon_ml.progress.run import (
    CROSS_RUN_LABEL,
    ProgressError,
    VersusSpec,
    summary_of,
)

from tests.ml.loop.support import reseal
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

ENCODER = "enc-other"


def evaluate(
    run: Path, generation: int, root: Path, fake: Any, versus: VersusSpec | None, **kwargs: Any
) -> dict[str, Any]:
    found: dict[str, Any] = run_module.progress_generation(
        run, generation, root, fake, 2, 1, (), 4, versus=versus, **kwargs
    )
    return found


def spec(run: Path, generation: int = 1, games: int = 4) -> VersusSpec:
    return VersusSpec(run, generation, games, ENCODER)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two committed runs: another key and other initial weights, one dataset."""
    base = tmp_path_factory.mktemp("two")
    make_run(base / "a", generations=2)
    make_run(base / "b", generations=2, config=dataset_config("other-run", (21, 22)))
    return base / "a", base / "b"


def fake_for(run: Path) -> FakeProgressPlayer:
    return FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))


def tree(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_the_other_runs_pair_plays_both_modes_from_both_seats(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    before = tree(b)
    fake = fake_for(a)
    done = evaluate(a, 1, tmp_path / "p", fake, spec(b, 1))
    (request,) = fake.requests
    other = request.versus
    assert other is not None
    manifest = layout.read_json(layout.generation_dir(b, 1) / layout.MANIFEST)["checkpoints"]
    assert (other.run_id, other.generation, other.games) == ("other-run", 1, 4)
    assert (other.preview, other.battle) == (manifest["preview"], manifest["battle"])
    assert other.encoder_version == ENCODER
    assert other.run_dir == b.resolve()
    assert other.policy_id_sample == policy_id(other.preview, other.battle, "sample")
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert record["plan"]["run_games_per_match"] == 4
    assert record["teams"]["pairs_used"] == 2
    found = {m["match_id"]: m for m in record["matches"] if m["opponent"]["kind"] == "run"}
    assert sorted(found) == ["run-argmax", "run-sample"]
    assert found["run-sample"]["opponent"] == {
        "kind": "run",
        "run_id": "other-run",
        "run_config_hash": other.run_config_hash,
        "generation": 1,
        "policy_id": other.policy_id_sample,
        "checkpoints": {"preview": other.preview, "battle": other.battle},
    }
    rows = [r for r in record["games"] if r["match_id"] == "run-argmax"]
    assert [(r["game"], r["pair"], r["candidate_seat"]) for r in rows] == [
        (0, 0, "p1"),
        (1, 0, "p2"),
        (2, 1, "p1"),
        (3, 1, "p2"),
    ]
    summary = {m["match_id"]: m for m in done["matches"]}
    assert summary["run-sample"]["comparison"] == CROSS_RUN_LABEL
    assert summary["run-sample"]["pairs"] == 2
    assert "diff_mean_ppm" in summary["run-sample"]
    assert "comparison" not in summary["actor-rule-sample"]
    assert done["head_to_head"] == {
        "label": CROSS_RUN_LABEL,
        "run_id": "other-run",
        "generation": 1,
        "games_per_match": 4,
        "lineage": {
            "label": "independent lineage",
            "reason": "an independent lineage (C6)",
        },
    }
    assert tree(b) == before  # the other run is only read
    assert not (tmp_path / "p" / ".lock").read_bytes()


def test_two_evaluations_give_the_same_bytes(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = runs
    for name in ("x", "y"):
        evaluate(a, 1, tmp_path / name, fake_for(a), spec(b, 0))
    assert tree(tmp_path / "x") == tree(tmp_path / "y")


def test_a_committed_directory_is_verified_and_reported(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    root = tmp_path / "p"
    evaluate(a, 1, root, fake_for(a), spec(b))
    again = evaluate(a, 1, root, fake_for(a), spec(b))
    assert again["skipped"] is True
    assert again["head_to_head"]["lineage"]["label"] == "independent lineage"
    # Without the flag the committed record is kept as it is (it may hold the matches).
    plain = evaluate(a, 1, root, fake_for(a), None)
    assert plain["skipped"] is True
    assert "run-sample" in {m["match_id"] for m in plain["matches"]}
    assert "head_to_head" in plain
    # Another game count is another record: refused, not rewritten.
    with pytest.raises(ProgressError, match="not the record asked for"):
        evaluate(a, 1, root, fake_for(a), spec(b, 1, 6))
    # A different generation of the other run is refused.
    with pytest.raises(ProgressError, match="not the other run's pair"):
        evaluate(a, 1, root, fake_for(a), spec(b, 0))


def test_a_record_without_cross_run_matches_is_kept_whatever_the_flag(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    root = tmp_path / "p"
    evaluate(a, 1, root, fake_for(a), None)
    again = evaluate(a, 1, root, fake_for(a), spec(b))
    assert again["skipped"] is True
    assert "head_to_head" not in again


def test_the_summary_is_derived_from_the_record_alone(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    evaluate(a, 1, tmp_path / "p", fake_for(a), spec(b))
    record = layout.read_json(tmp_path / "p" / "g000001" / "progress.json")
    assert {s["opponent"] for s in summary_of(record)} == {"baseline", "run"}


@pytest.mark.parametrize("games", [0, 3, -2])
def test_the_game_count_is_even_and_at_least_two(
    runs: tuple[Path, Path], tmp_path: Path, games: int
) -> None:
    a, b = runs
    fake = fake_for(a)
    with pytest.raises(ProgressError, match="--run-games"):
        evaluate(a, 1, tmp_path / "p", fake, spec(b, 1, games))
    assert fake.requests == []


def test_the_ablation_is_played_alone(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = runs
    with pytest.raises(ProgressError, match="alone"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(b), preview_ablation=True)


def test_more_games_than_pairs_is_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = runs
    fake = fake_for(a)
    with pytest.raises(ProgressError, match="disjoint"):
        evaluate(a, 1, tmp_path / "p", fake, spec(b, 1, 2_000_000))
    assert fake.requests == []


def test_the_same_run_is_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, _ = runs
    with pytest.raises(ProgressError, match="itself"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(a))
    copy = tmp_path / "copy"
    shutil.copytree(a, copy)
    with pytest.raises(ProgressError, match="same run key"):
        evaluate(a, 1, tmp_path / "q", fake_for(a), spec(copy))


def test_another_team_dataset_is_refused(tmp_path: Path) -> None:
    make_run(tmp_path / "a", generations=1)
    make_run(
        tmp_path / "c",
        generations=1,
        config=dataset_config("third-run", (31, 32), dataset_hash="6" * 64),
    )
    fake = fake_for(tmp_path / "a")
    with pytest.raises(ProgressError, match="different team datasets"):
        evaluate(tmp_path / "a", 0, tmp_path / "p", fake, spec(tmp_path / "c", 0))
    assert fake.requests == []


def test_a_run_without_a_team_dataset_is_refused(tmp_path: Path) -> None:
    from tests.ml.learner.support import edited_config  # noqa: PLC0415

    def rekey(raw: dict[str, Any]) -> None:
        raw["seeds"]["run_key"] = "no-dataset-run"
        raw["seeds"]["init_preview"] = 51

    make_run(tmp_path / "a", generations=1)
    make_run(tmp_path / "n", generations=1, config=edited_config(rekey))
    with pytest.raises(ProgressError, match="names no team dataset"):
        evaluate(
            tmp_path / "a", 0, tmp_path / "p", fake_for(tmp_path / "a"), spec(tmp_path / "n", 0)
        )


def test_another_format_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_run(tmp_path / "a", generations=1)
    make_run(tmp_path / "f", generations=1, config=dataset_config("fmt-run", (41, 42)))
    original = run_module._read_config

    def read(run_dir: Path) -> RunConfig:
        found = original(run_dir)
        if run_dir.name == "f":
            return read_run_config({**found.raw, "format_id": "gen9other"})
        return found

    monkeypatch.setattr(run_module, "_read_config", read)
    with pytest.raises(ProgressError, match="different formats"):
        evaluate(
            tmp_path / "a", 0, tmp_path / "p", fake_for(tmp_path / "a"), spec(tmp_path / "f", 0)
        )


def test_a_generation_the_other_run_has_not_committed_is_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    with pytest.raises(ProgressError, match="not committed"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(b, 9))


def test_a_non_synthetic_record_of_the_other_run_is_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    copy = tmp_path / "b"
    shutil.copytree(b, copy)
    path = layout.generation_dir(copy, 2) / "records" / "battle.json"
    found = layout.read_json(path)
    found["data_class"] = "private"
    layout.write_json(path, found)
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="synthetic"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(copy, 2))


def test_two_runs_that_share_a_checkpoint_are_refused(tmp_path: Path) -> None:
    make_run(tmp_path / "a", generations=1)
    # The same initial weights under another key: generation 0 is the same checkpoint.
    make_run(tmp_path / "s", generations=1, config=dataset_config("twin-run"))
    with pytest.raises(ProgressError, match="share a checkpoint"):
        evaluate(
            tmp_path / "a", 0, tmp_path / "p", fake_for(tmp_path / "a"), spec(tmp_path / "s", 0)
        )


def test_the_other_runs_training_seeds_are_refused(runs: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = runs
    copy = tmp_path / "b"
    shutil.copytree(b, copy)
    config = dataset_config()
    from areneon_ml.progress.seeds import progress_game_seed  # noqa: PLC0415

    plan = layout.generation_dir(copy, 1) / "plan.json"
    layout.write_json(plan, [{"seed": progress_game_seed(config.run_key, 0)}])
    reseal(copy, 1)
    with pytest.raises(ProgressError, match="equals a seed"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(copy, 2 - 1))


def test_files_that_name_another_opponent_are_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs

    def change(out: Path, _request: Any) -> None:
        record = layout.read_json(out / "progress.json")
        for match in record["matches"]:
            if match["opponent"]["kind"] == "run":
                match["opponent"]["generation"] = 0
        layout.write_json(out / "progress.json", record)

    fake = FakeProgressPlayer(dataset_config(), committed_manifest(a, 0), change)
    with pytest.raises(ProgressError, match="not the other run's pair"):
        evaluate(a, 1, tmp_path / "p", fake, spec(b))
    assert not (tmp_path / "p" / "g000001").exists()


def test_cross_run_matches_nobody_asked_for_are_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs

    class Eager(FakeProgressPlayer):
        def progress(self, request: Any) -> dict[str, Any]:
            import dataclasses  # noqa: PLC0415

            other = run_module.RunPair(
                "other-run",
                "0" * 64,
                1,
                "1" * 64,
                "2" * 64,
                policy_id("1" * 64, "2" * 64, "sample"),
                policy_id("1" * 64, "2" * 64, "argmax"),
                b,
                a,
                ENCODER,
                4,
            )
            return super().progress(dataclasses.replace(request, versus=other))

    with pytest.raises(ProgressError, match="asks for no cross-run"):
        evaluate(a, 1, tmp_path / "p", Eager(dataset_config(), committed_manifest(a, 0)), None)


def test_a_half_set_of_cross_run_matches_is_refused(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs

    def change(out: Path, _request: Any) -> None:
        record = layout.read_json(out / "progress.json")
        record["matches"] = [m for m in record["matches"] if m["match_id"] != "run-sample"]
        record["games"] = [g for g in record["games"] if g["match_id"] != "run-sample"]
        layout.write_json(out / "progress.json", record)

    fake = FakeProgressPlayer(dataset_config(), committed_manifest(a, 0), change)
    with pytest.raises(ProgressError, match="run-sample and run-argmax"):
        evaluate(a, 1, tmp_path / "p", fake, spec(b))


def test_the_other_runs_team_set_must_be_the_one_its_manifest_names(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    copy = tmp_path / "b"
    shutil.copytree(b, copy)
    path = layout.generation_dir(copy, 2) / "team-set.json"
    found = layout.read_json(path)
    found["teams"] = [*found["teams"], "0" * 64]
    layout.write_json(path, found)
    reseal(copy, 2)
    with pytest.raises(ProgressError, match="other run's team-set"):
        evaluate(a, 1, tmp_path / "p", fake_for(a), spec(copy, 2))


def test_the_progress_root_is_never_inside_or_above_the_other_run(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    fake = fake_for(a)
    with pytest.raises(ProgressError, match="inside the run"):
        evaluate(a, 1, b / "progress", fake, spec(b))
    with pytest.raises(ProgressError, match="contain the run"):
        evaluate(a, 1, b.parent, fake, spec(b))
    assert fake.requests == []


def test_the_cross_run_matches_combine_with_the_damage_aware_baseline(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    a, b = runs
    done = evaluate(a, 1, tmp_path / "p", fake_for(a), spec(b), damage_aware_budget_minutes=30)
    ids = {m["match_id"] for m in done["matches"]}
    assert {"run-sample", "run-argmax", "damage-aware-sample", "damage-aware-argmax"} <= ids
    assert done["head_to_head"]["label"] == CROSS_RUN_LABEL
    assert done["damage_aware"] == "played"
