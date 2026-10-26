"""``python -m areneon_ml.loop``: exit codes and the printed line (ADR-010 §1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.loop import __main__ as module
from areneon_ml.loop.__main__ import EXIT_PLAY, EXIT_REFUSED, main
from areneon_ml.loop.driver import Loop
from areneon_ml.loop.play import PlayError

from tests.ml.learner.support import tiny_config
from tests.ml.loop.support import FakePlayer
from tests.ml.model.support import FORMAT_DIR

DATA = FORMAT_DIR / "engine-data.json"


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_config().raw))
    return path


def fake_node(
    monkeypatch: pytest.MonkeyPatch, player: FakePlayer | None = None
) -> list[dict[str, Any]]:
    """Replace the Node player by the fake; the kwargs it was built with are recorded."""
    built: list[dict[str, Any]] = []

    def make(**kwargs: Any) -> FakePlayer:
        built.append(kwargs)
        return player or FakePlayer()

    monkeypatch.setattr(module, "NodePlayer", make)
    return built


def run(config_file: Path, run_dir: Path, generations: int, *extra: str) -> list[str]:
    return [
        "run", "--config", str(config_file), "--run-dir", str(run_dir),
        "--engine-data", str(DATA), "--generations", str(generations), *extra,
    ]  # fmt: skip


def test_run_prints_one_canonical_line_and_builds_the_node_player_from_its_flags(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    built = fake_node(monkeypatch)
    code = main(
        run(config_file, tmp_path / "r", 1, "--node", "/n/node", "--worker-python", "/w/py")
    )
    out = capsys.readouterr().out
    assert code == 0
    printed = json.loads(out)
    assert out == jcs(printed).decode() + "\n"
    assert printed["generation"] == 1
    assert printed["played"] == [1]
    assert printed["resumed_from"] is None
    assert printed["policy_id"].startswith("policy:")
    assert set(printed["checkpoints"]) == {"battle", "preview"}
    (kwargs,) = built
    assert kwargs["node"] == "/n/node"
    assert kwargs["worker_python"] == "/w/py"
    assert kwargs["run_dir"] == (tmp_path / "r").resolve()
    assert kwargs["config_path"] == (tmp_path / "r").resolve() / "run-config.json"
    assert kwargs["engine_data"] == DATA.resolve()
    assert kwargs["encoder_version"].startswith("enc1:")
    assert kwargs["play_script"] == "packages/showdown-bridge/scripts/play-generation.ts"
    assert (kwargs["policy_workers"], kwargs["game_processes"]) == (1, 1)


def test_worker_and_process_counts_are_flags_of_the_player_not_of_the_config(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = fake_node(monkeypatch)
    before = config_file.read_text()
    assert (
        main(run(config_file, tmp_path / "r", 1, "--policy-workers", "3", "--game-processes", "4"))
        == 0
    )
    assert (built[0]["policy_workers"], built[0]["game_processes"]) == (3, 4)
    assert config_file.read_text() == before
    # the run's own copy of the config, which is hashed, is the same bytes for any count
    other = fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "s", 1)) == 0
    assert (tmp_path / "r" / "run-config.json").read_bytes() == (
        tmp_path / "s" / "run-config.json"
    ).read_bytes()
    assert other[0]["policy_workers"] == 1


def test_the_node_heap_cap_is_a_flag_of_the_player_only(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "r", 1, "--node-heap-mb", "640")) == 0
    assert built[0]["node_heap_mb"] == 640
    other = fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "s", 1)) == 0
    assert other[0]["node_heap_mb"] is None


def test_the_generation_time_limit_is_a_flag_of_the_player_only(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "r", 1)) == 0
    assert built[0]["generation_timeout_s"] == 180 * 60.0
    other = fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "s", 1, "--generation-timeout-minutes", "5")) == 0
    assert other[0]["generation_timeout_s"] == 300.0
    assert (tmp_path / "r" / "run-config.json").read_bytes() == (
        tmp_path / "s" / "run-config.json"
    ).read_bytes()


def test_the_load_workers_are_a_flag_of_the_loop_only(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_node(monkeypatch)
    seen: list[tuple[int, int | None]] = []
    real = Loop

    class Spy(real):  # type: ignore[valid-type, misc]
        def run(self, generations: int) -> Any:
            seen.append((self.load_processes, self.load_memory_mb))
            return super().run(generations)

    monkeypatch.setattr("areneon_ml.loop.__main__.Loop", Spy)
    assert (
        main(
            run(config_file, tmp_path / "r", 1, "--load-processes", "3", "--load-memory-mb", "900")
        )
        == 0
    )
    assert main(run(config_file, tmp_path / "s", 1)) == 0
    assert seen == [(3, 900), (1, None)]
    assert (tmp_path / "r" / "run-config.json").read_bytes() == (
        tmp_path / "s" / "run-config.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "flag",
    [
        "--policy-workers",
        "--game-processes",
        "--node-heap-mb",
        "--generation-timeout-minutes",
        "--load-processes",
        "--load-memory-mb",
    ],
)
@pytest.mark.parametrize("value", ["0", "-2"])
def test_a_count_below_one_is_a_usage_error(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch, flag: str, value: str
) -> None:
    built = fake_node(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        main(run(config_file, tmp_path / "r", 1, flag, value))
    assert stop.value.code == 2
    assert built == []


def test_a_placeholder_run_key_is_refused_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = fake_node(monkeypatch)
    raw = tiny_config().raw
    placeholder = {**raw, "seeds": {**raw["seeds"], "run_key": "CHANGE-ME-PER-RUN"}}
    config = tmp_path / "placeholder.json"
    config.write_text(json.dumps(placeholder))
    assert main(run(config, tmp_path / "r", 1)) != 0
    assert "CHANGE-ME" in capsys.readouterr().err
    assert built == []


def test_a_config_naming_a_team_dataset_starts_the_script_that_can_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = fake_node(monkeypatch)
    named = {**tiny_config().raw, "team_dataset": {"path": "d", "dataset_hash": "a" * 64}}
    config = tmp_path / "named.json"
    config.write_text(json.dumps(named))
    main(
        run(config, tmp_path / "r", 1)
    )  # the fake player's records are the tiny config's: the run itself is not the point
    assert built[0]["play_script"] == "packages/team-dataset/scripts/play-generation.ts"


def test_a_second_call_resumes_and_says_so(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "r", 1)) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(run(config_file, tmp_path / "r", 1)) == 0
    again = json.loads(capsys.readouterr().out)
    assert (again["resumed_from"], again["played"]) == (1, [])
    assert again["learner_state_hash"] == first["learner_state_hash"]


def test_a_refused_config_or_directory_exits_3(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    fake_node(monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**tiny_config().raw, "notes": "x"}))
    assert main(run(bad, tmp_path / "r", 1)) == EXIT_REFUSED
    assert "training-run-config" in capsys.readouterr().err
    assert main(run(tmp_path / "missing.json", tmp_path / "r", 1)) == EXIT_REFUSED
    assert main(run(config_file, tmp_path / "r", 1)) == 0
    other = tmp_path / "other.json"
    raw = tiny_config().raw
    raw["ppo"]["epochs"] = 3
    other.write_text(json.dumps(raw))
    capsys.readouterr()
    assert main(run(other, tmp_path / "r", 1)) == EXIT_REFUSED
    assert "another TrainingRunConfig" in capsys.readouterr().err


def test_a_simulator_failure_exits_4(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    class Broken(FakePlayer):
        def play(self, request: Any) -> None:
            raise PlayError("play-generation.ts failed (exit 1): boom")

    fake_node(monkeypatch, Broken())
    assert main(run(config_file, tmp_path / "r", 1)) == EXIT_PLAY
    assert "boom" in capsys.readouterr().err


def test_provenance_exports_the_records_and_prints_the_registry_hash(
    tmp_path: Path, config_file: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    fake_node(monkeypatch)
    assert main(run(config_file, tmp_path / "r", 2)) == 0
    capsys.readouterr()
    assert (
        main(["provenance", "--run-dir", str(tmp_path / "r"), "--out", str(tmp_path / "out")]) == 0
    )
    assert len(json.loads(capsys.readouterr().out)["sources_hash"]) == 64
    assert len(list((tmp_path / "out" / "records").iterdir())) == 6
    assert (
        main(["provenance", "--run-dir", str(tmp_path / "none"), "--out", str(tmp_path / "o2")])
        == EXIT_REFUSED
    )
