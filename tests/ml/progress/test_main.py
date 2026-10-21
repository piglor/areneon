"""``python -m areneon_ml.loop progress``: flags, exit codes and the printed line (ADR-017)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.loop import __main__ as module
from areneon_ml.loop.__main__ import EXIT_PLAY, EXIT_REFUSED, main
from areneon_ml.loop.play import PlayError
from areneon_ml.progress.run import NodeProgressPlayer, PreviousPair, ProgressRequest

from tests.ml.model.support import FORMAT_DIR
from tests.ml.progress.support import (
    FakeProgressPlayer,
    committed_manifest,
    dataset_config,
    make_run,
)

DATA = FORMAT_DIR / "engine-data.json"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("main") / "run"
    make_run(run_dir, generations=2)
    return run_dir


def fake_node(monkeypatch: pytest.MonkeyPatch, run: Path) -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []
    fake = FakeProgressPlayer(dataset_config(), committed_manifest(run, 0))

    def make(**kwargs: Any) -> FakeProgressPlayer:
        built.append(kwargs)
        return fake

    monkeypatch.setattr(module, "NodeProgressPlayer", make)
    return built


def args(run: Path, *extra: str) -> list[str]:
    return ["progress", "--run-dir", str(run), "--engine-data", str(DATA), *extra]


def test_one_generation_prints_one_canonical_line_and_builds_the_player_from_its_flags(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = fake_node(monkeypatch, run)
    code = main(
        args(
            run,
            *("--generation", "1", "--games", "2", "--record-games", "1"),
            *(
                "--progress-dir",
                str(tmp_path / "p"),
                "--node",
                "/n/node",
                "--worker-python",
                "/w/py",
            ),
        )
    )
    out = capsys.readouterr().out
    assert code == 0
    printed = json.loads(out)
    assert out == jcs(printed).decode() + "\n"
    assert printed["progress_dir"] == str((tmp_path / "p").resolve())
    assert [g["generation"] for g in printed["generations"]] == [1]
    (kwargs,) = built
    assert kwargs["node"] == "/n/node"
    assert kwargs["worker_python"] == "/w/py"
    assert kwargs["run_dir"] == run.resolve()
    assert kwargs["config_path"] == run.resolve() / "run-config.json"
    assert kwargs["engine_data"] == DATA.resolve()
    assert kwargs["encoder_version"].startswith("enc1:")
    assert (tmp_path / "p" / "g000001" / "progress.json").is_file()


def test_all_evaluates_every_committed_generation_and_skips_the_ones_already_done(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    common = ["--games", "2", "--record-games", "0", "--progress-dir", str(tmp_path / "p")]
    assert main(args(run, "--generation", "1", *common)) == 0
    capsys.readouterr()
    assert main(args(run, "--all", *common)) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [(g["generation"], g["skipped"]) for g in printed["generations"]] == [
        (0, False),
        (1, True),
        (2, False),
    ]


def test_the_default_root_is_the_sibling_of_the_run(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    assert main(args(copy, "--generation", "0", "--games", "2", "--record-games", "0")) == 0
    assert json.loads(capsys.readouterr().out)["progress_dir"] == str(
        (tmp_path / "copy.progress").resolve()
    )


def test_a_refusal_exits_3_and_a_failed_simulator_side_exits_4(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    code = main(args(run, "--generation", "9", "--progress-dir", str(tmp_path / "p")))
    assert code == EXIT_REFUSED
    assert "is not committed" in capsys.readouterr().err

    class Dies(FakeProgressPlayer):
        def progress(self, request: ProgressRequest) -> dict[str, Any]:
            raise PlayError("progress.ts failed (exit 1): boom")

    monkeypatch.setattr(module, "NodeProgressPlayer", lambda **_: Dies(dataset_config(), {}))
    code = main(
        args(
            run,
            "--generation",
            "1",
            "--games",
            "2",
            "--record-games",
            "1",
            "--progress-dir",
            str(tmp_path / "q"),
        )
    )
    assert code == EXIT_PLAY
    assert "boom" in capsys.readouterr().err
    assert not (tmp_path / "q" / ".g000001.partial").exists()


def test_the_default_games_stored_never_exceed_the_games_played(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    assert (
        main(args(run, "--generation", "1", "--games", "2", "--progress-dir", str(tmp_path / "p")))
        == 0
    )
    capsys.readouterr()
    assert len(list((tmp_path / "p" / "g000001" / "games").glob("*.json"))) == 12  # 6 matches x 2


def test_a_generation_or_all_is_required_and_the_two_exclude_each_other(run: Path) -> None:
    with pytest.raises(SystemExit):
        main(args(run))
    with pytest.raises(SystemExit):
        main(args(run, "--generation", "1", "--all"))


def test_the_node_player_passes_the_whole_request_and_the_run_as_flags(tmp_path: Path) -> None:
    scripts = tmp_path / "packages" / "team-dataset" / "scripts"
    scripts.mkdir(parents=True)
    out = tmp_path / "args.json"
    (scripts / "progress.ts").write_text(
        f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]));"
        "print(json.dumps({'wall_ms': {'m': 3}}))"
    )
    player = NodeProgressPlayer(
        repo=tmp_path,
        node=sys.executable,
        worker_python="/venv/python",
        run_dir=tmp_path / "run",
        config_path=tmp_path / "run" / "run-config.json",
        engine_data=tmp_path / "data.json",
        encoder_version="enc1:x",
    )
    request = ProgressRequest(
        generation=3,
        preview="p" * 64,
        battle="b" * 64,
        policy_id_sample="policy:s",
        policy_id_argmax="policy:a",
        manifest_sha256="m" * 64,
        team_set=tmp_path / "team-set.json",
        games=7,
        record_games=2,
        out_dir=tmp_path / "work",
    )
    assert player.progress(request) == {"wall_ms": {"m": 3}}
    argv = json.loads(out.read_text())
    flags = dict(zip(argv[::2], argv[1::2], strict=True))
    assert flags == {
        "--config": str(tmp_path / "run" / "run-config.json"),
        "--generation": "3",
        "--preview": "p" * 64,
        "--battle": "b" * 64,
        "--policy-id-sample": "policy:s",
        "--policy-id-argmax": "policy:a",
        "--generation-manifest-sha256": "m" * 64,
        "--team-set": str(tmp_path / "team-set.json"),
        "--run-dir": str(tmp_path / "run"),
        "--engine-data": str(tmp_path / "data.json"),
        "--encoder-version": "enc1:x",
        "--worker-python": "/venv/python",
        "--games": "7",
        "--record-games": "2",
        "--out": str(tmp_path / "work"),
    }
    with_previous = dataclasses.replace(
        request, previous=PreviousPair("q" * 64, "c" * 64, "policy:ps", "policy:pa")
    )
    player.progress(with_previous)
    argv = json.loads(out.read_text())
    assert dict(zip(argv[::2], argv[1::2], strict=True)) | {} == flags | {
        "--previous-preview": "q" * 64,
        "--previous-battle": "c" * 64,
        "--previous-policy-id-sample": "policy:ps",
        "--previous-policy-id-argmax": "policy:pa",
    }


@pytest.mark.parametrize("printed", ["not json", "[1, 2]"])
def test_a_script_that_prints_no_object_is_a_play_error(tmp_path: Path, printed: str) -> None:
    scripts = tmp_path / "packages" / "team-dataset" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "progress.ts").write_text(f"print({printed!r})")
    player = NodeProgressPlayer(
        repo=tmp_path,
        node=sys.executable,
        worker_python="x",
        run_dir=tmp_path,
        config_path=tmp_path / "c.json",
        engine_data=tmp_path / "d.json",
        encoder_version="e",
    )
    request = ProgressRequest(0, "p", "b", "s", "a", "m", tmp_path, 1, 0, tmp_path)
    with pytest.raises(PlayError):
        player.progress(request)
