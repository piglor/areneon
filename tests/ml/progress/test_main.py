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
from areneon_ml.progress import run as run_module
from areneon_ml.progress.looks import read_looks
from areneon_ml.progress.run import (
    AblationSpec,
    AnchorPair,
    NodeProgressPlayer,
    PreviousPair,
    ProgressRequest,
    RunPair,
)

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


@pytest.fixture(scope="module")
def other(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second committed run: another key, other initial weights, the same dataset."""
    run_dir = tmp_path_factory.mktemp("other") / "run"
    make_run(run_dir, generations=2, config=dataset_config("other-run", (21, 22)))
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
    # The fixture dataset is small: anchor matches of 4 games (the default is 200).
    return [
        "progress",
        "--run-dir",
        str(run),
        "--engine-data",
        str(DATA),
        "--anchor-games",
        "4",
        *extra,
    ]


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
    # the move-use diagnostic (#1164) rides in the summary, over the stored games
    used = printed["generations"][0]["move_use"]
    assert used["label"] == "descriptive move-use diagnostic: not a strength measure"
    assert used["stored_games"] == len(used["matches"]) > 0
    (kwargs,) = built
    assert kwargs["node"] == "/n/node"
    assert kwargs["worker_python"] == "/w/py"
    assert kwargs["run_dir"] == run.resolve()
    assert kwargs["config_path"] == run.resolve() / "run-config.json"
    assert kwargs["engine_data"] == DATA.resolve()
    assert kwargs["encoder_version"].startswith("enc1:")
    assert (tmp_path / "p" / "g000001" / "progress.json").is_file()


def test_all_evaluates_exactly_the_multiples_of_five_present_and_skips_committed_ones(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    common = ["--games", "2", "--record-games", "0", "--progress-dir", str(tmp_path / "p")]
    assert main(args(run, "--generation", "0", *common)) == 0
    capsys.readouterr()
    assert main(args(run, "--all", *common)) == 0
    printed = json.loads(capsys.readouterr().out)
    # The run holds generations 0 to 2: only generation 0 is a multiple of 5, and it is committed.
    assert [(g["generation"], g["skipped"]) for g in printed["generations"]] == [(0, True)]
    assert not (tmp_path / "p" / "g000001").exists()


def test_the_schedule_is_every_fifth_committed_generation_from_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for newest, expected in ((-1, []), (0, [0]), (4, [0]), (5, [0, 5]), (12, [0, 5, 10])):
        monkeypatch.setattr(
            run_module, "committed_generations", lambda _r, n=newest: list(range(n + 1))
        )
        assert run_module.scheduled_generations(tmp_path) == expected
    assert run_module.SCHEDULE_EVERY == 5


def test_the_previous_matches_flag_restores_them_and_the_default_leaves_them_out(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    common = ["--games", "2", "--record-games", "0"]
    assert main(args(run, "--generation", "1", *common, "--progress-dir", str(tmp_path / "a"))) == 0
    default = json.loads(capsys.readouterr().out)["generations"][0]
    assert not [m for m in default["matches"] if m["match_id"].startswith("previous")]
    flag = [*common, "--previous-matches", "--progress-dir", str(tmp_path / "b")]
    assert main(args(run, "--generation", "1", *flag)) == 0
    asked = json.loads(capsys.readouterr().out)["generations"][0]
    assert asked["previous"] == "played"
    assert len([m for m in asked["matches"] if m["match_id"].startswith("previous")]) == 2


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
    assert len(list((tmp_path / "p" / "g000001" / "games").glob("*.json"))) == 8  # 4 matches x 2


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

    with_anchors = dataclasses.replace(
        request,
        anchors=(
            AnchorPair(0, "a" * 64, "d" * 64, "policy:0s", "policy:0a"),
            AnchorPair(10, "e" * 64, "f" * 64, "policy:1s", "policy:1a"),
        ),
        anchor_games=200,
    )
    player.progress(with_anchors)
    argv = json.loads(out.read_text())
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--anchor"] == [
        f"0,{'a' * 64},{'d' * 64},policy:0s,policy:0a",
        f"10,{'e' * 64},{'f' * 64},policy:1s,policy:1a",
    ]
    assert argv[argv.index("--anchor-games") + 1] == "200"
    with_ablation = dataclasses.replace(
        request, ablation=AblationSpec(0, "a" * 64, "policy:ab", "policy:rd")
    )
    player.progress(with_ablation)
    argv = json.loads(out.read_text())
    assert argv[argv.index("--ablation-preview") + 1] == f"0,{'a' * 64},policy:ab"
    assert argv[argv.index("--ablation-random-policy-id") + 1] == "policy:rd"
    assert "--anchor" not in argv


def test_anchors_and_anchor_games_reach_the_evaluation(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    code = main(
        args(
            run,
            *("--generation", "1", "--games", "2", "--record-games", "1"),
            *("--anchors", "0,7", "--anchor-games", "4", "--progress-dir", str(tmp_path / "p")),
        )
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    (done,) = printed["generations"]
    assert done["anchors"].startswith("played: g0; skipped")  # 7 is not earlier
    ids = [m["match_id"] for m in done["matches"]]
    assert "anchor-g0-sample" in ids
    assert "anchor-g7-sample" not in ids


def test_the_head_to_head_flags_reach_the_evaluation_and_the_look_log(
    run: Path,
    other: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_node(monkeypatch, run)
    code = main(
        args(
            run,
            *("--generation", "1", "--games", "2", "--record-games", "1"),
            *("--versus-run-dir", str(other), "--versus-generation", "1", "--run-games", "4"),
            *("--progress-dir", str(tmp_path / "p")),
        )
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    (done,) = printed["generations"]
    assert done["head_to_head"]["label"] == "cross-run comparison"
    assert done["head_to_head"]["lineage"]["label"] == "independent lineage"
    assert {m["match_id"]: m["games"] for m in done["matches"]}["run-sample"] == 4
    assert printed["looks"] == 1


def test_the_head_to_head_flags_go_together_and_never_with_all(
    run: Path, other: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    assert main(args(run, "--generation", "1", "--versus-run-dir", str(other))) == EXIT_REFUSED
    assert "go together" in capsys.readouterr().err
    both = ("--versus-run-dir", str(other), "--versus-generation", "1")
    assert main(args(run, "--all", *both)) == EXIT_REFUSED
    assert "needs --generation" in capsys.readouterr().err


def test_the_node_player_passes_the_other_run_as_flags(tmp_path: Path) -> None:
    scripts = tmp_path / "packages" / "team-dataset" / "scripts"
    scripts.mkdir(parents=True)
    out = tmp_path / "args.json"
    (scripts / "progress.ts").write_text(
        f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]));"
        "print(json.dumps({'wall_ms': {}}))"
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
    pair = RunPair(
        "other",
        "c" * 64,
        4,
        "a" * 64,
        "b" * 64,
        "policy:s",
        "policy:a",
        tmp_path / "other",
        tmp_path / "other" / "team-set.json",
        "enc1:y",
        200,
    )
    request = ProgressRequest(
        0, "p", "b", "s", "a", "m", tmp_path / "ts.json", 2, 0, tmp_path / "w", versus=pair
    )
    player.progress(request)
    argv = json.loads(out.read_text())
    flags = dict(zip(argv[::2], argv[1::2], strict=True))
    assert flags["--versus"] == f"other,{'c' * 64},4,{'a' * 64},{'b' * 64},policy:s,policy:a"
    assert flags["--versus-dir"] == str(tmp_path / "other")
    assert flags["--versus-encoder-version"] == "enc1:y"
    assert flags["--versus-team-set"] == str(tmp_path / "other" / "team-set.json")
    assert flags["--run-games"] == "200"
    player.progress(dataclasses.replace(request, versus=None))
    assert "--versus" not in json.loads(out.read_text())


def test_the_preview_ablation_flag_reaches_the_evaluation_and_refuses_all(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    code = main(
        args(
            run,
            *("--generation", "1", "--games", "2", "--record-games", "0", "--preview-ablation"),
            *("--progress-dir", str(tmp_path / "e1")),
        )
    )
    assert code == 0
    (done,) = json.loads(capsys.readouterr().out)["generations"]
    assert sorted(done["ablation"]["baselines"]) == ["actor-rule", "random-legal"]
    assert all(m["match_id"].startswith("preview-") for m in done["matches"])
    assert main(args(run, "--all", "--preview-ablation", "--games", "2")) == EXIT_REFUSED
    assert "needs --generation" in capsys.readouterr().err


def test_a_malformed_anchor_list_is_a_usage_error(run: Path) -> None:
    with pytest.raises(SystemExit):
        main(args(run, "--generation", "1", "--anchors", "0,x"))


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


def tree_of(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_a_progress_command_logs_one_look_beside_the_root_and_writes_nothing_in_it_or_the_run(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    before_run = tree_of(copy)
    common = ["--games", "2", "--record-games", "0"]
    assert main(args(copy, "--generation", "1", *common)) == 0
    assert json.loads(capsys.readouterr().out)["looks"] == 1
    root = tmp_path / "copy.progress"
    log = tmp_path / "copy.progress.looks.jsonl"
    assert log.is_file()
    assert not any("looks" in p.name for p in root.rglob("*"))
    assert tree_of(copy) == before_run
    root_files = {k: v for k, v in tree_of(root).items() if k != ".lock"}
    # A second look (a skipped, committed generation is still a result shown) is a second line.
    assert main(args(copy, "--generation", "1", *common)) == 0
    assert json.loads(capsys.readouterr().out)["looks"] == 2
    assert {k: v for k, v in tree_of(root).items() if k != ".lock"} == root_files
    assert tree_of(copy) == before_run
    assert read_looks(log).count == 2
    entry = json.loads(log.read_text().splitlines()[0])
    assert (entry["command"], entry["run_id"], entry["generations"]) == (
        "progress",
        json.loads((copy / "run-config.json").read_text())["seeds"]["run_key"],
        [1],
    )


def test_a_tampered_look_log_stops_the_command_with_exit_3(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_node(monkeypatch, run)
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    (tmp_path / "copy.progress.looks.jsonl").write_bytes(b"not a look\n")
    assert (
        main(args(copy, "--generation", "0", "--games", "2", "--record-games", "0")) == EXIT_REFUSED
    )


def test_the_damage_aware_flags_pass_a_budget_and_refuse_a_zero_or_a_lone_budget(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_node(monkeypatch, run)
    base = ("--generation", "1", "--games", "2", "--record-games", "1")
    code = main(args(run, *base, "--damage-aware", "--progress-dir", str(tmp_path / "a")))
    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert printed["generations"][0]["damage_aware"] == "played"
    for extra in (
        ("--damage-aware", "--damage-aware-budget-minutes", "0"),
        ("--damage-aware-budget-minutes", "5"),
    ):
        assert main(args(run, *base, *extra, "--progress-dir", str(tmp_path / "b"))) == EXIT_REFUSED
    capsys.readouterr()


def test_the_eval_dataset_option_reaches_the_record_and_is_logged_as_a_look(
    run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-020 E-17.1: the command passes the option on, the record names it, the look is logged."""
    fake_node(monkeypatch, run)
    other = tmp_path / "other-dataset"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps({"splits": [{"split": "validation", "teams": 6}]})
    )
    option = ["--eval-dataset-dir", str(other), "--eval-dataset-hash", "7" * 64]
    common = ["--generation", "1", "--games", "2", "--record-games", "0"]
    root = tmp_path / "p"
    assert main(args(run, *common, *option, "--progress-dir", str(root))) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["looks"] == 1
    record = json.loads((root / "g000001" / "progress.json").read_text())
    assert record["teams"]["evaluation_dataset_hash"] == "7" * 64
    assert read_looks(tmp_path / "p.looks.jsonl").count == 1
    # one flag without the other, the run's own dataset and a malformed hash are refused: no look
    for bad in (option[:2], option[2:], [*option[:3], "7" * 63], [*option[:3], "5" * 64]):
        assert main(args(run, *common, *bad, "--progress-dir", str(tmp_path / "q"))) == EXIT_REFUSED
    assert not (tmp_path / "q.looks.jsonl").exists()
