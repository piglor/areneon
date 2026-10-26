"""The Node side of the loop: the arguments it is given, and how its failures surface (ADR-010)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from areneon_ml.loop.play import NodePlayer, PlayError, PlayRequest

# "node" here is Python and the "scripts" are small Python programs, so the arguments the
# player builds can be seen without a Node install; the real scripts run in the end-to-end test.
SCRIPTS = "packages/showdown-bridge/scripts"


def player(tmp_path: Path, manifest: str, play: str, node: str = sys.executable) -> NodePlayer:
    scripts = tmp_path / SCRIPTS
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "format-manifest.ts").write_text(manifest)
    (scripts / "play-generation.ts").write_text(play)
    return NodePlayer(
        repo=tmp_path,
        node=node,
        worker_python="/venv/python",
        run_dir=tmp_path / "run",
        config_path=tmp_path / "run" / "run-config.json",
        engine_data=tmp_path / "data.json",
        encoder_version="enc1:x",
    )


def test_the_manifest_is_the_scripts_json_line(tmp_path: Path) -> None:
    script = "import sys, json; print(json.dumps({'argv': sys.argv[1:]}))"
    found = player(tmp_path, script, "").manifest("gen9fmt")
    assert found == {"argv": ["--format", "gen9fmt"]}


def test_play_passes_the_whole_request_and_the_run_as_flags(tmp_path: Path) -> None:
    out = tmp_path / "args.json"
    script = f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]))"
    request = PlayRequest(3, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    player(tmp_path, "", script).play(request)
    argv = json.loads(out.read_text())
    flags = dict(zip(argv[::2], argv[1::2], strict=True))
    assert flags == {
        "--config": str(tmp_path / "run" / "run-config.json"),
        "--generation": "3",
        "--policy-id": "policy:abc",
        "--preview": "p" * 64,
        "--battle": "b" * 64,
        "--run-dir": str(tmp_path / "run"),
        "--engine-data": str(tmp_path / "data.json"),
        "--encoder-version": "enc1:x",
        "--worker-python": "/venv/python",
        "--out": str(tmp_path / "work"),
    }


def test_play_passes_worker_and_process_counts_only_when_not_one(tmp_path: Path) -> None:
    out = tmp_path / "args.json"
    script = f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]))"
    request = PlayRequest(3, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    chosen = player(tmp_path, "", script)
    NodePlayer(**{**chosen.__dict__, "policy_workers": 3, "game_processes": 2}).play(request)
    argv = json.loads(out.read_text())
    assert argv[-4:] == ["--policy-workers", "3", "--game-processes", "2"]
    chosen.play(request)
    assert "--policy-workers" not in json.loads(out.read_text())
    assert "--game-processes" not in json.loads(out.read_text())


def test_play_runs_the_script_it_is_given(tmp_path: Path) -> None:
    out = tmp_path / "which.txt"
    other = tmp_path / "packages/team-dataset/scripts"
    other.mkdir(parents=True)
    (other / "play-generation.ts").write_text(f"open({str(out)!r}, 'w').write('dataset')")
    chosen = player(tmp_path, "", "raise SystemExit(1)")
    chosen = NodePlayer(
        **{
            **chosen.__dict__,
            "play_script": "packages/team-dataset/scripts/play-generation.ts",
        }
    )
    chosen.play(PlayRequest(1, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work"))
    assert out.read_text() == "dataset"


def test_a_failing_script_surfaces_the_end_of_its_stderr(tmp_path: Path) -> None:
    script = "import sys; sys.stderr.write('x' * 5000 + 'the cause'); sys.exit(1)"
    request = PlayRequest(1, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    with pytest.raises(
        PlayError, match=r"play-generation.ts failed \(exit 1\): x+the cause"
    ) as info:
        player(tmp_path, "", script).play(request)
    assert len(str(info.value)) < 2200


def test_a_node_that_cannot_start_is_a_play_error(tmp_path: Path) -> None:
    with pytest.raises(PlayError, match="cannot start"):
        player(tmp_path, "", "", node=str(tmp_path / "missing")).manifest("f")


def test_a_manifest_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlayError, match="did not print an object"):
        player(tmp_path, "print('[1]')", "").manifest("f")


def fake_node(tmp_path: Path, stdout: str = "") -> tuple[str, Path]:
    """An executable "node" that records its whole command line and prints ``stdout``."""
    record = tmp_path / "node-argv.txt"
    exe = tmp_path / "fake-node"
    exe.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {record}\ncat <<'EOF'\n{stdout}\nEOF\n")
    exe.chmod(0o755)
    return str(exe), record


def test_the_node_heap_cap_goes_before_the_play_script_only(tmp_path: Path) -> None:
    node, record = fake_node(tmp_path, "{}")
    request = PlayRequest(3, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    base = player(tmp_path, "", "", node=node)
    NodePlayer(**{**base.__dict__, "node_heap_mb": 512}).play(request)
    argv = record.read_text().splitlines()
    assert argv[0] == "--max-old-space-size=512"
    assert argv[1].endswith("play-generation.ts")
    base.play(request)
    assert not any(a.startswith("--max-old") for a in record.read_text().splitlines())
    # The version manifest is not a game process and gets no cap.
    NodePlayer(**{**base.__dict__, "node_heap_mb": 512}).manifest("gen9fmt")
    assert not any(a.startswith("--max-old") for a in record.read_text().splitlines())


def test_the_resident_sets_the_script_reports_are_kept_and_junk_is_ignored(tmp_path: Path) -> None:
    lines = "\n".join(
        [
            '{"generation": 3}',
            "not json",
            '{"resource_usage": {"game_process_peak_rss_bytes": 11,'
            ' "policy_worker_peak_rss_bytes": true, "other": 5}}',
            '{"resource_usage": 4}',
            "[1]",
        ]
    )
    node, _ = fake_node(tmp_path, lines)
    request = PlayRequest(3, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    chosen = player(tmp_path, "", "", node=node)
    chosen.play(request)
    assert chosen.resource_usage == {"game_process": 11}
    # A later call starts afresh.
    quiet, _ = fake_node(tmp_path, "{}")
    again = NodePlayer(
        **{**chosen.__dict__, "node": quiet, "resource_usage": chosen.resource_usage}
    )
    again.play(request)
    assert again.resource_usage == {}


def test_a_generation_past_its_time_limit_is_killed_with_its_children(tmp_path: Path) -> None:
    pids = tmp_path / "pids.txt"
    script = (
        "import subprocess, sys, time, os\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pids)!r}, 'w').write(str(child.pid))\n"
        "sys.stderr.write('still playing')\n"
        "sys.stderr.flush()\n"
        "time.sleep(60)\n"
    )
    chosen = player(tmp_path, "", script)
    timed = NodePlayer(**{**chosen.__dict__, "generation_timeout_s": 1.0})
    request = PlayRequest(1, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    start = time.monotonic()
    with pytest.raises(
        PlayError, match=r"exceeded the generation time limit of 1 s.*still playing"
    ):
        timed.play(request)
    assert time.monotonic() - start < 20
    grandchild = int(pids.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the grandchild outlived the limit")


def test_a_game_that_ends_inside_its_limit_is_not_touched(tmp_path: Path) -> None:
    node, _ = fake_node(tmp_path, "{}")
    chosen = player(tmp_path, "", "", node=node)
    NodePlayer(**{**chosen.__dict__, "generation_timeout_s": 60.0}).play(
        PlayRequest(1, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work")
    )


def test_an_interrupt_ends_the_children_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = "import time; time.sleep(60)"
    chosen = player(tmp_path, "", script)

    def interrupted(self: subprocess.Popen[str], timeout: float | None = None) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        chosen.play(PlayRequest(1, "policy:abc", "p" * 64, "b" * 64, tmp_path / "work"))
