"""The operational memory guard and ``--max-concurrent-starts`` of the driver (#1394)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from areneon_ml.loop import memory_guard as guard
from areneon_ml.loop.play import (
    MemberPair,
    NodePlayer,
    PlayError,
    PlayRequest,
    PopulationPlayRequest,
)

GIB = 1 << 30


def meminfo(tmp_path: Path, available_gib: float, swap_gib: float = 0.0) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       99999999 kB\nMemAvailable:   {int(available_gib * GIB) // 1024} kB\n"
        f"SwapFree:       {int(swap_gib * GIB) // 1024} kB\n",
        encoding="ascii",
    )
    return path


def metrics(run: Path, name: str, body: object) -> None:
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    (run / "metrics" / name).write_text(json.dumps(body), encoding="utf-8")


def peaks(game: object, worker: object) -> dict[str, object]:
    return {"peak_rss_bytes": {"game_process": game, "policy_worker": worker}}


def test_available_is_memavailable_and_swap_is_not_counted(tmp_path: Path) -> None:
    assert guard.available_bytes(meminfo(tmp_path, 10, 6)) == 10 * GIB


def test_available_without_swap_or_without_the_file(tmp_path: Path) -> None:
    only = tmp_path / "m"
    only.write_text("MemAvailable: 1024 kB\nBogus line\nSwapFree: x kB\n")
    assert guard.available_bytes(only) == 1024 * 1024
    only.write_text("MemTotal: 5 kB\n")
    assert guard.available_bytes(only) is None
    assert guard.available_bytes(tmp_path / "missing") is None


def test_default_estimate_when_nothing_was_measured(tmp_path: Path) -> None:
    found = guard.estimate(tmp_path / "metrics", 2, 4)
    assert not found.learned
    assert found.per_process_bytes == (
        guard.DEFAULT_GAME_PROCESS_BYTES + 8 * guard.DEFAULT_POLICY_WORKER_BYTES
    )


def test_estimate_is_learned_from_the_newest_file_that_has_both_peaks(tmp_path: Path) -> None:
    metrics(
        tmp_path, "g000001.json", {"peak_rss_bytes": {"game_process": 100, "policy_worker": 10}}
    )
    metrics(tmp_path, "g000002.json", {"peak_rss_bytes": {"driver": 5}})  # no worker peaks
    metrics(tmp_path, "g000003.json", {"peak_rss_bytes": "junk"})
    metrics(tmp_path, "g000004.json", [])
    (tmp_path / "metrics" / "g000005.json").write_text("{not json")
    metrics(
        tmp_path, "g000006.json", {"peak_rss_bytes": {"game_process": True, "policy_worker": 1}}
    )
    found = guard.estimate(tmp_path / "metrics", 3, 2)
    assert found.learned
    assert found.per_process_bytes == 100 + 10 * 3 * 2
    metrics(tmp_path, "g000007.json", peaks(7, 1))
    assert guard.estimate(tmp_path / "metrics", 1, 1).per_process_bytes == 8


def test_largest_fit_leaves_the_driver_reserve() -> None:
    assert guard.largest_fit(guard.DRIVER_RESERVE_BYTES - 1, 100) == 0
    assert guard.largest_fit(guard.DRIVER_RESERVE_BYTES + 1000, 100) == 10


def test_check_passes_at_the_limit_and_refuses_above_with_the_largest_fit(tmp_path: Path) -> None:
    run = tmp_path / "run"
    per = 1 * GIB
    metrics(
        run,
        "g000001.json",
        {"peak_rss_bytes": {"game_process": per // 2, "policy_worker": per // 2}},
    )
    info = meminfo(tmp_path, 11, 6)  # 11 GiB - 4 reserve = 7 processes of 1 GiB; swap ignored

    def check(processes: int) -> None:
        guard.check(
            game_processes=processes,
            policy_workers=1,
            members=1,
            metrics_dir=run / "metrics",
            meminfo=info,
        )

    check(7)
    with pytest.raises(guard.MemoryGuardError) as err:
        check(9)
    text = str(err.value)
    assert "largest --game-processes that fits is 7" in text
    assert "measured in the last generation" in text
    assert "--no-memory-guard" in text


def test_check_names_the_default_and_zero_fit(tmp_path: Path) -> None:
    with pytest.raises(guard.MemoryGuardError, match=r"default.*fits is 0"):
        guard.check(
            game_processes=1,
            policy_workers=1,
            members=1,
            metrics_dir=tmp_path / "metrics",
            meminfo=meminfo(tmp_path, 1),
        )


def test_check_is_silent_without_meminfo(tmp_path: Path) -> None:
    guard.check(
        game_processes=999,
        policy_workers=1,
        members=1,
        metrics_dir=tmp_path,
        meminfo=tmp_path / "missing",
    )


def player(tmp_path: Path, script: str, **extra: object) -> NodePlayer:
    scripts = tmp_path / "packages/showdown-bridge/scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "play-generation.ts").write_text(script)
    return NodePlayer(
        repo=tmp_path,
        node=sys.executable,
        worker_python="/venv/python",
        run_dir=tmp_path / "run",
        config_path=tmp_path / "run" / "run-config.json",
        engine_data=tmp_path / "data.json",
        encoder_version="enc1:x",
        **extra,  # type: ignore[arg-type]
    )


def argv_script(out: Path) -> str:
    return f"import sys, json; open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]))"


def test_max_concurrent_starts_is_passed_only_when_given(tmp_path: Path) -> None:
    out = tmp_path / "argv.json"
    request = PlayRequest(1, "policy:x", "p" * 64, "b" * 64, tmp_path / "w")
    player(tmp_path, argv_script(out)).play(request)
    assert "--max-concurrent-starts" not in json.loads(out.read_text())
    player(tmp_path, argv_script(out), max_concurrent_starts=5).play(request)
    argv = json.loads(out.read_text())
    assert argv[argv.index("--max-concurrent-starts") + 1] == "5"


def test_population_play_passes_it_and_guards_with_the_member_count(tmp_path: Path) -> None:
    out = tmp_path / "argv.json"
    members = tuple(MemberPair(f"m{k}", f"policy:{k}", "p" * 64, "b" * 64) for k in range(4))
    request = PopulationPlayRequest(1, members, tmp_path / "w")
    info = meminfo(tmp_path, 9)
    # default estimate with 4 members, 1 worker: 0.6 + 4 x 0.43 = 2.32 GB; 5 GB / 2.32 = 2 fit
    ok = player(tmp_path, argv_script(out), game_processes=2, max_concurrent_starts=3, meminfo=info)
    ok.play_population(request)
    argv = json.loads(out.read_text())
    assert argv[argv.index("--max-concurrent-starts") + 1] == "3"
    out.unlink()
    with pytest.raises(PlayError, match="largest --game-processes that fits is 2"):
        player(tmp_path, argv_script(out), game_processes=9, meminfo=info).play_population(request)
    assert not out.exists()  # refused before anything started
    player(
        tmp_path, argv_script(out), game_processes=9, meminfo=info, memory_guard=False
    ).play_population(request)
    assert out.exists()


def test_single_lineage_play_is_refused_by_the_guard(tmp_path: Path) -> None:
    out = tmp_path / "argv.json"
    request = PlayRequest(1, "policy:x", "p" * 64, "b" * 64, tmp_path / "w")
    with pytest.raises(PlayError, match="memory guard"):
        player(tmp_path, argv_script(out), game_processes=4, meminfo=meminfo(tmp_path, 3)).play(
            request
        )
    assert not out.exists()


def test_the_smoke_numbers_s6_passes_and_s9_is_refused_at_idle(tmp_path: Path) -> None:
    """A 4-member population at idle, 19 GB available and 6 GB swap (S=9 was OOM-killed)."""
    info = meminfo(tmp_path, 19, 6)

    def check(processes: int) -> None:
        guard.check(
            game_processes=processes,
            policy_workers=1,
            members=4,
            metrics_dir=tmp_path / "metrics",
            meminfo=info,
        )

    check(6)
    with pytest.raises(guard.MemoryGuardError, match="fits is 6"):
        check(9)


def test_a_single_lineage_run_at_s6_passes_at_idle(tmp_path: Path) -> None:
    info = meminfo(tmp_path, 19, 6)
    for workers in (1, 2):
        guard.check(
            game_processes=6,
            policy_workers=workers,
            members=1,
            metrics_dir=tmp_path / "metrics",
            meminfo=info,
        )


def test_learned_peaks_count_only_the_members_a_process_plays(tmp_path: Path) -> None:
    metrics(tmp_path, "g000001.json", peaks(GIB, GIB))
    wide = guard.estimate(tmp_path / "metrics", 1, 4, 1)  # one process: all 4 members
    split = guard.estimate(tmp_path / "metrics", 1, 4, 9)  # min(4, 1 + ceil(4 / 9)) = 2
    assert wide.per_process_bytes == 5 * GIB
    assert split.per_process_bytes == 3 * GIB
    # without a measurement every member counts whatever S is
    assert guard.estimate(tmp_path / "none", 1, 4, 9) == guard.estimate(tmp_path / "none", 1, 4, 1)
