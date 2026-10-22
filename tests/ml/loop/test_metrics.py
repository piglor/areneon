"""The run-metrics record (ADR-010 erratum A1, item 3): outside every hash, after the commit."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Loop
from areneon_ml.loop.metrics import (
    Counts,
    MetricsError,
    build_metrics,
    metrics_path,
    per_hour,
    read_run_metrics,
    write_metrics,
)

from tests.ml.learner.support import edited_config, tiny_config
from tests.ml.loop.support import Killed, killer, make_loop, manifests


def ticking(step: float = 1.5) -> Any:
    counter = itertools.count()
    return lambda: next(counter) * step


def timed_loop(run_dir: Path, on_stage: Any = None) -> Loop:
    loop = make_loop(run_dir, on_stage=on_stage)
    loop.clock = ticking()
    return loop


def test_every_committed_generation_of_a_clean_run_has_a_record(tmp_path: Path) -> None:
    timed_loop(tmp_path).run(2)
    config = tiny_config()
    assert sorted(p.name for p in (tmp_path / "metrics").iterdir()) == [
        "g000000.json",
        "g000001.json",
        "g000002.json",
    ]
    for g in range(3):
        raw = layout.read_json(metrics_path(tmp_path, g))
        record = read_run_metrics(raw, config, g)
        assert record["peak_accelerator_memory_bytes"] is None
        assert (record["device"], record["threads"]) == ("cpu", 1)
    first = layout.read_json(metrics_path(tmp_path, 1))
    report = layout.read_json(layout.generation_dir(tmp_path, 1) / "train-report.json")
    assert first["counts"] == {
        "games": 2,
        "decisions": sum(r["steps"] for r in report.values()),
        "optimiser_steps": sum(r["optimiser_steps"] for r in report.values()),
    }
    # The fake clock advances 1.5 s per reading, so a phase read twice lasts 1500 ms.
    assert first["phase_ms"] == {
        "play": 1500,
        "write": 3000,
        "load": 1500,
        "train": 1500,
        "seal": 1500,
    }
    assert first["per_hour"]["games"] == 2 * 3_600_000 // 1500
    zero = layout.read_json(metrics_path(tmp_path, 0))
    assert zero["phase_ms"]["play"] == zero["phase_ms"]["train"] == 0
    assert zero["counts"] == {"games": 0, "decisions": 0, "optimiser_steps": 0}
    assert zero["per_hour"] == {"games": 0, "decisions": 0, "optimiser_steps": 0}


def test_the_metrics_are_outside_every_hash(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    timed = tmp_path / "timed"
    make_loop(plain).run(2)
    timed_loop(timed).run(2)
    assert manifests(plain) == manifests(timed)
    for g in range(3):
        files = layout.read_json(layout.generation_dir(timed, g) / "manifest.json")["files"]
        assert not any("metrics" in name for name in files)
        assert "metrics" not in {p.name for p in layout.generation_dir(timed, g).iterdir()}
    # Two runs with other clocks differ only in the metrics.
    other = tmp_path / "other"
    loop = make_loop(other)
    loop.clock = ticking(0.25)
    loop.run(2)
    assert manifests(other) == manifests(timed)
    assert metrics_path(other, 1).read_bytes() != metrics_path(timed, 1).read_bytes()


def test_a_run_killed_after_the_commit_reports_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(Killed):
        timed_loop(tmp_path, killer(1, "committed")).run(2)
    assert metrics_path(tmp_path, 0).is_file()
    assert layout.generation_dir(tmp_path, 1).is_dir()
    assert not metrics_path(tmp_path, 1).exists()
    timed_loop(tmp_path).run(2)
    # The resume does not write the lost generation's timings; later generations have theirs.
    assert not metrics_path(tmp_path, 1).exists()
    assert metrics_path(tmp_path, 2).is_file()


def test_a_run_killed_before_the_commit_writes_no_record_for_it(tmp_path: Path) -> None:
    with pytest.raises(Killed):
        timed_loop(tmp_path, killer(1, "written")).run(2)
    assert not metrics_path(tmp_path, 1).exists()


def test_the_cuda_peak_is_read_around_the_train_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def reset(device: object) -> None:
        calls.append("reset")

    def peak(device: object) -> int:
        calls.append("peak")
        return 7

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    loops: list[Loop] = []

    def to_cuda(generation: int, stage: str) -> None:
        # After the learner exists and before it trains: only the driver's reads see "cuda".
        if (generation, stage) == (1, "played"):
            loops[0].device = torch.device("cuda")

    loops.append(make_loop(tmp_path, on_stage=to_cuda))
    loops[0].run(1)
    assert calls == ["reset", "peak"]


def test_build_metrics_on_a_cuda_config_keeps_the_peak() -> None:
    config = edited_config(lambda raw: raw["learner"].update(device="cuda", threads=2))
    record = build_metrics(config, 3, {"train": 2000}, Counts(0, 10, 4), 5_000)
    assert record["peak_accelerator_memory_bytes"] == 5_000
    assert (record["device"], record["threads"]) == ("cuda", 2)
    assert record["per_hour"]["decisions"] == 18_000
    assert read_run_metrics(record, config, 3) == record
    untrained = build_metrics(config, 0, {}, Counts(), None)
    assert untrained["peak_accelerator_memory_bytes"] == 0
    assert read_run_metrics(untrained, config, 0) == untrained


def test_per_hour_truncates_and_is_zero_without_time() -> None:
    assert per_hour(1, 3) == 1_200_000
    assert per_hour(7, 41_250) == 610
    assert per_hour(5, 0) == 0


def test_writes_are_atomic_and_leave_no_partial(tmp_path: Path) -> None:
    record = build_metrics(tiny_config(), 1, {}, Counts(), None)
    path = write_metrics(tmp_path, record)
    assert path == metrics_path(tmp_path, 1)
    assert sorted(p.name for p in path.parent.iterdir()) == ["g000001.json"]
    again = write_metrics(tmp_path, record)  # the directory exists now
    assert again.read_bytes() == path.read_bytes()


def _record() -> dict[str, Any]:
    return build_metrics(tiny_config(), 1, {"play": 1000, "train": 500}, Counts(2, 30, 6), None)


def _change(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(json.dumps(_record()))
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return record


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_change(("run_config_hash",), "0" * 64), "config's hash"),
        (_change(("generation",), 2), "another generation"),
        (_change(("per_hour", "games"), 1), "per_hour"),
        (_change(("device",), "cuda"), "device and threads"),
        (_change(("threads",), 2), "device and threads"),
        (_change(("peak_accelerator_memory_bytes",), 5), "null exactly on a CPU"),
        (_change(("phase_ms", "play"), 1.5), "run-metrics"),
        ({"schema_version": 1}, "run-metrics"),
    ],
)
def test_the_reader_refuses(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(MetricsError, match=message):
        read_run_metrics(raw, tiny_config(), 1)
