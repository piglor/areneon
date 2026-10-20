"""The learner throughput tool (ADR-010 §4): the figures it reports and the device facts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from areneon_ml.learner.__main__ import EXIT_REFUSED, main
from areneon_ml.learner.bench import Recorded, benchmark, device_facts, time_passes
from areneon_ml.learner.trainer import RoleReport, TrainReport

from tests.ml.learner.support import make_stream, tiny_config
from tests.ml.model.support import FORMAT_DIR, game, sample


def recorded() -> Recorded:
    text, manifest = make_stream([(0, "p1")])
    return Recorded(text, manifest)


def test_it_times_the_repeats_and_reports_what_it_ran() -> None:
    ticks = iter([10.0, 14.0])
    found = benchmark(tiny_config(), game(), recorded(), 3, clock=lambda: next(ticks))
    assert found["seconds"] == 4.0
    assert found["repeats"] == 3
    assert found["decisions"] == 3 * (found["decisions"] // 3) > 0
    assert found["decisions_per_second"] == found["decisions"] / 4.0
    assert found["optimiser_steps_per_second"] == found["optimiser_steps"] / 4.0
    assert found["device"] == "cpu"
    assert found["peak_memory_bytes"] is None
    assert found["run_config_hash"] == tiny_config().hash
    assert (found["d_model"], found["n_layers"], found["minibatch_size"]) == (16, 1, 16)


def report(steps: int = 4) -> TrainReport:
    return TrainReport({"battle": RoleReport(steps, 2, 0.0, 0.0, 0.0, 0.0, 0.0)})


def test_it_warms_up_before_the_clock_starts() -> None:
    calls: list[str] = []

    def train() -> TrainReport:
        calls.append("train")
        return report()

    def clock() -> float:
        calls.append("clock")
        return float(len(calls))

    timing = time_passes(train, torch.device("cpu"), 2, clock)
    assert calls == ["train", "clock", "train", "train", "clock"]
    assert (timing.decisions, timing.optimiser_steps, timing.peak_memory_bytes) == (8, 4, None)


def test_a_non_positive_repeat_count_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        time_passes(report, torch.device("cpu"), 0)


def test_the_cuda_facts_name_the_device_its_capability_and_the_kernels_built_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "A GPU")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90", "sm_120"])
    facts = device_facts(torch.device("cuda"))
    assert facts["device_name"] == "A GPU"
    assert facts["capability"] == "sm_120"
    assert facts["arch_list"] == ["sm_90", "sm_120"]
    assert facts["torch"] == torch.__version__


def test_on_cuda_the_clock_stops_after_the_device_has_finished_and_memory_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: log.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: log.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 12345)

    def train() -> TrainReport:
        log.append("train")
        return report()

    def clock() -> float:
        log.append("clock")
        return float(len(log))

    timing = time_passes(train, torch.device("cuda"), 1, clock)
    assert timing.peak_memory_bytes == 12345
    assert log == ["train", "sync", "reset", "clock", "train", "sync", "clock"]


def test_the_cli_prints_one_canonical_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text, manifest = make_stream([(0, "p1")])
    (tmp_path / "stream.jsonl").write_text(text)
    (tmp_path / "training-set.json").write_text(json.dumps(manifest))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_config().raw))
    args = [
        "bench", "--config", str(config), "--engine-data", str(FORMAT_DIR / "engine-data.json"),
        "--stream-dir", str(tmp_path), "--repeats", "1",
    ]  # fmt: skip
    assert main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["device"] == "cpu"
    assert printed["decisions"] > 0
    assert sample("version-manifest/valid/golden.json")  # the corpus is reachable from here


def test_the_cli_refuses_what_it_cannot_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "bench", "--config", str(tmp_path / "missing.json"),
        "--engine-data", str(FORMAT_DIR / "engine-data.json"), "--stream-dir", str(tmp_path),
    ]  # fmt: skip
    assert main(args) == EXIT_REFUSED
    assert "missing.json" in capsys.readouterr().err
