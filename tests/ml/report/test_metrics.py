"""The dashboard's run-metrics input (ADR-010 erratum A1, item 3)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.inspector.readers import InspectError
from areneon_ml.loop import layout
from areneon_ml.loop.metrics import Counts, Peaks, build_metrics, metrics_path, write_metrics
from areneon_ml.report.model import NOT_RECORDED, ReportError, read_report_model
from areneon_ml.report.render import build_report

from tests.ml.report.support import fixture_copy, make_run


@pytest.fixture
def run(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    make_run(path, generations=2)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(run: Path, generation: int, change: Any) -> None:
    path = metrics_path(run, generation)
    record = layout.read_json(path)
    change(record)
    layout.write_json(path, record)


def test_the_figures_and_the_files_hash_are_on_the_page(run: Path) -> None:
    rewrite(
        run,
        1,
        lambda r: r.update(
            phase_ms={"play": 12_345, "write": 7, "load": 2_000, "train": 4_500, "seal": 90},
            counts={"games": 4, "decisions": 90, "optimiser_steps": 9},
            per_hour={"games": 1166, "decisions": 72_000, "optimiser_steps": 7200},
        ),
    )
    model = read_report_model(run, None)
    page = build_report(model)
    assert [g.metrics is not None for g in model.generations] == [True, True, True]
    for n in range(3):
        name = f"metrics/g{n:06d}.json"
        assert model.inputs[name] == sha(run / name)
        assert name in page
        assert model.inputs[name] in page
    assert "12.345 s" in page
    assert "72000" in page
    assert "none (CPU learner)" in page
    assert "cpu, 1 thread" in page


def test_a_generation_without_a_file_says_not_recorded(run: Path) -> None:
    metrics_path(run, 1).unlink()
    model = read_report_model(run, None)
    assert [g.metrics is None for g in model.generations] == [False, True, False]
    assert "metrics/g000001.json" not in model.inputs
    page = build_report(model)
    assert page.count(f">{NOT_RECORDED}<") >= 11
    assert "metrics/g000002.json" in page


def test_a_run_from_before_the_records_has_none(tmp_path: Path) -> None:
    model = read_report_model(fixture_copy(tmp_path), None)
    assert all(g.metrics is None for g in model.generations)
    assert not any(name.startswith("metrics/") for name in model.inputs)
    assert "Wall-clock time and throughput" in build_report(model)


def test_a_record_of_another_device_than_the_config_is_refused(run: Path) -> None:
    rewrite(run, 2, lambda r: r.update(device="cuda", peak_accelerator_memory_bytes=3 * 2**30))
    with pytest.raises(ReportError, match="device and threads"):
        read_report_model(run, None)


def test_the_metrics_are_not_part_of_any_manifest(run: Path) -> None:
    before = {n: sha(layout.generation_dir(run, n) / layout.MANIFEST) for n in range(3)}
    rewrite(run, 1, lambda r: r.update(threads=1))
    write_metrics(run, layout.read_json(metrics_path(run, 1)))
    assert before == {n: sha(layout.generation_dir(run, n) / layout.MANIFEST) for n in range(3)}
    model = read_report_model(run, None)  # no manifest check can fail on a metrics change
    assert model.generations[1].manifest_sha256 == before[1]


def test_interrupted_writes_are_ignored(run: Path) -> None:
    (run / "metrics" / ".g000001.json.partial").write_text("{")
    assert read_report_model(run, None).generations[1].metrics is not None


def test_a_file_for_a_generation_the_run_lacks_is_refused(run: Path) -> None:
    shutil.copyfile(metrics_path(run, 2), run / "metrics" / "g000007.json")
    with pytest.raises(ReportError, match="no such generation"):
        read_report_model(run, None)


def test_a_file_that_is_not_named_like_a_record_is_refused(run: Path) -> None:
    (run / "metrics" / "notes.txt").write_text("x")
    with pytest.raises(ReportError, match="not a metrics file"):
        read_report_model(run, None)


def test_a_symlinked_record_is_refused(run: Path) -> None:
    target = run / "elsewhere.json"
    shutil.copyfile(metrics_path(run, 1), target)
    metrics_path(run, 1).unlink()
    metrics_path(run, 1).symlink_to(target)
    with pytest.raises(ReportError, match="not a metrics file"):
        read_report_model(run, None)


def test_a_metrics_path_that_is_not_a_directory_is_refused(run: Path) -> None:
    shutil.rmtree(run / "metrics")
    (run / "metrics").write_text("x")
    with pytest.raises(ReportError, match="not a directory"):
        read_report_model(run, None)


def test_a_record_of_another_run_or_one_that_breaks_the_contract_is_refused(run: Path) -> None:
    rewrite(run, 1, lambda r: r.update(run_config_hash="0" * 64))
    with pytest.raises(ReportError, match="config's hash") as raised:
        read_report_model(run, None)
    assert isinstance(raised.value, InspectError)
    path = metrics_path(run, 1)
    layout.write_json(path, {"schema_version": 1})
    with pytest.raises(ReportError, match="run-metrics"):
        read_report_model(run, None)
    path.write_bytes(b"not json")
    with pytest.raises(ReportError, match="canonical"):
        read_report_model(run, None)


def test_build_metrics_default_record_is_readable_by_the_report(tmp_path: Path) -> None:
    other = tmp_path / "run"
    config = make_run(other, generations=1)
    record = build_metrics(config, 1, {"play": 1}, Counts(1, 2, 3), Peaks())
    write_metrics(other, record)
    assert read_report_model(other, None).generations[1].metrics == record


def test_the_optional_memory_figures_are_shown_when_recorded_and_absent_otherwise(
    run: Path,
) -> None:
    rewrite(
        run,
        1,
        lambda r: r.update(
            peak_rss_bytes={
                "driver": 2 * 2**30,
                "game_process": 600 * 2**20,
                "policy_worker": 430 * 2**20,
            }
        ),
    )
    page = build_report(read_report_model(run, None))
    assert "driver 2048 MiB, game process 600 MiB, worker 430 MiB" in page
    # The other generations recorded neither figure.
    assert "Peak reserved accelerator memory" in page
    assert page.count(f">{NOT_RECORDED}<") >= 4
