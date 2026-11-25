"""The teacher stream's verified marker through the loops (#1501, ADR-010 erratum A11.3).

A queue of experiments starts a new driver process for every generation, and each one used to
check and encode the whole teacher stream before it could train. Here a new loop object stands for
a new process: the stream is validated once for the run directory, every later start finds the
marker, and not one byte of any generation, checkpoint or learner state differs from the run that
validates at every start (the marker is only a cache of a verdict, never an input).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.training import check_stream_file
from areneon_ml.learner import imitation
from areneon_ml.learner.config import RunConfig
from areneon_ml.learner.data import _Collector
from areneon_ml.loop import layout
from areneon_ml.loop.teacher import import_teacher, teacher_dir
from areneon_ml.loop.warm_start import import_clone

from tests.ml.learner.support import make_stream
from tests.ml.loop.test_imitation import imitation_config
from tests.ml.loop.test_population import make_population, tree_bytes
from tests.ml.loop.test_population_imitation import config_for
from tests.ml.loop.test_warm_start import clone_ids, clone_output, warm_loop

GENERATIONS = 3
BATTLE = ("battle",)


@pytest.fixture(scope="module")
def teacher_out(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("teacher")
    text, manifest = make_stream(policy="damage-aware@v3")
    (out / "stream.jsonl").write_text(text)
    layout.write_json(out / "training-set.json", manifest)
    return out, sha256_hex(jcs(manifest))


@pytest.fixture(scope="module")
def clone_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return clone_output(tmp_path_factory.mktemp("clone"))


class Passes:
    """How often the teacher stream's line checks ran and how many examples were encoded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.checked = 0
        self.encoded = 0
        real_check, real_add = check_stream_file, _Collector.add

        def check(*args: Any, **kwargs: Any) -> Any:
            self.checked += 1
            return real_check(*args, **kwargs)

        def add(collector: Any, example: dict[str, Any]) -> None:
            self.encoded += 1
            real_add(collector, example)

        monkeypatch.setattr(imitation, "check_stream_file", check)
        monkeypatch.setattr(_Collector, "add", add)


def without_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The behaviour before the marker: no marker is read and none is written."""
    from areneon_ml.learner import teacher_marker  # noqa: PLC0415

    monkeypatch.setattr(teacher_marker, "read_marker", lambda *args, **kwargs: None)
    monkeypatch.setattr(teacher_marker, "write_marker", lambda *args, **kwargs: False)


def single(root: Path, clone_dir: Path, teacher: Path, config: RunConfig, per_process: int) -> Path:
    """A single-lineage run of ``GENERATIONS`` generations, ``per_process`` per driver."""
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher, run_dir)
    for upto in range(per_process, GENERATIONS + per_process, per_process):
        warm_loop(run_dir, config).run(min(upto, GENERATIONS))
    return run_dir


def population(root: Path, clone_dir: Path, teacher: Path, config: RunConfig, per: int) -> Path:
    run_dir = root / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher, run_dir)
    for upto in range(per, GENERATIONS + per, per):
        make_population(run_dir, config).run(min(upto, GENERATIONS))
    return run_dir


def markers(run_dir: Path, digest: str) -> list[Path]:
    return sorted(teacher_dir(run_dir, digest).glob("verified-*.json"))


def test_a_driver_started_for_every_generation_checks_the_teacher_stream_once(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = imitation_config(clone_ids(clone_dir), teacher_out[1])
    passes = Passes(monkeypatch)
    run_dir = single(tmp_path, clone_dir, teacher_out[0], config, per_process=1)
    assert passes.checked == 1  # three drivers read the stream; one paid the pass
    assert len(markers(run_dir, teacher_out[1])) == 1
    encoded_once = passes.encoded
    assert encoded_once > 0
    # a fourth start finds the marker: nothing is checked or encoded for the stream's own sake
    warm_loop(run_dir, config).run(GENERATIONS)
    assert (passes.checked, passes.encoded) == (1, encoded_once)


def test_the_generation_files_are_byte_identical_with_and_without_the_marker(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = imitation_config(clone_ids(clone_dir), teacher_out[1])
    with_marker = single(tmp_path / "with", clone_dir, teacher_out[0], config, per_process=1)
    assert markers(with_marker, teacher_out[1])
    continuous = single(
        tmp_path / "one", clone_dir, teacher_out[0], config, per_process=GENERATIONS
    )
    with monkeypatch.context() as plain:
        without_markers(plain)
        without = single(tmp_path / "without", clone_dir, teacher_out[0], config, per_process=1)
    assert not markers(without, teacher_out[1])
    assert tree_bytes(with_marker) == tree_bytes(without) == tree_bytes(continuous)


def test_a_population_started_for_every_generation_checks_the_teacher_stream_once(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(clone_ids(clone_dir), teacher_out[1], key="marker-pop")
    passes = Passes(monkeypatch)
    run_dir = population(tmp_path, clone_dir, teacher_out[0], config, per=1)
    assert passes.checked == 1
    assert len(markers(run_dir, teacher_out[1])) == 1


def test_the_population_files_are_byte_identical_with_and_without_the_marker(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(clone_ids(clone_dir), teacher_out[1], key="marker-pop")
    with_marker = population(tmp_path / "with", clone_dir, teacher_out[0], config, per=1)
    with monkeypatch.context() as plain:
        without_markers(plain)
        without = population(tmp_path / "without", clone_dir, teacher_out[0], config, per=1)
    assert tree_bytes(with_marker) == tree_bytes(without)


def test_a_damaged_marker_between_two_drivers_changes_no_byte(
    tmp_path: Path, clone_dir: Path, teacher_out: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = imitation_config(clone_ids(clone_dir), teacher_out[1])
    run_dir = tmp_path / "run"
    import_clone(clone_dir, run_dir, BATTLE)
    import_teacher(teacher_out[0], run_dir)
    warm_loop(run_dir, config).run(1)
    (marker,) = markers(run_dir, teacher_out[1])
    marker.write_bytes(marker.read_bytes()[:40])  # torn
    passes = Passes(monkeypatch)
    warm_loop(run_dir, config).run(2)
    assert passes.checked == 1  # the full path ran again and wrote the marker back
    warm_loop(run_dir, config).run(GENERATIONS)
    assert passes.checked == 1
    reference = single(tmp_path / "ref", clone_dir, teacher_out[0], config, per_process=GENERATIONS)
    assert tree_bytes(run_dir) == tree_bytes(reference)
