"""The dashboard of a population run with the progress root of a member (#1335)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_ml.loop import layout
from areneon_ml.progress.run import progress_dir_of
from areneon_ml.report.command import report_run
from areneon_ml.report.model import ReportError, _member_of, read_report_model
from areneon_ml.report.render import build_report

from tests.ml.loop.test_population import gen_path
from tests.ml.progress.test_population import population, run  # noqa: F401  (fixture and helper)


def test_a_member_root_renders_beside_the_ladder(
    population: tuple[Path, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    run_dir, _ = population
    run(population, tmp_path / "p", 2)  # the ladder top
    model = read_report_model(run_dir, tmp_path / "p")
    assert model.progress_members == {2: "m0"}
    assert set(model.progress) == {2}
    html = build_report(model)
    assert "Progress evaluation: members" in html
    assert "Progress evaluation</h2>" in html
    assert "Population report" in html


def test_a_named_member_and_the_command(
    population: tuple[Path, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    run_dir, _ = population
    run(population, tmp_path / "p", 2, "m1")
    run(population, tmp_path / "p", 1, "m1")
    model = read_report_model(run_dir, tmp_path / "p")
    assert model.progress_members == {1: "m1", 2: "m1"}
    out = tmp_path / "out" / "r.html"
    report_run(run_dir, tmp_path / "p", out)
    assert out.is_file()


def test_an_empty_root_says_so(population: tuple[Path, Any], tmp_path: Path) -> None:  # noqa: F811
    run_dir, _ = population
    root = tmp_path / "empty"
    root.mkdir()
    html = build_report(read_report_model(run_dir, root))
    assert "holds no committed generation" in html


def test_no_root_has_no_member_section(population: tuple[Path, Any]) -> None:  # noqa: F811
    html = build_report(read_report_model(population[0], None))
    assert "Progress evaluation: members" not in html


def test_a_root_of_a_pair_that_is_no_members_is_refused(
    population: tuple[Path, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    run_dir, _ = population
    run(population, tmp_path / "p", 2)
    # A copy of the run whose generation manifest holds other checkpoint ids for every member.
    other = tmp_path / "run"
    shutil.copytree(run_dir, other)
    path = gen_path(other, 2) / "manifest.json"
    manifest = layout.read_json(path)
    for entry in manifest["members"].values():
        entry["checkpoints"] = {"preview": "0" * 64, "battle": "1" * 64}
    layout.write_json(path, manifest)
    with pytest.raises(ReportError):
        read_report_model(other, tmp_path / "p")
    assert progress_dir_of(tmp_path / "p", 2).is_dir()


def test_the_member_of_a_pair_is_exactly_one() -> None:
    pair = {"preview": "a" * 64, "battle": "b" * 64}
    manifest = {"members": {"m0": {"checkpoints": pair}, "m1": {"checkpoints": {"x": "y"}}}}
    assert _member_of(manifest, 2, pair) == "m0"
    both = {"members": {"m0": {"checkpoints": pair}, "m1": {"checkpoints": pair}}}
    with pytest.raises(ReportError, match="several members share"):
        _member_of(both, 2, pair)
    with pytest.raises(ReportError, match="no member's frozen pair"):
        _member_of({"members": {"m0": {"checkpoints": {"x": "y"}}}}, 2, pair)
    odds: list[dict[str, Any]] = [{}, {"members": []}, {"members": {"m0": 3}}]
    for odd in odds:
        with pytest.raises(ReportError, match="no well-formed members"):
            _member_of(odd, 2, pair)
