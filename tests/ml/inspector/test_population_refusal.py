"""The inspector and the dashboard refuse a population run in fixed words."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.inspector import readers
from areneon_ml.inspector.model import read_run_dir_config
from areneon_ml.inspector.readers import POPULATION_REFUSAL, InspectError
from areneon_ml.report import model as report_model
from areneon_ml.report.model import ReportError

from tests.ml.inspector.support import FIXTURE

#: Files written by the Node population play; population-generation.test.ts requires them to equal
#: a fresh tiny M = 2 generation.
POPULATION = Path(__file__).resolve().parents[3] / "fixtures" / "loop" / "population-play"


def load(name: str) -> Any:
    return json.loads((POPULATION / name).read_text())


def test_the_words_are_fixed() -> None:
    assert POPULATION_REFUSAL == (
        "a population run is not supported by the inspector yet (ADR-021 slices 11 and 12)"
    )


def test_the_readers_refuse_a_population_generation_and_plan() -> None:
    assert load("generation.json")["schema_version"] == 2
    with pytest.raises(InspectError, match="population run is not supported by the inspector yet"):
        readers.read_generation(load("generation.json"), 1)
    plan = load("plan.json")
    assert "seats" in plan[0]
    with pytest.raises(InspectError, match="population run is not supported by the inspector yet"):
        readers.read_plan(plan)


def test_a_single_lineage_generation_is_still_read() -> None:
    gen = FIXTURE / "generations" / "g000001"
    assert readers.read_plan(json.loads((gen / "plan.json").read_text()))
    assert readers.read_generation(json.loads((gen / "generation.json").read_text()), 1)


def test_the_config_reader_refuses_a_population_config(tmp_path: Path) -> None:
    config = json.loads((FIXTURE / "run-config.json").read_text())
    config["population"] = {"members": 2}
    (tmp_path / "run-config.json").write_bytes(bytes(jcs(config)) + b"\n")
    with pytest.raises(InspectError, match="population run is not supported by the inspector yet"):
        read_run_dir_config(tmp_path)


def test_the_dashboard_refuses_a_population_generation(tmp_path: Path) -> None:
    source = FIXTURE / "generations" / "g000001"
    gen = tmp_path / "g000001"
    shutil.copytree(source, gen)
    for name in ("plan.json", "generation.json"):
        shutil.copy(POPULATION / name, gen / name)
    config = read_run_dir_config(FIXTURE)
    with pytest.raises(ReportError, match="g000001: a population run is not supported"):
        report_model._played_generation(1, gen, config, "0" * 64, {})
