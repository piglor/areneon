from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import contract_models as cm

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:areneon:schema:contracts:v1:sample-thing",
    "title": "SampleThing",
    "type": "object",
    "additionalProperties": False,
    "required": ["n"],
    "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 9}},
}


@pytest.fixture
def contracts(tmp_path: Path) -> Path:
    root = tmp_path / "contracts"
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "sample-thing.schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    (root / "v1" / "notes.md").write_text("ignored", encoding="utf-8")
    return root


def test_module_paths_use_underscores() -> None:
    assert cm.module_path(Path("x/v1/legal-choice-set.schema.json")) == Path(
        "v1/legal_choice_set.py"
    )


def test_generates_a_closed_strict_model(contracts: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    cm.generate(contracts, out)
    source = (out / "v1" / "sample_thing.py").read_text(encoding="utf-8")
    assert "class SampleThing(BaseModel):" in source
    assert 'extra="forbid"' in source
    assert "StrictInt" in source
    assert sorted(p.name for p in out.rglob("*.py")) == [
        "__init__.py",
        "__init__.py",
        "sample_thing.py",
    ]


def test_write_then_check_finds_no_drift(contracts: Path, tmp_path: Path) -> None:
    models = tmp_path / "models"
    assert cm.main(["--write"], contracts, models) == 0
    assert cm.main([], contracts, models) == 0


def test_check_reports_missing_changed_and_orphaned_files(
    contracts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tmp_path / "models"
    assert cm.main([], contracts, models) == 1
    assert "v1/sample_thing.py: generated file is missing" in capsys.readouterr().err
    cm.main(["--write"], contracts, models)
    (models / "v1" / "sample_thing.py").write_text("# edited\n", encoding="utf-8")
    (models / "v1" / "orphan.py").write_text("", encoding="utf-8")
    assert cm.main([], contracts, models) == 1
    err = capsys.readouterr().err
    assert "v1/sample_thing.py: generated file differs from its schema" in err
    assert "v1/orphan.py: generated file has no schema" in err


def test_the_repository_models_match_their_schemas() -> None:
    assert cm.main([]) == 0
