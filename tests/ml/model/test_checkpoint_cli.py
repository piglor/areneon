"""``python -m areneon_ml.checkpoint``: init, classify and independent."""

from __future__ import annotations

import hashlib
import io
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.checkpoint.__main__ import EXIT_REFUSED, main
from areneon_ml.checkpoint.files import checkpoint_bytes, load_checkpoint
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.network import init_network

from .support import FORMAT_DIR, TINY_MODEL, game
from .test_provenance import RECORDS, SOURCES, h

ENGINE_DATA = str(FORMAT_DIR / "engine-data.json")


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(TINY_MODEL.as_dict()))
    return path


def run(*argv: str) -> tuple[int, dict[str, Any]]:
    stdout = io.BytesIO()
    real = sys.stdout
    sys.stdout = type("Out", (), {"buffer": stdout})()
    try:
        code = main(list(argv))
    finally:
        sys.stdout = real
    text = stdout.getvalue()
    return code, (json.loads(text) if text else {})


def test_init_writes_a_seeded_checkpoint_under_its_id(tmp_path: Path, config_file: Path) -> None:
    args = ["init", "--role", "battle", "--seed", "5", "--engine-data", ENGINE_DATA]
    args += ["--model-config", str(config_file), "--run-dir", str(tmp_path)]
    code, out = run(*args)
    assert code == 0
    path = tmp_path / "checkpoints" / f"{out['checkpoint_id']}.safetensors"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == out["checkpoint_id"]
    assert out["role"] == "battle"
    assert out["encoder_version"] == encoder_version(game(), TINY_MODEL.encoder)
    assert out["engine_data_hash"] == game().engine_data_hash
    assert load_checkpoint(path.read_bytes(), game(), out["checkpoint_id"], role="battle")
    expected = checkpoint_bytes(init_network("battle", game(), TINY_MODEL, 5), game())
    assert path.read_bytes() == expected
    again, same = run(*args)
    assert (again, same) == (0, out)


def test_init_needs_a_config_file_there_are_no_default_sizes(tmp_path: Path) -> None:
    args = ["init", "--role", "preview", "--seed", "1", "--engine-data", ENGINE_DATA]
    with pytest.raises(SystemExit) as stop:
        run(*args, "--run-dir", str(tmp_path))
    assert stop.value.code == 2


def test_init_refuses_a_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"d_model": 3}))
    args = ["init", "--role", "battle", "--seed", "1", "--engine-data", ENGINE_DATA]
    code, _ = run(*args, "--model-config", str(bad), "--run-dir", str(tmp_path))
    assert code == EXIT_REFUSED
    assert "keys must be exactly" in capsys.readouterr().err


def test_init_refuses_a_shared_trunk_role_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({**TINY_MODEL.as_dict(), "options": {"shared_trunk": True}}))
    args = ["init", "--role", "preview", "--seed", "1", "--engine-data", ENGINE_DATA]
    code, _ = run(*args, "--model-config", str(shared), "--run-dir", str(tmp_path))
    assert code == EXIT_REFUSED
    assert "made by the generation loop" in capsys.readouterr().err
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("seed", ["-1", str(2**64)])
def test_init_refuses_a_seed_out_of_range(
    tmp_path: Path, config_file: Path, capsys: pytest.CaptureFixture[str], seed: str
) -> None:
    args = ["init", "--role", "battle", "--seed", seed, "--engine-data", ENGINE_DATA]
    code, _ = run(*args, "--model-config", str(config_file), "--run-dir", str(tmp_path))
    assert code == EXIT_REFUSED
    assert "seed must be between" in capsys.readouterr().err
    assert not (tmp_path / "checkpoints").exists()


@pytest.fixture
def provenance(tmp_path: Path) -> tuple[Path, Path]:
    records = tmp_path / "records"
    records.mkdir()
    for record in RECORDS:
        (records / f"{record.lineage_id}.json").write_bytes(jcs(record.to_json()))
    sources = tmp_path / "sources.json"
    sources.write_bytes(jcs(SOURCES))
    return records, sources


def test_classify_prints_the_c1_class(provenance: tuple[Path, Path]) -> None:
    records, sources = provenance
    base = ["classify", "--records", str(records), "--sources", str(sources)]
    code, out = run(*base, "--candidate", h("A2"), "--opponent", h("B1"))
    assert (code, out["label"]) == (0, "held_out")
    assert out["sources_hash"] == hashlib.sha256(jcs(SOURCES)).hexdigest()
    _, out = run(*base, "--candidate", h("A2"), "--opponent", h("C0"))
    assert out["label"] == "seen"
    _, out = run(*base, "--candidate", h("A2"), "--opponent", "first-legal@golden")
    assert out["label"] == "seen"
    _, out = run(*base, "--candidate", h("A2"), "--opponent", h("B1"), h("B0"))
    assert out["label"] == "held_out"


def test_independent_prints_c6(provenance: tuple[Path, Path]) -> None:
    records, sources = provenance
    base = ["independent", "--records", str(records), "--sources", str(sources)]
    code, out = run(*base, "--a", h("A2"), "--b", h("C0"))
    assert code == 0
    assert out == {
        "independent": False,
        "shared_checkpoints": [],
        "shared_sources": [h("bc")],
        "sources_hash": hashlib.sha256(jcs(SOURCES)).hexdigest(),
    }
    _, out = run(*base, "--a", h("A2"), "--b", h("B1"))
    assert out["independent"] is True


def test_unusable_records_are_refused_with_exit_3(
    tmp_path: Path, provenance: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    records, sources = provenance
    base = ["classify", "--sources", str(sources), "--candidate", h("A2"), "--opponent", h("B1")]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(*base, "--records", str(empty))[0] == EXIT_REFUSED
    assert "holds no CheckpointRecord" in capsys.readouterr().err
    (records / "A1.json").unlink()
    assert run(*base, "--records", str(records))[0] == EXIT_REFUSED
    assert "has no record" in capsys.readouterr().err
    (records / "bad.json").write_text("{}")
    assert run(*base, "--records", str(records))[0] == EXIT_REFUSED
    sources.write_text("[]")
    assert run(*base, "--records", str(records))[0] == EXIT_REFUSED
    sources.write_text(json.dumps(SOURCES, indent=2))  # valid JSON, not canonical
    assert run(*base, "--records", str(records))[0] == EXIT_REFUSED
    assert "not canonical" in capsys.readouterr().err
    sources.write_text("not json")
    assert run(*base, "--records", str(records))[0] == EXIT_REFUSED


def test_the_module_runs_as_a_program(
    provenance: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    records, sources = provenance
    argv = ["areneon_ml.checkpoint", "independent", "--records", str(records)]
    argv += ["--sources", str(sources), "--a", h("A2"), "--b", h("B1")]
    out = io.BytesIO()
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdout", type("Out", (), {"buffer": out})())
    monkeypatch.delitem(sys.modules, "areneon_ml.checkpoint.__main__", raising=False)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("areneon_ml.checkpoint", run_name="__main__")
    assert exited.value.code == 0
    assert json.loads(out.getvalue())["independent"] is True
