"""Binding the provenance of a clone directory made before #853 (#1356)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from areneon_contracts.contracts import jcs
from areneon_ml.checkpoint.binding import BINDING_NAME
from areneon_ml.checkpoint.files import pack_safetensors
from areneon_ml.learner.__main__ import EXIT_REFUSED, main
from areneon_ml.learner.clone_bind import CloneBindError, bind_clone
from areneon_ml.learner.state import state_hash
from areneon_ml.loop.warm_start import import_clone

from tests.ml.loop.test_warm_start import TEACHER, clone_output

RUN_CONFIG = "f" * 64
SIDECAR: dict[str, Any] = {"schema_version": 1, "kind": "clone", "run_config_hash": RUN_CONFIG}


def _edit(role: str, raw: dict[str, Any]) -> None:
    raw["learner_state_hash"] = state_hash(PACKED, SIDECAR)


PACKED = pack_safetensors({"w": torch.zeros(2)})


def unbound(root: Path) -> Path:
    """A clone directory as written before #853: no ``provenance.json``, a real state."""
    clone_output(root, record_edit=_edit)
    (root / BINDING_NAME).unlink()
    (root / "clone-state.safetensors").write_bytes(PACKED)
    (root / "clone-state.json").write_bytes(jcs(SIDECAR))
    return root


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    return unbound(tmp_path / "clone")


def test_an_unbound_clone_is_bound_then_imports(clone: Path, tmp_path: Path) -> None:
    with pytest.raises(Exception, match="not bound"):
        import_clone(clone, tmp_path / "before")
    assert bind_clone(clone).written
    assert import_clone(clone, tmp_path / "run").keys() == {"preview", "battle"}


def test_a_rerun_is_a_no_op(clone: Path) -> None:
    bind_clone(clone)
    before = (clone / BINDING_NAME).read_bytes()
    assert not bind_clone(clone).written
    assert (clone / BINDING_NAME).read_bytes() == before


def test_an_existing_different_binding_is_refused(clone: Path) -> None:
    (clone / BINDING_NAME).write_bytes(b"{}")
    with pytest.raises(CloneBindError, match="never overwritten"):
        bind_clone(clone)
    assert (clone / BINDING_NAME).read_bytes() == b"{}"


def _record_files(clone: Path) -> list[Path]:
    return sorted((clone / "records").glob("*.json"))


def test_a_doctored_checkpoint_is_refused(clone: Path) -> None:
    path = next((clone / "checkpoints").glob("*.safetensors"))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(CloneBindError, match="does not hash"):
        bind_clone(clone)
    assert not (clone / BINDING_NAME).exists()


def test_a_missing_checkpoint_is_refused(clone: Path) -> None:
    next((clone / "checkpoints").glob("*.safetensors")).unlink()
    with pytest.raises(CloneBindError, match="record"):
        bind_clone(clone)


def test_a_misnamed_or_missing_or_unreadable_record_is_refused(clone: Path) -> None:
    first = _record_files(clone)[0]
    first.rename(first.with_name("other.json"))
    with pytest.raises(CloneBindError, match="named by its checkpoint id"):
        bind_clone(clone)
    other = first.with_name("other.json")
    other.unlink()
    with pytest.raises(CloneBindError, match="exactly one"):
        bind_clone(clone)
    first.write_text("not json")
    with pytest.raises(CloneBindError):
        bind_clone(clone)


def test_two_records_of_one_role_are_refused(clone: Path, tmp_path: Path) -> None:
    other = clone_output(tmp_path / "other", seed=500, record_edit=_edit)
    roles = {json.loads(p.read_text())["role"]: p for p in _record_files(other)}
    for path in _record_files(clone):
        if json.loads(path.read_text())["role"] == "preview":
            continue
        path.unlink()
    shutil.copy(roles["preview"], clone / "records" / roles["preview"].name)
    with pytest.raises(CloneBindError, match="two preview records"):
        bind_clone(clone)


def test_a_non_clone_record_is_refused(clone: Path) -> None:
    path = _record_files(clone)[0]
    raw = json.loads(path.read_text())
    raw["training_set_hashes"] = []
    path.write_text(json.dumps(raw))
    with pytest.raises(CloneBindError, match="not a clone record"):
        bind_clone(clone)


def test_a_doctored_registry_is_refused(clone: Path) -> None:
    (clone / "sources.json").write_bytes(jcs({TEACHER: [TEACHER]}))
    with pytest.raises(CloneBindError, match=r"sources\.json"):
        bind_clone(clone)


def test_state_that_disagrees_with_the_records_is_refused(clone: Path) -> None:
    (clone / "clone-state.safetensors").write_bytes(pack_safetensors({"w": torch.ones(2)}))
    with pytest.raises(CloneBindError, match="another clone state"):
        bind_clone(clone)


def test_a_state_sidecar_of_another_run_config_is_refused(tmp_path: Path) -> None:
    root = unbound(tmp_path / "c")
    sidecar = {**SIDECAR, "run_config_hash": "1" * 64}
    (root / "clone-state.json").write_bytes(jcs(sidecar))
    # the records' state hash no longer matches either: the sidecar is part of it
    with pytest.raises(CloneBindError):
        bind_clone(root)


def test_run_config_mismatch_with_a_consistent_state(tmp_path: Path) -> None:
    def edit(role: str, raw: dict[str, Any]) -> None:
        raw["learner_state_hash"] = state_hash(PACKED, SIDECAR)
        raw["run_config_hash"] = "2" * 64

    root = tmp_path / "c"
    clone_output(root, record_edit=edit)
    (root / BINDING_NAME).unlink()
    (root / "clone-state.safetensors").write_bytes(PACKED)
    (root / "clone-state.json").write_bytes(jcs(SIDECAR))
    with pytest.raises(CloneBindError, match="another run config"):
        bind_clone(root)


def test_a_missing_or_foreign_state_is_refused(clone: Path) -> None:
    (clone / "clone-state.json").write_bytes(jcs({**SIDECAR, "kind": "ppo"}))
    with pytest.raises(CloneBindError, match="not a clone state"):
        bind_clone(clone)
    (clone / "clone-state.json").unlink()
    with pytest.raises(CloneBindError, match="clone state"):
        bind_clone(clone)


def test_the_command(clone: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bind", "--clone-dir", str(clone)]) == 0
    assert "written" in capsys.readouterr().out
    assert main(["bind", "--clone-dir", str(clone)]) == 0
    assert "already bound" in capsys.readouterr().out
    (clone / "clone-state.json").unlink()
    (clone / BINDING_NAME).unlink()
    assert main(["bind", "--clone-dir", str(clone)]) == EXIT_REFUSED
    assert "clone state" in capsys.readouterr().err
