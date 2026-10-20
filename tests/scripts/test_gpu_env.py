"""The GPU training environment stays separate and in step with the shipped one (ADR-010 §5.2)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from scripts import check_cuda_separation as guard

REPO = Path(__file__).resolve().parents[2]
GPU = REPO / guard.GPU_ENV


def packages(lock: Path) -> dict[str, dict[str, object]]:
    found = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    return {p["name"]: p for p in found}


def test_it_is_its_own_uv_project_with_its_own_lock() -> None:
    project = tomllib.loads((GPU / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == guard.GPU_PROJECT_NAME
    assert (GPU / "uv.lock").is_file()
    assert (GPU / "README.md").is_file()
    assert "areneon-ml" in packages(GPU / "uv.lock")


def test_the_root_workspace_excludes_it_and_the_root_lock_has_no_trace_of_it() -> None:
    root = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    workspace = root["tool"]["uv"]["workspace"]
    assert guard.GPU_ENV in workspace["exclude"]
    assert guard.GPU_ENV not in workspace["members"]
    assert guard.GPU_PROJECT_NAME not in packages(REPO / "uv.lock")


def test_it_takes_the_cuda_build_of_the_same_torch_version_as_the_shipped_lock() -> None:
    shipped = packages(REPO / "uv.lock")["torch"]
    gpu = packages(GPU / "uv.lock")["torch"]
    assert str(shipped["version"]).split("+")[0] == str(gpu["version"]).split("+")[0]
    assert str(shipped["version"]).endswith("+cpu")
    assert "+cu" in str(gpu["version"])
    assert "/whl/cu" in str(gpu["source"])


def test_the_cuda_packages_are_in_its_lock_and_only_there() -> None:
    gpu = packages(GPU / "uv.lock")
    assert any(guard.is_cuda_package(name) for name in gpu)
    assert not any(guard.is_cuda_package(name) for name in packages(REPO / "uv.lock"))
