from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from scripts import check_cuda_separation as guard
from scripts import check_licenses
from scripts import python_licences as pl

EXIT_FAILED_CLOSED = 2
REPO = Path(__file__).resolve().parents[2]

PYLOCK = """
lock-version = "1.0"
[[packages]]
name = "areneon-contracts"
[[packages]]
name = "areneon-ml"
[[packages]]
name = "torch"
"""


def lock(*names: str) -> str:
    return "".join(f'[[package]]\nname = "{name}"\nversion = "1"\n' for name in names)


def pylock_with(*extra: str) -> str:
    return PYLOCK + "".join(f'[[packages]]\nname = "{name}"\n' for name in extra)


@pytest.mark.parametrize(
    "name",
    [
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
        "NVIDIA_NCCL_CU13",
        "cuda-toolkit",
        "cuda-bindings",
        "triton",
    ],
)
def test_the_named_cuda_packages_are_recognised(name: str) -> None:
    assert guard.is_cuda_package(name)


@pytest.mark.parametrize("name", ["torch", "numpy", "cuda-python-docs", "nvidia", "tritonclient"])
def test_other_packages_are_not(name: str) -> None:
    assert not guard.is_cuda_package(name)


def test_a_lock_with_a_cuda_package_is_refused() -> None:
    assert guard.lock_violations(lock("torch", "jinja2")) == []
    assert guard.lock_violations(lock("torch", "triton", "nvidia-cublas")) == [
        "uv.lock contains nvidia-cublas",
        "uv.lock contains triton",
    ]
    assert guard.lock_violations("") == []


def test_a_shipped_closure_with_a_cuda_package_is_refused() -> None:
    assert guard.closure_violations(lambda _command: pylock_with()) == []
    assert guard.closure_violations(lambda _command: pylock_with("triton")) == [
        "the closure of areneon-contracts contains triton",
        "the closure of areneon-ml contains triton",
    ]


def workspace(*members: str) -> str:
    listed = ", ".join(f'"{m}"' for m in members)
    return f"[tool.uv.workspace]\nmembers = [{listed}]\n"


@pytest.mark.parametrize("member", ["ml/gpu-env", "ml/gpu-env/", "ml/*", "ml/**"])
def test_a_workspace_member_covering_the_gpu_environment_is_refused(member: str) -> None:
    assert guard.workspace_violations(workspace("ml", member)) == [
        f"workspace member {member!r} covers ml/gpu-env"
    ]


def test_the_workspace_may_list_the_shipped_packages_only() -> None:
    assert guard.workspace_violations(workspace("packages/schema/python", "ml")) == []
    assert guard.workspace_violations("[project]\nname = 'x'\n") == []


def test_shipped_must_not_name_the_gpu_environment() -> None:
    assert guard.workspace_violations("", ("areneon-ml", "areneon-gpu-training")) == []
    assert guard.workspace_violations("", ("areneon-ml", "areneon_gpu_env")) == [
        "SHIPPED names the GPU environment: areneon_gpu_env"
    ]


def test_no_image_recipe_may_refer_to_the_gpu_environment(tmp_path: Path) -> None:
    (tmp_path / "svc").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "svc" / "Dockerfile").write_text("RUN uv sync --project ml/gpu-env\n")
    (tmp_path / "svc" / "app.Dockerfile").write_text("RUN uv sync --package areneon-ml\n")
    (tmp_path / "svc" / "Dockerfile.e2e").write_text("RUN pip install areneon-gpu-env\n")
    (tmp_path / "node_modules" / "Dockerfile").write_text("ml/gpu-env\n")
    (tmp_path / "notes.txt").write_text("ml/gpu-env\n")
    assert guard.dockerfile_violations(tmp_path) == [
        "svc/Dockerfile refers to the GPU environment (ml/gpu-env)",
        "svc/Dockerfile.e2e refers to the GPU environment (ml/gpu-env)",
    ]


@pytest.mark.parametrize(
    ("file", "text"),
    [
        ("Dockerfile", "COPY ml/GPU-ENV /app"),
        ("Dockerfile", "COPY ml/gpu_env /app"),
        ("Dockerfile", "RUN pip install Areneon_GPU_Env"),
        ("dockerfile", "COPY ml/gpu-env /app"),
        ("api.dockerfile", "COPY ml/gpu-env /app"),
        ("Containerfile", "COPY ml/gpu-env /app"),
        ("containerfile.prod", "COPY ml/gpu-env /app"),
        ("compose.yaml", "build:\n  dockerfile_inline: |\n    COPY ml/gpu-env /app\n"),
        ("docker-compose.yml", "volumes: ['./ml/gpu-env:/gpu']"),
        ("compose.override.yaml", "x: ml/gpu-env"),
    ],
)
def test_the_bypasses_of_the_name_check_are_closed(tmp_path: Path, file: str, text: str) -> None:
    (tmp_path / "a" / "vendor").mkdir(parents=True)
    (tmp_path / "a" / "vendor" / file).write_text(text)
    assert guard.dockerfile_violations(tmp_path) == [
        f"a/vendor/{file} refers to the GPU environment (ml/gpu-env)"
    ]


def test_only_the_root_vendor_directory_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "vendor" / "pokemon-showdown").mkdir(parents=True)
    (tmp_path / "vendor" / "pokemon-showdown" / "Dockerfile").write_text("ml/gpu-env")
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "compose.yaml").write_text("services: {}\n")
    assert guard.dockerfile_violations(tmp_path) == []


@pytest.mark.parametrize(
    "member", ["./ml/gpu-env", "ml//gpu-env", "ml/../ml/gpu-env", " ml/gpu-env "]
)
def test_c1_normalises_workspace_members(member: str) -> None:
    assert guard.workspace_violations(workspace("ml", member)) == [
        f"workspace member {member!r} covers ml/gpu-env"
    ]


def test_an_excluded_gpu_environment_is_not_a_member() -> None:
    text = workspace("ml", "ml/*") + 'exclude = ["ml/gpu-env"]\n'
    assert guard.workspace_violations(text) == []
    assert guard.workspace_violations(workspace("ml", "ml/*") + 'exclude = ["other"]\n') != []


def make_root(tmp_path: Path, packages: tuple[str, ...] = ("torch",)) -> Path:
    (tmp_path / "pyproject.toml").write_text(workspace("ml"))
    (tmp_path / "uv.lock").write_text(lock(*packages))
    return tmp_path


def test_main_passes_a_clean_root_and_prints_each_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean = make_root(tmp_path)
    assert guard.main(clean, lambda _command: pylock_with()) == 0
    (clean / "uv.lock").write_text(lock("torch", "nvidia-cublas"))
    assert guard.main(clean, lambda _command: pylock_with("nvidia-cublas")) == 1
    err = capsys.readouterr().err
    assert "CUDA separation violated: uv.lock contains nvidia-cublas" in err
    assert "the closure of areneon-ml contains nvidia-cublas" in err


def test_main_fails_closed_on_unreadable_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert guard.main(tmp_path, lambda _command: pylock_with()) == EXIT_FAILED_CLOSED
    assert "failed closed" in capsys.readouterr().err
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "uv.lock").write_text("not = [toml")
    assert guard.main(tmp_path, lambda _command: pylock_with()) == EXIT_FAILED_CLOSED
    root = make_root(tmp_path)
    assert guard.main(root, lambda _command: "garbage = [") == EXIT_FAILED_CLOSED


def test_the_repository_passes_its_own_guard() -> None:
    assert guard.violations(REPO) == []


def test_the_gpu_environment_is_not_a_workspace_member_or_shipped() -> None:
    members = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["uv"]["workspace"][
        "members"
    ]
    assert "ml" in members
    assert guard.GPU_ENV not in members
    assert guard.workspace_violations((REPO / "pyproject.toml").read_text()) == []
    assert all("gpu" not in name for name in pl.SHIPPED)


def test_the_cuda_names_are_never_allowlisted() -> None:
    for licence in check_licenses.ALLOWED:
        assert "nvidia" not in licence.lower()
        assert "cuda" not in licence.lower()
    assert not any(guard.is_cuda_package(name) for name in pl.SHIPPED)
