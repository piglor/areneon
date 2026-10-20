"""The evaluation store isolation check (ADR-010 O11, protocol I1): it passes a
correctly locked store, fails each way a store can leak, and its training-user
probe passes only where the store is really unreadable."""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-evaluation-store.sh"
ME = getpass.getuser()
pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="root reads everything")


def which(tool: str) -> str:
    found = shutil.which(tool)
    assert found is not None
    return found


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, check=False, cwd=cwd
    )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A store the way the runbook leaves it: 0700, files 0600, outside any repository."""
    root = tmp_path / "store"
    (root / "sealed").mkdir(parents=True)
    (root / "register").mkdir()
    (root / "sealed" / "key").write_text("secret\n")
    (root / "register" / "register.jsonl").write_text("")
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)
    return root


def test_a_locked_store_passes(store: Path) -> None:
    result = run("--store", str(store), "--owner", ME)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all checks passed" in result.stdout
    assert "FAIL" not in result.stdout


def test_the_check_changes_nothing_and_prints_no_content(store: Path) -> None:
    before = sorted((p, p.stat().st_mode, p.stat().st_mtime_ns) for p in store.rglob("*"))
    result = run("--store", str(store), "--owner", ME)
    assert "secret" not in result.stdout + result.stderr
    assert before == sorted((p, p.stat().st_mode, p.stat().st_mtime_ns) for p in store.rglob("*"))


def test_group_or_other_bits_on_the_directory_fail(store: Path) -> None:
    store.chmod(0o750)
    result = run("--store", str(store), "--owner", ME)
    assert result.returncode == 1
    assert "mode 750" in result.stdout


def test_a_readable_file_inside_fails(store: Path) -> None:
    (store / "sealed" / "key").chmod(0o644)
    result = run("--store", str(store), "--owner", ME)
    assert result.returncode == 1
    assert "has a group or other bit" in result.stdout


def test_a_store_owned_by_someone_else_fails(store: Path) -> None:
    result = run("--store", str(store), "--owner", "nobody")
    assert result.returncode == 1
    assert f"is owned by {ME}, not nobody" in result.stdout


def test_a_store_in_a_git_working_tree_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    inside = repo / "store"
    inside.mkdir(mode=0o700)
    inside.chmod(0o700)
    result = run("--store", str(inside), "--owner", ME, "--repo", str(repo))
    assert result.returncode == 1
    assert "inside a git working tree" in result.stdout
    assert "is under the repository" in result.stdout


def test_a_world_writable_parent_fails(tmp_path: Path) -> None:
    parent = tmp_path / "open"
    parent.mkdir()
    child = parent / "store"
    child.mkdir()
    child.chmod(0o700)
    parent.chmod(0o777)
    result = run("--store", str(child), "--owner", ME)
    assert result.returncode == 1
    assert "writable by group or other" in result.stdout


def test_a_missing_store_fails(tmp_path: Path) -> None:
    result = run("--store", str(tmp_path / "none"), "--owner", ME)
    assert result.returncode == 1
    assert "not a directory" in result.stdout


def test_the_probe_passes_where_the_user_is_refused(store: Path) -> None:
    # A directory this user cannot enter stands in for the store as seen by the training user.
    store.chmod(0o000)
    try:
        result = run("--store", str(store / "sealed"), "--probe-denied")
    finally:
        store.chmod(0o700)
    assert result.returncode == 0, result.stdout
    assert "FAIL" not in result.stdout
    assert result.stdout.count("Permission denied") == 2


def test_the_probe_fails_where_the_user_can_read(store: Path) -> None:
    result = run("--store", str(store), "--probe-denied")
    assert result.returncode == 1
    assert "succeeds" in result.stdout


def test_the_probe_fails_for_a_path_that_does_not_exist(tmp_path: Path) -> None:
    """A typo is not a refusal: ENOENT must not pass."""
    result = run("--store", str(tmp_path / "nonexistent" / "zzz"), "--probe-denied")
    assert result.returncode == 1
    assert "PASS" not in result.stdout
    assert "not with Permission denied" in result.stdout


def test_the_container_checks_need_docker(store: Path, tmp_path: Path) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    for tool in ("stat", "find", "dirname", "readlink", "git", "grep", "id"):
        (empty_bin / tool).symlink_to(which(tool))
    result = subprocess.run(
        [
            which("bash"),
            str(SCRIPT),
            "--store",
            str(store),
            "--owner",
            ME,
            "--training-container",
            "t1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": str(empty_bin)},
    )
    assert result.returncode == 1
    assert "docker is not on PATH" in result.stdout


FAKE_DOCKER = """#!/usr/bin/env bash
# A stand-in `docker inspect`: the container name is the last argument.
name="${@: -1}"
case "$*" in
  *"{{.Source}} {{.RW}}"*)
    case "$name" in
      eval-ok) printf "STORE/sealed false\\nSTORE/runs true\\n" ;;
      eval-rw) printf "STORE/sealed true\\n" ;;
      *) exit 1 ;;
    esac ;;
  *"{{.Config.User}}"*)
    case "$name" in
      train-root) printf "" ;;
      train-zero) printf "0:0" ;;
      nope) exit 1 ;;
      *) printf "1001:1001" ;;
    esac ;;
  *"{{.HostConfig.Privileged}}"*)
    case "$name" in
      train-priv) printf "true" ;;
      *) printf "false" ;;
    esac ;;
  *)
    case "$name" in
      train-clean|train-root|train-zero|train-priv) printf "/data/train\\n" ;;
      train-leaky) printf "STORE/sealed\\n" ;;
      train-store) printf "STORE\\n" ;;
      train-parent) printf "PARENT\\n" ;;
      train-everything) printf "/\\n" ;;
      train-below) printf "STORE/sealed/teams\\n" ;;
      train-sock) printf "/var/run/docker.sock\\n" ;;
      *) exit 1 ;;
    esac ;;
esac
"""


@pytest.fixture
def docker_env(store: Path, tmp_path: Path) -> dict[str, str]:
    """A PATH whose `docker` answers from the table above, with the store's real paths filled in."""
    fake = tmp_path / "fakebin"
    fake.mkdir()
    real = store.resolve()
    text = FAKE_DOCKER.replace("STORE", str(real)).replace("PARENT", str(real.parent))
    (fake / "docker").write_text(text)
    (fake / "docker").chmod(0o755)
    return {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}


def check_with(env: dict[str, str], store: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), "--store", str(store), "--owner", ME, *extra],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_a_clean_training_container_and_a_read_only_evaluation_container_pass(
    store: Path, docker_env: dict[str, str]
) -> None:
    good = check_with(
        docker_env,
        store,
        "--training-container",
        "train-clean",
        "--evaluation-container",
        "eval-ok",
    )
    assert good.returncode == 0, good.stdout
    assert "runs as 1001:1001" in good.stdout
    assert "read-only" in good.stdout


@pytest.mark.parametrize(
    ("container", "message"),
    [
        ("train-leaky", "contains or lies under"),
        ("train-store", "contains or lies under"),
        ("train-parent", "contains or lies under"),
        ("train-everything", "mounts /, which contains"),
        ("train-below", "contains or lies under"),
        ("train-sock", "Docker socket"),
        ("train-root", "runs as root or an unknown user"),
        ("train-zero", "runs as root or an unknown user"),
        ("train-priv", "is privileged"),
    ],
)
def test_a_training_container_that_can_reach_the_store_fails(
    store: Path, docker_env: dict[str, str], container: str, message: str
) -> None:
    result = check_with(docker_env, store, "--training-container", container)
    assert result.returncode == 1
    assert message in result.stdout


def test_a_read_write_sealed_mount_and_an_unknown_container_fail(
    store: Path, docker_env: dict[str, str]
) -> None:
    rw = check_with(docker_env, store, "--evaluation-container", "eval-rw")
    assert rw.returncode == 1
    assert "read-write" in rw.stdout
    unknown = check_with(
        docker_env, store, "--training-container", "nope", "--evaluation-container", "nope"
    )
    assert unknown.returncode == 1
    assert unknown.stdout.count("cannot inspect") == 2


def test_a_group_writable_parent_fails(tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir()
    child = parent / "store"
    child.mkdir()
    child.chmod(0o700)
    parent.chmod(0o775)
    result = run("--store", str(child), "--owner", ME)
    assert result.returncode == 1
    assert "writable by group or other" in result.stdout


def test_the_code_checkout_must_not_be_writable_by_others(tmp_path: Path, store: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "tool.ts").write_text("x")
    code.chmod(0o755)
    (code / "tool.ts").chmod(0o644)
    good = run("--store", str(store), "--owner", ME, "--code", str(code))
    assert good.returncode == 0, good.stdout
    assert "is owned by" in good.stdout
    (code / "tool.ts").chmod(0o664)
    bad = run("--store", str(store), "--owner", ME, "--code", str(code))
    assert bad.returncode == 1
    assert "writable by group or other" in bad.stdout
    missing = run("--store", str(store), "--owner", ME, "--code", str(tmp_path / "none"))
    assert missing.returncode == 1


def test_a_code_checkout_owned_by_someone_else_fails(tmp_path: Path, store: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    code.chmod(0o755)
    result = run("--store", str(store), "--owner", "nobody", "--code", str(code))
    assert result.returncode == 1
    assert "owned by neither nobody nor root" in result.stdout


def test_usage_errors_exit_2() -> None:
    assert run().returncode == 2
    assert run("--store", "/tmp").returncode == 2
    assert run("--bogus").returncode == 2
