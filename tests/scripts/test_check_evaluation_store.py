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
    assert "world-writable" in result.stdout


def test_a_missing_store_fails(tmp_path: Path) -> None:
    result = run("--store", str(tmp_path / "none"), "--owner", ME)
    assert result.returncode == 1
    assert "not a directory" in result.stdout


def test_the_probe_passes_where_the_user_cannot_enter(store: Path) -> None:
    # A directory this user cannot enter stands in for the store as seen by the training user.
    store.chmod(0o000)
    try:
        result = run("--store", str(store / "sealed"), "--probe-denied")
    finally:
        store.chmod(0o700)
    assert result.returncode == 0, result.stdout
    assert "FAIL" not in result.stdout


def test_the_probe_fails_where_the_user_can_read(store: Path) -> None:
    result = run("--store", str(store), "--probe-denied")
    assert result.returncode == 1
    assert "succeeds" in result.stdout


def test_the_container_checks_need_docker(store: Path, tmp_path: Path) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()
    bash = which("bash")
    env = {"PATH": str(empty_bin)}
    for tool in ("stat", "find", "dirname", "readlink", "git", "grep", "id"):
        (empty_bin / tool).symlink_to(which(tool))
    result = subprocess.run(
        [bash, str(SCRIPT), "--store", str(store), "--owner", ME, "--training-container", "t1"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 1
    assert "docker is not on PATH" in result.stdout


def test_a_fake_docker_shows_the_mount_rules(store: Path, tmp_path: Path) -> None:
    """A stand-in `docker inspect` answers; the script reads mounts and read-only flags from it."""
    fake = tmp_path / "fakebin"
    fake.mkdir()
    real = store.resolve()
    (fake / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *"{{.Source}} {{.RW}}"*) case "${@: -1}" in\n'
        f'    eval-ok) printf "{real}/sealed false\\n{real}/runs true\\n" ;;\n'
        f'    eval-rw) printf "{real}/sealed true\\n" ;;\n'
        "    *) exit 1 ;;\n"
        "  esac ;;\n"
        '  *) case "${@: -1}" in\n'
        '    train-clean) printf "/data/train\\n" ;;\n'
        f'    train-leaky) printf "{real}/sealed\\n" ;;\n'
        '    train-sock) printf "/var/run/docker.sock\\n" ;;\n'
        "    *) exit 1 ;;\n"
        "  esac ;;\n"
        "esac\n"
    )
    (fake / "docker").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}

    def check(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), "--store", str(store), "--owner", ME, *extra],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    good = check("--training-container", "train-clean", "--evaluation-container", "eval-ok")
    assert good.returncode == 0, good.stdout
    assert "mounts nothing under" in good.stdout
    assert "read-only" in good.stdout
    leaky = check("--training-container", "train-leaky")
    assert leaky.returncode == 1
    assert "mounts a path under" in leaky.stdout
    sock = check("--training-container", "train-sock")
    assert sock.returncode == 1
    assert "Docker socket" in sock.stdout
    rw = check("--evaluation-container", "eval-rw")
    assert rw.returncode == 1
    assert "read-write" in rw.stdout
    unknown = check("--training-container", "nope", "--evaluation-container", "nope")
    assert unknown.returncode == 1
    assert unknown.stdout.count("cannot inspect") == 2


def test_usage_errors_exit_2() -> None:
    assert run().returncode == 2
    assert run("--store", "/tmp").returncode == 2
    assert run("--bogus").returncode == 2
