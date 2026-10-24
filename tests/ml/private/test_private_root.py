"""The private output root (Python twin of ``private-root.ts``)."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from areneon_ml.private.private_root import (
    PRIVATE_ROOT_REFUSALS,
    PrivateRootError,
    assert_private_root,
    place_refusal,
    private_root_file,
    write_private_file,
)

REPO = Path(__file__).resolve().parents[3]
SHARED = json.loads((REPO / "fixtures/private-guard/refusals.json").read_text(encoding="utf-8"))
ME = os.getuid()


def root(tmp_path: Path, mode: int = 0o700, name: str = "out") -> Path:
    tmp_path.chmod(0o700)
    directory = tmp_path / "private-data" / name
    directory.mkdir(parents=True)
    directory.chmod(mode)
    return directory


def code_of(tmp_path: Path, run: Callable[[], object]) -> str:
    try:
        run()
    except PrivateRootError as error:
        caught = error
    else:
        return "ok"
    assert str(caught) == PRIVATE_ROOT_REFUSALS[caught.code]
    assert str(tmp_path) not in str(caught)
    return caught.code


def test_words_equal_the_shared_fixture() -> None:
    assert SHARED["private_root"] == PRIVATE_ROOT_REFUSALS


def test_accepts_a_private_directory(tmp_path: Path) -> None:
    directory = root(tmp_path)
    assert assert_private_root(directory) == directory.resolve()
    assert assert_private_root(str(directory)) == directory.resolve()


def test_refuses_relative_missing_and_file(tmp_path: Path) -> None:
    directory = root(tmp_path)
    assert code_of(tmp_path, lambda: assert_private_root("private-data/out")) == "not_absolute"
    assert code_of(tmp_path, lambda: assert_private_root(directory / "nope")) == "unreadable"
    (directory / "f").write_text("x")
    assert code_of(tmp_path, lambda: assert_private_root(directory / "f")) == "not_a_directory"


def test_refuses_a_symlink_and_a_path_under_one(tmp_path: Path) -> None:
    directory = root(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(directory)
    assert code_of(tmp_path, lambda: assert_private_root(link)) == "symbolic_link"
    via = tmp_path / "via"
    via.symlink_to(directory.parent)
    (directory / "inner").mkdir(mode=0o700)
    assert code_of(tmp_path, lambda: assert_private_root(via / "out" / "inner")) == "symbolic_link"


def test_refuses_another_owner(tmp_path: Path) -> None:
    assert code_of(tmp_path, lambda: assert_private_root(root(tmp_path), uid=ME + 1)) == (
        "wrong_owner"
    )


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o705, 0o770])
def test_refuses_group_or_other_access(tmp_path: Path, mode: int) -> None:
    directory = root(tmp_path, mode)
    assert code_of(tmp_path, lambda: assert_private_root(directory)) == "group_or_other_access"


def test_writable_parent_refused_sticky_accepted(tmp_path: Path) -> None:
    directory = root(tmp_path)
    directory.parent.chmod(0o777)
    assert code_of(tmp_path, lambda: assert_private_root(directory)) == "unsafe_parent"
    directory.parent.chmod(0o1777)
    assert code_of(tmp_path, lambda: assert_private_root(directory)) == "ok"
    directory.parent.chmod(0o700)


def test_foreign_parent_owner_refused_root_accepted(tmp_path: Path) -> None:
    directory = root(tmp_path)
    parent = directory.parent
    refuse = code_of(
        tmp_path,
        lambda: assert_private_root(directory, parent_uid=lambda p: ME + 1 if p == parent else ME),
    )
    assert refuse == "unsafe_parent"
    assert (
        code_of(tmp_path, lambda: assert_private_root(directory, parent_uid=lambda _p: 0)) == "ok"
    )


def test_refuses_no_private_data_segment(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    plain = tmp_path / "plain"
    plain.mkdir(mode=0o700)
    assert code_of(tmp_path, lambda: assert_private_root(plain)) == "no_private_data_segment"


def test_place_refusal() -> None:
    assert place_refusal("/mnt/c/private-data/out") == "windows_mount"
    assert place_refusal("/mnt/d") == "windows_mount"
    assert place_refusal("/mnt/cc/private-data") is None
    assert place_refusal("/home/u/private-data-x/out") == "no_private_data_segment"
    assert place_refusal("/home/u/private-data/out") is None


def test_writes_a_new_0600_file_and_never_overwrites(tmp_path: Path) -> None:
    directory = root(tmp_path)
    path = write_private_file(directory, "result.json", "{}")
    assert path == directory.resolve() / "result.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert code_of(tmp_path, lambda: write_private_file(directory, "result.json", "x")) == "exists"
    assert path.read_text() == "{}"
    assert write_private_file(directory, "b.bin", b"\x01\x02").read_bytes() == b"\x01\x02"


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../x", "a\\b", "a\x00b", "/abs"])
def test_only_a_plain_file_name(tmp_path: Path, name: str) -> None:
    directory = root(tmp_path)
    assert code_of(tmp_path, lambda: private_root_file(directory, name)) == "bad_file_name"


def test_the_root_is_rechecked_on_each_call(tmp_path: Path) -> None:
    directory = root(tmp_path)
    directory.chmod(0o755)
    assert code_of(tmp_path, lambda: write_private_file(directory, "a", "x")) == (
        "group_or_other_access"
    )


@pytest.mark.skipif(os.getuid() == 0, reason="root can write anywhere")
def test_a_failed_write_is_unreadable(tmp_path: Path) -> None:
    directory = root(tmp_path)
    directory.chmod(0o500)
    try:
        assert code_of(tmp_path, lambda: write_private_file(directory, "a", "x")) == "unreadable"
    finally:
        directory.chmod(0o700)


def test_the_file_helpers_forward_the_parent_seam(tmp_path: Path) -> None:
    directory = root(tmp_path)

    def foreign(_parent: Path) -> int:
        return ME + 1

    assert code_of(tmp_path, lambda: private_root_file(directory, "a", parent_uid=foreign)) == (
        "unsafe_parent"
    )
    assert code_of(
        tmp_path, lambda: write_private_file(directory, "a", "x", parent_uid=foreign)
    ) == ("unsafe_parent")
