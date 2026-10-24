"""The private output root (ADR-015 erratum E1), Python twin of ``private-root.ts``.

A directory is accepted only if it is absolute, a real directory (no symbolic
link in the path), owned by the invoking user, mode 0700, below no group- or
world-writable parent (a sticky bit is fine), every parent owned by the user or
root, not on a Windows drive mounted into WSL, and has a ``private-data`` path
segment. The codes and fixed words equal the TypeScript ones
(``fixtures/private-guard/refusals.json``); a refusal never carries a path.
``uid`` and ``parent_uid`` are seams for tests only.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Final

PRIVATE_ROOT_REFUSALS: Final[dict[str, str]] = {
    "not_absolute": "refusing: the private output root must be an absolute path",
    "unreadable": "refusing: the private output root cannot be read",
    "not_a_directory": "refusing: the private output root is not a real directory",
    "symbolic_link": "refusing: the private output root is, or lies under, a symbolic link",
    "wrong_owner": "refusing: the private output root is not owned by the invoking user",
    "group_or_other_access": "refusing: the private output root must have mode 0700",
    "unsafe_parent": (
        "refusing: a parent of the private output root is writable by others "
        "or owned by another user"
    ),
    "windows_mount": "refusing: the private output root is on a Windows drive mount",
    "no_private_data_segment": "refusing: the private output root has no private-data path segment",
    "bad_file_name": "refusing: not a plain file name for the private output root",
    "exists": "refusing: the private output file already exists",
}

_GROUP_OTHER = 0o077
_GROUP_OTHER_WRITE = 0o022
_WINDOWS_MOUNT = re.compile(r"^/mnt/[A-Za-z](/|$)")


class PrivateRootError(Exception):
    """A refusal with a fixed code and fixed words."""

    def __init__(self, code: str) -> None:
        super().__init__(PRIVATE_ROOT_REFUSALS[code])
        self.code = code


def assert_private_root(
    directory: str | os.PathLike[str],
    *,
    uid: int | None = None,
    parent_uid: Callable[[Path], int] | None = None,
) -> Path:
    """Refuse unless ``directory`` is an acceptable private output root; returns its real path."""
    text = os.fspath(directory)
    if not os.path.isabs(text):
        raise PrivateRootError("not_absolute")
    me = os.getuid() if uid is None else uid
    try:
        info = os.lstat(text)  # lstat: a symbolic link to a directory is not the directory
        if stat.S_ISLNK(info.st_mode):
            raise PrivateRootError("symbolic_link")
        if not stat.S_ISDIR(info.st_mode):
            raise PrivateRootError("not_a_directory")
        real = os.path.realpath(text)
        if real != os.path.normpath(text):
            raise PrivateRootError("symbolic_link")
        if info.st_uid != me:
            raise PrivateRootError("wrong_owner")
        if info.st_mode & _GROUP_OTHER:
            raise PrivateRootError("group_or_other_access")
        for parent in Path(real).parents:
            above = os.lstat(parent)
            owner = parent_uid(parent) if parent_uid is not None else above.st_uid
            if owner not in (me, 0):
                raise PrivateRootError("unsafe_parent")
            if above.st_mode & _GROUP_OTHER_WRITE and not above.st_mode & stat.S_ISVTX:
                raise PrivateRootError("unsafe_parent")
    except OSError:
        raise PrivateRootError("unreadable") from None
    place = place_refusal(real)
    if place is not None:
        raise PrivateRootError(place)
    return Path(real)


def place_refusal(real: str) -> str | None:
    """The refusal for where a real path lies (a Windows drive, or no ``private-data`` segment)."""
    if _WINDOWS_MOUNT.match(real):
        return "windows_mount"
    return None if "private-data" in PurePosixPath(real).parts else "no_private_data_segment"


def private_root_file(
    root: str | os.PathLike[str],
    name: str,
    *,
    uid: int | None = None,
    parent_uid: Callable[[Path], int] | None = None,
) -> Path:
    """The path of a file in a checked private root; only a plain file name is accepted."""
    real = assert_private_root(root, uid=uid, parent_uid=parent_uid)
    if not re.fullmatch(r"[^/\\\x00]+", name) or name in (".", ".."):
        raise PrivateRootError("bad_file_name")
    return real / name


def write_private_file(
    root: str | os.PathLike[str],
    name: str,
    data: bytes | str,
    *,
    uid: int | None = None,
    parent_uid: Callable[[Path], int] | None = None,
) -> Path:
    """Write a new file (mode 0600) in the private root; an existing file is never overwritten."""
    path = private_root_file(root, name, uid=uid, parent_uid=parent_uid)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise PrivateRootError("exists") from None
    except OSError:
        raise PrivateRootError("unreadable") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return path
