from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_binary_files as guard

TEXT = "# notes\n\nplain UTF-8, with accents: é ü ✓\n".encode()
PARQUET = b"PAR1" + bytes(range(256)) + b"PAR1"
AGE = b"age-encryption.org/v1\n-> X25519 abc\n" + bytes([0x80, 0xFF, 0x00])
SIGNATURE = "starts with a Parquet, age or DuckDB signature"
DUCKDB = bytes(8) + b"DUCK" + bytes([0x01, 0x00, 0x00, 0x00])


def write(root: Path, name: str, data: bytes) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path.relative_to(root))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_text_files_pass(repo: Path) -> None:
    paths = [
        write(repo, "README.md", TEXT),
        write(repo, "fixtures/teams/a.json", b'{"a": 1}\n'),
        write(repo, "empty.txt", b""),
        write(repo, "data.csv", b"a,b\n1,2\n"),
    ]
    assert guard.violations(paths) == []


def test_the_allowlist_starts_empty() -> None:
    assert frozenset() == guard.ALLOWED


@pytest.mark.parametrize(
    "name",
    [
        "teams/shard.parquet",
        "teams/shard.PARQUET",
        "key.age",
        "db/teams.duckdb",
        "db/teams.duckdb.wal",
        "db/teams.DUCKDB.WAL",
        "x.arrow",
        "x.feather",
        "x.sqlite",
        "x.db",
        "deep/er/x.db",
    ],
)
def test_binary_names_are_rejected_even_with_text_content(repo: Path, name: str) -> None:
    path = write(repo, name, TEXT)
    assert guard.violations([path]) == [f"{path}: is named like a binary data file"]


def test_a_binary_name_is_rejected_when_the_file_is_gone(repo: Path) -> None:
    assert guard.violations(["gone.parquet"]) == ["gone.parquet: is named like a binary data file"]
    assert guard.violations(["gone.txt"]) == []
    (repo / "dir.db").mkdir()
    assert guard.violations(["dir.db"]) == ["dir.db: is named like a binary data file"]


@pytest.mark.parametrize(
    ("data", "reasons"),
    [
        (PARQUET, ["is not valid UTF-8", "holds a NUL byte", SIGNATURE]),
        (AGE, ["is not valid UTF-8", "holds a NUL byte", SIGNATURE]),
        (DUCKDB, ["holds a NUL byte", SIGNATURE]),
    ],
)
def test_a_renamed_binary_is_rejected_by_content(
    repo: Path, data: bytes, reasons: list[str]
) -> None:
    # The mutation-style negative control: the same bytes under an innocent name.
    path = write(repo, "docs/notes.txt", data)
    assert guard.violations([path]) == [f"{path}: {reason}" for reason in reasons]


def test_a_file_with_a_nul_byte_is_rejected(repo: Path) -> None:
    path = write(repo, "notes.md", b"looks like text\x00but is not")
    assert guard.violations([path]) == [f"{path}: holds a NUL byte"]


def test_a_file_that_is_not_utf8_is_rejected(repo: Path) -> None:
    path = write(repo, "notes.md", "café".encode("latin-1"))
    assert guard.violations([path]) == [f"{path}: is not valid UTF-8"]


@pytest.mark.parametrize(
    "data",
    [
        b"PAR1 but otherwise plain text",
        b"age-encryption.org/v1\nplain text after the header\n",
        b"-----BEGIN AGE ENCRYPTED FILE-----\nYWdlLWVuY3J5cHRpb24ub3Jn\n",
        b"12345678DUCK and plain text",
    ],
)
def test_a_signature_alone_is_rejected_even_in_valid_text(repo: Path, data: bytes) -> None:
    path = write(repo, "notes.md", data)
    assert guard.violations([path]) == [f"{path}: starts with a Parquet, age or DuckDB signature"]


@pytest.mark.parametrize(
    "data",
    [
        b"a PAR1 inside text",
        b"see age-encryption.org/v1 in the docs",
        b"DUCK at offset zero is only a word",
        b"1234567DUCK at offset seven",
        b"123456789DUCK at offset nine",
    ],
)
def test_a_signature_elsewhere_is_only_text(repo: Path, data: bytes) -> None:
    assert guard.violations([write(repo, "notes.md", data)]) == []


def test_the_allowlist_is_exact(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "ALLOWED", frozenset({"fixtures/golden.parquet"}))
    allowed = write(repo, "fixtures/golden.parquet", PARQUET)
    sibling = write(repo, "fixtures/golden.parquet.bak.parquet", PARQUET)
    other_dir = write(repo, "other/fixtures/golden.parquet", PARQUET)
    renamed = write(repo, "fixtures/golden.txt", PARQUET)
    found = guard.violations([allowed, sibling, other_dir, renamed])
    assert not any(line.startswith(f"{allowed}:") for line in found)
    assert {line.split(":")[0] for line in found} == {sibling, other_dir, renamed}


def test_main_reports_and_exits_nonzero(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = write(repo, "ok.txt", TEXT)
    bad = write(repo, "x.parquet", PARQUET)
    assert guard.main([clean]) == 0
    assert guard.main([clean, bad]) == 1
    assert "object storage" in capsys.readouterr().err


def test_the_repo_has_no_tracked_binary_file() -> None:
    root = Path(__file__).resolve().parents[2]
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.split(b"\0")
    paths = [str(root / name.decode()) for name in listed if name]
    assert guard.violations(paths) == []
