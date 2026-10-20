from __future__ import annotations

import base64
import sqlite3
import subprocess
from pathlib import Path

import pytest

from scripts import check_binary_files as guard

TEXT = "# notes\n\nplain UTF-8, with accents: é ü ✓\n".encode()
PARQUET = b"PAR1" + bytes(range(256)) + b"PAR1"
AGE = b"age-encryption.org/v1\n-> X25519 abc\n" + bytes([0x80, 0xFF, 0x00])
SIGNATURE = "starts with a Parquet, SQLite, age or DuckDB signature"
ARMOR = "holds armored age text"
WRAPPED = "holds a base64-wrapped Parquet, SQLite, age or DuckDB file"
# Built in two parts: this file must not contain the header the guard forbids.
ARMOR_HEADER = b"-----BEGIN AGE " + b"ENCRYPTED FILE-----"
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
        "teams/shard.parquet.gz",
        "teams/shard.parquet.zst",
        "x.sqlite3",
        "x.sqlite-wal",
        "x.sqlite3-shm",
        "x.db-wal",
        "x.DB-SHM",
        "x.ddb",
        "teams/shard.parquet.snappy",
        "teams/shard.parquet.bz2",
        "x.db-journal",
        "x.sqlite-journal",
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
        b"12345678DUCK and plain text",
    ],
)
def test_a_signature_alone_is_rejected_even_in_valid_text(repo: Path, data: bytes) -> None:
    path = write(repo, "notes.md", data)
    assert guard.violations([path]) == [f"{path}: {SIGNATURE}"]


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


def sqlite_bytes(directory: Path) -> bytes:
    """A real SQLite database, generated at test time (never committed)."""
    db = directory / "generated.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (a TEXT)")
    con.execute("INSERT INTO t VALUES ('x')")
    con.commit()
    con.close()
    return db.read_bytes()


def test_a_real_sqlite_database_is_rejected_under_any_name(repo: Path, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    data = sqlite_bytes(scratch)
    assert data.startswith(b"SQLite format 3\0")
    path = write(repo, "docs/notes.txt", data)
    assert guard.violations([path]) == [
        f"{path}: is not valid UTF-8",
        f"{path}: holds a NUL byte",
        f"{path}: {SIGNATURE}",
    ]


def guard_reasons(data: bytes) -> list[str]:
    return guard.reasons_for("docs/notes.txt", data)


def test_a_sqlite_header_alone_is_a_signature(repo: Path) -> None:
    # Text up to the NUL: the NUL is also reported, the signature must be too.
    assert SIGNATURE in guard_reasons(b"SQLite format 3\0")


def test_the_encrypted_parquet_footer_signature_is_a_signature(repo: Path) -> None:
    assert SIGNATURE in guard_reasons(b"PARE" + bytes(range(0x80, 0xC0)) + b"PARE")
    assert SIGNATURE in guard_reasons(b"PARE\0")


def test_text_starting_with_pare_is_not_an_encrypted_footer(repo: Path) -> None:
    # A CSV may start PARENT,CHILD; negative control for the binary requirement.
    for text in (b"PARENT,CHILD\n1,2\n", b"PARE but otherwise plain text", "PARE é".encode()):
        assert guard.violations([write(repo, "data.csv", text)]) == []


def test_text_starting_with_par1_is_refused_and_allowed_by_exact_path(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(repo, "docs/par1.txt", b"PAR1 is a word here\n")
    assert guard.violations([path]) == [f"{path}: {SIGNATURE}"]
    monkeypatch.setattr(guard, "ALLOWED", frozenset({path}))
    assert guard.violations([path]) == []


def test_armored_age_text_is_rejected_anywhere_in_a_file(repo: Path) -> None:
    body = b"YWdlLWVuY3J5cHRpb24ub3Jn\n-----END AGE ENCRYPTED FILE-----\n"
    for prefix in (b"", b"some notes\n", b"x" * 5000 + b"\n"):
        path = write(repo, "docs/notes.md", prefix + ARMOR_HEADER + b"\n" + body)
        assert guard.violations([path]) == [f"{path}: {ARMOR}"], prefix
    # Negative control: the same text without the header is fine.
    plain = write(
        repo, "docs/ok.md", b"YWdlLWVuY3J5cHRpb24ub3Jn\n-----END AGE ENCRYPTED FILE-----\n"
    )
    assert guard.violations([plain]) == []


def test_the_guard_source_does_not_hold_the_header_it_forbids() -> None:
    assert ARMOR_HEADER not in Path(guard.__file__).read_bytes()


def wrap(data: bytes, width: int | None = None, urlsafe: bool = False) -> bytes:
    text = (base64.urlsafe_b64encode if urlsafe else base64.b64encode)(data).decode()
    if width is not None:
        text = "\n".join(text[i : i + width] for i in range(0, len(text), width))
    return text.encode()


BLOBS = {
    "parquet": PARQUET,
    "pare": b"PARE" + bytes(range(0x80, 0xFF)),
    "age": b"age-encryption.org/v1\n-> X25519 abc\n--- def\n" + bytes(range(1, 200)),
    "sqlite": b"SQLite format 3\0" + bytes(range(1, 200)),
    "duckdb": bytes(8) + b"DUCK" + bytes(range(1, 200)),
}


@pytest.mark.parametrize("name", sorted(BLOBS))
@pytest.mark.parametrize("width", [None, 64, 76])
@pytest.mark.parametrize("urlsafe", [False, True])
def test_a_base64_wrapped_binary_is_rejected_in_text(
    repo: Path, name: str, width: int | None, urlsafe: bool
) -> None:
    blob = wrap(BLOBS[name], width, urlsafe)
    for text in (
        blob,
        b'{"blob": "' + blob.replace(b"\n", b"\\n") + b'"}' if width is None else blob,
        b"-----BEGIN DATA-----\n" + blob + b"\n-----END DATA-----\n",
        b"data:application/octet-stream;base64," + blob,
    ):
        path = write(repo, "docs/notes.md", text)
        assert guard.violations([path]) == [f"{path}: {WRAPPED}"], text[:60]


def test_a_base64_wrapped_binary_after_other_base64_is_rejected(repo: Path) -> None:
    # Four-character aligned inside one run.
    path = write(repo, "docs/notes.md", wrap(b"prefix bytes" + PARQUET))
    assert guard.violations([path]) == [f"{path}: {WRAPPED}"]


@pytest.mark.parametrize(
    "data",
    [
        wrap(b"just some ordinary text, not a signature at all" * 3),
        wrap(bytes(range(1, 250))),  # random-looking bytes without a signature
        b"sha512-" + wrap(bytes(range(7, 71))),  # a lockfile integrity value
        b"short UEFSMQ token",
        b"abcdefghijklmnopqrstuvwxyz0123456789 words and more words",
        b"UEFSMQ" + b"=" * 3,  # too short to be a run
    ],
)
def test_ordinary_base64_and_words_are_text(repo: Path, data: bytes) -> None:
    assert guard.violations([write(repo, "docs/notes.md", data)]) == []


def test_a_run_with_padding_in_the_middle_is_not_base64_and_is_skipped(repo: Path) -> None:
    # Padding may only end the data; a wrapped run with it mid-way cannot decode.
    text = b"A" * 18 + b"==\n" + b"A" * 20 + b"\n"
    assert guard.reasons_for("x.md", text) == []


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
