# pokepaste-guard: allow  (synthetic sets used only to exercise the guard)
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_no_pokepaste as guard

SET = """Pikachu @ Light Ball
Ability: Static
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Fake Out
- Volt Tackle
- Protect
- Feint
"""

# No Ability line: still a team (item header + stat spread).
SET_WITHOUT_ABILITY = """Pikachu @ Light Ball
SPs: 32 Atk / 32 Spe
- Fake Out
- Protect
"""


@pytest.mark.parametrize("text", [SET, SET_WITHOUT_ABILITY, "Ability: X\n- A\n- B\nTimid Nature\n"])
def test_detects_sets(text: str) -> None:
    assert guard.looks_like_pokepaste(text)


@pytest.mark.parametrize(
    "text",
    [
        "Ability: Static\nJolly Nature\n- Fake Out\n",  # one move only
        "Ability: Static\n- Fake Out\n- Protect\n",  # one marker only
        "# Notes\n- item one\n- item two\n",
        "Contact: a @ b.com\n- one\n- two\n",  # '@' line alone is one marker
    ],
)
def test_ignores_non_sets(text: str) -> None:
    assert not guard.looks_like_pokepaste(text)


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("fixtures/teams/sample.txt", True),
        ("./fixtures/sample.txt", True),
        ("docs/fixtures/team.txt", False),
        ("docs/team.txt", False),
        (".", False),
    ],
)
def test_allowed_paths(path: str, allowed: bool) -> None:
    assert guard.is_allowed_path(path) is allowed


def test_allow_marker_only_counts_in_header() -> None:
    assert guard.has_allow_marker("# pokepaste-guard: allow\n" + SET)
    assert not guard.has_allow_marker(SET + "\n# pokepaste-guard: allow\n")


def write(tmp: Path, name: str, text: str) -> str:
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_violations_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bad = write(tmp_path, "team.txt", SET)
    buried = write(tmp_path, "buried.txt", SET + "\npokepaste-guard: allow\n")
    marked = write(tmp_path, "doc.md", "pokepaste-guard: allow\n" + SET)
    write(tmp_path, "fixtures/sample.txt", SET)
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00")
    paths = [bad, buried, marked, "fixtures/sample.txt", str(binary), "missing.txt", str(tmp_path)]
    assert guard.violations(paths) == [bad, buried]
    assert guard.main(paths) == 1
    assert guard.main([marked]) == 0
