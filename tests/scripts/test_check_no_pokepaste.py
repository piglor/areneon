from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts import check_no_pokepaste as guard

# pokepaste-guard: allow  (synthetic set used only to exercise the guard)
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


def test_detects_set() -> None:
    assert guard.looks_like_pokepaste(SET)


@pytest.mark.parametrize(
    "text",
    [
        "Ability: Static\n- Fake Out\n",  # one move only
        "- Fake Out\n- Protect\nJolly Nature\n",  # no ability
        "Ability: Static\n- Fake Out\n- Protect\n",  # no stats or nature
        "# Notes\n- item one\n- item two\n",
    ],
)
def test_ignores_non_sets(text: str) -> None:
    assert not guard.looks_like_pokepaste(text)


def test_nature_without_stats_counts() -> None:
    assert guard.looks_like_pokepaste("Ability: Static\n- A\n- B\nTimid Nature\n")


@given(st.text(alphabet=st.characters(blacklist_characters=":"), max_size=200))
def test_no_colon_never_matches(text: str) -> None:
    assert not guard.looks_like_pokepaste(text)


def test_allowed_paths() -> None:
    assert guard.is_allowed_path("fixtures/teams/sample.txt")
    assert not guard.is_allowed_path("docs/team.txt")


def write(tmp: Path, name: str, text: str) -> str:
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_violations_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    bad = write(tmp_path, "team.txt", SET)
    marked = write(tmp_path, "doc.md", SET + "\npokepaste-guard: allow\n")
    fixture = write(tmp_path, "fixtures/sample.txt", SET)
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00")
    paths = [bad, marked, "fixtures/sample.txt", str(binary), "missing.txt", str(tmp_path)]
    assert Path(fixture).exists()
    assert guard.violations(paths) == [bad]
    assert guard.main(paths) == 1
    assert guard.main([marked]) == 0
