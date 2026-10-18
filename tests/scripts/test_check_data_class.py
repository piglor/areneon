from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_data_class as guard

TEAM = {"species": "Garchomp", "moves": ["Earthquake"], "evs": {"atk": 32}}
REQUEST_MON = {"details": "Garchomp, L50", "stats": {"atk": 182}, "moves": ["earthquake"]}


PRIVATE = guard.PRIVATE
UNLABELLED = guard.UNLABELLED_TEAM


def write(root: Path, name: str, text: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(root))


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("document", "in_fixtures", "expected"),
    [
        ({"data_class": "synthetic", "team": [TEAM]}, False, []),
        ({"team": [TEAM]}, True, []),
        ({"team": [TEAM]}, False, [UNLABELLED]),
        ({"data_class": "synthetic", "inner": {"data_class": "private"}}, True, [PRIVATE]),
        # The label must sit at the document root, not on the team itself.
        ([{"data_class": "synthetic", **TEAM}], False, [UNLABELLED]),
        ({"species": "Garchomp", "moves": []}, False, []),
        # A raw |request| side reveals the spread through stats.
        ({"request": {"side": {"pokemon": [REQUEST_MON]}}}, False, [UNLABELLED]),
        ({"details": "Garchomp, L50", "moves": []}, False, []),
        # A JSON Schema describing a team is not a team.
        ({"properties": {"species": {}, "moves": {"type": "array"}, "evs": {}}}, False, []),
        ("plain", False, []),
    ],
)
def test_problems_in(document: object, in_fixtures: bool, expected: list[str]) -> None:
    assert guard.problems_in(document, in_fixtures) == expected


def test_private_data_is_rejected_even_in_fixtures(repo: Path) -> None:
    path = write(repo, "fixtures/teams/real.json", json.dumps({"data_class": "private", **TEAM}))
    assert guard.violations([path]) == [f"{path}: {PRIVATE}"]


def test_unlabelled_team_outside_fixtures_is_rejected(repo: Path) -> None:
    path = write(repo, "packages/x/team.json", json.dumps({"bring": [TEAM]}))
    assert guard.violations([path]) == [f"{path}: {UNLABELLED}"]


def test_labelled_and_teamless_files_pass(repo: Path) -> None:
    example = {"data_class": "synthetic", "team": [TEAM]}
    labelled = write(repo, "docs/example.json", json.dumps(example))
    teamless = write(repo, "rules/formats/x.json", json.dumps({"format_id": "x"}))
    assert guard.violations([labelled, teamless, "README.md", "missing.json"]) == []


def test_jsonl_lines_are_checked_one_by_one(repo: Path) -> None:
    lines = [{"data_class": "synthetic", "team": TEAM}, {"team": TEAM}]
    path = write(repo, "episodes/run.jsonl", "".join(json.dumps(line) + "\n\n" for line in lines))
    assert guard.violations([path]) == [f"{path}: {UNLABELLED}"]


def test_unreadable_json_fails_closed(repo: Path) -> None:
    (repo / "bad.json").write_text("{", encoding="utf-8")
    (repo / "latin1.json").write_bytes(b'{"a": "\xe9"}')
    (repo / "adir.json").mkdir()
    assert guard.violations(["bad.json", "latin1.json", "adir.json"]) == [
        "bad.json: is not valid JSON, so it cannot be checked",
        "latin1.json: is not valid JSON, so it cannot be checked",
    ]


def test_main_reports_and_exits_nonzero(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = write(repo, "ok.json", json.dumps({"a": 1}))
    bad = write(repo, "team.json", json.dumps(TEAM))
    assert guard.main([clean]) == 0
    assert guard.main([clean, bad]) == 1
    assert "private-data/" in capsys.readouterr().err
