from __future__ import annotations

import json
from collections.abc import Mapping
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


def episode(kind: str = "evaluation", sealed: object = True, data_class: str = "synthetic") -> str:
    collection: dict[str, object] = {"kind": kind, "run_id": "r1"}
    if sealed is not None:
        collection["sealed"] = sealed
    return json.dumps({"data_class": data_class, "provenance": {"collection": collection}})


MARKER = "fixtures/contracts/v1/episode/valid/evaluation-sealed.json"
SEALED = guard.SEALED


@pytest.mark.parametrize(
    "name",
    [
        "packages/x/episode.json",  # real-looking, outside fixtures/
        "fixtures/other/episode.json",  # in fixtures/ but not the corpus
        "fixtures/contracts/v1/episode/other/e.json",  # corpus, not a listed marker
        "fixtures/contracts/v1/episode/valid/new-sealed.json",  # same dir, not listed
        "fixtures/contracts/v2/episode/invalid/e.json",  # other version, not listed
        "fixtures/contracts/v1/episode/valid/deep/e.json",  # too deep
        "fixtures/contracts/v1/team/valid/e.json",  # other contract
        "fixtures/contracts/latest/episode/valid/e.json",  # not v<N>
        "fixtures/contracts/v1/episode/valid/e.jsonl",  # JSONL never
    ],
)
def test_sealed_episode_outside_the_marker_path_is_rejected(repo: Path, name: str) -> None:
    path = write(repo, name, episode() + "\n")
    assert guard.violations([path]) == [f"{path}: {SEALED}"]


@pytest.mark.parametrize("name", sorted(guard.SEALED_MARKERS))
def test_sealed_marker_in_the_corpus_is_allowed(repo: Path, name: str) -> None:
    path = write(repo, name, episode())
    assert guard.violations([path]) == []


def test_sealed_marker_needs_a_synthetic_root(repo: Path) -> None:
    unlabelled = write(
        repo,
        MARKER,
        json.dumps({"provenance": {"collection": {"kind": "evaluation", "sealed": True}}}),
    )
    assert guard.violations([unlabelled]) == [f"{unlabelled}: {SEALED}"]


def test_private_sealed_episode_is_rejected_everywhere(repo: Path) -> None:
    for name in ("packages/x/e.json", MARKER):
        path = write(repo, name, episode(data_class="private"))
        assert guard.violations([path]) == sorted([f"{path}: {PRIVATE}", f"{path}: {SEALED}"])


@pytest.mark.parametrize(
    ("kind", "sealed"),
    [
        ("evaluation", False),
        ("evaluation", None),
        ("evaluation", "true"),
        ("league", True),  # the schema refuses it; the guard is about evaluation
        ("sparring", None),
    ],
)
def test_other_collections_are_allowed(repo: Path, kind: str, sealed: object) -> None:
    path = write(repo, "packages/x/e.json", episode(kind, sealed))
    assert guard.violations([path]) == []


def test_episode_without_provenance_is_allowed(repo: Path) -> None:
    path = write(repo, "packages/x/e.json", json.dumps({"data_class": "synthetic"}))
    assert guard.violations([path]) == []


def test_sealed_in_a_list_or_later_jsonl_line_or_list_root_is_rejected(repo: Path) -> None:
    nested = write(repo, "packages/x/n.json", json.dumps({"runs": [json.loads(episode())]}))
    lines = write(repo, "packages/x/l.jsonl", "{}\n" + episode() + "\n")
    list_root = write(repo, MARKER, json.dumps([json.loads(episode())]))
    assert guard.violations([nested, lines, list_root]) == [
        f"{nested}: {SEALED}",
        f"{lines}: {SEALED}",
        f"{list_root}: {SEALED}",
    ]


def test_allowlist_matches_the_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    sealed = set()
    for file in (root / "fixtures").rglob("*.json*"):
        rel = file.relative_to(root).as_posix()
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".json":
            docs = [json.loads(text)]
        else:
            docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        if any(guard._is_sealed(obj) for doc in docs for obj in guard._walk(doc)):
            sealed.add(rel)
    assert sealed == set(guard.SEALED_MARKERS)
    assert all((root / marker).is_file() for marker in guard.SEALED_MARKERS)


# ADR-013 §4, §8: the team system's records and manifests.
ROOT = Path(__file__).resolve().parents[2]
RECORD = {
    "schema_version": 1,
    "data_class": "synthetic",
    "team_hash": "a" * 64,
    "paste": "Garchomp\nAbility: Rough Skin\n",
}
DATASET = {
    "data_class": "synthetic",
    "dataset_id": "d1",
    "generator": {"config_hash": "a" * 64},
    "key": {"kind": "public", "generation_key": "b" * 64},
    "splits": [],
}
SNAPSHOT = {"data_class": "synthetic", "snapshot_id": "s1", "shards": [], "cells": []}
UNLABELLED_RECORD = guard.UNLABELLED_RECORD
UNKNOWN_KEY_KIND = guard.UNKNOWN_KEY_KIND


def without(document: Mapping[str, object], key: str) -> dict[str, object]:
    return {k: v for k, v in document.items() if k != key}


@pytest.mark.parametrize("document", [RECORD, DATASET, SNAPSHOT])
@pytest.mark.parametrize("in_fixtures", [False, True])
def test_labelled_team_records_and_manifests_pass_everywhere(
    document: dict[str, object], in_fixtures: bool
) -> None:
    assert guard.problems_in(document, in_fixtures) == []


@pytest.mark.parametrize("record", [RECORD, DATASET, SNAPSHOT])
@pytest.mark.parametrize("in_fixtures", [False, True])
def test_a_record_without_data_class_is_rejected_even_in_fixtures(
    record: dict[str, object], in_fixtures: bool
) -> None:
    # Negative controls: remove the label, or set it to anything but "synthetic".
    assert guard.problems_in(without(record, "data_class"), in_fixtures) == [UNLABELLED_RECORD]
    assert guard.problems_in({**record, "data_class": "public"}, in_fixtures) == [UNLABELLED_RECORD]
    assert guard.problems_in({**record, "data_class": None}, in_fixtures) == [UNLABELLED_RECORD]


def test_the_label_must_sit_on_the_record_not_the_document_root() -> None:
    assert guard.problems_in(
        {"data_class": "synthetic", "teams": [without(RECORD, "data_class")]}, True
    ) == [UNLABELLED_RECORD]


def test_a_private_record_is_rejected_as_private() -> None:
    assert guard.problems_in({**RECORD, "data_class": "private"}, False) == [
        PRIVATE,
        UNLABELLED_RECORD,
    ]


@pytest.mark.parametrize(
    "key",
    [
        {"kind": "sealed", "generation_key": "b" * 64},
        {"kind": "secret"},
        {"kind": "Public"},
        {"generation_key": "b" * 64},
        "b" * 64,
        None,
        [],
    ],
)
def test_a_dataset_manifest_key_must_be_a_public_key(key: object) -> None:
    assert guard.problems_in({**DATASET, "key": key}, True) == [UNKNOWN_KEY_KIND]


def test_a_dataset_manifest_without_a_key_is_not_a_manifest_but_a_schema_failure() -> None:
    # The shape needs `key`; a manifest missing it is refused by its schema, and
    # carries no key to leak. An unlabelled one is still not shaped like the others.
    assert guard.problems_in(without(DATASET, "key"), True) == []


def test_a_record_inside_a_jsonl_line_is_checked_per_line(repo: Path) -> None:
    lines = [RECORD, without(RECORD, "data_class")]
    path = write(
        repo, "fixtures/teams/dev.jsonl", "".join(json.dumps(line) + "\n" for line in lines)
    )
    assert guard.violations([path]) == [f"{path}: {UNLABELLED_RECORD}"]


def test_a_json_schema_describing_the_records_is_not_a_record() -> None:
    properties = {"paste": {"type": "string"}, "team_hash": {"type": "string"}}
    assert guard.problems_in({"properties": properties}, False) == []
    assert (
        guard.problems_in({"properties": {"shards": {}, "cells": {}, "snapshot_id": {}}}, False)
        == []
    )
    schema: dict[str, object] = {
        "properties": {"dataset_id": {}, "generator": {}, "key": {}, "splits": {}}
    }
    assert guard.problems_in(schema, False) == []


@pytest.mark.parametrize(
    "path", sorted((ROOT / "fixtures/contracts/v1").glob("team-*/valid/*.json"))
)
def test_the_valid_corpus_samples_pass(path: Path) -> None:
    assert guard.violations([str(path)]) == []


def test_every_team_record_in_the_corpus_is_labelled_synthetic() -> None:
    labelled = 0
    for path in (ROOT / "fixtures/contracts/v1").glob("team-*/*/*.json"):
        for obj in guard._walk(json.loads(path.read_text(encoding="utf-8"))):
            if guard._is_team_record(obj) or guard._is_snapshot_manifest(obj):
                assert obj.get("data_class") == "synthetic" or "invalid" in path.parts[-2]
                labelled += 1
    assert labelled > 0
