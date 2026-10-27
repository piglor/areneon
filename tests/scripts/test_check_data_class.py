from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

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
    assert SEALED in [r.split(": ", 1)[1] for r in guard.violations([path])]


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
    # A paste-shaped record outside fixtures/ is also an unlabelled team (ADR-015 §6).
    extra = [UNLABELLED] if record is RECORD and not in_fixtures else []
    expected = [UNLABELLED_RECORD, *extra]
    assert guard.problems_in(without(record, "data_class"), in_fixtures) == expected
    assert guard.problems_in({**record, "data_class": "public"}, in_fixtures) == expected
    assert guard.problems_in({**record, "data_class": None}, in_fixtures) == expected


def test_the_label_must_sit_on_the_record_not_the_document_root() -> None:
    assert guard.problems_in(
        {"data_class": "synthetic", "teams": [without(RECORD, "data_class")]}, True
    ) == [UNLABELLED_RECORD]


def test_a_private_record_is_rejected_as_private() -> None:
    assert guard.problems_in({**RECORD, "data_class": "private"}, True) == [
        PRIVATE,
        UNLABELLED_RECORD,
    ]
    assert guard.problems_in({**RECORD, "data_class": "private"}, False) == [
        PRIVATE,
        UNLABELLED_RECORD,
        UNLABELLED,
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


# ADR-015 §6: the paste shape and the private sample markers.
PASTE_ONLY = {"paste": "Garchomp\nAbility: Rough Skin\n"}
PRIVATE_RECORD = {
    "schema_version": 1,
    "data_class": "private",
    "team_id": "0" * 32,
    "revision": 1,
    "supersedes_record_hash": None,
    "team_hash": "a" * 64,
    "canonical_team_hash": "b" * 64,
    "paste": "Garchomp\nAbility: Rough Skin\n",
    "status": "active",
    "tags": ["synthetic"],
    "labels": [{"label_id": "fake_out", "value": "yes"}],
    "holdout": False,
}
SHAPE = guard.MARKER_SHAPE
MARKER = "fixtures/contracts/v1/private-team-record/valid/sample.json"
NOT_IN_CORPUS = guard.PASTE_NOT_IN_CORPUS


@pytest.fixture
def corpus(repo: Path, monkeypatch: pytest.MonkeyPatch) -> frozenset[str]:
    write(repo, "corpus/a.txt", PASTE_ONLY["paste"])
    write(repo, "corpus/d/teams-dev.jsonl", json.dumps({"paste": "From a jsonl line\n"}) + "\n")
    write(repo, "corpus/b.json", json.dumps({"data_class": "synthetic", **PASTE_ONLY}))
    monkeypatch.setattr(guard, "TEAM_CORPUS", repo / "corpus")
    monkeypatch.setattr(guard, "PRIVATE_SAMPLE_MARKERS", frozenset({MARKER}))
    return guard.team_corpus()


def test_a_paste_without_other_team_keys_is_team_shaped() -> None:
    assert guard.problems_in(PASTE_ONLY, False) == [UNLABELLED]
    assert guard.problems_in({"x": [PASTE_ONLY]}, False) == [UNLABELLED]
    assert guard.problems_in({"data_class": "synthetic", **PASTE_ONLY}, False) == []
    # Negative controls: inside fixtures, a schema, a non-string paste.
    assert guard.problems_in(PASTE_ONLY, True) == []
    assert guard.problems_in({"properties": {"paste": {"type": "string"}}}, False) == []
    assert guard.problems_in({"paste": 3}, False) == []


def test_the_paste_shape_is_enforced_on_files(repo: Path) -> None:
    path = write(repo, "docs/example.json", json.dumps(PASTE_ONLY))
    assert guard.violations([path]) == [f"{path}: {UNLABELLED}"]


def test_the_corpus_is_the_committed_one() -> None:
    corpus = guard.team_corpus()
    assert (ROOT / "fixtures/teams/gen9championsvgc2026regmc/legal-synthetic.txt").read_text(
        encoding="utf-8"
    ) in corpus


def test_the_markers_are_the_record_samples_of_the_contract_corpus() -> None:
    assert guard.PRIVATE_SAMPLE_MARKERS
    for marker in guard.PRIVATE_SAMPLE_MARKERS:
        path = PurePosixPath(marker)
        assert path.parts[:4] == ("fixtures", "contracts", "v1", "private-team-record")
        assert path.parts[4] in ("valid", "reader-invalid")
        assert (ROOT / marker).is_file(), marker


def test_the_marked_samples_pass_the_guard_as_committed() -> None:
    assert guard.violations(sorted(guard.PRIVATE_SAMPLE_MARKERS)) == []


def test_the_marker_record_keys_and_types_follow_the_contract_schema() -> None:
    schema = json.loads(
        (ROOT / "packages/schema/contracts/v1/private-team-record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    props = schema["properties"]
    assert guard.PRIVATE_RECORD_KEYS == set(props) == set(schema["required"])
    assert props["team_id"]["pattern"] == f"^{guard.HEX32.pattern}$"
    assert props["supersedes_record_hash"]["pattern"] == f"^{guard.HEX64.pattern}$"
    assert props["revision"]["maximum"] == guard.MAX_REVISION
    assert props["revision"]["minimum"] == 1
    assert props["holdout"]["type"] == "boolean"
    assert props["status"]["enum"] == ["active", "retired"]
    assert props["tags"]["maxItems"] == guard.MAX_TAGS
    assert props["tags"]["items"]["pattern"] == f"^{guard.TAG.pattern}$"
    assert props["labels"]["maxItems"] == guard.MAX_LABELS
    label = props["labels"]["items"]["properties"]
    token = f"^{guard.LABEL_TOKEN.pattern}$"
    assert label["label_id"]["pattern"] == label["value"]["pattern"] == token


def test_a_marked_private_record_with_corpus_pastes_passes(
    repo: Path, corpus: frozenset[str]
) -> None:
    path = write(repo, MARKER, json.dumps(PRIVATE_RECORD))
    assert guard.violations([path]) == []
    in_json = write(repo, MARKER, json.dumps({**PRIVATE_RECORD, "paste": PASTE_ONLY["paste"]}))
    assert guard.violations([in_json]) == []


def test_the_same_record_at_an_unmarked_path_is_rejected(
    repo: Path, corpus: frozenset[str]
) -> None:
    path = write(repo, MARKER.replace("sample", "other"), json.dumps(PRIVATE_RECORD))
    assert guard.violations([path]) == [f"{path}: {PRIVATE}", f"{path}: {UNLABELLED_RECORD}"]


def test_the_marker_is_an_exact_file_not_a_directory(repo: Path, corpus: frozenset[str]) -> None:
    sibling = write(repo, MARKER + ".bak.json", json.dumps(PRIVATE_RECORD))
    nested = write(repo, "x/" + MARKER, json.dumps(PRIVATE_RECORD))
    parent = write(repo, str(Path(MARKER).parent / "sub" / "b.json"), json.dumps(PRIVATE_RECORD))
    for path in (sibling, nested, parent):
        assert PRIVATE in [r.split(": ", 1)[1] for r in guard.violations([path])]


@pytest.mark.parametrize(
    "record",
    [
        {**PRIVATE_RECORD, "paste": "Not a corpus paste\n"},
        {**PRIVATE_RECORD, "paste": ""},
        {k: v for k, v in PRIVATE_RECORD.items() if k != "paste"},
        {**PRIVATE_RECORD, "paste": 3},
        {**PRIVATE_RECORD, "extra": [{"paste": "Not a corpus paste\n"}]},
        {**PRIVATE_RECORD, "data_class": "synthetic"},
        [PRIVATE_RECORD],
    ],
)
def test_a_marked_file_must_be_a_private_record_with_only_corpus_pastes(
    repo: Path, corpus: frozenset[str], record: object
) -> None:
    path = write(repo, MARKER, json.dumps(record))
    assert NOT_IN_CORPUS in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_corpus_reads_text_json_and_jsonl_pastes(corpus: frozenset[str]) -> None:
    assert corpus == {PASTE_ONLY["paste"], "From a jsonl line\n"}


def test_the_marker_record_keys_are_the_private_team_record_shape() -> None:
    assert set(PRIVATE_RECORD) == guard.PRIVATE_RECORD_KEYS


def test_the_marker_waives_only_the_root_private_label(repo: Path, corpus: frozenset[str]) -> None:
    # An extra key is refused (the shape), and its private label is not waived.
    nested = {**PRIVATE_RECORD, "other": {"data_class": "private"}}
    path = write(repo, MARKER, json.dumps(nested))
    assert {PRIVATE, SHAPE} <= {r.split(": ", 1)[1] for r in guard.violations([path])}


REAL = "Garchomp @ Sitrus Berry\nAbility: Rough Skin\n- Earthquake\n"


@pytest.mark.parametrize(
    "extra",
    [
        # The reviewer's three attacks, and variants.
        {"sets": [{"species": "Garchomp", "moves": ["Earthquake"], "evs": {"atk": 32}}]},
        {"pastes": [REAL]},
        {"notes": "Garchomp @ Sitrus Berry"},
        {"species": "Garchomp", "moves": ["Earthquake"], "evs": {"atk": 32}},
        {"nested": {"deeper": [{"paste": PASTE_ONLY["paste"]}]}},
    ],
)
def test_a_marked_file_cannot_carry_extra_content(
    repo: Path, corpus: frozenset[str], extra: dict[str, object]
) -> None:
    path = write(repo, MARKER, json.dumps({**PRIVATE_RECORD, **extra}))
    assert SHAPE in [r.split(": ", 1)[1] for r in guard.violations([path])]


@pytest.mark.parametrize(
    "field",
    [
        {"tags": ["Ability: Rough Skin"]},
        {"tags": ["- Earthquake"]},
        {"labels": [{"label_id": "x", "value": "Garchomp @ Sitrus Berry"}]},
        {"status": "Garchomp @ Sitrus Berry"},
        {"team_id": "line one\nAbility: Intimidate"},
    ],
)
def test_a_paste_hidden_in_an_allowed_key_is_refused(
    repo: Path, corpus: frozenset[str], field: dict[str, object]
) -> None:
    path = write(repo, MARKER, json.dumps({**PRIVATE_RECORD, **field}))
    assert SHAPE in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_a_paste_hidden_in_a_key_name_is_refused(repo: Path, corpus: frozenset[str]) -> None:
    record = {**PRIVATE_RECORD, "labels": [{"Garchomp @ Sitrus Berry": "x"}]}
    path = write(repo, MARKER, json.dumps(record))
    assert SHAPE in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_a_key_missing_from_the_record_is_refused(repo: Path, corpus: frozenset[str]) -> None:
    record = {k: v for k, v in PRIVATE_RECORD.items() if k != "tags"}
    path = write(repo, MARKER, json.dumps(record))
    assert SHAPE in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_marker_does_not_waive_the_sealed_rule(repo: Path, corpus: frozenset[str]) -> None:
    sealed = {
        **PRIVATE_RECORD,
        "provenance": {"collection": {"kind": "evaluation", "sealed": True}},
    }
    path = write(repo, MARKER, json.dumps(sealed))
    assert SEALED in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_marker_does_not_waive_the_unlabelled_team_rule(
    repo: Path, monkeypatch: pytest.MonkeyPatch, corpus: frozenset[str]
) -> None:
    # Outside fixtures/ a team needs a synthetic root, which a private record lacks.
    outside = "packages/x/sample.json"
    monkeypatch.setattr(guard, "PRIVATE_SAMPLE_MARKERS", frozenset({outside}))
    path = write(repo, outside, json.dumps(PRIVATE_RECORD))
    assert guard.violations([path]) == [f"{path}: {UNLABELLED}"]


def test_a_marked_record_without_a_label_is_still_unlabelled(
    repo: Path, corpus: frozenset[str]
) -> None:
    record = {k: v for k, v in PRIVATE_RECORD.items() if k != "data_class"}
    path = write(repo, MARKER, json.dumps(record))
    assert {r.split(": ", 1)[1] for r in guard.violations([path])} == {
        UNLABELLED_RECORD,
        NOT_IN_CORPUS,
        SHAPE,
    }


def test_the_private_markers_match_the_repo() -> None:
    # The whole tracked tree, not only fixtures/: a private label anywhere needs a marker.
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.split(b"\0")
    private = set()
    for raw in listed:
        name = raw.decode()
        if not name.endswith(guard.SUFFIXES) or not (ROOT / name).is_file():
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        docs = guard._documents(name, text)
        if any(obj.get("data_class") == "private" for doc in docs for obj in guard._walk(doc)):
            private.add(name)
    every = (
        set(guard.PRIVATE_SAMPLE_MARKERS)
        | set(guard.PRIVATE_MANIFEST_MARKERS)
        | set(guard.PRIVATE_TEST_RECORD_MARKERS)
    )
    assert private == every
    assert not set(guard.PRIVATE_SAMPLE_MARKERS) & set(guard.PRIVATE_MANIFEST_MARKERS)
    assert not set(guard.PRIVATE_TEST_RECORD_MARKERS) & set(guard.PRIVATE_MANIFEST_MARKERS)
    assert not set(guard.PRIVATE_TEST_RECORD_MARKERS) & set(guard.PRIVATE_SAMPLE_MARKERS)
    assert all((ROOT / marker).is_file() for marker in every)
    assert all(marker.startswith("fixtures/") for marker in every)


BAD_TYPES = [
    ("schema_version", 2),
    ("schema_version", True),
    ("schema_version", "1"),
    ("data_class", "synthetic"),
    ("team_id", "0" * 31),
    ("team_id", "A" * 32),
    ("team_id", "g" * 32),
    ("revision", 0),
    ("revision", 2**31),
    ("revision", True),
    ("revision", 1.5),
    ("supersedes_record_hash", "a" * 63),
    ("supersedes_record_hash", ""),
    ("team_hash", "A" * 64),
    ("team_hash", None),
    ("canonical_team_hash", "a" * 65),
    ("status", "Active"),
    ("status", "deleted"),
    ("holdout", "false"),
    ("holdout", 0),
    ("holdout", None),
    ("tags", "synthetic"),
    ("tags", ["Garchomp"]),
    ("tags", ["-leading"]),
    ("tags", ["a" * 33]),
    ("tags", ["ok"] * 17),
    ("tags", [1]),
    ("labels", {"team": {"mon1": {"species": "Garchomp", "moves": "Earthquake, Dragon Claw"}}}),
    ("labels", [{"label_id": "x", "value": "y", "extra": "z"}]),
    ("labels", [{"label_id": "x"}]),
    ("labels", [{"label_id": "x", "value": "has space"}]),
    ("labels", [{"label_id": "x" * 129, "value": "y"}]),
    ("labels", [{"label_id": "x", "value": "y"}] * 65),
    ("labels", ["Garchomp"]),
    ("labels", {"m": ["Earthquake", "Dragon Claw"]}),
    # Opaque encodings of a paste in a free-form field.
    ("team_id", "R2FyY2hvbXAgQCBTaXRydXMgQmVycnk="),
    ("status", "47617263686f6d7020402053697472757320426572727920"),
    # Look-alikes of the line-shaped signatures.
    ("tags", ["Garchomp\u00a0@\u00a0Sitrus Berry"]),
    ("tags", ["Garchomp \uff20 Sitrus Berry"]),
    ("tags", ["Ability:Rough Skin"]),
    ("tags", ["ability: rough skin"]),
    ("tags", ["-Earthquake"]),
    ("tags", ["\u2013 Earthquake"]),
    ("tags", ["EVs: 4 HP / 252 Atk"]),
    ("tags", ["Adamant Nature"]),
]


@pytest.mark.parametrize(("key", "value"), BAD_TYPES)
def test_every_marked_value_is_type_checked(
    repo: Path, corpus: frozenset[str], key: str, value: object
) -> None:
    path = write(repo, MARKER, json.dumps({**PRIVATE_RECORD, key: value}))
    assert SHAPE in [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_record_with_every_field_valid_passes_and_edge_values_pass(
    repo: Path, corpus: frozenset[str]
) -> None:
    edges = {
        "revision": 2**31 - 1,
        "supersedes_record_hash": "c" * 64,
        "status": "retired",
        "tags": ["a", "a" + "b" * 31] + ["x"] * 14,
        "labels": [{"label_id": "A.b_c:d-e", "value": "0"}] * 64,
    }
    path = write(repo, MARKER, json.dumps({**PRIVATE_RECORD, **edges}))
    assert guard.violations([path]) == []


# Duplicate keys: json.loads keeps the last value, the earlier one stays in the bytes.
REAL_PASTE = json.dumps("Garchomp @ Sitrus Berry\nAbility: Rough Skin\n- Earthquake\n")
CORPUS_PASTE = json.dumps(PASTE_ONLY["paste"])


def record_text(**replace: str) -> str:
    fields = {k: json.dumps(v) for k, v in PRIVATE_RECORD.items()} | replace
    return "{" + ", ".join(f'"{k}": {v}' for k, v in fields.items()) + "}"


@pytest.mark.parametrize(
    "text",
    [
        # The reviewer's attack: a real paste first, the corpus paste last.
        record_text()[:-1] + f', "paste": {CORPUS_PASTE}}}',
        '{"paste": ' + REAL_PASTE + ", " + record_text()[1:],
        '{"data_class": "synthetic", ' + record_text()[1:],
        '{"a": {"b": 1, "b": 2}}',
        '{"list": [{"x": 1}, {"y": 1, "y": 2}]}',
    ],
)
@pytest.mark.parametrize("name", [MARKER, "docs/example.json", "fixtures/other/x.json"])
def test_duplicate_keys_are_refused_everywhere(
    repo: Path, corpus: frozenset[str], text: str, name: str
) -> None:
    path = write(repo, name, text)
    assert guard.violations([path]) == [f"{path}: {guard.DUPLICATE_KEY}"]


def test_duplicate_keys_in_one_jsonl_line_are_refused(repo: Path) -> None:
    path = write(repo, "episodes/run.jsonl", '{"a": 1}\n{"a": 1, "a": 2}\n')
    assert guard.violations([path]) == [f"{path}: {guard.DUPLICATE_KEY}"]
    clean = write(repo, "episodes/ok.jsonl", '{"a": 1}\n{"a": 2}\n')
    assert guard.violations([clean]) == []


def test_no_tracked_json_has_a_duplicate_key() -> None:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.split(b"\0")
    names = [n.decode() for n in listed if n.endswith((b".json", b".jsonl"))]
    for name in names:
        if (ROOT / name).is_file():
            guard._documents(name, (ROOT / name).read_text(encoding="utf-8"))


# ADR-013 slice (g): a sealed-set reveal or a sealed team is secret until the set retires.
REVEAL_DOC = {
    "schema_version": 1,
    "data_class": "synthetic",
    "set_id": "set-1",
    "set_manifest_hash": "a" * 64,
    "reveal": {"key": "b" * 64, "salt": "c" * 64},
    "files": {"teams_sha256": "d" * 64},
}
SEALED_TEAM_LINE = {"k": 3, "paste": "Garchomp\n", "stratum": 4}


@pytest.mark.parametrize("name", sorted(guard.REVEAL_ALLOWED))
def test_reveal_corpus_samples_are_allowed(repo: Path, name: str) -> None:
    assert guard.violations([write(repo, name, json.dumps(REVEAL_DOC))]) == []


@pytest.mark.parametrize(
    "name",
    [
        "packages/x/retired.json",
        "fixtures/teams/x/retired-manifest.json",
        "fixtures/contracts/v1/retired-sealed-manifest/valid/other.json",
        "fixtures/contracts/v1/retired-sealed-manifest/valid/revealed.jsonl",
    ],
)
def test_reveal_outside_the_reviewed_paths_is_rejected(repo: Path, name: str) -> None:
    path = write(repo, name, json.dumps(REVEAL_DOC))
    assert guard.violations([path]) == [f"{path}: {guard.REVEAL}"]


def test_reveal_needs_its_own_synthetic_label(repo: Path) -> None:
    doc = {key: value for key, value in REVEAL_DOC.items() if key != "data_class"}
    path = write(repo, sorted(guard.REVEAL_ALLOWED)[0], json.dumps(doc))
    assert guard.violations([path]) == [f"{path}: {guard.UNLABELLED_REVEAL}"]


def test_a_sealed_team_with_an_extra_key_is_still_a_sealed_team(repo: Path) -> None:
    path = write(repo, "fixtures/other/t.json", json.dumps({**SEALED_TEAM_LINE, "extra": 1}))
    assert guard.violations([path]) == [f"{path}: {guard.REVEAL}"]


def test_sealed_team_lines_are_rejected_wherever_they_are(repo: Path) -> None:
    line = json.dumps(SEALED_TEAM_LINE) + "\n"
    path = write(repo, "fixtures/teams/x/teams.jsonl", line)
    nested = write(repo, "fixtures/other/sealed.json", json.dumps({"teams": [SEALED_TEAM_LINE]}))
    assert guard.violations([path, nested]) == [
        f"{path}: {guard.REVEAL}",
        f"{nested}: {guard.REVEAL}",
    ]


@pytest.mark.parametrize(
    "document",
    [
        {"set_id": "s", "files": {}, "reveal": {"key": "k"}},  # no salt: not a reveal
        {"set_id": "s", "reveal": {"key": "k", "salt": "s"}},  # no files
        {"k": "1", "paste": "x", "stratum": 2},
        {"k": 1, "paste": 2, "stratum": 2},
    ],
)
def test_other_shapes_are_not_reveals(repo: Path, document: dict[str, object]) -> None:
    assert guard.violations([write(repo, "fixtures/other/o.json", json.dumps(document))]) == []


def test_reveal_allowlist_matches_the_repo() -> None:
    present = {
        file.relative_to(ROOT).as_posix()
        for file in (ROOT / "fixtures").rglob("*.json")
        if any(
            guard._is_reveal(obj)
            for obj in guard._walk(json.loads(file.read_text(encoding="utf-8")))
        )
    }
    assert present == set(guard.REVEAL_ALLOWED)


# ADR-015 section 2: the strict shape of a private snapshot manifest.
MANIFEST_MARKER = "fixtures/contracts/v1/private-snapshot-manifest/valid/sample.json"
MANIFEST_SHAPE = guard.MANIFEST_SHAPE
MANIFEST = json.loads(
    (ROOT / "fixtures/contracts/v1/private-snapshot-manifest/valid/extended.json").read_text(
        encoding="utf-8"
    )
)
MANIFEST_FIRST = json.loads(
    (ROOT / "fixtures/contracts/v1/private-snapshot-manifest/valid/first.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture
def marked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "PRIVATE_MANIFEST_MARKERS", frozenset({MANIFEST_MARKER}))


def manifest_reasons(repo: Path, document: object, name: str = MANIFEST_MARKER) -> list[str]:
    path = write(repo, name, json.dumps(document))
    return [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_manifest_markers_are_manifest_samples_of_the_contract_corpus() -> None:
    assert guard.PRIVATE_MANIFEST_MARKERS
    for marker in guard.PRIVATE_MANIFEST_MARKERS:
        path = PurePosixPath(marker)
        assert path.parts[:4] == ("fixtures", "contracts", "v1", "private-snapshot-manifest")
        assert path.parts[4] in ("valid", "reader-invalid")
        assert (ROOT / marker).is_file(), marker


def test_the_marked_manifest_samples_pass_the_guard_as_committed() -> None:
    assert guard.violations(sorted(guard.PRIVATE_MANIFEST_MARKERS)) == []
    invalid = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "fixtures/contracts/v1/private-snapshot-manifest/invalid").glob("*.json")
    )
    assert invalid
    assert guard.violations(invalid) == []


def test_the_manifest_spec_follows_the_contract_schema() -> None:
    schema = json.loads(
        (ROOT / "packages/schema/contracts/v1/private-snapshot-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    required: Any
    required, optional = guard.PRIVATE_MANIFEST_SPEC
    props = schema["properties"]
    assert set(required) == set(schema["required"])
    assert set(required) | set(optional) == set(props)
    assert set(optional) == {"parent_snapshot_hash", "first"}
    assert props["parent_snapshot_hash"]["pattern"] == f"^{guard.HEX64.pattern}$"
    assert props["first"]["const"] is True
    assert props["shards"]["maxItems"] == required["shards"][1]
    assert props["cells"]["maxItems"] == required["cells"][1]
    assert set(props["holdout"]["properties"]) == set(required["holdout"][0])
    assert props["holdout"]["properties"]["ppm"]["const"] == required["holdout"][0]["ppm"][1]
    assert props["holdout"]["properties"]["rule"]["const"] == required["holdout"][0]["rule"][1]
    assert props["writer"]["properties"]["version"]["pattern"] == f"^{guard.VERSION.pattern}$"
    shard = schema["$defs"]["Shard"]
    assert set(shard["properties"]) == set(required["shards"][2][0])
    assert set(shard["properties"]["cell_ranges"]["items"]["properties"]) == set(
        required["shards"][2][0]["cell_ranges"][2][0]
    )
    assert shard["properties"]["index"]["maximum"] == required["shards"][2][0]["index"][2]
    assert set(props["cells"]["items"]["properties"]) == set(required["cells"][2][0])
    assert set(schema["$defs"]["VersionManifest"]["properties"]) == set(required["manifest"][0])


@pytest.mark.usefixtures("marked")
def test_a_marked_manifest_in_both_lineage_forms_passes(repo: Path) -> None:
    assert manifest_reasons(repo, MANIFEST) == []
    assert manifest_reasons(repo, MANIFEST_FIRST) == []
    empty = {**MANIFEST_FIRST, "cells": [], "records": {"active": 0, "retired": 1}}
    assert manifest_reasons(repo, empty) == []


@pytest.mark.usefixtures("marked")
def test_the_same_manifest_at_an_unmarked_path_is_rejected(repo: Path) -> None:
    other = MANIFEST_MARKER.replace("sample", "other")
    assert manifest_reasons(repo, MANIFEST, other) == [PRIVATE, UNLABELLED_RECORD]


@pytest.mark.usefixtures("marked")
def test_the_manifest_marker_is_an_exact_file(repo: Path) -> None:
    for name in (MANIFEST_MARKER + ".bak.json", "x/" + MANIFEST_MARKER):
        assert PRIVATE in manifest_reasons(repo, MANIFEST, name)


@pytest.mark.usefixtures("marked")
def test_a_manifest_marker_does_not_free_a_private_record(repo: Path) -> None:
    # The two marker sets are separate: a record at a manifest marker is not a manifest.
    assert MANIFEST_SHAPE in manifest_reasons(repo, PRIVATE_RECORD)


EXTRA = [
    {"notes": "Garchomp @ Sitrus Berry"},
    {"paste": PASTE_ONLY["paste"]},
    {"pastes": ["Garchomp @ Sitrus Berry\nAbility: Rough Skin\n- Earthquake\n"]},
    {"species": "Garchomp", "moves": ["Earthquake"], "evs": {"atk": 32}},
    {"dataset_hashes": ["a" * 64]},
    {"uploaded_by": "x"},
    {"nested": {"data_class": "private"}},
]


@pytest.mark.usefixtures("marked")
@pytest.mark.parametrize("extra", EXTRA)
def test_a_manifest_with_an_extra_key_is_refused(repo: Path, extra: dict[str, object]) -> None:
    assert MANIFEST_SHAPE in manifest_reasons(repo, {**MANIFEST, **extra})


def with_path(change: Mapping[str, object], *path: str | int) -> dict[str, object]:
    """A copy of MANIFEST with the value at ``path`` replaced (or removed for a ``...`` marker)."""
    copy = json.loads(json.dumps(MANIFEST))
    node = copy
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = change["value"]
    return copy  # type: ignore[no-any-return]


FREE_TEXT = [
    (("snapshot_id",), "my team notes"),
    (("snapshot_id",), "Garchomp @ Sitrus Berry"),
    (("snapshot_id",), "Ability: Rough Skin"),
    (("snapshot_id",), "- Earthquake"),
    (("snapshot_id",), "line one\nline two"),
    (("snapshot_id",), "R2FyY2hvbXAgQCBTaXRydXMgQmVycnk=" + "A" * 200),
    (("manifest", "regulation"), "M C"),
    (("manifest", "format_id"), "Gen9 Champions"),
    (("manifest", "showdown_commit"), "a" * 39),
    (("writer", "name"), "duck db"),
    (("writer", "version"), "1.5.6 beta"),
    (("holdout", "rule"), "hmac-sha256-team-id/2"),
    (("holdout", "key"), "A" * 64),
    (("holdout", "ppm"), 300000),
    (("data_class",), "synthetic"),
    (("schema_version",), 2),
    (("schema_version",), True),
    (("parent_snapshot_hash",), "abc"),
    (("first",), False),
    (("first",), "true"),
    (("shard_key",), "z" * 64),
    (("label_set_hash",), None),
    (("records", "active"), -1),
    (("records", "retired"), 1.5),
    (("records", "active"), True),
    (("records", "active"), 2**53),
    (("parquet_schema_version",), 0),
    (("shards", 0, "index"), 4096),
    (("shards", 0, "split"), "dev"),
    (("shards", 0, "rows"), 0),
    (("shards", 0, "file_sha256"), "g" * 64),
    (("shards", 0, "cell_ranges", 0, "active_rows"), -1),
    (("shards", 0, "cell_ranges", 0, "stratum"), 65536),
    (("cells", 0, "split"), "dev"),
    (("cells", 0, "rows"), 0),
    (("shards",), "none"),
    (("cells",), {"0": 1}),
    (("holdout",), "hmac"),
    (("shards", 0), "Garchomp @ Sitrus Berry"),
]


@pytest.mark.usefixtures("marked")
@pytest.mark.parametrize(("path", "value"), FREE_TEXT)
def test_every_manifest_value_is_type_checked(
    repo: Path, path: tuple[str | int, ...], value: object
) -> None:
    document = with_path({"value": value}, *path)
    assert MANIFEST_SHAPE in manifest_reasons(repo, document)


@pytest.mark.usefixtures("marked")
def test_a_paste_hidden_in_a_manifest_key_name_is_refused(repo: Path) -> None:
    document = json.loads(json.dumps(MANIFEST))
    document["holdout"]["Garchomp @ Sitrus Berry"] = "x"
    assert MANIFEST_SHAPE in manifest_reasons(repo, document)
    document = json.loads(json.dumps(MANIFEST))
    document["shards"][0]["cell_ranges"][0]["Ability: Rough Skin"] = 1
    assert MANIFEST_SHAPE in manifest_reasons(repo, document)


@pytest.mark.usefixtures("marked")
def test_a_missing_manifest_key_or_a_wrong_container_is_refused(repo: Path) -> None:
    for key in guard.PRIVATE_MANIFEST_SPEC[0]:
        assert MANIFEST_SHAPE in manifest_reasons(
            repo, {k: v for k, v in MANIFEST.items() if k != key}
        )
    assert MANIFEST_SHAPE in manifest_reasons(repo, [MANIFEST])
    too_many = {**MANIFEST, "cells": [MANIFEST["cells"][0]] * 65537}
    assert MANIFEST_SHAPE in manifest_reasons(repo, too_many)


@pytest.mark.usefixtures("marked")
def test_a_marked_manifest_does_not_waive_a_sealed_marker_or_a_team(repo: Path) -> None:
    sealed = {**MANIFEST, "provenance": {"collection": {"kind": "evaluation", "sealed": True}}}
    reasons = manifest_reasons(repo, sealed)
    assert SEALED in reasons
    assert MANIFEST_SHAPE in reasons


def test_the_manifest_leaf_kinds() -> None:
    assert guard._leaf_ok("hex64", "a" * 64)
    assert not guard._leaf_ok("hex64", "a" * 63)
    assert guard._leaf_ok(("const", True), True)
    assert not guard._leaf_ok(("const", True), 1)
    assert not guard._leaf_ok(("const", 1), True)
    assert guard._leaf_ok(("int", 0, 3), 3)
    assert not guard._leaf_ok(("int", 0, 3), 4)
    assert not guard._leaf_ok(("int", 0, 3), False)


ROLE_SET_TABLE: dict[str, object] = {
    "schema_version": 1,
    "source_path": "data/random-battles/champions/doubles-sets.json",
    "source_sha256": "a" * 64,
    "species": [
        {
            "id": "venusaur",
            "sets": [
                {"role": "Offensive Protect", "moves": ["protect"], "abilities": ["overgrow"]}
            ],
        }
    ],
}


@pytest.mark.parametrize("in_fixtures", [True, False])
def test_a_role_set_table_without_team_text_passes(in_fixtures: bool) -> None:
    assert guard.problems_in(ROLE_SET_TABLE, in_fixtures) == []


@pytest.mark.parametrize(
    "intruder",
    [
        {"paste": "Venusaur @ Leftovers"},
        {"paste": 7},
        {"species": "Venusaur", "moves": ["Protect"], "evs": {"hp": 1}},
        {"evs": {"hp": 1}},
    ],
)
@pytest.mark.parametrize("in_fixtures", [True, False])
def test_a_role_set_table_may_not_hold_a_paste_or_a_team(
    intruder: dict[str, object], in_fixtures: bool
) -> None:
    table = {**ROLE_SET_TABLE, "note": intruder}
    assert guard.ROLE_SET_TEAM in guard.problems_in(table, in_fixtures)


def test_a_team_shaped_set_inside_a_species_entry_is_caught() -> None:
    entry = {"id": "venusaur", "sets": [{"species": "Venusaur", "moves": ["Protect"], "evs": {}}]}
    table = {**ROLE_SET_TABLE, "species": [entry]}
    assert guard.ROLE_SET_TEAM in guard.problems_in(table, True)


def test_other_documents_are_not_role_set_tables() -> None:
    other = {"source_path": "x", "species": []}
    assert guard.ROLE_SET_TEAM not in guard.problems_in({**other, "paste": "x"}, True)


# ADR-025 section 2: a usage prior is aggregate marginals, never a team.
USAGE_PRIOR: dict[str, object] = json.loads(
    (ROOT / "fixtures/contracts/v1/usage-prior/valid/smogon-chaos.json").read_text(encoding="utf-8")
)


def _forme(prior: dict[str, object]) -> dict[str, object]:
    species: list[dict[str, list[dict[str, object]]]] = prior["species"]  # type: ignore[assignment]
    return species[0]["formes"][0]


@pytest.mark.parametrize("in_fixtures", [True, False])
def test_a_usage_prior_without_team_text_passes(in_fixtures: bool) -> None:
    assert guard.problems_in(USAGE_PRIOR, in_fixtures) == []


@pytest.mark.parametrize(
    "intruder",
    [
        {"paste": "Venusaur @ Leftovers"},
        {"paste": 7},
        {"species": "Venusaur", "moves": ["Protect"], "evs": {"hp": 1}},
        {"evs": {"hp": 1}},
        {"species": "Venusaur", "moves": ["Protect"], "item": "leftovers", "evs": {}},
    ],
)
@pytest.mark.parametrize("in_fixtures", [True, False])
def test_a_usage_prior_may_not_hold_a_paste_an_evs_key_or_a_team(
    intruder: dict[str, object], in_fixtures: bool
) -> None:
    prior = copy.deepcopy(USAGE_PRIOR)
    _forme(prior)["sets"] = [intruder]
    assert guard.USAGE_PRIOR_TEAM in guard.problems_in(prior, in_fixtures)
    prior = copy.deepcopy(USAGE_PRIOR)
    prior["note"] = intruder
    assert guard.USAGE_PRIOR_TEAM in guard.problems_in(prior, in_fixtures)


def test_a_usage_prior_of_every_source_kind_passes_the_guard_as_committed() -> None:
    for kind in ("smogon-chaos", "role-sets", "generator-matched"):
        path = ROOT / f"fixtures/contracts/v1/usage-prior/valid/{kind}.json"
        assert guard.problems_in(json.loads(path.read_text(encoding="utf-8")), True) == []


def test_other_documents_are_not_usage_priors() -> None:
    other = {"source": {"kind": "x"}, "species": []}  # no derivation
    assert guard.USAGE_PRIOR_TEAM not in guard.problems_in({**other, "paste": "x"}, True)


# ADR-015 erratum E1: the strict shape of a private test record.
TEST_MARKER = "fixtures/contracts/v1/private-test-record/valid/sample.json"
TEST_SHAPE = guard.TEST_RECORD_SHAPE
TEST_DIR = ROOT / "fixtures/contracts/v1/private-test-record"
TEST_RECORD = json.loads((TEST_DIR / "valid/snapshot-holdout.json").read_text(encoding="utf-8"))


@pytest.fixture
def marked_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "PRIVATE_TEST_RECORD_MARKERS", frozenset({TEST_MARKER}))


def record_reasons(repo: Path, document: object, name: str = TEST_MARKER) -> list[str]:
    path = write(repo, name, json.dumps(document))
    return [r.split(": ", 1)[1] for r in guard.violations([path])]


def test_the_test_record_markers_are_exactly_the_private_labelled_corpus_samples() -> None:
    labelled = {
        str(p.relative_to(ROOT))
        for kind in ("valid", "invalid", "reader-invalid")
        for p in (TEST_DIR / kind).glob("*.json")
        if json.loads(p.read_text(encoding="utf-8"))["data_class"] == "private"
    }
    assert labelled == set(guard.PRIVATE_TEST_RECORD_MARKERS)
    assert guard.violations(sorted(guard.PRIVATE_TEST_RECORD_MARKERS)) == []
    invalid = sorted(str(p.relative_to(ROOT)) for p in (TEST_DIR / "invalid").glob("*.json"))
    assert invalid
    assert guard.violations(invalid) == []


def test_the_test_record_spec_follows_the_contract_schema() -> None:
    schema = json.loads(
        (ROOT / "packages/schema/contracts/v1/private-test-record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    required: Any
    required, optional = guard.PRIVATE_TEST_RECORD_SPEC
    assert set(required) == set(schema["required"])
    assert set(required) | set(optional) == set(schema["properties"])
    defs = schema["$defs"]
    assert set(defs["TestSet"]["properties"]) == set(required["test_set"][0]) | set(
        required["test_set"][1]
    )
    assert set(defs["Plan"]["properties"]) == set(required["plan"][0])
    match = required["matches"][2]
    assert set(defs["Match"]["properties"]) == set(match[0]) | set(match[1])
    assert set(defs["ForfeitCount"]["properties"]) == set(match[0]["forfeits"][2][0])
    assert (
        defs["Match"]["properties"]["pair_score_counts"]["maxItems"]
        == match[0]["pair_score_counts"][1]
    )


@pytest.mark.usefixtures("marked_record")
def test_a_marked_test_record_is_accepted_only_in_its_strict_shape(repo: Path) -> None:
    assert record_reasons(repo, TEST_RECORD) == []
    extra = {**TEST_RECORD, "notes": "x"}
    assert TEST_SHAPE in record_reasons(repo, extra)
    pasted = json.loads(json.dumps(TEST_RECORD))
    pasted["test_set"]["Garchomp @ Sitrus Berry"] = 1
    assert TEST_SHAPE in record_reasons(repo, pasted)
    teams = json.loads(json.dumps(TEST_RECORD))
    teams["test_set"]["teams"] = ["a" * 64]
    assert TEST_SHAPE in record_reasons(repo, teams)
    free = json.loads(json.dumps(TEST_RECORD))
    free["matches"][0]["forfeits"] = [{"side": "candidate", "cause": "Worker said x", "count": 1}]
    assert TEST_SHAPE in record_reasons(repo, free)
    assert TEST_SHAPE in record_reasons(repo, [TEST_RECORD])
    for key in guard.PRIVATE_TEST_RECORD_SPEC[0]:
        assert TEST_SHAPE in record_reasons(
            repo, {k: v for k, v in TEST_RECORD.items() if k != key}
        )
    for path, value in (
        (("generation",), -1),
        (("test_set", "source_kind"), "bucket"),
        (("matches", 0, "pair_score_counts"), [0, 0, 0, 0, 0, 0]),
        (("matches", 0, "mode"), "greedy"),
        (("matches", 0, "opponent_policy_id"), "Not A Policy"),
        (("matches", 0, "match_id"), "Has Upper"),
    ):
        assert TEST_SHAPE in record_reasons(repo, record_with({"value": value}, *path))


def record_with(patch: dict[str, object], *path: str | int) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(json.dumps(TEST_RECORD))
    cursor: Any = document
    for step in path[:-1]:
        cursor = cursor[step]
    cursor[path[-1]] = patch["value"]
    return document


def test_a_test_record_is_refused_outside_its_marker(repo: Path) -> None:
    assert PRIVATE in record_reasons(repo, TEST_RECORD, "fixtures/elsewhere.json")
    assert PRIVATE in record_reasons(repo, TEST_RECORD, TEST_MARKER)  # not marked in this run


@pytest.mark.usefixtures("marked_record")
def test_a_marked_test_record_does_not_waive_a_sealed_marker_or_a_team(repo: Path) -> None:
    sealed = {**TEST_RECORD, "provenance": {"collection": {"kind": "evaluation", "sealed": True}}}
    reasons = record_reasons(repo, sealed)
    assert SEALED in reasons
    assert TEST_SHAPE in reasons


def test_the_enum_leaf() -> None:
    assert guard._leaf_ok(("enum", ("a", "b")), "a")
    assert not guard._leaf_ok(("enum", ("a", "b")), "c")
    assert not guard._leaf_ok(("enum", ("a", "b")), 1)
