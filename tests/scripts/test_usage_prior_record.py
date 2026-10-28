"""The committed Smogon usage prior records of 2026-09, cutoff 1630 (#1120, #1209).

Two records: the first (``usage-prior.2026-09-1630.json``) and its successor (``...v2.json``:
names mapped by the engine's own id normalisation, and the species the source lacks filled from
the generator-matched prior). The first is kept as it was, since a consumer may name its hash.
The records are reviewed aggregate data. These tests pin them: they pass the reader against the
golden team vocabulary, their hashes and inputs are the stated ones, the guards accept them, and
the raw files (which are not committed) are nowhere in the tracked tree. The re-derivation test
runs only where the raw files are present in the gitignored ``.tmp/chaos``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import jcs, sha256_hex
from areneon_contracts.usage_prior import read_usage_prior, usage_prior_hash

from scripts import check_data_class as guard
from scripts.derive_usage_prior import run

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "gen9championsvgc2026regmc"
DIR = ROOT / "config" / "opponent-model" / FORMAT
RECORD = DIR / "usage-prior.2026-09-1630.json"
RECORD2 = DIR / "usage-prior.2026-09-1630.v2.json"
INPUTS = DIR / "smogon-chaos.2026-09-1630.inputs.json"
RAW = ROOT / ".tmp" / "chaos"
PRIOR_HASH = "1949b487e7685c94f0ec9d96ee39889d04396d64c1665483e18ed18906ebc846"
PRIOR_HASH2 = "910736ec4d738cb022b7aee64110028210c6d31338c926f9579434d3304eeda7"
VOCABULARY = json.loads(
    (ROOT / "fixtures" / "showdown" / FORMAT / "team-vocabulary.json").read_text(encoding="utf-8")
)


def load() -> dict[str, object]:
    loaded: dict[str, object] = json.loads(RECORD.read_text(encoding="utf-8"))
    return loaded


def test_the_record_is_the_jcs_bytes_with_its_pinned_hash_and_passes_the_reader() -> None:
    record = load()
    assert RECORD.read_bytes() == jcs(record) + b"\n"
    assert usage_prior_hash(record) == PRIOR_HASH
    read_usage_prior(record, VOCABULARY)


def test_the_second_record_is_the_jcs_bytes_with_its_pinned_hash_and_passes_the_reader() -> None:
    record = json.loads(RECORD2.read_text(encoding="utf-8"))
    assert RECORD2.read_bytes() == jcs(record) + b"\n"
    assert usage_prior_hash(record) == PRIOR_HASH2
    read_usage_prior(record, VOCABULARY)
    assert guard.problems_in(record, False) == []
    assert guard.violations([str(RECORD2)]) == []


def test_the_second_record_covers_every_preview_species_and_labels_the_fallback() -> None:
    first: dict[str, Any] = load()
    second: dict[str, Any] = json.loads(RECORD2.read_text(encoding="utf-8"))
    generator = json.loads((DIR / "generator-matched.uniform-v1.json").read_text(encoding="utf-8"))
    ids = [s["id"] for s in second["species"]]
    assert ids == [s["id"] for s in generator["species"]]  # all 300 preview species
    label = second["source"]["fallback"]
    assert label["kind"] == "generator-matched"
    assert label["prior_hash"] == sha256_hex(jcs(generator))
    assert len(label["species"]) == 67
    first_ids = {s["id"] for s in first["species"]}
    assert set(label["species"]) == set(ids) - first_ids - {"farfetchd", "sirfetchd"}
    assert {"farfetchd", "sirfetchd"} <= set(ids) - set(label["species"])  # mapped, not filled
    # the source's own species are the first record's; the source block differs by the label only
    own = {s["id"]: s for s in second["species"] if s["id"] not in label["species"]}
    assert {k: v for k, v in own.items() if k in first_ids} == {
        s["id"]: s for s in first["species"] if s["id"] in own
    }
    assert {k: v for k, v in second["source"].items() if k != "fallback"} == first["source"]
    assert second["derivation"]["script_version"] == "usage-prior-chaos.v2"


def test_the_record_states_its_source_and_inputs() -> None:
    record = load()
    source = record["source"]
    assert isinstance(source, dict)
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    assert (source["kind"], source["month"], source["cutoff"]) == ("smogon-chaos", "2026-09", 1630)
    assert source["number_of_battles"] == 1_631_943
    assert source["inputs"] == [
        {k: f[k] for k in ("source_id", "url", "retrieved_at", "size", "sha256")}
        for f in inputs["files"]
    ]
    assert all(
        f["url"].startswith("https://www.smogon.com/stats/2026-09/") for f in inputs["files"]
    )
    assert record["vocabulary_hash"] == sha256_hex(jcs(VOCABULARY))
    assert record["derivation"] == {
        "script_version": "usage-prior-chaos.v1",
        "k_items": 16,
        "k_moves": 24,
        "k_spreads": 32,
    }


def test_the_second_notice_says_which_species_are_not_smogon_data() -> None:
    notice = (DIR / "usage-prior.2026-09-1630.v2.NOTICE.txt").read_text(encoding="utf-8")
    for needle in (
        "Smogon",
        "generator-matched",
        "67 preview species",
        "say nothing about how often",
    ):
        assert needle in notice


def test_the_notice_names_the_source_and_the_basis() -> None:
    notice = (DIR / "usage-prior.2026-09-1630.NOTICE.txt").read_text(encoding="utf-8")
    for needle in (
        "Smogon",
        "ADR-009",
        "not a legal opinion",
        "gen9championsvgc2026regmc-1630.json",
    ):
        assert needle in notice


def test_the_committed_record_passes_the_data_class_guard() -> None:
    assert guard.problems_in(load(), False) == []
    assert guard.violations([str(RECORD)]) == []


def test_the_raw_files_are_ignored_and_in_no_tracked_file() -> None:
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".tmp/chaos/chaos.json"], cwd=ROOT, check=False
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    sizes = {f["size"]: f["sha256"] for f in inputs["files"]}
    for name in tracked:
        path = ROOT / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size in sizes:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest != sizes[path.stat().st_size], f"{name} is a raw input file"
    # A raw chaos file is the only thing that carries these keys in JSON form.
    marker = '"Viability' + ' Ceiling":'
    found = subprocess.run(
        ["git", "grep", "-l", "-F", marker], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert found.stdout == ""


@pytest.mark.skipif(
    not (RAW / "chaos.json").exists(), reason="the raw files are not on this machine"
)
def test_the_second_record_is_re_derived_byte_for_byte_from_the_raw_files(tmp_path: Path) -> None:
    # The first record was made by script version v1 (commit d0b8e9c); only v2 is reproducible here.
    out = tmp_path / "usage-prior.json"
    assert run(INPUTS, RAW, ROOT, out, None, True) == 0
    assert out.read_bytes() == RECORD2.read_bytes()
