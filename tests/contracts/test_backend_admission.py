"""The BackendAdmission reader (ADR-030 groundwork G1; #1461).

The Python reader refuses exactly what the TS one does: every reader-invalid corpus sample with the
rule it names, the schema-invalid ones, and a ledger of another pin. The valid samples hash to the
digests the TS side recorded, and the fresh seed is the TS one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.backend_admission import (
    AdmissionError,
    admission_hash,
    fresh_seed,
    read_backend_admission,
)

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures/contracts"
HASHES: dict[str, str] = json.loads((SAMPLES / "hashes.json").read_text(encoding="utf-8"))
PIN = "7244fe8e28fa8bc4d0809ae30d183c608b62bb2b"
DIR = "v1/backend-admission"


def load(path: str) -> Any:
    return json.loads((SAMPLES / path).read_text(encoding="utf-8"))


READER_INVALID = {
    "count-mismatch": "count mismatch",
    "counts-not-nested": "must nest",
    "fresh-below-minimum": "minimum size",
    "fresh-repeated-items": "each item once",
    "kinds-not-query-kind": "sorted and unique",
    "kinds-on-query-kind": "only an entity the fresh corpus holds",
    "missing-revocation": "revoked exactly when",
    "nineteen-exercising": "admitted flag",
    "other-pin": "another pinned engine",
    "pair-flag": "pair's admitted flag",
    "pair-not-admitted-entity": "two admitted entities",
    "rare-pair": "1 in 10\\^4",
    "regression-evidence": "must come from the fresh corpus",
    "revocation-past-corpus": "within its corpus",
    "two-fresh": "at most one fresh",
    "unsorted-entities": "entities must be sorted",
    "wrong-fresh-seed": "fresh seed",
}


def test_every_reader_invalid_sample_is_listed() -> None:
    found = sorted(
        p.split("/")[-1].removesuffix(".json")
        for p in HASHES
        if p.startswith(f"{DIR}/reader-invalid/")
    )
    assert found == sorted(READER_INVALID)


@pytest.mark.parametrize("name", sorted(READER_INVALID))
def test_reader_invalid_samples_are_refused(name: str) -> None:
    with pytest.raises(AdmissionError, match=READER_INVALID[name]):
        read_backend_admission(load(f"{DIR}/reader-invalid/{name}.json"), PIN)


@pytest.mark.parametrize("path", sorted(p for p in HASHES if p.startswith(f"{DIR}/valid/")))
def test_valid_samples_are_read_and_hash_as_on_the_ts_side(path: str) -> None:
    ledger = load(path)
    assert read_backend_admission(ledger, PIN) == ledger
    assert admission_hash(ledger) == HASHES[path]


@pytest.mark.parametrize("path", sorted(p for p in HASHES if p.startswith(f"{DIR}/invalid/")))
def test_invalid_samples_are_refused_by_the_schema(path: str) -> None:
    with pytest.raises(AdmissionError, match="backend-admission"):
        read_backend_admission(load(path), PIN)


def test_the_fresh_seed_is_the_ts_one() -> None:
    ledger = load(f"{DIR}/valid/admitted.json")
    fresh = next(c for c in ledger["corpora"] if c["kind"] == "fresh")
    assert fresh["seed"] == fresh_seed(
        ledger["backend"]["backend_commit"], ledger["backend"]["backend_format"]
    )
    assert fresh["id"] == f"fresh-{fresh['seed'][:16]}"


def _doctor(change: Any) -> dict[str, Any]:
    ledger: dict[str, Any] = copy.deepcopy(load(f"{DIR}/valid/shadow-revoked.json"))
    change(ledger)
    return ledger


def _entity(ledger: dict[str, Any], entity_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(e for e in ledger["entities"] if e["id"] == entity_id)
    return found


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda lg: lg["corpora"].reverse(), "corpora must be sorted"),
        (lambda lg: lg["revoked"].append(dict(lg["revoked"][0])), "revocations must be sorted"),
        (
            lambda lg: lg["revoked"].append(
                {"id": "move:zzz", "corpus": lg["corpora"][0]["id"], "position": 0}
            ),
            "known entity",
        ),
        (
            lambda lg: _entity(lg, "move:protect")["counts"][0].update(corpus="nope"),
            "known corpora",
        ),
        (
            lambda lg: _entity(lg, "move:protect")["counts"][0].update(present=0),
            "present in at least one",
        ),
        (
            lambda lg: _entity(lg, "move:protect")["counts"][0].update(exercised=1),
            "unequal exercising",
        ),
        (lambda lg: lg["pairs"].append(dict(lg["pairs"][0])), "sorted and unique"),
        (lambda lg: lg["pairs"][0].update(co_equal=30), "must nest"),
        (
            lambda lg: _entity(lg, "species:garchomp")["kinds"][0].update(co_exercised=22),
            "kind counts must nest",
        ),
    ],
)
def test_doctored_ledgers_are_refused(change: Any, message: str) -> None:
    with pytest.raises(AdmissionError, match=message):
        read_backend_admission(_doctor(change), PIN)


def test_an_empty_ledger_admits_nothing() -> None:
    ledger = load(f"{DIR}/valid/empty-after-pin-bump.json")
    assert read_backend_admission(ledger, PIN)["entities"] == []


# ---------------------------------------------------------------- the committed lines form (#1457)

from areneon_contracts.backend_admission import (  # noqa: E402
    backend_admission_lines,
    read_backend_admission_lines,
)


@pytest.mark.parametrize("name", sorted(p.name for p in (SAMPLES / DIR / "valid").iterdir()))
def test_lines_round_trip(name: str) -> None:
    ledger = read_backend_admission(load(f"{DIR}/valid/{name}"), PIN)
    text = backend_admission_lines(ledger)
    assert text.endswith("\n")
    assert read_backend_admission_lines(text, PIN) == ledger
    assert admission_hash(read_backend_admission_lines(text, PIN)) == admission_hash(ledger)


def test_lines_refusals() -> None:
    name = sorted(p.name for p in (SAMPLES / DIR / "valid").iterdir())[0]
    ledger = read_backend_admission(load(f"{DIR}/valid/{name}"), PIN)
    lines = backend_admission_lines(ledger).splitlines()

    def read(ls: list[str]) -> Any:
        return read_backend_admission_lines("".join(f"{line}\n" for line in ls), PIN)

    with pytest.raises(AdmissionError):
        read([])
    with pytest.raises(AdmissionError, match="header"):
        read(lines[1:])
    with pytest.raises(AdmissionError, match="one kind"):
        read([lines[0], lines[0]])
    with pytest.raises(AdmissionError, match="one kind"):
        read([*lines, '{"other":{}}'])
    with pytest.raises(AdmissionError):
        read([*lines, "not json"])
    with pytest.raises(AdmissionError, match="canonical"):
        read([*lines[:-1], " " + lines[-1]])
    with pytest.raises(AdmissionError, match="newline"):
        read_backend_admission_lines(backend_admission_lines(ledger).rstrip("\n"), PIN)


# The kernel version's committed ledger (ADR-030 step 3): its lines form, read by the
# Python reader, has the pinned hash.
KERNEL_LEDGER = (
    Path(__file__).resolve().parents[2]
    / "config/backends/gen9championsvgc2026regmc.damage-kernel.admission.jsonl"
)
KERNEL_ADMISSION_HASH = "0" * 64


def test_the_committed_kernel_ledger_has_the_pinned_hash() -> None:
    text = KERNEL_LEDGER.read_text(encoding="utf-8")
    ledger = read_backend_admission_lines(text, PIN)
    assert backend_admission_lines(ledger) == text
    assert admission_hash(ledger) == KERNEL_ADMISSION_HASH
    assert ledger["backend"]["kind"] == "damage-kernel"
    fresh = [c for c in ledger["corpora"] if c["kind"] == "fresh"]
    assert len(fresh) == 1
    assert fresh[0]["distinct_items"] >= 200_000
    backend = ledger["backend"]
    assert fresh[0]["seed"] == fresh_seed(backend["backend_commit"], backend["backend_format"])
