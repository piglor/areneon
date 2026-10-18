"""The Python side of the shared contract corpus (ADR-008 §2, §4).

Every sample under ``fixtures/contracts/`` must be classified by
python-jsonschema exactly as the TS side's Ajv classifies it. Every valid
sample must parse into its generated pydantic model and dump back to the same
JCS bytes, and every sample's JCS digest must equal the one the TS side
recorded in ``hashes.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from areneon_contracts.contracts import (
    CanonicalError,
    ContractError,
    _model,
    jcs,
    parse,
    request_hash,
    schema,
    sha256_hex,
    validate,
)

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "fixtures/contracts"
GOLDEN = REPO / "fixtures/showdown/gen9championsvgc2026regmc"
HASHES: dict[str, str] = json.loads((SAMPLES / "hashes.json").read_text(encoding="utf-8"))
VECTORS: list[dict[str, Any]] = json.loads(
    (SAMPLES / "jcs-vectors.json").read_text(encoding="utf-8")
)["vectors"]


def load(path: str) -> Any:
    return json.loads((SAMPLES / path).read_text(encoding="utf-8"))


def contract_of(path: str) -> str:
    return path.split("/")[1]


VALID = sorted(p for p in HASHES if "/valid/" in p)
INVALID = sorted(p for p in HASHES if "/invalid/" in p)


def test_the_corpus_has_both_kinds_for_every_contract() -> None:
    contracts = sorted(
        p.name.removesuffix(".schema.json")
        for p in (REPO / "packages/schema/contracts/v1").iterdir()
    )
    assert sorted({contract_of(p) for p in VALID}) == contracts
    assert sorted({contract_of(p) for p in INVALID}) == contracts


@pytest.mark.parametrize("path", VALID)
def test_valid_samples_validate_parse_and_round_trip(path: str) -> None:
    sample = load(path)
    validate(contract_of(path), sample)
    parsed = parse(contract_of(path), sample)
    assert jcs(parsed.model_dump(mode="json", by_alias=True)) == jcs(sample)


@pytest.mark.parametrize("path", INVALID)
def test_invalid_samples_are_refused(path: str) -> None:
    with pytest.raises(ContractError, match=contract_of(path)):
        validate(contract_of(path), load(path))
    with pytest.raises(ContractError):
        parse(contract_of(path), load(path))


@pytest.mark.parametrize("path", sorted(HASHES))
def test_jcs_digests_equal_the_ts_side(path: str) -> None:
    assert sha256_hex(jcs(load(path))) == HASHES[path]


@pytest.mark.parametrize("fixture", ["preview", "turn", "force-switch"])
def test_request_hash_equals_the_ts_side(fixture: str) -> None:
    golden = json.loads((GOLDEN / f"{fixture}.request.json").read_text(encoding="utf-8"))
    assert request_hash(golden["request"]) == golden["request_hash"]
    assert request_hash({**golden["request"], "rqid": 9}) == golden["request_hash"]


def test_model_and_schema_resolve_by_name() -> None:
    assert _model("episode").__name__ == schema("episode")["title"] == "Episode"


def test_the_model_refuses_coercion_after_validation() -> None:
    sample = load("v1/request-identity/valid/a1.json")
    with pytest.raises(ValueError, match="request_seq"):
        _model("request-identity").model_validate({**sample, "request_seq": "1"}, strict=True)


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: v["name"])
def test_jcs_reproduces_the_shared_vectors(vector: dict[str, Any]) -> None:
    value = json.loads(vector["json"])
    if vector["jcs"] is None:
        with pytest.raises(CanonicalError):
            jcs(value)
    else:
        assert jcs(value).decode("utf-8") == vector["jcs"]


def test_jcs_refuses_an_integer_a_js_number_cannot_hold() -> None:
    assert jcs({"a": [True, {"b": 2**53 - 1}]}) == b'{"a":[true,{"b":9007199254740991}]}'
    with pytest.raises(CanonicalError):
        jcs({"a": [{"b": 2**53 + 1}]})


def test_request_hash_validates_the_request_first() -> None:
    with pytest.raises(ContractError, match="engine-request"):
        request_hash({"side": {}, "turn": 3})
