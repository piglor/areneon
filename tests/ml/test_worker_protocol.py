"""Framing, the contract set and the candidate parser of the policy worker (ADR-010 §2)."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from areneon_contracts.contracts import jcs
from areneon_ml.worker.candidates import UnsupportedCandidate, parse_choice
from areneon_ml.worker.protocol import (
    FrameError,
    QueryError,
    check,
    contract_versions,
    decode_frame,
    encode_frame,
    read_frame,
)

from tests.ml.helpers import REPO, battle_query, sample

MANY = 100


def test_the_contracts_it_speaks_are_those_the_ts_side_recorded() -> None:
    # The TS corpus capture computes the hello from its own copy of the schemas.
    recorded = sample("policy-handshake/valid/hello.json")["message"]["contracts"]
    assert contract_versions() == recorded
    names = [c["name"] for c in recorded]
    assert names == sorted(names)


@pytest.mark.parametrize(
    "line",
    [
        b"\xff\xfe",
        b"not json",
        b"[1]",
        b'{"a":1,"a":2}',
        b'{"a":1.5}',
        b'{"a":NaN}',
        b'{"a": 1}',
        b'{"b":1,"a":2}',
        b'{"a":"\\ud800"}',
    ],
)
def test_a_frame_that_is_not_canonical_json_is_refused(line: bytes) -> None:
    with pytest.raises(FrameError):
        decode_frame(line)


def test_a_canonical_frame_is_read() -> None:
    assert decode_frame(b'{"a":[1,"b"]}') == {"a": [1, "b"]}


def test_read_frame_reads_lines_and_ends_cleanly() -> None:
    stream = io.BytesIO(b'{"a":1}\n{"b":2}\n')
    assert read_frame(stream, 100) == {"a": 1}
    assert read_frame(stream, 100) == {"b": 2}
    assert read_frame(stream, 100) is None


def test_a_line_may_be_exactly_the_maximum_and_no_longer() -> None:
    assert read_frame(io.BytesIO(b'{"a":1}\n'), 7) == {"a": 1}
    with pytest.raises(FrameError, match="longer than 6"):
        read_frame(io.BytesIO(b'{"a":1}\n'), 6)
    with pytest.raises(FrameError, match="longer than 3"):
        read_frame(io.BytesIO(b'{"a":1}'), 3)


def test_input_that_ends_in_the_middle_of_a_line_is_refused() -> None:
    with pytest.raises(FrameError, match="middle of a line"):
        read_frame(io.BytesIO(b'{"a":1}'), 100)


def test_encode_frame_is_jcs_and_bounded() -> None:
    assert encode_frame({"b": 1, "a": 2}, 20) == b'{"a":2,"b":1}\n'
    with pytest.raises(FrameError, match="exceeds"):
        encode_frame({"a": "x" * 30}, 20)


def test_a_value_without_a_canonical_form_is_refused() -> None:
    with pytest.raises(FrameError, match="canonical form"):
        decode_frame(b'{"a":{"\\ud800":1}}')


def test_check_names_where_a_message_is_wrong_but_never_its_content() -> None:
    query = {**battle_query(), "seed": "sodium,SECRET-SEED"}
    with pytest.raises(QueryError, match="additionalProperties") as caught:
        check("policy-query", query)
    assert "SECRET-SEED" not in str(caught.value)


def test_every_candidate_of_the_corpus_has_a_known_shape() -> None:
    seen = 0
    for path in (REPO / "fixtures/showdown").glob("*/choices/*.json"):
        for candidate in json.loads(path.read_text())["candidates"]:
            parse_choice(candidate["choice"])
            seen += 1
    for name in ("policy-query/valid/battle.json", "policy-query/valid/preview.json"):
        for candidate in sample(name)["candidates"]:
            parse_choice(candidate["choice"])
            seen += 1
    assert seen > MANY


def test_parse_choice_reads_the_enumerators_shapes() -> None:
    assert parse_choice("team 1234").kind == "team"
    parsed = parse_choice("move 1 -2 mega, switch 3")
    assert (parsed.kind, parsed.parts) == ("joint", ("move 1 -2 mega", "switch 3"))
    assert parse_choice("pass, move 4").kind == "joint"


@pytest.mark.parametrize(
    "choice",
    [
        "",
        "default",
        "team",
        "team 0",
        "move 0",
        "move 1 mega2",
        "switch 1",
        "move 1, move 2, move 3",
        "move 1,move 2",
        "shift, pass",
        "move 1 2 3",
    ],
)
def test_an_unknown_shape_is_unsupported(choice: str) -> None:
    with pytest.raises(UnsupportedCandidate):
        parse_choice(choice)


def test_jcs_of_a_reply_round_trips_through_the_frame_reader() -> None:
    value: dict[str, Any] = {"schema_version": 1, "query_id": 3}
    assert read_frame(io.BytesIO(jcs(value) + b"\n"), 100) == value
