"""Framing and the contract set of the worker protocol (ADR-010 §2, ADR-008 §3).

One JCS JSON object per line, UTF-8, ``\\n``-terminated, no longer than the
run's maximum line length. A frame that is anything else is refused: the
worker does not guess what a malformed line meant.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any, BinaryIO

from areneon_contracts.contracts import CanonicalError, jcs, schema, sha256_hex
from jsonschema import Draft202012Validator

# The contracts both sides must agree on (the policy-handshake schema's name enum).
SPOKEN = ("BattleObservation", "PolicyQuery", "PolicyReply", "PreviewObservation", "Recommendation")


class FrameError(ValueError):
    """A line is not a valid frame: too long, not UTF-8 JCS, or not an object."""


class _Refused(ValueError):
    pass


def _refuse(what: str) -> Any:
    raise _Refused(what)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _Refused("duplicate key")
    return dict(pairs)


def decode_frame(line: bytes) -> dict[str, Any]:
    """The object a frame line (without its newline) holds, if it is canonical."""
    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_float=lambda text: _refuse(f"non-integer number {text}"),
            parse_constant=lambda text: _refuse(f"constant {text}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _Refused, RecursionError) as exc:
        raise FrameError(f"not UTF-8 JSON: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise FrameError("a frame must be a JSON object")
    try:
        canonical = jcs(value)
    except CanonicalError as exc:
        raise FrameError(f"no canonical form: {exc}") from exc
    if canonical != line:
        raise FrameError("a frame must be in RFC 8785 (JCS) canonical form")
    return value


def read_frame(stream: BinaryIO, max_line_bytes: int) -> dict[str, Any] | None:
    """The next frame, or ``None`` at a clean end of input; anything else raises ``FrameError``."""
    line = stream.readline(max_line_bytes + 2)
    if line == b"":
        return None
    if not line.endswith(b"\n"):
        if len(line) > max_line_bytes:
            raise FrameError(f"a line is longer than {max_line_bytes} bytes")
        raise FrameError("the input ended in the middle of a line")
    if len(line) - 1 > max_line_bytes:
        raise FrameError(f"a line is longer than {max_line_bytes} bytes")
    return decode_frame(line[:-1])


def encode_frame(value: dict[str, Any], max_line_bytes: int) -> bytes:
    """The JCS bytes of ``value`` and a newline."""
    line = jcs(value)
    if len(line) > max_line_bytes:
        raise FrameError(
            f"a frame of {len(line)} bytes exceeds the maximum line length {max_line_bytes}"
        )
    return line + b"\n"


def _kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


@cache
def contract_versions() -> list[dict[str, Any]]:
    """The contracts this side speaks, sorted by name: version and schema hash (JCS bytes)."""
    out = []
    for name in SPOKEN:
        loaded = schema(_kebab(name))
        out.append(
            {
                "name": name,
                "schema_version": loaded["properties"]["schema_version"]["const"],
                "schema_sha256": sha256_hex(jcs(loaded)),
            }
        )
    return out


@cache
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(schema(name))


class QueryError(ValueError):
    """A message breaks its contract or a reader rule. The text holds no instance content."""


def check(name: str, instance: object) -> None:
    """Validate against a contract; the error names where and which keyword, never the value."""
    errors = sorted(_validator(name).iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        found = "; ".join(
            f"/{'/'.join(map(str, e.absolute_path))} {e.validator}" for e in errors[:5]
        )
        raise QueryError(f"not a {name}: {found}"[:300])
