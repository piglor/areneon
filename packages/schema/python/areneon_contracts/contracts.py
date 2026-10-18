"""Validate, then parse (ADR-008 §2): the Python side of Areneon's contracts.

The schemas in ``packages/schema/contracts/v<N>/`` are the source of truth.
``validate`` checks an instance against one with python-jsonschema (Draft
2020-12) and fails closed; ``parse`` validates first and only then builds the
generated pydantic model, so the model is a typed view of data that already
passed the schema; the model classes are not exported, so nothing builds one
without validating first. ``jcs`` and ``sha256_hex`` give the RFC 8785 bytes and the
digest every language computes identically (ADR-008 §4).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import rfc8785
from jsonschema import Draft202012Validator
from pydantic import BaseModel

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


class ContractError(ValueError):
    """An instance does not match its contract schema."""


@cache
def schema(name: str, version: int = 1) -> dict[str, Any]:
    """The schema ``contracts/v<version>/<name>.schema.json``."""
    path = CONTRACTS / f"v{version}" / f"{name}.schema.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@cache
def _validator(name: str, version: int) -> Draft202012Validator:
    return Draft202012Validator(schema(name, version))


def validate(name: str, instance: object, version: int = 1) -> None:
    """Raise ``ContractError`` naming every way ``instance`` misses the schema."""
    errors = sorted(
        _validator(name, version).iter_errors(instance), key=lambda e: list(e.absolute_path)
    )
    if errors:
        found = "; ".join(f"/{'/'.join(map(str, e.absolute_path))} {e.message}" for e in errors)
        raise ContractError(f"{name} v{version}: {found}")


def _model(name: str, version: int = 1) -> type[BaseModel]:
    """The generated pydantic model of a contract: its schema's title."""
    module = importlib.import_module(
        f"areneon_contracts.models.v{version}.{name.replace('-', '_')}"
    )
    found: type[BaseModel] = getattr(module, schema(name, version)["title"])
    return found


def parse(name: str, instance: object, version: int = 1) -> BaseModel:
    """Validate against the schema, then build the typed model."""
    validate(name, instance, version)
    return _model(name, version).model_validate(instance, strict=True)


class CanonicalError(ValueError):
    """A value has no JCS form shared by every language."""


# A decoded Python str holds a surrogate code point only when it was unpaired.
SURROGATES = range(0xD800, 0xE000)


def _has_lone_surrogate(text: str) -> bool:
    return any(ord(char) in SURROGATES for char in text)


def _assert_ijson(value: Any) -> None:
    """The I-JSON checks every language applies alike, keys included."""
    if isinstance(value, str) and _has_lone_surrogate(value):
        raise CanonicalError("jcs: lone surrogate in string")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_ijson(key)
            _assert_ijson(item)
    elif isinstance(value, list):
        for item in value:
            _assert_ijson(item)


def jcs(value: Any) -> bytes:
    """The RFC 8785 (JCS) bytes of a JSON value.

    Like the TS ``jcs``, it refuses a lone surrogate in any string, object
    keys included (rfc8785 itself checks only values); both reproduce
    ``fixtures/contracts/jcs-vectors.json``. rfc8785 also refuses an integer
    beyond +-(2^53-1), which a JS number cannot hold exactly; contract
    schemas bound every integer inside that range, and only validated data
    is hashed, so no contract value reaches either side with one.
    """
    _assert_ijson(value)
    try:
        out: bytes = rfc8785.dumps(value)
    except rfc8785.CanonicalizationError as exc:
        raise CanonicalError(f"jcs: {exc}") from exc
    return out


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_hash(request: dict[str, Any]) -> str:
    """ADR-004 r2 request identity: SHA-256 of the canonical request without ``rqid``.

    The request is validated against the engine-mirror schema first: only a
    validated request is hashed (ADR-008 §4).
    """
    validate("engine-request", request)
    return sha256_hex(jcs({k: v for k, v in request.items() if k != "rqid"}))
