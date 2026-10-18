"""Validate, then parse (ADR-008 §2): the Python side of Areneon's contracts.

The schemas in ``packages/schema/contracts/v<N>/`` are the source of truth.
``validate`` checks an instance against one with python-jsonschema (Draft
2020-12) and fails closed; ``parse`` validates first and only then builds the
generated pydantic model, so the model is a typed view of data that already
passed the schema. ``jcs`` and ``sha256_hex`` give the RFC 8785 bytes and the
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


def model(name: str, version: int = 1) -> type[BaseModel]:
    """The generated pydantic model of a contract: its schema's title."""
    module = importlib.import_module(
        f"areneon_contracts.models.v{version}.{name.replace('-', '_')}"
    )
    found: type[BaseModel] = getattr(module, schema(name, version)["title"])
    return found


def parse(name: str, instance: object, version: int = 1) -> BaseModel:
    """Validate against the schema, then build the typed model."""
    validate(name, instance, version)
    return model(name, version).model_validate(instance, strict=True)


def jcs(value: Any) -> bytes:
    """The RFC 8785 (JCS) bytes of a JSON value."""
    out: bytes = rfc8785.dumps(value)
    return out


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request_hash(request: dict[str, Any]) -> str:
    """ADR-004 r2 request identity: SHA-256 of the canonical request without ``rqid``."""
    return sha256_hex(jcs({k: v for k, v in request.items() if k != "rqid"}))
