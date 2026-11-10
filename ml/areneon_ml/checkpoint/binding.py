"""The hash binding of an exported records directory and its sources registry (#853).

The C1/C6 checker's ``sources`` registry says which checkpoints generated each training set. A
wrong entry (an empty list for a set that really came from lineage X) makes two lineages look
independent, so the registry is never a free input: the exporters (``loop provenance``,
``learner clone``) derive it from the run's own manifests and write ``provenance.json`` beside
``sources.json``, naming the registry's SHA-256 and the SHA-256 of every record file by
checkpoint id. ``verify_binding`` refuses a registry, a record or a record set that differs from
what was bound; ``python -m areneon_ml.checkpoint classify|independent --run-dir R`` derives
everything from the run again instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from areneon_contracts.contracts import jcs, sha256_hex

BINDING_NAME = "provenance.json"
BINDING_VERSION = 1


class BindingError(ValueError):
    """The registry or the records are not what the exporter bound."""


def binding_bytes(records: Mapping[str, bytes], registry: bytes) -> bytes:
    """The binding file: the registry's hash and each record file's hash by checkpoint id."""
    return jcs(
        {
            "schema_version": BINDING_VERSION,
            "sources_hash": sha256_hex(registry),
            "records": {key: sha256_hex(data) for key, data in sorted(records.items())},
        }
    )


def verify_binding(binding: Path, registry: bytes, records: Mapping[str, bytes]) -> None:
    """Refuse a registry or record set that differs from the binding at ``binding``."""
    if not binding.is_file():
        raise BindingError(
            f"the sources registry is not bound to any run: {binding.name} is missing "
            "(derive it with `loop provenance` or pass --run-dir)"
        )
    raw = binding.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise BindingError(f"{binding.name} is not JSON") from exc
    if not isinstance(value, dict) or raw != jcs(value):
        raise BindingError(f"{binding.name} is not a canonical (RFC 8785) JSON object")
    if (
        set(value) != {"schema_version", "sources_hash", "records"}
        or value["schema_version"] != BINDING_VERSION
        or not isinstance(value["records"], dict)
    ):
        raise BindingError(f"{binding.name} is not a version {BINDING_VERSION} binding")
    if value["sources_hash"] != sha256_hex(registry):
        raise BindingError("the sources registry differs from the one the run exported")
    bound: dict[str, object] = value["records"]
    if set(bound) != set(records):
        raise BindingError("the records directory differs from the exported record set")
    for key, data in records.items():
        if bound[key] != sha256_hex(data):
            raise BindingError(f"record {key} differs from the exported one")
