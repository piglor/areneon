"""Python checks: validator verdicts (jsonschema, fastjsonschema) and the
generated models' parse verdicts + JCS round trip. One JSON line per sample.

Run: python harness/check_py.py <out>   (see run.sh for the pinned --with list)
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fastjsonschema
import jsonschema
import rfc8785

OUT = Path(sys.argv[1])
SPIKE = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((SPIKE / "schema" / "contracts.schema.json").read_text())
sys.path.insert(0, str(OUT / "py"))


def jcs(value: Any) -> bytes:
    return rfc8785.dumps(value)


def pydantic_runner(module: str, **dump: Any) -> Callable[[str], Any]:
    model = importlib.import_module(module).AreneonContracts

    def run(raw: str) -> Any:
        return model.model_validate_json(raw).model_dump(mode="json", by_alias=True, **dump)

    return run


def quicktype_runner() -> Callable[[str], Any]:
    mod = importlib.import_module("quicktype_models")
    return lambda raw: mod.areneon_contracts_to_dict(
        mod.areneon_contracts_from_dict(json.loads(raw))
    )


def main() -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    try:
        fast: Callable[[Any], Any] | None = fastjsonschema.compile(SCHEMA)
        fast_error = ""
    except Exception as exc:  # recorded, not fatal: the spike compares tools
        fast, fast_error = None, f"{type(exc).__name__}: {exc}"
    runners = {
        "dmcg_exclude_unset": pydantic_runner("dmcg_models", exclude_unset=True),
        "dmcg_exclude_none": pydantic_runner("dmcg_models", exclude_none=True),
        "dmcg_missing": pydantic_runner("dmcg_missing_models"),
        "quicktype": quicktype_runner(),
    }
    for kind in ("valid", "invalid"):
        for path in sorted((OUT / "samples" / kind).iterdir()):
            key = path.name.split(".")[0]
            catalog = {key: json.loads(path.read_text())}
            raw = json.dumps(catalog)
            row: dict[str, Any] = {"lang": "py", "sample": f"{kind}/{path.name}"}
            row["jsonschema"] = validator.is_valid(catalog)
            if fast is None:
                row["fastjsonschema"] = f"compile failed: {fast_error[:80]}"
            else:
                try:
                    fast(catalog)
                    row["fastjsonschema"] = True
                except fastjsonschema.JsonSchemaException:
                    row["fastjsonschema"] = False
            for name, run in runners.items():
                try:
                    back = run(raw)
                except Exception:
                    row[name] = False
                    continue
                row[name] = True
                row[f"{name}_jcs_equal"] = jcs(back) == jcs(catalog)
            print(json.dumps(row))
    # request_hash = SHA-256 of the canonical request without rqid (request-identity.ts).
    for path in sorted((OUT / "samples" / "valid").glob("request_fixture.*.json")):
        fixture = json.loads(path.read_text())
        request = {k: v for k, v in fixture["request"].items() if k != "rqid"}
        same = hashlib.sha256(jcs(request)).hexdigest() == fixture["request_hash"]
        print(
            json.dumps({"lang": "py", "request_hash": path.name, "matches": same}), file=sys.stderr
        )
    versions = {
        m: importlib.metadata.version(m)
        for m in ("pydantic", "jsonschema", "fastjsonschema", "rfc8785")
    }
    print(json.dumps(versions), file=sys.stderr)


if __name__ == "__main__":
    main()
