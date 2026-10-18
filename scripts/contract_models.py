"""Generate the pydantic models for Areneon's contract schemas (ADR-008 §2).

Every ``packages/schema/contracts/v<N>/<name>.schema.json`` becomes
``packages/schema/python/areneon_contracts/models/v<N>/<name_with_underscores>.py``
through the pinned datamodel-code-generator, with fixed options. The output
is checked in; ``--check`` (``make contracts``) regenerates into a temporary
directory and fails on any difference, ``--write`` (``make contracts-update``)
replaces it.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACTS = REPO / "packages/schema/contracts"
MODELS = REPO / "packages/schema/python/areneon_contracts/models"

# ADR-008 §2: strict types (nothing coerced), closed models, the MISSING
# sentinel for optional fields, no timestamp (byte-identical reruns).
OPTIONS = (
    "--input-file-type=jsonschema",
    "--output-model-type=pydantic_v2.BaseModel",
    "--target-python-version=3.12",
    "--target-pydantic-version=2.14",
    "--use-annotated",
    "--field-constraints",
    "--use-double-quotes",
    "--disable-timestamp",
    "--use-schema-description",
    "--use-field-description",
    "--enum-field-as-literal=all",
    "--use-union-operator",
    "--use-standard-collections",
    "--formatters=builtin",
    "--use-missing-sentinel",
    "--extra-fields=forbid",
    "--strict-types",
    "str",
    "int",
    "bool",
)
SUFFIX = ".schema.json"


def schemas(contracts: Path = CONTRACTS) -> list[Path]:
    """Every ``v<N>/*.schema.json``, in a fixed order."""
    return sorted(contracts.glob(f"v*/*{SUFFIX}"))


def module_path(schema: Path) -> Path:
    """``v1/legal-choice-set.schema.json`` -> ``v1/legal_choice_set.py``."""
    name = schema.name.removesuffix(SUFFIX).replace("-", "_")
    return Path(schema.parent.name) / f"{name}.py"


def generate(contracts: Path, out: Path) -> None:
    """Write every model module, and an ``__init__.py`` per version, under ``out``."""
    for schema in schemas(contracts):
        target = out / module_path(schema)
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / "__init__.py").write_text(
            f'"""Generated pydantic models for contracts {target.parent.name}; do not edit."""\n',
            encoding="utf-8",
        )
        subprocess.run(
            ["datamodel-codegen", "--input", str(schema), "--output", str(target), *OPTIONS],
            check=True,
            capture_output=True,
        )
    (out / "__init__.py").write_text(
        '"""Generated pydantic models; do not edit."""\n', encoding="utf-8"
    )


def drift(expected: Path, actual: Path) -> list[str]:
    """Files that differ, are missing from ``actual`` or have no schema."""

    def files(root: Path) -> set[Path]:
        return {p.relative_to(root) for p in root.rglob("*.py")} if root.exists() else set()

    want, have = files(expected), files(actual)
    found = [f"{p}: generated file is missing" for p in sorted(want - have)]
    found += [f"{p}: generated file has no schema" for p in sorted(have - want)]
    found += [
        f"{p}: generated file differs from its schema"
        for p in sorted(want & have)
        if not filecmp.cmp(expected / p, actual / p, shallow=False)
    ]
    return found


def main(
    argv: Sequence[str] | None = None, contracts: Path = CONTRACTS, models: Path = MODELS
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "models"
        generate(contracts, fresh)
        if "--write" in args:
            shutil.rmtree(models, ignore_errors=True)
            shutil.copytree(fresh, models)
            print("contracts: Python models written")
            return 0
        found = drift(fresh, models)
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print("Run `make contracts-update` after fixing the schemas.", file=sys.stderr)
        return 1
    print("contracts: Python models have no drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
