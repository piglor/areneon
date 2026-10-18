"""List the licences of the shipped Python packages' production dependencies.

ADR-008 §7 / #710: every Python runtime dependency of a shipped workspace
package, transitive ones included, must be on ``check_licenses.py``'s
allowlist. This prints the same JSON shape as ``pnpm licenses list --json``
(``{licence: [{"name": ...}]}``) so one checker judges both ecosystems:

    uv run python scripts/python_licences.py | python scripts/check_licenses.py

The closure comes from the lockfile (``uv export --no-dev --format
pylock.toml``): every package in it ships, workspace members and path
dependencies included, the shipped package itself too. A package's
licence is its ``License-Expression`` metadata. Without one, the trove
classifiers are mapped to SPDX through ``CLASSIFIERS``, an explicit table; an
unmapped or missing licence is reported as ``UNKNOWN``, which the checker
refuses (fail closed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata

# The shipped Python workspace packages (ADR-002): their production closure ships.
SHIPPED = ("areneon-contracts",)

CLASSIFIERS: Mapping[str, str] = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
UNKNOWN = "UNKNOWN"


class ClosureError(ValueError):
    """The exported closure is not a pylock this script can read."""


def closure(package: str, run: Callable[[list[str]], str] | None = None) -> list[str]:
    """Every package that ships with ``package``, itself included, from the lockfile."""
    command = [
        "uv",
        "export",
        "--package",
        package,
        "--no-dev",
        "--frozen",
        "--format",
        "pylock.toml",
    ]
    output = (
        run(command)
        if run
        else subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )
    try:
        entries = tomllib.loads(output)["packages"]
        names = {entry["name"] for entry in entries}
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        # Fail closed: an unread closure could hide an unchecked dependency.
        raise ClosureError(f"cannot read the closure of {package}: {exc!r}") from exc
    if package not in names:
        raise ClosureError(f"the closure of {package} does not list {package}")
    return sorted(names)


def licence_of(meta: metadata.PackageMetadata) -> str:
    """The SPDX licence a distribution declares, else its mapped classifier."""
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()
    mapped = {CLASSIFIERS[c] for c in meta.get_all("Classifier") or [] if c in CLASSIFIERS}
    return " AND ".join(sorted(mapped)) if mapped else UNKNOWN


def report(
    packages: Sequence[str], lookup: Callable[[str], metadata.PackageMetadata] = metadata.metadata
) -> dict[str, list[dict[str, str]]]:
    """``{licence: [{"name": package}]}`` for every package, as pnpm prints it."""
    out: dict[str, list[dict[str, str]]] = {}
    for name in packages:
        try:
            licence = licence_of(lookup(name))
        except metadata.PackageNotFoundError:
            licence = UNKNOWN
        out.setdefault(licence, []).append({"name": name})
    return out


def main(run: Callable[[list[str]], str] | None = None) -> int:
    try:
        packages = sorted({dep for package in SHIPPED for dep in closure(package, run)})
    except ClosureError as exc:
        print(f"python licences failed closed: {exc}", file=sys.stderr)
        return 1
    json.dump(report(packages), sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
