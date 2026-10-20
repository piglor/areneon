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
classifiers are mapped to SPDX through ``CLASSIFIERS``, an explicit table. The
bare BSD classifier is never mapped: it does not say which BSD variant
applies. A package that carries only that classifier is admitted through the
named per-package list in ``licence_exceptions.json`` (ADR-001 clarification,
ADR-010 §5.1). An unmapped or missing licence is reported as ``UNKNOWN``,
which the checker refuses (fail closed).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from importlib import metadata
from pathlib import Path

# The shipped Python workspace packages (ADR-002): their production closure ships.
SHIPPED = ("areneon-contracts", "areneon-ml")

CLASSIFIERS: Mapping[str, str] = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}
UNKNOWN = "UNKNOWN"
BSD_CLASSIFIER = "License :: OSI Approved :: BSD License"
EXCEPTIONS_FILE = Path(__file__).with_name("licence_exceptions.json")
_ENTRY_KEYS = frozenset({"package", "licence", "version", "evidence", "verified"})


class ClosureError(ValueError):
    """The exported closure is not a pylock this script can read."""


class ExceptionListError(ValueError):
    """The per-package exception list is missing or malformed."""


def normalise(name: str) -> str:
    """The PEP 503 form of a package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load_exceptions(path: Path = EXCEPTIONS_FILE) -> dict[str, str]:
    """``{normalised package: licence}`` from the reviewed exception list.

    Fails closed: a missing file, a malformed entry (wrong keys, a blank field,
    a date that is not ISO) or a package listed twice is an error.
    """
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["exceptions"]
        if not isinstance(entries, list):
            raise TypeError("exceptions is not a list")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExceptionListError(f"cannot read {path.name}: {exc!r}") from exc
    out: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise ExceptionListError(
                f"{path.name}: an entry must have exactly {sorted(_ENTRY_KEYS)}"
            )
        if not all(isinstance(v, str) and v.strip() for v in entry.values()):
            raise ExceptionListError(f"{path.name}: {entry.get('package')!r} has a blank field")
        try:
            date.fromisoformat(entry["verified"])
        except ValueError as exc:
            raise ExceptionListError(f"{path.name}: {entry['package']!r} has a bad date") from exc
        name = normalise(entry["package"])
        if name in out:
            raise ExceptionListError(f"{path.name}: {name!r} is listed twice")
        out[name] = entry["licence"].strip()
    return out


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


def licence_of(meta: metadata.PackageMetadata, exceptions: Mapping[str, str] | None = None) -> str:
    """The SPDX licence a distribution declares, else its mapped classifier.

    A distribution whose only licence metadata is the bare BSD classifier gets
    the licence its entry in ``exceptions`` names, and ``UNKNOWN`` if unlisted.
    """
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()
    classifiers = meta.get_all("Classifier") or []
    mapped = {CLASSIFIERS[c] for c in classifiers if c in CLASSIFIERS}
    if mapped:
        return " AND ".join(sorted(mapped))
    if BSD_CLASSIFIER in classifiers and exceptions:
        return exceptions.get(normalise(meta.get("Name") or ""), UNKNOWN)
    return UNKNOWN


def report(
    packages: Sequence[str],
    lookup: Callable[[str], metadata.PackageMetadata] = metadata.metadata,
    exceptions: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """``{licence: [{"name": package}]}`` for every package, as pnpm prints it."""
    out: dict[str, list[dict[str, str]]] = {}
    for name in packages:
        try:
            licence = licence_of(lookup(name), exceptions)
        except metadata.PackageNotFoundError:
            licence = UNKNOWN
        out.setdefault(licence, []).append({"name": name})
    return out


def main(run: Callable[[list[str]], str] | None = None) -> int:
    try:
        packages = sorted({dep for package in SHIPPED for dep in closure(package, run)})
        exceptions = load_exceptions()
    except (ClosureError, ExceptionListError) as exc:
        print(f"python licences failed closed: {exc}", file=sys.stderr)
        return 1
    json.dump(report(packages, exceptions=exceptions), sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
