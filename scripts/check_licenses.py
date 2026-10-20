"""Fail unless every shipped (production) dependency has an allowed licence.

Reads the JSON printed by ``pnpm -r licenses list --prod --json`` on stdin,
or the same shape from ``scripts/python_licences.py`` for Python.
ADR-001: Areneon is Apache-2.0, so only permissive licences may ship with
it. The check is an allowlist and fails closed: an unknown licence, a
missing licence or unparseable output is an error.

SPDX expressions are honoured: ``A OR B`` passes if either side is allowed,
``A AND B`` only if both are, and ``A WITH exception`` is judged on ``A``
(an exception only grants extra permissions). Operators are matched in
upper case as SPDX specifies; anything else fails closed.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

ALLOWED = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BlueOak-1.0.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        # ADR-001 clarification (2026-10-20), ADR-010 Q9: OSI-approved, permissive.
        "BSL-1.0",
        "CC0-1.0",
        "CC-BY-4.0",
        "ISC",
        "MIT",
        # typing-extensions (via pydantic): OSI-approved, permissive (ADR-008 §7).
        "PSF-2.0",
        "Python-2.0",
        "Unlicense",
        "Zlib",
    }
)

EXIT_DENIED = 1
EXIT_BAD_OUTPUT = 2

# Exact text pnpm prints instead of JSON when there are no production packages.
NO_PACKAGES = "No licenses in packages found"

_OPERATORS = frozenset({"AND", "OR", "WITH", ")"})
_TOKEN = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bWITH\b|[^\s()]+")


class LicenceOutputError(ValueError):
    """The pnpm output could not be understood."""


class _MalformedExpressionError(ValueError):
    """An SPDX expression is missing an operand or a parenthesis."""


def is_allowed(expression: str) -> bool:
    """Evaluate an SPDX licence expression against ``ALLOWED``."""
    try:
        value, rest = _parse_or(_TOKEN.findall(expression))
    except _MalformedExpressionError:
        return False
    return value and not rest


def _parse_or(tokens: list[str]) -> tuple[bool, list[str]]:
    value, rest = _parse_and(tokens)
    while rest and rest[0] == "OR":
        right, rest = _parse_and(rest[1:])
        value = value or right
    return value, rest


def _parse_and(tokens: list[str]) -> tuple[bool, list[str]]:
    value, rest = _parse_atom(tokens)
    while rest and rest[0] == "AND":
        right, rest = _parse_atom(rest[1:])
        value = value and right
    return value, rest


def _parse_atom(tokens: list[str]) -> tuple[bool, list[str]]:
    if not tokens or tokens[0] in _OPERATORS:
        raise _MalformedExpressionError
    head, rest = tokens[0], tokens[1:]
    if head == "(":
        value, rest = _parse_or(rest)
        if not rest or rest[0] != ")":
            raise _MalformedExpressionError
        return value, rest[1:]
    if rest and rest[0] == "WITH":
        exception = rest[1:2]
        if not exception or exception[0] in _OPERATORS:
            raise _MalformedExpressionError
        rest = rest[2:]
    return head in ALLOWED, rest


def parse_report(raw: str) -> Mapping[str, Sequence[Mapping[str, object]]]:
    text = raw.strip()
    if text == NO_PACKAGES:
        return {}
    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LicenceOutputError(f"pnpm licences output is not JSON: {text[:80]!r}") from exc
    if not isinstance(report, dict):
        raise LicenceOutputError("pnpm licences output is not a JSON object")
    return report


def denied(report: Mapping[str, Sequence[Mapping[str, object]]]) -> list[str]:
    """Return ``name@licence`` for every package whose licence is not allowed."""
    bad: list[str] = []
    for licence, packages in report.items():
        if is_allowed(licence):
            continue
        for package in packages:
            bad.append(f"{package.get('name', '?')}@{licence}")
    return sorted(bad)


def main(stdin: TextIO | None = None) -> int:
    source = sys.stdin if stdin is None else stdin
    try:
        report = parse_report(source.read())
    except LicenceOutputError as exc:
        print(f"licence check failed closed: {exc}", file=sys.stderr)
        return EXIT_BAD_OUTPUT
    bad = denied(report)
    for entry in bad:
        print(f"licence not on the allowlist in production dependency: {entry}", file=sys.stderr)
    return EXIT_DENIED if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
