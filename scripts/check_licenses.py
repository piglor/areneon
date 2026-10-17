"""Fail when a shipped (production) dependency uses a copyleft licence.

Reads the JSON printed by ``pnpm licenses list --prod --json`` on stdin.
ADR-001: Areneon is Apache-2.0; AGPL/GPL/SSPL code must not ship with it.
Dev-only tooling is out of scope.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

DENY = re.compile(r"\b(A?GPL|SSPL|EUPL|OSL)\b", re.IGNORECASE)


def denied(report: Mapping[str, Sequence[Mapping[str, object]]]) -> list[str]:
    """Return ``name@licence`` for every package whose licence is denied."""
    bad: list[str] = []
    for licence, packages in report.items():
        if not DENY.search(licence):
            continue
        for package in packages:
            bad.append(f"{package.get('name', '?')}@{licence}")
    return sorted(bad)


def main(stdin: TextIO = sys.stdin) -> int:
    raw = stdin.read().strip()
    # pnpm prints plain text, not JSON, when there are no production packages.
    report = json.loads(raw) if raw.startswith("{") else {}
    bad = denied(report)
    for entry in bad:
        print(f"denied licence in production dependency: {entry}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
