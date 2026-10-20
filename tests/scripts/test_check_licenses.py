from __future__ import annotations

import io
import json

import pytest

from scripts import check_licenses

# Shape of `pnpm -r licenses list --prod --json` output.
PNPM_AGPL = {
    "MIT": [{"name": "ok", "versions": ["1.0.0"], "license": "MIT"}],
    "AGPL-3.0-or-later": [
        {"name": "pokemon-showdown-client", "versions": ["0.0.0"], "license": "AGPL-3.0-or-later"}
    ],
}


@pytest.mark.parametrize(
    ("expression", "allowed"),
    [
        ("MIT", True),
        ("Apache-2.0", True),
        ("MIT OR GPL-3.0", True),
        ("(MIT OR Apache-2.0) AND BSD-3-Clause", True),
        ("MIT AND GPL-3.0", False),
        ("GPL-3.0", False),
        ("LGPL-2.1", False),
        ("UNKNOWN", False),
        ("", False),
        ("(MIT", False),
        ("MIT OR", False),
        ("MIT GPL-3.0", False),
        ("OR MIT", False),
        ("MIT AND )", False),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("GPL-2.0 WITH Classpath-exception-2.0", False),
        ("MIT WITH", False),
        ("MIT WITH OR ISC", False),
        ("MIT or ISC", False),
        # ADR-001 clarification: BSL-1.0 and Zlib are allowed (torch, numpy expressions).
        ("BSL-1.0", True),
        ("Zlib", True),
        (
            "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause"
            " AND BSL-1.0 AND MIT",
            True,
        ),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", True),
        # No CUDA or proprietary licence is allowlisted (ADR-010 §5.2).
        ("LicenseRef-NVIDIA-Proprietary", False),
        ("LicenseRef-NVIDIA-SOFTWARE-LICENSE", False),
        ("MIT AND LicenseRef-NVIDIA-Proprietary", False),
    ],
)
def test_is_allowed(expression: str, allowed: bool) -> None:
    assert check_licenses.is_allowed(expression) is allowed


def test_denied_lists_injected_agpl() -> None:
    assert check_licenses.denied(PNPM_AGPL) == ["pokemon-showdown-client@AGPL-3.0-or-later"]


def test_main_passes_permissive_and_no_packages() -> None:
    assert check_licenses.main(io.StringIO(check_licenses.NO_PACKAGES + "\n")) == 0
    assert check_licenses.main(io.StringIO(json.dumps({"MIT": [{"name": "a"}]}))) == 0


def test_main_fails_on_injected_agpl() -> None:
    assert check_licenses.main(io.StringIO(json.dumps(PNPM_AGPL))) == check_licenses.EXIT_DENIED


@pytest.mark.parametrize("raw", ["", "ERR_PNPM_SOMETHING failed", "[]"])
def test_main_fails_closed_on_bad_output(raw: str) -> None:
    assert check_licenses.main(io.StringIO(raw)) == check_licenses.EXIT_BAD_OUTPUT


def test_main_reads_stdin_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(check_licenses.NO_PACKAGES))
    assert check_licenses.main() == 0
