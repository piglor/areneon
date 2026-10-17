from __future__ import annotations

import io
import json

from scripts import check_licenses


def test_allows_permissive() -> None:
    report = {"MIT": [{"name": "a"}], "Apache-2.0": [{"name": "b"}]}
    assert check_licenses.denied(report) == []


def test_denies_copyleft() -> None:
    report = {
        "AGPL-3.0-or-later": [{"name": "client"}],
        "GPL-2.0": [{"name": "x"}],
        "MIT": [{"name": "ok"}],
        "LGPL-2.1": [{"name": "lib"}],
    }
    assert check_licenses.denied(report) == ["client@AGPL-3.0-or-later", "x@GPL-2.0"]


def test_main_exit_codes() -> None:
    assert check_licenses.main(io.StringIO("")) == 0
    assert check_licenses.main(io.StringIO("No licenses in packages found")) == 0
    assert check_licenses.main(io.StringIO(json.dumps({"MIT": [{"name": "a"}]}))) == 0
    assert check_licenses.main(io.StringIO(json.dumps({"SSPL-1.0": [{}]}))) == 1
