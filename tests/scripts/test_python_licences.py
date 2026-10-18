from __future__ import annotations

import io
import json
from email.message import Message
from importlib import metadata

import pytest

from scripts import check_licenses
from scripts import python_licences as pl

PYLOCK = """
lock-version = "1.0"
[[packages]]
name = "areneon-contracts"
directory = { path = "packages/schema/python" }
[[packages]]
name = "pydantic"
version = "2.14.1"
[[packages]]
name = "agpl-py"
directory = { path = "agpl-py" }
"""


def meta(**fields: str | list[str]) -> metadata.PackageMetadata:
    message = Message()
    for key, value in fields.items():
        for item in value if isinstance(value, list) else [value]:
            message[key.replace("_", "-")] = item
    return message  # type: ignore[return-value]


def test_closure_lists_every_package_of_the_pylock_workspace_ones_included() -> None:
    seen: list[list[str]] = []

    def run(command: list[str]) -> str:
        seen.append(command)
        return PYLOCK

    assert pl.closure("areneon-contracts", run) == ["agpl-py", "areneon-contracts", "pydantic"]
    assert seen[0][:4] == ["uv", "export", "--package", "areneon-contracts"]
    assert "pylock.toml" in seen[0]


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("not = [toml", "cannot read the closure"),
        ('lock-version = "1.0"\n', "cannot read the closure"),
        ('[[packages]]\nversion = "1"\n', "cannot read the closure"),
        ('[[packages]]\nname = "other"\n', "does not list areneon-contracts"),
    ],
)
def test_closure_fails_closed(output: str, reason: str) -> None:
    with pytest.raises(pl.ClosureError, match=reason):
        pl.closure("areneon-contracts", lambda _command: output)


@pytest.mark.parametrize(
    ("fields", "licence"),
    [
        ({"License_Expression": " MIT "}, "MIT"),
        ({"Classifier": ["License :: OSI Approved :: Apache Software License"]}, "Apache-2.0"),
        (
            {
                "Classifier": [
                    "License :: OSI Approved :: MIT License",
                    "License :: OSI Approved :: Python Software Foundation License",
                ]
            },
            "MIT AND PSF-2.0",
        ),
        ({"Classifier": ["License :: OSI Approved :: BSD License"]}, pl.UNKNOWN),
        ({"License": "MIT"}, pl.UNKNOWN),
    ],
)
def test_licence_of_prefers_the_expression_then_mapped_classifiers(
    fields: dict[str, str | list[str]], licence: str
) -> None:
    assert pl.licence_of(meta(**fields)) == licence


def test_report_groups_by_licence_and_marks_missing_packages_unknown() -> None:
    known = {"a": meta(License_Expression="MIT"), "b": meta(License_Expression="AGPL-3.0-only")}

    def lookup(name: str) -> metadata.PackageMetadata:
        if name not in known:
            raise metadata.PackageNotFoundError(name)
        return known[name]

    assert pl.report(["a", "b", "c"], lookup) == {
        "MIT": [{"name": "a"}],
        "AGPL-3.0-only": [{"name": "b"}],
        pl.UNKNOWN: [{"name": "c"}],
    }


def test_main_prints_the_report_or_fails_closed(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pl, "report", lambda packages: {"MIT": [{"name": p} for p in packages]})
    assert pl.main(lambda _command: PYLOCK) == 0
    assert json.load(io.StringIO(capsys.readouterr().out))["MIT"][0] == {"name": "agpl-py"}
    assert pl.main(lambda _command: "garbage = [") == 1
    assert "failed closed" in capsys.readouterr().err


def test_the_shipped_closure_is_on_the_allowlist() -> None:
    report = pl.report(pl.closure("areneon-contracts"))
    assert check_licenses.denied(report) == []
