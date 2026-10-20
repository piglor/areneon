from __future__ import annotations

import io
import json
from email.message import Message
from importlib import metadata
from pathlib import Path

import pytest

from scripts import check_licenses
from scripts import python_licences as pl

PYLOCK = """
lock-version = "1.0"
[[packages]]
name = "areneon-contracts"
directory = { path = "packages/schema/python" }
[[packages]]
name = "areneon-ml"
directory = { path = "ml" }
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

    assert pl.closure("areneon-contracts", run) == [
        "agpl-py",
        "areneon-contracts",
        "areneon-ml",
        "pydantic",
    ]
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
        ({"Name": "Sympy", "Classifier": ["License :: OSI Approved :: BSD License"]}, pl.UNKNOWN),
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
    monkeypatch.setattr(
        pl, "report", lambda packages, **_kw: {"MIT": [{"name": p} for p in packages]}
    )
    assert pl.main(lambda _command: PYLOCK) == 0
    assert json.load(io.StringIO(capsys.readouterr().out))["MIT"][0] == {"name": "agpl-py"}
    assert pl.main(lambda _command: "garbage = [") == 1
    assert "failed closed" in capsys.readouterr().err


def test_the_shipped_closure_is_on_the_allowlist() -> None:
    report = pl.report(
        sorted({d for p in pl.SHIPPED for d in pl.closure(p)}), exceptions=pl.load_exceptions()
    )
    assert check_licenses.denied(report) == []


BSD = "License :: OSI Approved :: BSD License"
LISTED = {"sym-py": "BSD-3-Clause"}


def test_a_bsd_classifier_only_package_takes_its_named_exception() -> None:
    assert pl.licence_of(meta(Name="Sym_Py", Classifier=[BSD]), LISTED) == "BSD-3-Clause"
    assert pl.licence_of(meta(Name="sym.py", Classifier=[BSD]), LISTED) == "BSD-3-Clause"


def test_an_unlisted_bsd_classifier_only_package_is_still_refused() -> None:
    other = meta(Name="other", Classifier=[BSD])
    assert pl.licence_of(other, LISTED) == pl.UNKNOWN
    assert pl.licence_of(other) == pl.UNKNOWN
    assert pl.licence_of(meta(Name="", Classifier=[BSD]), LISTED) == pl.UNKNOWN
    unlisted = {"BSD-3-Clause": [{"name": "other"}]}
    assert check_licenses.denied(pl.report(["other"], lambda _n: other, LISTED)) == [
        "other@UNKNOWN"
    ]
    assert check_licenses.denied(unlisted) == []


def test_an_exception_does_not_cover_other_metadata() -> None:
    # Only a package with the bare BSD classifier and nothing else is excepted.
    assert pl.licence_of(meta(Name="sym-py"), LISTED) == pl.UNKNOWN
    assert pl.licence_of(meta(Name="sym-py", License="BSD"), LISTED) == pl.UNKNOWN
    assert pl.licence_of(meta(Name="sym-py", License_Expression="AGPL-3.0-only"), LISTED) == (
        "AGPL-3.0-only"
    )


def write_exceptions(path: Path, entries: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps({"exceptions": entries}))
    return path


GOOD = {
    "package": "Sym_Py",
    "licence": "BSD-3-Clause",
    "version": "1",
    "evidence": "LICENSE in the wheel",
    "verified": "2026-10-20",
}


def test_the_exception_list_is_loaded_by_normalised_name(tmp_path: Path) -> None:
    assert pl.load_exceptions(write_exceptions(tmp_path / "e.json", [GOOD])) == LISTED


@pytest.mark.parametrize(
    "entry",
    [
        {**GOOD, "extra": "x"},
        {k: v for k, v in GOOD.items() if k != "evidence"},
        {**GOOD, "licence": " "},
        {**GOOD, "verified": "yesterday"},
        {**GOOD, "version": 1},
    ],
)
def test_a_malformed_exception_entry_is_refused(tmp_path: Path, entry: dict[str, str]) -> None:
    with pytest.raises(pl.ExceptionListError):
        pl.load_exceptions(write_exceptions(tmp_path / "e.json", [entry]))


def test_a_duplicate_or_unreadable_exception_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(pl.ExceptionListError, match="listed twice"):
        pl.load_exceptions(
            write_exceptions(tmp_path / "e.json", [GOOD, {**GOOD, "package": "sym-py"}])
        )
    with pytest.raises(pl.ExceptionListError, match="cannot read"):
        pl.load_exceptions(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text('{"exceptions": "no"}')
    with pytest.raises(pl.ExceptionListError, match="cannot read"):
        pl.load_exceptions(tmp_path / "bad.json")
    with pytest.raises(pl.ExceptionListError, match="exactly"):
        pl.load_exceptions(write_exceptions(tmp_path / "n.json", ["x"]))  # type: ignore[list-item]


def test_main_fails_closed_on_a_bad_exception_list(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    load, missing = pl.load_exceptions, tmp_path / "missing.json"
    monkeypatch.setattr(pl, "load_exceptions", lambda: load(missing))
    assert pl.main(lambda _command: PYLOCK) == 1
    assert "cannot read missing.json" in capsys.readouterr().err


def test_the_committed_exceptions_are_the_verified_bsd_packages() -> None:
    assert pl.load_exceptions() == {
        "jinja2": "BSD-3-Clause",
        "mpmath": "BSD-3-Clause",
        "sympy": "BSD-3-Clause",
    }
