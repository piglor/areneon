from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import migrate_check as mc

# A golden-comparing "check": exits 1 when the file differs from the expected text,
# exactly what a vitest golden test does. The command is a real subprocess.
COMPARE = (
    sys.executable,
    "-c",
    "import sys,pathlib;"
    "ok=pathlib.Path('golden.json').read_text()==pathlib.Path('expected.json').read_text();"
    "print('' if ok else 'FAIL  golden.test.ts > matches a fresh capture');"
    "sys.exit(0 if ok else 1)",
)
RECAPTURE = (
    sys.executable,
    "-c",
    "import pathlib as p;p.Path('golden.json').write_text(p.Path('expected.json').read_text())",
)
REWRITER = (
    sys.executable,
    "-c",
    "import pathlib;pathlib.Path('golden.json').write_text('silently rewritten')",
)


def check(
    commands: tuple[tuple[str, ...], ...] = (COMPARE,), recapture: tuple[tuple[str, ...], ...] = ()
) -> mc.Check:
    return mc.Check("golden", "golden compare", commands, ("golden.json",), recapture)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "golden.json").write_text('{"v": 1}')
    (tmp_path / "expected.json").write_text('{"v": 1}')
    return tmp_path


def env(root: Path) -> mc.Env:
    return mc.Env(root, nice=False)


def stub_runner(results: list[mc.Result]) -> list[mc.Result]:
    """A suite that ignores drift (the thing the proof must reject)."""
    return [mc.Result(r.id, r.title, mc.PASS, 0.0) for r in results]


def drift_is_reported(results: list[mc.Result]) -> bool:
    (r,) = results
    return r.status == mc.DRIFT and any("golden.test.ts" in d for d in r.drift)


def test_clean_golden_passes(root: Path) -> None:
    (r,) = mc.run_checks([check()], env(root))
    assert r.status == mc.PASS
    assert r.drift == []


def test_doctored_golden_is_reported_as_drift(root: Path) -> None:
    (root / "golden.json").write_text('{"v": 2}')  # the doctoring
    results = mc.run_checks([check()], env(root))
    assert drift_is_reported(results)
    assert "DRIFT" in mc.summary(results)
    assert "golden.test.ts" in mc.summary(results)
    assert mc.main(["--no-nice"], env(root), [check()]) == 1


def test_a_stub_that_ignores_drift_fails_the_proof(root: Path) -> None:
    (root / "golden.json").write_text('{"v": 2}')
    results = mc.run_checks([check()], env(root))
    assert not drift_is_reported(stub_runner(results))  # negative control: the assertion has teeth


def test_compare_never_rewrites_a_golden_even_when_it_passes(root: Path) -> None:
    (r,) = mc.run_checks([check((REWRITER,))], env(root))
    assert r.status == mc.REWRITTEN
    assert any("golden.json" in d for d in r.drift)


def test_recapture_rewrites_and_lists_changed_files(root: Path) -> None:
    (root / "golden.json").write_text('{"v": 2}')
    (r,) = mc.run_checks([check(recapture=(RECAPTURE,))], env(root), recapture=True)
    assert r.status == mc.PASS
    assert r.changed_files == ["golden.json"]
    assert (root / "golden.json").read_text() == '{"v": 1}'
    assert "golden.json" in mc.summary([r])


def test_recapture_failure_is_an_error(root: Path) -> None:
    bad = (sys.executable, "-c", "import sys;print('boom');sys.exit(3)")
    (r,) = mc.run_checks([check(recapture=(bad,))], env(root), recapture=True)
    assert r.status == mc.ERROR


def test_missing_tool_and_timeout_are_errors(root: Path) -> None:
    (r,) = mc.run_checks([check(commands=(("no-such-tool-xyz",),))], env(root))
    assert r.status == mc.ERROR
    slow = (sys.executable, "-c", "import time;time.sleep(5)")
    (r,) = mc.run_checks([check(commands=(slow,))], mc.Env(root, nice=False, timeout_s=1))
    assert r.status == mc.ERROR
    assert "timed out" in r.drift[0]


def test_absent_check_is_listed_not_failed(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    absent = mc.Check("later", "not built", absent="not on main")
    assert mc.main(["--no-nice"], env(root), [absent]) == 0
    assert "ABSENT" in capsys.readouterr().out


def test_cli_options(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    assert mc.main(["--no-nice", "--json", str(out), "--only", "gold*"], env(root), [check()]) == 0
    assert json.loads(out.read_text())[0]["status"] == mc.PASS
    assert mc.main(["--list"], env(root), [check()]) == 0
    assert "golden: golden compare" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        mc.main(["--only", "nope"], env(root), [check()])
    (root / "golden.json").write_text("x")
    assert mc.main(["--no-nice", "--recapture"], env(root), [check(recapture=(RECAPTURE,))]) == 0


def test_drift_lines_formats() -> None:
    assert mc.drift_lines(" FAIL  a.test.ts > x\n FAILED tests/b.py::t - boom\n") == [
        "a.test.ts > x",
        "tests/b.py::t - boom",
    ]
    assert mc.drift_lines("noise\nAssertionError: bad\n") == ["AssertionError: bad"]
    assert mc.drift_lines("a\nb\nc\nd\n") == ["b", "c", "d"]


def test_expand_and_environ(root: Path) -> None:
    e = mc.Env(root, python="PY", nice=True)
    assert "PY" in mc.expand(("{py}", "-m", "x"), e)
    assert str(root / "ml") in e.environ()["PYTHONPATH"]
    assert e.environ().get("ARENEON_PYTHON")


def test_manifest_is_in_dependency_order_and_references_real_files() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    ids = [c.id for c in mc.CHECKS]
    assert ids[0] == "pin-and-rules"
    assert ids.index("engine-tables") < ids.index("mechanics-probe") < ids.index("contract-corpus")
    assert len(ids) == len(set(ids))
    for c in mc.CHECKS:
        for argv in c.commands:
            for a in argv:
                if (a.startswith(("packages/", "tests/")) and a.endswith((".ts", ".py"))) or a in (
                    "tests/contracts",
                ):
                    assert (root / a).exists(), a
