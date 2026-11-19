# ruff: noqa: E501
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts import migrate_check as mc

# A golden-comparing "check": exits 1 when the golden differs from the expected text, exactly
# what a vitest golden test does. Honours ARENEON_UPDATE_GOLDEN the way test_golden.py does.
COMPARE = (
    sys.executable,
    "-c",
    "import os,sys,pathlib as p;g=p.Path('fixtures/golden.json');e=p.Path('fixtures/expected.json');"
    "g.write_text('rewritten') if os.environ.get('ARENEON_UPDATE_GOLDEN')=='1' else None;"
    "ok=g.read_text()==e.read_text();"
    "print('' if ok else 'FAIL  golden.test.ts > matches a fresh capture');"
    "sys.exit(0 if ok else 1)",
)
RECAPTURE = (
    sys.executable,
    "-c",
    "import pathlib as p;g=p.Path('fixtures/golden.json');g.write_text(p.Path('fixtures/expected.json').read_text())",
)


def writer(path: str, text: str = "silently rewritten") -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        f"import pathlib as p;f=p.Path({path!r});f.parent.mkdir(parents=True,exist_ok=True);f.write_text({text!r})",
    )


def check(
    commands: tuple[tuple[str, ...], ...] = (COMPARE,),
    recapture: tuple[tuple[str, ...], ...] = (),
    needs: tuple[str, ...] = (),
) -> mc.Check:
    return mc.Check("golden", "golden compare", commands, recapture, needs=needs)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "fixtures/golden.json").write_text('{"v": 1}')
    (tmp_path / "fixtures/expected.json").write_text('{"v": 1}')
    (tmp_path / "config/other.json").write_text("{}")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "x")
    return tmp_path


def env(root: Path, **kw: object) -> mc.Env:
    return mc.Env(root, nice=False, **kw)  # type: ignore[arg-type]


def doctor(root: Path) -> None:
    (root / "fixtures/golden.json").write_text('{"v": 2}')


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
    doctor(root)
    results = mc.run_checks([check()], env(root))
    assert drift_is_reported(results)
    assert "DRIFT" in mc.summary(results)
    assert "golden.test.ts" in mc.summary(results)
    assert mc.main(["--no-nice"], env(root), [check()]) == 1


def test_a_stub_that_ignores_drift_fails_the_proof(root: Path) -> None:
    doctor(root)
    results = mc.run_checks([check()], env(root))
    assert not drift_is_reported(stub_runner(results))  # negative control: the assertion has teeth


def test_update_golden_in_the_environment_cannot_rewrite_a_golden(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "ARENEON_UPDATE_GOLDEN",
        "ARENEON_HISTORY_PARITY_DUMP",
        "ARENEON_PRINT_PAGE_DIGESTS",
    ):
        monkeypatch.setenv(name, "1")
    e = env(root)
    scrubbed = e.environ()
    assert not [
        k for k in scrubbed if k.startswith(("ARENEON_UPDATE", "ARENEON_PRINT", "ARENEON_HISTORY"))
    ]
    (r,) = mc.run_checks([check()], e)
    assert r.status == mc.PASS
    assert (root / "fixtures/golden.json").read_text() == '{"v": 1}'
    assert git_clean(root)


def test_recapture_environment_enables_the_update_variable_only_there() -> None:
    e = mc.Env(Path("."), nice=False)
    assert e.environ(recapture=True)["ARENEON_UPDATE_GOLDEN"] == "1"
    assert "ARENEON_UPDATE_GOLDEN" not in e.environ()


def git_clean(root: Path) -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    )
    return out.stdout == ""


@pytest.mark.parametrize(
    "path",
    [
        "fixtures/golden.json",
        "config/other.json",
        "docs/new.md",
        "ml/areneon_ml/reducer/data/x.json",
    ],
)
def test_any_write_to_watched_data_is_reported_even_when_the_check_passes(
    root: Path, path: str
) -> None:
    (r,) = mc.run_checks([check((writer(path),))], env(root))
    assert r.status == mc.REWRITTEN
    assert any(path in d for d in r.drift)


def test_a_file_already_dirty_and_changed_again_is_reported(root: Path) -> None:
    (root / "config/other.json").write_text("dirty")
    (r,) = mc.run_checks([check((writer("config/other.json", "dirtier"),))], env(root))
    assert r.status == mc.REWRITTEN


def test_recapture_rewrites_and_lists_changed_files(root: Path) -> None:
    doctor(root)  # the committed golden is stale against what the engine now produces
    (r,) = mc.run_checks([check(recapture=(RECAPTURE,))], env(root), recapture=True)
    assert r.status == mc.PASS
    assert (root / "fixtures/golden.json").read_text() == '{"v": 1}'
    (root / "fixtures/expected.json").write_text('{"v": 3}')
    (r,) = mc.run_checks([check(recapture=(RECAPTURE,))], env(root), recapture=True)
    assert r.status == mc.PASS
    assert "fixtures/golden.json" in r.changed_files
    assert "fixtures/golden.json" in mc.summary([r])


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


def test_timeout_kills_the_whole_process_group(root: Path, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    cmd = ("sh", "-c", f"(sleep 2; touch {marker}) & sleep 30")
    (r,) = mc.run_checks([check(commands=(cmd,))], mc.Env(root, nice=False, timeout_s=1))
    assert r.status == mc.ERROR
    time.sleep(3)
    assert not marker.exists()


def test_a_missing_interpreter_turns_the_check_red(root: Path) -> None:
    gone = env(root, python="/nonexistent/venv/bin/python")
    (r,) = mc.run_checks([check(needs=("python",))], gone)
    assert r.status == mc.ERROR
    assert "python interpreter missing" in r.drift[0]


def test_a_missing_engine_checkout_turns_the_check_red(root: Path) -> None:
    (r,) = mc.run_checks([check(needs=("engine",))], env(root))
    assert r.status == mc.ERROR
    assert "engine checkout" in r.drift[0]
    (root / "vendor/pokemon-showdown/dist/sim").mkdir(parents=True)
    (r,) = mc.run_checks([check(needs=("engine",))], env(root))
    assert r.status == mc.PASS


def fake_vitest(results: list[dict[str, Any]], returncode: int = 0):  # type: ignore[no-untyped-def]
    """An env.run stand-in that writes vitest's JSON report."""

    def run(argv: Sequence[str], cwd: Path, environ: dict[str, str], timeout: float) -> mc.Proc:
        out = next(a.split("=", 1)[1] for a in argv if a.startswith("--outputFile="))
        Path(out).write_text(json.dumps({"testResults": results}))
        return mc.Proc(returncode, "")

    return run


def vfile(root: Path, name: str, *statuses: str) -> dict[str, Any]:
    return {
        "name": str(root / name),
        "status": "failed" if "failed" in statuses else "passed",
        "assertionResults": [{"status": s, "fullName": f"t{i}"} for i, s in enumerate(statuses)],
    }


VT = mc.Check("v", "v", (mc.vitest("packages/a/test/x.test.ts"),), needs=())


def test_vitest_all_passed(root: Path) -> None:
    e = env(root, run=fake_vitest([vfile(root, "packages/a/test/x.test.ts", "passed")]))
    assert mc.run_checks([VT], e)[0].status == mc.PASS


def test_a_skipped_vitest_test_fails_the_gate(root: Path) -> None:
    e = env(root, run=fake_vitest([vfile(root, "packages/a/test/x.test.ts", "passed", "skipped")]))
    (r,) = mc.run_checks([VT], e)
    assert r.status == mc.SKIPPED
    assert "1 skipped" in r.drift[0]


def test_a_vitest_failure_names_the_test(root: Path) -> None:
    e = env(root, run=fake_vitest([vfile(root, "packages/a/test/x.test.ts", "failed")], 1))
    (r,) = mc.run_checks([VT], e)
    assert r.status == mc.DRIFT
    assert "packages/a/test/x.test.ts > t0" in r.drift


def test_a_named_vitest_file_that_did_not_run_is_drift(root: Path) -> None:
    e = env(root, run=fake_vitest([vfile(root, "packages/a/test/x.test.ts.other", "passed")]))
    (r,) = mc.run_checks([VT], e)
    assert r.status == mc.DRIFT
    assert any("did not run" in d for d in r.drift)


def test_a_suite_that_failed_to_load_is_drift(root: Path) -> None:
    broken = {
        "name": str(root / "packages/a/test/x.test.ts"),
        "status": "failed",
        "message": "boom",
    }
    (r,) = mc.run_checks([VT], env(root, run=fake_vitest([broken], 1)))
    assert r.status == mc.DRIFT
    assert "boom" in r.drift[0]


def test_a_missing_vitest_report_is_drift(root: Path) -> None:
    def run(argv: Sequence[str], cwd: Path, environ: dict[str, str], timeout: float) -> mc.Proc:
        return mc.Proc(1, "FAIL  x.test.ts > broke\n")

    (r,) = mc.run_checks([VT], env(root, run=run))
    assert r.status == mc.DRIFT
    assert "x.test.ts > broke" in r.drift[0]


def fake_pytest(output: str, returncode: int = 0):  # type: ignore[no-untyped-def]
    def run(argv: Sequence[str], cwd: Path, environ: dict[str, str], timeout: float) -> mc.Proc:
        return mc.Proc(returncode, output)

    return run


PT = mc.Check("p", "p", (mc.pytest("tests/x.py"),), needs=())


def test_a_skipped_pytest_test_fails_the_gate(root: Path) -> None:
    out = "SKIPPED [1] tests/x.py:3: needs the engine\n3 passed, 1 skipped in 0.1s\n"
    (r,) = mc.run_checks([PT], env(root, run=fake_pytest(out)))
    assert r.status == mc.SKIPPED
    assert "needs the engine" in r.drift[0]
    out = "3 passed, 2 skipped in 0.1s\n"
    (r,) = mc.run_checks([PT], env(root, run=fake_pytest(out)))
    assert r.status == mc.SKIPPED


def test_pytest_pass_and_failure(root: Path) -> None:
    assert (
        mc.run_checks([PT], env(root, run=fake_pytest("3 passed in 0.1s\n")))[0].status == mc.PASS
    )
    (r,) = mc.run_checks([PT], env(root, run=fake_pytest("FAILED tests/x.py::t - boom\n", 1)))
    assert r.status == mc.DRIFT


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
    assert mc.main(["--no-nice", "--recapture"], env(root), [check(recapture=(RECAPTURE,))]) == 0


def test_drift_lines_formats() -> None:
    out = mc.drift_lines(" FAIL  a.test.ts > x\n FAILED tests/b.py::t - boom\n")
    assert out == ["a.test.ts > x", "tests/b.py::t - boom"]
    assert mc.drift_lines("noise\nAssertionError: bad\n") == ["AssertionError: bad"]
    assert mc.drift_lines("a\nb\nc\nd\n") == ["b", "c", "d"]


def test_expand_and_environ(root: Path) -> None:
    e = mc.Env(root, python="PY", nice=True)
    assert "PY" in mc.expand(("{py}", "-m", "x"), e)
    assert str(root / "ml") in e.environ()["PYTHONPATH"]
    assert e.environ()["ARENEON_PYTHON"] == "PY"
    assert e.environ()["ARENEON_REQUIRE_PYTHON_WORKER"] == "1"


# The frozen manifest: the checks, in dependency order, and the test files and scripts of each.
# A ticketed item (mechanics inventory, volatile copy, stall moves, the choice oracle, the contract
# corpus, the gap doc) disappearing from a check fails here and must be a reviewed edit.
B = "packages/showdown-bridge"
EXPECTED: dict[str, tuple[str, ...]] = {
    "pin-and-rules": (
        f"{B}/test/engine.test.ts",
        f"{B}/test/rules-channel.test.ts",
        f"{B}/test/gaps-doc.test.ts",
        f"{B}/scripts/check-rules.ts",
    ),
    "golden-fixtures": (
        f"{B}/test/golden.test.ts",
        f"{B}/test/teams.test.ts",
        f"{B}/test/random-corpus.test.ts",
    ),
    "choice-corpus": (
        f"{B}/test/choice-corpus.test.ts",
        f"{B}/test/choice-channels.test.ts",
        f"{B}/test/choices.test.ts",
    ),
    "engine-tables": (
        f"{B}/test/engine-data.test.ts",
        f"{B}/test/team-vocabulary.test.ts",
        f"{B}/test/role-sets.test.ts",
        f"{B}/test/move-classes.test.ts",
        f"{B}/test/volatile-copy.test.ts",
        f"{B}/test/stall-moves.test.ts",
        f"{B}/test/history-facts-data.test.ts",
        f"{B}/test/mechanics-inventory.test.ts",
        f"{B}/test/mechanics-inventory-weights.test.ts",
        "tests/ml/reducer/test_protocol_volatiles.py",
    ),
    "mechanics-probe": (
        f"{B}/test/mechanics-speed.test.ts",
        f"{B}/test/mechanics-speed-battle.test.ts",
        f"{B}/test/mechanics-damage.test.ts",
        f"{B}/test/mechanics-damage-paths.test.ts",
        f"{B}/test/mechanics-ko.test.ts",
        f"{B}/test/mechanics-ko-candidates.test.ts",
        f"{B}/test/mechanics-stats.test.ts",
        f"{B}/test/mechanics-parity.test.ts",
        f"{B}/scripts/check-mechanics-config.ts",
        f"{B}/scripts/check-damage-aware-config.ts",
    ),
    "calc-differential": (f"{B}/test/mechanics-differential.test.ts",),
    "reader-parity": (
        "packages/public-state/test/parity.test.ts",
        "packages/public-state/test/history-facts-parity.test.ts",
        "packages/public-state/test/baton-pass.test.ts",
        f"{B}/test/world-facts-parity.test.ts",
        f"{B}/test/history-facts-parity.test.ts",
        "tests/ml/reducer/test_golden.py",
        "tests/ml/reducer/test_corpus.py",
        "tests/ml/reducer/test_unavailable.py",
        "tests/ml/reducer/test_illusion_battle.py",
        "tests/ml/reducer/test_fling.py",
        "tests/ml/reducer/test_baton_pass.py",
        "tests/ml/reducer/test_baton_pass_volatiles.py",
    ),
    "battle-state-spec-oracle": (
        f"{B}/test/battle-state-spec.test.ts",
        "tests/ml/model/test_battle_state_spec.py",
    ),
    "encoder-golden": (
        "tests/ml/model/test_encoder.py",
        "tests/ml/model/test_options.py",
        "tests/ml/model/test_gamedata.py",
    ),
    "a1-a2-equality": (
        "packages/bot-client/test/a1-a2-equality.test.ts",
        "packages/bot-client/test/encoder-pairs.test.ts",
    ),
    "fidelity-corpus": (
        f"{B}/test/search-fidelity-corpus.test.ts",
        f"{B}/test/fidelity.test.ts",
    ),
    "contract-corpus": (
        f"{B}/test/samples.test.ts",
        "packages/schema/scripts/contracts.ts",
        "tests/contracts",
    ),
}


def files_of(c: mc.Check) -> tuple[str, ...]:
    return tuple(
        a
        for argv in c.commands
        for a in argv
        if a.endswith((".ts", ".py")) or a == "tests/contracts"
        if "/" in a and not a.startswith("--")
    )


def test_manifest_is_the_frozen_expected_list() -> None:
    assert [c.id for c in mc.CHECKS] == list(EXPECTED)
    for c in mc.CHECKS:
        assert files_of(c) == EXPECTED[c.id], c.id
        assert not c.absent
        assert set(c.needs) == {"python", "engine"}


def test_manifest_files_exist() -> None:
    repo = Path(__file__).resolve().parent.parent.parent
    for files in EXPECTED.values():
        for f in files:
            assert (repo / f).exists(), f
    for c in mc.CHECKS:
        for argv in c.recapture:
            for a in argv:
                if a.endswith(".ts"):
                    assert (repo / a).exists(), a


def test_recapture_covers_the_corpora_and_reduced_snapshots() -> None:
    wired = {a.rsplit("/", 1)[-1] for c in mc.CHECKS for argv in c.recapture for a in argv}
    for script in (
        "capture-random-corpus.ts",
        "capture-unavailable-corpus.ts",
        "capture-illusion-battle.ts",
        "capture-fling-battle.ts",
        "capture-baton-pass-battle.ts",
        "capture-samples.ts",
        "capture-encoder-pairs.ts",
        "test_golden.py",
    ):
        assert script in wired, script


def test_the_real_environment_is_scrubbed_of_write_variables() -> None:
    os.environ["ARENEON_UPDATE_GOLDEN"] = "1"
    try:
        assert "ARENEON_UPDATE_GOLDEN" not in mc.Env(Path("."), nice=False).environ()
    finally:
        del os.environ["ARENEON_UPDATE_GOLDEN"]
