"""One entry point for the migration parity suite (#1466).

Runs every parity, golden and differential check that proves a migration safe
(a Showdown pin bump, a new simulation backend) in dependency order, never
rewrites a golden unless ``--recapture`` is given, and prints a summary of
check, status and what drifted. The checks are the repository's own tests and
scripts: nothing is reimplemented here. See docs/runbooks/migration.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PASS = "PASS"
DRIFT = "DRIFT"  # a command failed: a golden, table or parity result differs
REWRITTEN = "REWRITTEN"  # the check passed but a tracked data file changed while it ran
SKIPPED = "SKIPPED"  # a test in the check was skipped: a skip is not evidence
ERROR = "ERROR"  # the check could not run (interpreter or engine missing, timeout)
ABSENT = "ABSENT"  # listed but not built yet
FAILING = (DRIFT, REWRITTEN, SKIPPED, ERROR)

SHOWDOWN = "fixtures/showdown/gen9championsvgc2026regmc"
BRIDGE = "packages/showdown-bridge"
VITEST = ("pnpm", "exec", "vitest", "run")
PYTEST = ("{py}", "-m", "pytest", "--no-cov", "-q", "-rs", "-p", "no:cacheprovider")

# Every place a golden, table, generated type or generated doc lives. A compare run must
# leave all of them exactly as it found them, whatever the check's own globs say.
WATCHED = (
    "fixtures",
    "config",
    "docs",
    "ml/areneon_ml/reducer/data",
    "packages/*/generated",
    "packages/*/src",
)

# Variables that make a test or script write or dump a golden. Removed in compare mode.
WRITE_ENV = re.compile(
    r"^ARENEON_(UPDATE|RECAPTURE|PRINT|DUMP|WRITE)|^ARENEON_HISTORY_PARITY_DUMP$"
)


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    commands: tuple[tuple[str, ...], ...] = ()
    # Commands that rewrite the goldens, run only with --recapture (in this order).
    recapture: tuple[tuple[str, ...], ...] = ()
    absent: str = ""  # reason when the check does not exist yet
    needs: tuple[str, ...] = ("python", "engine")
    timeout_s: int = 1800


def vitest(*files: str) -> tuple[str, ...]:
    return (*VITEST, *[f"{BRIDGE}/test/{f}" if "/" not in f else f for f in files])


def node(script: str, *args: str) -> tuple[str, ...]:
    return ("node", f"{BRIDGE}/scripts/{script}", *args)


def pytest(*paths: str) -> tuple[str, ...]:
    return (*PYTEST, *paths)


CHECKS: tuple[Check, ...] = (
    Check(
        "pin-and-rules",
        "pinned engine, rules records, channels and generated gap doc agree",
        (
            vitest("engine.test.ts", "rules-channel.test.ts", "gaps-doc.test.ts"),
            node("check-rules.ts"),
        ),
    ),
    Check(
        "golden-fixtures",
        "golden request fixtures, team verdicts, battle summary, channels, random corpus",
        (vitest("golden.test.ts", "teams.test.ts", "random-corpus.test.ts"),),
        (node("capture-golden.ts"), node("capture-random-corpus.ts")),
    ),
    Check(
        "choice-corpus",
        "choice corpus replay, brute-force missedActions oracle, choice channels",
        (vitest("choice-corpus.test.ts", "choice-channels.test.ts", "choices.test.ts"),),
        (node("capture-choices.ts"), node("capture-choice-channels.ts")),
    ),
    Check(
        "engine-tables",
        "engine-data v1/v2, team vocabulary, role sets, move classes, volatile copy, stall moves, "
        "history facts, mechanics inventory re-exported byte for byte",
        (
            vitest(
                "engine-data.test.ts",
                "team-vocabulary.test.ts",
                "role-sets.test.ts",
                "move-classes.test.ts",
                "volatile-copy.test.ts",
                "stall-moves.test.ts",
                "history-facts-data.test.ts",
                "mechanics-inventory.test.ts",
                "mechanics-inventory-weights.test.ts",
            ),
            pytest("tests/ml/reducer/test_protocol_volatiles.py"),
        ),
        (
            node("export-engine-data.ts"),
            node("export-engine-data-2.ts"),
            node("export-team-vocabulary.ts"),
            node("export-role-sets.ts"),
            node("export-move-classes.ts"),
            node("export-volatile-copy.ts"),
            node("export-stall-moves.ts"),
            node("export-history-facts-data.ts"),
            node("mechanics-inventory.ts"),
        ),
    ),
    Check(
        "mechanics-probe",
        "mechanics probe fixtures (speed, damage, KO, stats), real-battle cross-checks, config",
        (
            vitest(
                "mechanics-speed.test.ts",
                "mechanics-speed-battle.test.ts",
                "mechanics-damage.test.ts",
                "mechanics-damage-paths.test.ts",
                "mechanics-ko.test.ts",
                "mechanics-ko-candidates.test.ts",
                "mechanics-stats.test.ts",
                "mechanics-parity.test.ts",
            ),
            node("check-mechanics-config.ts"),
            node("check-damage-aware-config.ts"),
        ),
        (node("mechanics-parity.ts"),),
    ),
    Check(
        "calc-differential",
        "@smogon/calc differential against the probe",
        (vitest("mechanics-differential.test.ts"),),
    ),
    Check(
        "reader-parity",
        "shared reader vs Python reducer, world facts and history facts parity, reduced goldens",
        (
            vitest(
                "packages/public-state/test/parity.test.ts",
                "packages/public-state/test/history-facts-parity.test.ts",
                "packages/public-state/test/baton-pass.test.ts",
                f"{BRIDGE}/test/world-facts-parity.test.ts",
                f"{BRIDGE}/test/history-facts-parity.test.ts",
            ),
            pytest(
                "tests/ml/reducer/test_golden.py",
                "tests/ml/reducer/test_corpus.py",
                "tests/ml/reducer/test_unavailable.py",
                "tests/ml/reducer/test_illusion_battle.py",
                "tests/ml/reducer/test_fling.py",
                "tests/ml/reducer/test_baton_pass.py",
                "tests/ml/reducer/test_baton_pass_volatiles.py",
            ),
        ),
        (
            node("capture-unavailable-corpus.ts"),
            node("capture-illusion-battle.ts"),
            node("capture-fling-battle.ts"),
            node("capture-baton-pass-battle.ts"),
            # Rewrite the reduced snapshots (ARENEON_UPDATE_GOLDEN=1 is set for recapture commands).
            pytest(
                "tests/ml/reducer/test_golden.py",
                "tests/ml/reducer/test_unavailable.py",
                "tests/ml/reducer/test_illusion_battle.py",
                "tests/ml/reducer/test_fling.py",
                "tests/ml/reducer/test_baton_pass.py",
                "tests/ml/reducer/test_baton_pass_volatiles.py",
            ),
        ),
    ),
    Check(
        "battle-state-spec-oracle",
        "battle-state spec oracle (ADR-030): engine fields, reducer map and encoder map agree;"
        " the G4 projection of record v2 is bit-equal to the encoder (canary, invariance)",
        (
            vitest("battle-state-spec.test.ts", "battle-state-projection.python.test.ts"),
            pytest("tests/ml/model/test_battle_state_spec.py", "tests/ml/projection"),
        ),
        (node("capture-projection-points.ts"),),
    ),
    Check(
        "encoder-golden",
        "encoder golden digests and encoder_version",
        (
            pytest(
                "tests/ml/model/test_encoder.py",
                "tests/ml/model/test_options.py",
                "tests/ml/model/test_gamedata.py",
            ),
        ),
    ),
    Check(
        "a1-a2-equality",
        "A1/A2 channel and observation equality, encoder pairs",
        (
            vitest(
                "packages/bot-client/test/a1-a2-equality.test.ts",
                "packages/bot-client/test/encoder-pairs.test.ts",
            ),
        ),
        (("node", "packages/bot-client/scripts/capture-encoder-pairs.ts"),),
    ),
    Check(
        "fidelity-corpus",
        "fidelity corpus replay and report freshness",
        (vitest("search-fidelity-corpus.test.ts", "fidelity.test.ts"),),
    ),
    Check(
        "battle-differential",
        "battle-level differential (ADR-030 G2): committed battles replayed with forced draws,"
        " broken backends caught",
        (
            vitest(
                "battle-trace.test.ts",
                "packages/team-dataset/test/differential-battle.test.ts",
                "packages/team-dataset/test/differential-battle-targeted.test.ts",
            ),
        ),
    ),
    Check(
        "contract-corpus",
        "contract corpus (Ajv and python-jsonschema agree), generated types, digests",
        (
            vitest("samples.test.ts"),
            ("node", "packages/schema/scripts/contracts.ts"),
            pytest("tests/contracts"),
        ),
        (node("capture-samples.ts"),),
    ),
)


@dataclass
class Result:
    id: str
    title: str
    status: str
    seconds: float
    drift: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)


@dataclass
class Proc:
    returncode: int
    output: str


def run_group(argv: Sequence[str], cwd: Path, env: dict[str, str], timeout: float) -> Proc:
    """Run argv in its own process group; kill the whole group on timeout."""
    proc = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return Proc(proc.returncode, out or "")


@dataclass
class Env:
    root: Path
    python: str = "python3"
    nice: bool = True
    timeout_s: int | None = None
    run: Callable[[Sequence[str], Path, dict[str, str], float], Proc] = run_group

    def environ(self, *, recapture: bool = False) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not WRITE_ENV.match(k)}
        parts = [self.root, self.root / "packages/schema/python", self.root / "ml"]
        extra = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([*map(str, parts), *([extra] if extra else [])])
        # The TS parity tests spawn this interpreter as their Python reference, and a missing
        # one must fail there instead of skipping.
        env["ARENEON_PYTHON"] = self.python
        env["ARENEON_REQUIRE_PYTHON_WORKER"] = "1"
        env["CI"] = "1"
        if recapture:
            env["ARENEON_UPDATE_GOLDEN"] = "1"
        return env

    def problems(self, check: Check) -> list[str]:
        out = []
        if "python" in check.needs and not (
            Path(self.python).exists() or shutil.which(self.python)
        ):
            out.append(f"python interpreter missing: {self.python}")
        if (
            "engine" in check.needs
            and not (self.root / "vendor/pokemon-showdown/dist/sim").is_dir()
        ):
            out.append("engine checkout missing or not built (make showdown)")
        return out


def expand(argv: Sequence[str], env: Env) -> list[str]:
    out = [env.python if a == "{py}" else a for a in argv]
    if env.nice and shutil.which("nice"):
        out = ["nice", "-n", "19", *out]
    return out


def tracked_state(root: Path) -> dict[str, str | None]:
    """Path -> content hash of every modified or untracked file under WATCHED (git status)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "-z", "--", *WATCHED],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    state: dict[str, str | None] = {}
    entries = iter(proc.stdout.split("\0"))
    for entry in entries:
        if not entry:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            next(entries, None)  # the rename's source path
        f = root / path
        state[path] = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
    return state


def changed(before: dict[str, str | None], after: dict[str, str | None]) -> list[str]:
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


_FAIL_LINE = re.compile(r"^\s*(?:FAIL|FAILED|✗|✘)\s+(.*\S)\s*$")
_ERROR_LINE = re.compile(r"^(?:\w*Error|AssertionError|Error):.*$")
_PYTEST_SKIPPED = re.compile(r"(\d+) skipped")
_PYTEST_SKIP_LINE = re.compile(r"^SKIPPED \[\d+\] (.*)$")


def drift_lines(output: str, limit: int = 8) -> list[str]:
    """The failing test names (vitest FAIL, pytest FAILED) and the first assertion text."""
    seen: list[str] = []
    for line in output.splitlines():
        m = _FAIL_LINE.match(line)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    if not seen:
        seen = [ln.strip() for ln in output.splitlines() if _ERROR_LINE.match(ln.strip())]
    if not seen:
        tail = [ln.strip() for ln in output.splitlines() if ln.strip()]
        seen = tail[-3:]
    return seen[:limit]


def is_vitest(argv: Sequence[str]) -> bool:
    return tuple(argv[:4]) == VITEST


def is_pytest(argv: Sequence[str]) -> bool:
    return tuple(argv[:3]) == PYTEST[:3] or tuple(argv[1:3]) == ("-m", "pytest")


@dataclass
class Outcome:
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def read_vitest(
    argv: Sequence[str], report: Path, root: Path, returncode: int, output: str
) -> Outcome:
    """Failures, skips and files that never ran, from vitest's JSON report (exact paths)."""
    out = Outcome()
    wanted = {a for a in argv[4:] if not a.startswith("-")}
    try:
        data = json.loads(report.read_text())
    except (OSError, ValueError):
        out.failed = drift_lines(output) if returncode else ["vitest wrote no report"]
        return out
    ran: set[str] = set()
    for file in data.get("testResults", []):
        rel = os.path.relpath(file["name"], root)
        ran.add(rel)
        tests = file.get("assertionResults", [])
        failed = [t for t in tests if t.get("status") == "failed"]
        out.failed += [f"{rel} > {t.get('fullName', '?')}" for t in failed]
        if file.get("status") == "failed" and not failed:
            out.failed.append(f"{rel}: {str(file.get('message') or 'suite failed')[:120]}")
        skipped = [t for t in tests if t.get("status") in ("skipped", "pending", "todo")]
        if skipped:
            out.skipped.append(f"{rel}: {len(skipped)} skipped")
    out.failed += [f"{w}: did not run" for w in sorted(wanted - ran)]
    if returncode and not out.failed:
        out.failed = drift_lines(output)
    return out


def read_pytest(returncode: int, output: str) -> Outcome:
    out = Outcome()
    if returncode:
        out.failed = drift_lines(output)
    lines = [m.group(1) for ln in output.splitlines() if (m := _PYTEST_SKIP_LINE.match(ln))]
    counted = [m for ln in output.splitlines() if (m := _PYTEST_SKIPPED.search(ln))]
    if lines or counted:
        out.skipped = lines[:5] or [f"{counted[-1].group(1)} skipped"]
    return out


def execute(
    argv: Sequence[str], env: Env, check: Check, *, recapture: bool = False
) -> Outcome | str:
    """Run one command; an Outcome (failures, skips) or a string when it could not run."""
    full = list(argv)
    report: Path | None = None
    tmp = None
    if is_vitest(full) and not recapture:
        tmp = tempfile.mkdtemp(prefix="migrate-check-")
        report = Path(tmp) / "report.json"
        full += ["--reporter=json", f"--outputFile={report}"]
    try:
        proc = env.run(
            expand(full, env),
            env.root,
            env.environ(recapture=recapture),
            env.timeout_s or check.timeout_s,
        )
    except FileNotFoundError as exc:
        return f"cannot run {argv[0]}: {exc.filename}"
    except subprocess.TimeoutExpired:
        return f"timed out: {' '.join(argv[:4])}"
    try:
        if report is not None:
            return read_vitest(full, report, env.root, proc.returncode, proc.output)
        if is_pytest(full) and not recapture:
            return read_pytest(proc.returncode, proc.output)
        return Outcome(failed=drift_lines(proc.output) if proc.returncode else [])
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def run_check(check: Check, env: Env, *, recapture: bool = False) -> Result:
    start = time.monotonic()
    if check.absent:
        return Result(check.id, check.title, ABSENT, 0.0, [check.absent])
    missing = env.problems(check)
    if missing:
        return Result(check.id, check.title, ERROR, 0.0, missing)
    before = tracked_state(env.root)
    status, drift = PASS, []
    changed_files: list[str] = []
    if recapture:
        for argv in check.recapture:
            res = execute(argv, env, check, recapture=True)
            if isinstance(res, str):
                status, drift = ERROR, [res]
                break
            if res.failed:
                status, drift = ERROR, [f"recapture failed: {' '.join(argv[:4])}", *res.failed[:3]]
                break
        after_recapture = tracked_state(env.root)
        changed_files = changed(before, after_recapture)
        before = after_recapture  # compare runs below start from the recaptured goldens
    if status == PASS:
        skips: list[str] = []
        for argv in check.commands:
            res = execute(argv, env, check)
            if isinstance(res, str):
                status, drift = ERROR, [res]
                break
            if res.failed:
                status = DRIFT
                drift += res.failed[:8]
            skips += res.skipped
        if status == PASS and skips:
            status, drift = SKIPPED, [f"skipped (not evidence): {s}" for s in skips]
    rewritten = changed(before, tracked_state(env.root))
    if rewritten:
        # A compare run must never touch tracked data: say so even when the check passed.
        status = REWRITTEN if status in (PASS, SKIPPED) else status
        drift += [f"modified while comparing: {p}" for p in rewritten]
    return Result(check.id, check.title, status, time.monotonic() - start, drift, changed_files)


def run_checks(
    checks: Sequence[Check],
    env: Env,
    *,
    recapture: bool = False,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    results = []
    for check in checks:
        result = run_check(check, env, recapture=recapture)
        results.append(result)
        if on_result:
            on_result(result)
    return results


def summary(results: Sequence[Result]) -> str:
    width = max([len(r.id) for r in results] + [5])
    lines = [f"{'check':<{width}}  {'status':<9}  {'time':>7}  what drifted"]
    for r in results:
        first = r.drift[0] if r.drift else "-"
        lines.append(f"{r.id:<{width}}  {r.status:<9}  {r.seconds:>6.1f}s  {first}")
        lines += [f"{'':<{width}}  {'':<9}  {'':>7}  {d}" for d in r.drift[1:]]
    changed_files = sorted({f for r in results for f in r.changed_files})
    if changed_files:
        lines += ["", "recaptured goldens that changed (review each diff before committing):"]
        lines += [f"  {f}" for f in changed_files]
    bad = [r.id for r in results if r.status in FAILING]
    tail = f"; REVIEW: {', '.join(bad)}" if bad else ""
    lines += ["", f"{len(results) - len(bad)}/{len(results)} checks without drift{tail}"]
    return "\n".join(lines)


def select(only: Sequence[str] | None, checks: Sequence[Check] = CHECKS) -> list[Check]:
    if not only:
        return list(checks)
    unknown = [o for o in only if not any(fnmatch.fnmatch(c.id, o) for c in checks)]
    if unknown:
        raise SystemExit(f"unknown check: {', '.join(unknown)}")
    return [c for c in checks if any(fnmatch.fnmatch(c.id, o) for o in only)]


def main(
    argv: Sequence[str] | None = None, env: Env | None = None, checks: Sequence[Check] = CHECKS
) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recapture", action="store_true", help="rewrite goldens, compare, list changes"
    )
    ap.add_argument("--only", action="append", help="check id (glob), repeatable")
    ap.add_argument("--list", action="store_true", help="list the checks and exit")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    ap.add_argument("--python", default=os.environ.get("MIGRATE_PYTHON", sys.executable))
    ap.add_argument("--no-nice", action="store_true")
    args = ap.parse_args(argv)
    chosen = select(args.only, checks)
    if args.list:
        for c in chosen:
            print(f"{c.id}: {c.title}")
        return 0
    env = env or Env(Path(__file__).resolve().parent.parent, args.python, not args.no_nice)
    results = run_checks(
        chosen,
        env,
        recapture=args.recapture,
        on_result=lambda r: print(f"[{r.status}] {r.id} ({r.seconds:.1f}s)", flush=True),
    )
    print()
    print(summary(results))
    if args.json:
        Path(args.json).write_text(json.dumps([r.__dict__ for r in results], indent=2) + "\n")
    return 1 if any(r.status in FAILING for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
