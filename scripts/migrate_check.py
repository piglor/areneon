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
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PASS = "PASS"
DRIFT = "DRIFT"  # the check failed: a golden, table or parity result differs
REWRITTEN = "REWRITTEN"  # the check passed but a golden changed on disk while it ran
ERROR = "ERROR"  # the check could not run (tool missing, timeout)
ABSENT = "ABSENT"  # listed by the ticket, not on main yet
FAILING = (DRIFT, REWRITTEN, ERROR)

SHOWDOWN = "fixtures/showdown/gen9championsvgc2026regmc"
BRIDGE = "packages/showdown-bridge"
VITEST = ("pnpm", "exec", "vitest", "run")
PYTEST = ("{py}", "-m", "pytest", "--no-cov", "-q", "-p", "no:cacheprovider")


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    commands: tuple[tuple[str, ...], ...] = ()
    # Glob patterns (repo relative) of the goldens this check compares against.
    goldens: tuple[str, ...] = ()
    # Commands that rewrite those goldens, run only with --recapture.
    recapture: tuple[tuple[str, ...], ...] = ()
    absent: str = ""  # reason when the check does not exist yet
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
        "pinned engine, rules records and channels agree",
        (
            vitest("engine.test.ts", "rules-channel.test.ts"),
            node("check-rules.ts"),
        ),
    ),
    Check(
        "golden-fixtures",
        "golden request fixtures, team verdicts, battle summary, channels (fresh == committed)",
        (vitest("golden.test.ts", "teams.test.ts"),),
        (
            f"{SHOWDOWN}/*.request.json",
            f"{SHOWDOWN}/battle.summary.json",
            f"{SHOWDOWN}/capture.provenance.json",
            f"{SHOWDOWN}/channels/*",
            "fixtures/teams/*/verdicts.json",
        ),
        (node("capture-golden.ts"),),
    ),
    Check(
        "choice-corpus",
        "choice corpus replay, brute-force missedActions oracle, choice channels",
        (vitest("choice-corpus.test.ts", "choice-channels.test.ts", "choices.test.ts"),),
        (f"{SHOWDOWN}/choices/*", f"{SHOWDOWN}/choice-channels/*"),
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
            (*PYTEST, "tests/ml/reducer/test_protocol_volatiles.py"),
        ),
        (
            f"{SHOWDOWN}/engine-data*.json",
            f"{SHOWDOWN}/team-vocabulary*.json",
            f"{SHOWDOWN}/role-sets*.json",
            f"{SHOWDOWN}/move-classes*.json",
            f"{SHOWDOWN}/mechanics-inventory*.json",
            "ml/areneon_ml/reducer/data/*.json",
            "docs/mechanics-inventory.md",
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
        ("fixtures/mechanics/*/*/cases.json", "docs/mechanics-parity.md"),
        (node("mechanics-parity.ts"),),
    ),
    Check(
        "calc-differential",
        "@smogon/calc differential against the probe",
        (vitest("mechanics-differential.test.ts"),),
        ("fixtures/mechanics/gen9championsvgc2026regmc/calc-divergences.json",),
    ),
    Check(
        "reader-parity",
        "shared reader vs Python reducer, world facts and history facts parity",
        (
            vitest(
                "packages/public-state/test/parity.test.ts",
                "packages/public-state/test/history-facts-parity.test.ts",
                "packages/public-state/test/baton-pass.test.ts",
                f"{BRIDGE}/test/world-facts-parity.test.ts",
                f"{BRIDGE}/test/history-facts-parity.test.ts",
            ),
            (*PYTEST, "tests/ml/reducer/test_golden.py", "tests/ml/reducer/test_corpus.py"),
        ),
        (f"{SHOWDOWN}/reduced/*",),
    ),
    Check(
        "battle-state-spec-oracle",
        "battle-state spec oracle (ADR-030): engine fields, reducer map and encoder map agree",
        (
            vitest("battle-state-spec.test.ts"),
            (*PYTEST, "tests/ml/model/test_battle_state_spec.py"),
        ),
        ("config/engine-state/*", "fixtures/contracts/v1/battle-state-spec/*"),
    ),
    Check(
        "encoder-golden",
        "encoder golden digests and encoder_version",
        (
            (
                *PYTEST,
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
        (f"{SHOWDOWN}/a2-pairs.json",),
    ),
    Check(
        "fidelity-corpus",
        "fidelity corpus replay and report freshness",
        (vitest("search-fidelity-corpus.test.ts", "fidelity.test.ts"),),
        (f"{SHOWDOWN}/search-fidelity/*.json", f"{SHOWDOWN}/search-fidelity/*.md"),
    ),
    Check(
        "contract-corpus",
        "contract corpus (Ajv and python-jsonschema agree), generated types, digests",
        (
            vitest("samples.test.ts"),
            ("node", "packages/schema/scripts/contracts.ts"),
            (*PYTEST, "tests/contracts"),
        ),
        ("fixtures/contracts/**/*.json",),
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
class Env:
    root: Path
    python: str = "python3"
    nice: bool = True
    timeout_s: int | None = None
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def environ(self) -> dict[str, str]:
        env = dict(os.environ)
        parts = [self.root, self.root / "packages/schema/python", self.root / "ml"]
        # The TS parity tests spawn this interpreter as their Python reference.
        env.setdefault("ARENEON_PYTHON", self.python)
        extra = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join([*map(str, parts), *([extra] if extra else [])])
        return env


def expand(argv: Sequence[str], env: Env) -> list[str]:
    out = [env.python if a == "{py}" else a for a in argv]
    if env.nice and shutil.which("nice"):
        out = ["nice", "-n", "19", *out]
    return out


def golden_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(found)


def digest(root: Path, patterns: Sequence[str]) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in golden_files(root, patterns)
    }


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


_FAIL_LINE = re.compile(r"^\s*(?:FAIL|FAILED|\u00d7|✗|✘)\s+(.*\S)\s*$")
_ERROR_LINE = re.compile(r"^(?:\w*Error|AssertionError|Error):.*$")


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


def run_check(check: Check, env: Env, *, recapture: bool = False) -> Result:
    start = time.monotonic()
    if check.absent:
        return Result(check.id, check.title, ABSENT, 0.0, [check.absent])
    before = digest(env.root, check.goldens)
    status, drift = PASS, []
    changed_files: list[str] = []
    if recapture:
        for argv in check.recapture:
            proc = _exec(argv, env, check)
            if isinstance(proc, str):
                status, drift = ERROR, [proc]
                break
            if proc.returncode != 0:
                status, drift = (
                    ERROR,
                    [
                        f"recapture failed: {' '.join(argv)}",
                        *drift_lines(proc.stdout + proc.stderr, 3),
                    ],
                )
                break
        changed_files = changed(before, digest(env.root, check.goldens))
        before = digest(
            env.root, check.goldens
        )  # compare runs below start from the recaptured goldens
    if status == PASS:
        for argv in check.commands:
            proc = _exec(argv, env, check)
            if isinstance(proc, str):
                status, drift = ERROR, [proc]
                break
            if proc.returncode != 0:
                status = DRIFT
                drift += [line for line in drift_lines(proc.stdout + proc.stderr)]
    after = digest(env.root, check.goldens)
    rewritten = changed(before, after)
    if rewritten:
        # A compare run must never touch a golden: say so even when the check passed.
        status = REWRITTEN if status == PASS else status
        drift += [f"golden modified while comparing: {p}" for p in rewritten]
    return Result(check.id, check.title, status, time.monotonic() - start, drift, changed_files)


def _exec(argv: Sequence[str], env: Env, check: Check) -> subprocess.CompletedProcess[str] | str:
    try:
        return env.run(
            expand(argv, env),
            cwd=env.root,
            env=env.environ(),
            capture_output=True,
            text=True,
            timeout=env.timeout_s or check.timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"cannot run {argv[0]}: {exc.filename}"
    except subprocess.TimeoutExpired:
        return f"timed out: {' '.join(argv[:4])}"


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
    lines += [
        "",
        f"{len(results) - len(bad)}/{len(results)} checks without drift"
        + (f"; REVIEW: {', '.join(bad)}" if bad else ""),
    ]
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
        "--recapture",
        action="store_true",
        help="rewrite the goldens, then compare and list changed files",
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
