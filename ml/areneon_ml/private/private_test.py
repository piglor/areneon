"""The owner's private test command (ADR-015 erratum E1, item 2).

``python -m areneon_ml.loop private-test`` measures one frozen pair (checkpoint ids and play mode)
on the owner's teams, from both seats, and writes ONE record of counts and scores under a private
output root. It is the top-level command: the owner-session guard runs before anything is read
(``assert_owner_session``: refused in an agent session and on non-terminal stdio, no override), the
output root must pass the private root rule, and only then are the run directory, the pair and the
team source touched. The teams are read by the Node child (``scripts/private-test.ts`` of the
package that owns the snapshot reader), which prints one JSON line of counts; Python never holds a
team. Nothing but the record, the 0600 salt file and the private look log is written, and only
under the private root:
no game record, omniscient log, Episode, replay or dashboard page.

Every failure is a fixed code (and, for a team, a 1-based position in the source's order). The
child's stderr is forwarded only when it is exactly such a line; anything else shows as ``failed``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from areneon_contracts.contracts import jcs

from areneon_ml.learner.config import RunConfig, read_run_config
from areneon_ml.loop import layout
from areneon_ml.loop.records import features_hash, policy_id
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.private.owner_session import OwnerSession, OwnerSessionError, assert_owner_session
from areneon_ml.private.private_root import (
    PrivateRootError,
    assert_private_root,
    private_root_file,
    write_private_file,
)
from areneon_ml.private.private_test_record import (
    INTERVAL_LABEL,
    LABEL,
    PrivateTestRecordError,
    private_test_allowed,
    read_private_test_record,
    record_hash,
    summary_of,
)
from areneon_ml.progress.looks import Look, append_look
from areneon_ml.progress.run import ProgressError, _source

#: The Node child, found by its file name: the one package that holds it owns the snapshot reader,
#: and no code outside that package names it (a packaging rule).
SCRIPT_GLOB: Final = "packages/*/scripts/private-test.ts"
LOOKS_NAME: Final = "private-test.looks.jsonl"
SOURCES: Final = ("staging", "snapshot", "dataset")
SUBSETS: Final = ("all", "holdout")
MODES: Final = ("sample", "argmax")
HEX64: Final = re.compile(r"[0-9a-f]{64}")
SALT_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
#: The only line of the child's stderr that is forwarded: a code and a position.
CHILD_LINE: Final = re.compile(r"private-test: ([a-z_]+)(?: ([1-9][0-9]{0,8}))?")

#: Fixed words for every refusal. A refusal may add a 1-based position, never text.
REFUSALS: Final[dict[str, str]] = {
    "usage": "usage error",
    "private_pair_source": (
        "a pair with private ancestry is tested on hold-out teams of a snapshot "
        "or on a public dataset only"
    ),
    "run_unverified": "the run directory or its generation does not verify",
    "opponent_unverified": "the opponent pair's run directory or generation does not verify",
    "salt_unreadable": "the salt file cannot be read or is not 64 hex digits",
    "record_mismatch": "the child's record does not match the request",
    "record_exists": "a different record of that name already exists in the private root",
    "source_unreadable": "the team source cannot be read",
    "source_empty": "the team source holds no team",
    "team_import_failed": "a team cannot be imported (position among the source's teams)",
    "team_invalid": "a team is not legal in the format (position among the source's teams)",
    "team_repeated": "a team repeats an earlier one (position among the source's teams)",
    "too_few_teams": "the test set needs at least two teams",
    "too_many_pairs": "more pairs asked for than the set gives",
    "snapshot_unverified": "the snapshot does not verify against the pinned engine and rules",
    "worker_failed": "the policy worker failed",
    "play_failed": "the test could not be played",
    "failed": "the child failed",
}


class PrivateTestError(Exception):
    """A refusal with a fixed code, fixed words and, for a team, a position."""

    def __init__(self, code: str, position: int | None = None) -> None:
        shown = code if code in REFUSALS else "failed"
        super().__init__(f"private-test: {shown}" + ("" if position is None else f" {position}"))
        self.code = shown
        self.position = position


@dataclass(frozen=True)
class PrivateTestRequest:
    """Everything the command takes, checked once."""

    run_dir: Path
    generation: int
    out_root: Path
    engine_data: Path
    source: str
    subset: str
    teams_dir: Path
    snapshots_dir: Path
    snapshot_hash: str | None = None
    salt_file: str | None = None
    max_pairs: int | None = None
    damage_aware_budget_ms: int | None = None
    opponent_run_dir: Path | None = None
    opponent_generation: int | None = None
    node: str = "node"
    worker_python: str = "python"
    repo: Path = Path(".")


class PrivateTestPlayer(Protocol):
    def play(self, request: ChildRequest) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ChildRequest:
    """What the Node child is given."""

    request: PrivateTestRequest
    config_path: Path
    encoder_version: str
    generation_manifest_sha256: str
    preview: str
    battle: str
    candidate_data_class: str
    salt_path: Path
    opponent: tuple[str, str, str, str, Path, str] | None = None
    #: The run's candidate feature set hash (ADR-022), in the pair's policy ids; None without one.
    features: str | None = None


@dataclass(frozen=True)
class NodePrivateTestPlayer:
    """Runs the Node child; shows only the lines of its stderr that are a fixed refusal."""

    repo: Path
    node: str
    worker_python: str

    def play(self, child: ChildRequest) -> dict[str, Any]:
        r = child.request
        args = [
            *("--config", str(child.config_path)),
            *("--generation", str(r.generation)),
            *("--preview", child.preview),
            *("--battle", child.battle),
            *(
                "--policy-id-sample",
                policy_id(child.preview, child.battle, "sample", features=child.features),
            ),
            *(
                "--policy-id-argmax",
                policy_id(child.preview, child.battle, "argmax", features=child.features),
            ),
            *("--generation-manifest-sha256", child.generation_manifest_sha256),
            *("--candidate-data-class", child.candidate_data_class),
            *("--run-dir", str(r.run_dir)),
            *("--engine-data", str(r.engine_data)),
            *("--encoder-version", child.encoder_version),
            *("--worker-python", self.worker_python),
            *("--salt-file", str(child.salt_path)),
            *("--source", r.source),
            *("--subset", r.subset),
            *("--teams-dir", str(r.teams_dir)),
            *("--snapshots-dir", str(r.snapshots_dir)),
        ]
        if r.snapshot_hash is not None:
            args += ["--snapshot-hash", r.snapshot_hash]
        if r.max_pairs is not None:
            args += ["--max-pairs", str(r.max_pairs)]
        if r.damage_aware_budget_ms is not None:
            args += ["--damage-aware-budget-ms", str(r.damage_aware_budget_ms)]
        if child.opponent is not None:
            preview, battle, sample, argmax, other_dir, other_encoder = child.opponent
            args += [
                *("--opponent-pair", f"{preview},{battle},{sample},{argmax}"),
                *("--opponent-run-dir", str(other_dir)),
                *("--opponent-encoder-version", other_encoder),
            ]
        scripts = sorted(self.repo.glob(SCRIPT_GLOB))
        if len(scripts) != 1:
            raise PrivateTestError("failed")
        try:
            done = subprocess.run(
                [self.node, str(scripts[0]), *args],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
                env=_child_env(),
            )
        except OSError:
            raise PrivateTestError("failed") from None
        if done.returncode != 0:
            last = next((ln for ln in reversed(done.stderr.splitlines()) if ln.strip()), "")
            found = CHILD_LINE.fullmatch(last)
            if found is None:
                raise PrivateTestError("failed")
            raise PrivateTestError(found[1], None if found[2] is None else int(found[2]))
        try:
            out = json.loads(done.stdout)
        except ValueError:
            raise PrivateTestError("failed") from None
        if not isinstance(out, dict):
            raise PrivateTestError("failed")
        return out


#: The only variables the Node child sees: no token of the owner's shell reaches it.
CHILD_ENV: Final = ("PATH", "HOME", "LANG", "TMPDIR", "SHOWDOWN_DIR", "CLAUDECODE", "AI_AGENT")


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in CHILD_ENV}


def _check_request(r: PrivateTestRequest) -> None:
    """The flag combinations, before anything is read."""
    if r.source not in SOURCES or r.subset not in SUBSETS:
        raise PrivateTestError("usage")
    if r.subset == "holdout" and r.source != "snapshot":
        raise PrivateTestError("usage")
    if (r.source == "snapshot") != (r.snapshot_hash is not None):
        raise PrivateTestError("usage")
    if r.snapshot_hash is not None and not HEX64.fullmatch(r.snapshot_hash):
        raise PrivateTestError("usage")
    if r.generation < 0 or (r.max_pairs is not None and r.max_pairs < 1):
        raise PrivateTestError("usage")
    if r.damage_aware_budget_ms is not None and r.damage_aware_budget_ms < 1:
        raise PrivateTestError("usage")
    if (r.opponent_run_dir is None) != (r.opponent_generation is None):
        raise PrivateTestError("usage")
    if r.salt_file is not None and not SALT_NAME.fullmatch(r.salt_file):
        raise PrivateTestError("usage")


def _salt(root: Path, name: str | None) -> tuple[Path, str]:
    """The salt file of the test: an existing one by name, or a new 256-bit one in the root."""
    if name is None:
        fresh = f"private-test-salt-{secrets.token_hex(4)}.key"
        return write_private_file(root, fresh, secrets.token_hex(32) + "\n"), fresh
    path = private_root_file(root, name)
    try:
        text = path.read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        raise PrivateTestError("salt_unreadable") from None
    if not HEX64.fullmatch(text) or path.stat().st_mode & 0o077:
        raise PrivateTestError("salt_unreadable")
    return path, name


def _tie(  # noqa: PLR0913, PLR0917
    record: dict[str, Any],
    request: PrivateTestRequest,
    config: RunConfig,
    pair: Any,
    opponent: Any,
    candidate_class: str,
    opponent_features: str | None = None,
) -> None:
    """The record is about this run, generation, pair, source and opponents, and nothing else."""
    test_set, plan = record["test_set"], record["plan"]
    generation = config.generation
    ok = (
        record["run_id"] == config.run_key
        and record["run_config_hash"] == config.hash
        and record["generation"] == request.generation
        and record["generation_manifest_sha256"] == pair.manifest_sha256
        and record["checkpoints"] == {"preview": pair.preview, "battle": pair.battle}
        and record["candidate_data_class"] == candidate_class
        and record["version_manifest"]["format_id"] == config.format_id
        and test_set["source_kind"] == request.source
        and test_set["subset"] == request.subset
        and test_set.get("snapshot_hash") == request.snapshot_hash
        and plan["max_turns"] == generation.max_turns
        and plan["latency_budget_ms"]
        == {"preview_ms": generation.preview_ms, "battle_ms": generation.battle_ms}
        and (request.max_pairs is None or test_set["pairs"] == request.max_pairs)
    )
    if request.source == "dataset":
        ok = ok and test_set.get("dataset_hash") == config.raw["team_dataset"]["dataset_hash"]
    expected = {"random-legal", "actor-rule"}
    expected |= {"damage-aware"} if request.damage_aware_budget_ms is not None else set()
    expected |= {"pair"} if opponent is not None else set()
    want = sorted(f"{name}-{mode}" for name in expected for mode in MODES)
    ok = ok and [m["match_id"] for m in record["matches"]] == want
    for match in record["matches"]:
        mode = match["mode"]
        ok = ok and match["policy_id"] == policy_id(
            pair.preview, pair.battle, mode, features=features_hash(config)
        )
        if "opponent_checkpoints" in match:
            ok = (
                ok
                and opponent is not None
                and match["opponent_policy_id"]
                == policy_id(
                    opponent.preview,
                    opponent.battle,
                    mode,
                    features=opponent_features,
                )
                and match["opponent_checkpoints"]
                == {"preview": opponent.preview, "battle": opponent.battle}
            )
    if not ok:
        raise PrivateTestError("record_mismatch")


def _write_record(root: Path, record: dict[str, Any]) -> tuple[str, str]:
    """The record under the private root, named by its hash; an identical rerun is a no-op."""
    digest = record_hash(record)
    name = f"private-test-{digest[:16]}.json"
    data = jcs(record) + b"\n"
    try:
        write_private_file(root, name, data)
    except PrivateRootError as exc:
        if exc.code != "exists" or private_root_file(root, name).read_bytes() != data:
            raise PrivateTestError("record_exists") from None
    return name, digest


def _touch_private(path: Path) -> None:
    """Create the look log with mode 0600 (the log is then only appended to)."""
    try:
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        return


def run_private_test(
    request: PrivateTestRequest,
    *,
    session: OwnerSession | None = None,
    player: PrivateTestPlayer | None = None,
) -> dict[str, Any]:
    """Guard, root, run, play, read, write, log the look; returns the display (counts only)."""
    assert_owner_session(session)  # before any team, any run file or any flag is looked at
    root = assert_private_root(request.out_root)
    _check_request(request)
    run_dir = request.run_dir.resolve()
    try:
        config = read_run_config(layout.read_json(run_dir / layout.CONFIG))
        pair = _source(
            run_dir,
            config,
            request.generation,
            allow_private=True,
            needs_dataset=request.source == "dataset",
        )
        encoder = _encoder_version(request.engine_data, config)
    except (ProgressError, layout.RunDirError, ValueError, OSError):
        raise PrivateTestError("run_unverified") from None
    if config.run_key.startswith("CHANGE-ME"):
        raise PrivateTestError("run_unverified")
    opponent = None
    opponent_arg = None
    opponent_features = None
    if request.opponent_run_dir is not None and request.opponent_generation is not None:
        other_dir = request.opponent_run_dir.resolve()
        try:
            other = read_run_config(layout.read_json(other_dir / layout.CONFIG))
            opponent = _source(
                other_dir,
                other,
                request.opponent_generation,
                allow_private=True,
                needs_dataset=False,
            )
            if other.format_id != config.format_id or (other_dir, opponent.generation) == (
                run_dir,
                request.generation,
            ):
                raise PrivateTestError("opponent_unverified")
            opponent_features = features_hash(other)
            opponent_arg = (
                opponent.preview,
                opponent.battle,
                policy_id(
                    opponent.preview, opponent.battle, "sample", features=features_hash(other)
                ),
                policy_id(
                    opponent.preview, opponent.battle, "argmax", features=features_hash(other)
                ),
                other_dir,
                _encoder_version(request.engine_data, other),
            )
        except (ProgressError, layout.RunDirError, ValueError, OSError):
            raise PrivateTestError("opponent_unverified") from None
    candidate_class = (
        "private"
        if pair.data_class == "private"
        or (opponent is not None and opponent.data_class == "private")
        else "synthetic"
    )
    if candidate_class == "private" and not private_test_allowed(request.source, request.subset):
        raise PrivateTestError("private_pair_source")
    salt_path, salt_name = _salt(root, request.salt_file)
    runner = player or NodePrivateTestPlayer(request.repo, request.node, request.worker_python)
    found = runner.play(
        ChildRequest(
            request,
            run_dir / layout.CONFIG,
            encoder,
            pair.manifest_sha256,
            pair.preview,
            pair.battle,
            candidate_class,
            salt_path,
            opponent_arg,
            features_hash(config),
        )
    )
    try:
        record = read_private_test_record(found.get("record"))
    except PrivateTestRecordError:
        raise PrivateTestError("record_mismatch") from None
    _tie(record, request, config, pair, opponent, candidate_class, opponent_features)
    name, digest = _write_record(root, record)
    # The result is verified and kept: log the look, then show it.
    looks_file = root / LOOKS_NAME
    _touch_private(looks_file)
    state = append_look(
        looks_file,
        "private-test",
        config.run_key,
        [Look(request.generation, m["match_id"], m["games"]) for m in record["matches"]],
    )
    test_set = record["test_set"]
    return {
        "label": LABEL,
        "interval": INTERVAL_LABEL,
        "record": name,
        "record_hash": digest,
        "salt_file": salt_name,
        "looks": state.count,
        "test_set": {
            k: test_set[k] for k in ("commitment", "team_count", "pairs", "source_kind", "subset")
        },
        "candidate_data_class": candidate_class,
        "matches": summary_of(record),
    }


def _encoder_version(engine_data: Path, config: RunConfig) -> str:
    game = GameData.from_table(json.loads(engine_data.read_text(encoding="utf-8")))
    return encoder_version(game, config.model.encoder)


def private_test_main(
    request: PrivateTestRequest,
    *,
    session: OwnerSession | None = None,
    player: PrivateTestPlayer | None = None,
    out: Callable[[bytes], object] | None = None,
    err: Callable[[str], object] | None = None,
) -> int:
    """The process-level wrapper: one canonical JSON line on success, one fixed line on failure.

    Exit 0 ok, 2 usage, 3 a refusal, 4 the child failed. Nothing but a fixed message, a code and
    a position is ever shown, whatever the cause.
    """
    write_out = out or sys.stdout.buffer.write
    write_err = err or (lambda text: print(text, file=sys.stderr))
    try:
        result = run_private_test(request, session=session, player=player)
    except PrivateTestError as exc:
        write_err(str(exc))
        return 2 if exc.code == "usage" else 4 if exc.code in CHILD_CODES else 3
    except (OwnerSessionError, PrivateRootError) as exc:
        write_err(str(exc))
        return 3
    except Exception:
        write_err("private-test: failed")
        return 4
    write_out(jcs(result) + b"\n")
    return 0


#: Codes that come from the Node child (a failed play), exit 4.
CHILD_CODES: Final = frozenset(
    {
        "source_unreadable",
        "source_empty",
        "team_import_failed",
        "team_invalid",
        "team_repeated",
        "too_few_teams",
        "too_many_pairs",
        "snapshot_unverified",
        "worker_failed",
        "play_failed",
        "failed",
    }
)

__all__ = [
    "CHILD_CODES",
    "LOOKS_NAME",
    "REFUSALS",
    "ChildRequest",
    "NodePrivateTestPlayer",
    "PrivateTestError",
    "PrivateTestRequest",
    "private_test_main",
    "run_private_test",
]
