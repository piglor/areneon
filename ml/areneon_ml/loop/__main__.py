"""``python -m areneon_ml.loop``: the training loop of one run directory.

* ``run``: continue the run in ``--run-dir`` until ``--generations`` generations are
  committed. A directory with committed generations is resumed from the last one; an
  unfinished generation is discarded and replayed. Prints one JCS JSON line.
* ``inspect``: write the training-data inspector's static pages (teams and games) for a run; with
  ``--resimulate G:K``, instead one engine re-simulation page of a played game (omniscient, local).
* ``replay``: write graphical replay pages of training games (each game's two own channels), or
  with ``--progress-dir`` of stored progress games (the omniscient view, for local use).
* ``import-clone``: copy a cloning run's checkpoints and records (``learner clone`` output) into a
  run directory, so a run config with ``warm_start`` can start generation 0 from them (ADR-010
  erratum A2). Never fetched: the clone directory is a local path. Prints the ids by role.
  ``--roles battle`` imports the Battle checkpoint and record alone (a Battle-only warm start).
* ``provenance``: write the run's CheckpointRecords and its sources registry into a
  directory, ready for ``python -m areneon_ml.checkpoint classify|independent``.
* ``report``: write the run dashboard (training curves and, with a progress root, the progress
  evaluation) as one offline HTML page.
* ``private-test``: OWNER ONLY (refused in an agent session and on non-terminal stdio): one frozen
  pair on the owner's own teams, from both seats, counts and scores only, written under a private
  output root (ADR-015 erratum E1).
* ``progress``: the progress evaluation of a committed generation (ADR-017): its frozen pair
  against the scripted baselines (and, from generation 1, the previous generation's pair and,
  with ``--anchors``, fixed earlier generations) on the validation teams, written to
  ``<run-dir>.progress/``.
  A progress view, never the Training gate; the run directory is only read.

The learner runs in this process (``learner.device`` of the config: ``cpu``, or ``cuda``
from the separate GPU environment); the A1 games run in Node, whose policy worker uses
the Python named by ``--worker-python`` (CPU torch from the shipped closure). Exit 0 ok,
2 usage, 3 a refused configuration, directory or hash, 4 the simulator side failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs

from areneon_ml.checkpoint.files import CheckpointFileError
from areneon_ml.checkpoint.provenance import ProvenanceError
from areneon_ml.checkpoint.record import RecordError
from areneon_ml.inspector.command import inspect_run
from areneon_ml.inspector.progress_replay_command import replay_progress
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_command import Selection, parse_games, replay_games
from areneon_ml.inspector.resimulate import node_runner
from areneon_ml.inspector.resimulate_command import parse_target, resimulate_game
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.state import LearnerStateError
from areneon_ml.learner.trainer import LearnerError
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Loop, LoopError, export_provenance
from areneon_ml.loop.play import NodePlayer, PlayError
from areneon_ml.loop.warm_start import WarmStartError, import_clone
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.network import ROLES
from areneon_ml.private.private_test import PrivateTestRequest, private_test_main
from areneon_ml.progress.looks import Look, LookError, append_look, looks_path
from areneon_ml.progress.records import ProgressRecordError
from areneon_ml.progress.run import (
    DEFAULT_ANCHOR_GAMES,
    DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES,
    DEFAULT_GAMES,
    DEFAULT_RECORD_GAMES,
    DEFAULT_RUN_GAMES,
    SCHEDULE_EVERY,
    NodeProgressPlayer,
    ProgressError,
    VersusSpec,
    progress_generation,
    progress_root,
    scheduled_generations,
)
from areneon_ml.report.command import report_run
from areneon_ml.report.move_use import progress_move_use

EXIT_REFUSED = 3
EXIT_PLAY = 4
REPO = Path(__file__).resolve().parents[3]


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _add_private_test_parser(
    commands: Any,
) -> None:
    """The ``private-test`` command (owner only)."""
    private = commands.add_parser(
        "private-test",
        help="OWNER ONLY: measure a frozen pair on your own teams (counts and scores only)",
    )
    private.add_argument("--run-dir", type=Path, required=True)
    private.add_argument("--generation", type=int, required=True)
    private.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="an existing 0700 directory with a private-data path segment: the record goes there",
    )
    private.add_argument("--engine-data", type=Path, required=True)
    private.add_argument("--source", choices=("staging", "snapshot", "dataset"), required=True)
    private.add_argument("--subset", choices=("all", "holdout"), default="all")
    private.add_argument(
        "--teams-dir", type=Path, default=REPO / "private-data" / "teams", help="source staging"
    )
    private.add_argument(
        "--snapshots-dir",
        type=Path,
        default=REPO / "private-data" / "snapshots",
        help="source snapshot: the verified local copies",
    )
    private.add_argument("--snapshot-hash", help="source snapshot: the pinned snapshot hash")
    private.add_argument("--salt-file", help="a salt file of the out root, to repeat a test")
    private.add_argument("--max-pairs", type=_positive, help="play only this many pairs")
    private.add_argument("--damage-aware", action="store_true", help="also play damage-aware@v1")
    private.add_argument("--damage-aware-budget-minutes", type=_positive, default=None)
    private.add_argument("--opponent-run-dir", type=Path, help="another frozen pair as opponent")
    private.add_argument("--opponent-generation", type=int)
    private.add_argument("--node", default="node", help="the node executable")
    private.add_argument("--worker-python", default=sys.executable, help="CPU torch worker python")


def _add_run(commands: Any) -> None:
    """The ``run`` command and its operational flags."""
    run = commands.add_parser("run", help="play, train and freeze generations")
    run.add_argument("--config", type=Path, required=True, help="a TrainingRunConfig JSON file")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--engine-data", type=Path, required=True)
    run.add_argument("--generations", type=int, required=True, help="generations to have committed")
    run.add_argument("--node", default="node", help="the node executable")
    run.add_argument(
        "--worker-python",
        default=sys.executable,
        help="the Python for the policy worker: CPU torch from the shipped closure",
    )
    run.add_argument(
        "--policy-workers",
        type=_positive,
        default=1,
        help="policy worker processes per game process (operational: the generation's files "
        "are the same for any number)",
    )
    run.add_argument(
        "--game-processes",
        type=_positive,
        default=1,
        help="Node processes sharing a generation's games, each with --policy-workers workers "
        "(operational: the generation's files are the same for any number)",
    )
    run.add_argument(
        "--generation-timeout-minutes",
        type=_positive,
        default=180,
        help="wall-clock limit of one generation's play phase: past it the Node process group "
        "is killed and the generation fails, nothing committed (operational: the generation's "
        "files are the same for any limit; default 180)",
    )
    run.add_argument(
        "--load-processes",
        type=_positive,
        default=1,
        help="worker processes that check and encode a generation's stream in the load phase "
        "(operational: the trajectories and every hash are the same for any number; each worker "
        "adds its piece's tensors to memory while it runs, see docs/training-run.md)",
    )
    run.add_argument(
        "--load-memory-mb",
        type=_positive,
        help="cap the memory the load workers may add (about 400 MB each): fewer workers run "
        "when --load-processes would not fit (operational)",
    )
    run.add_argument(
        "--node-heap-mb",
        type=_positive,
        help="cap each Node game process's heap (--max-old-space-size, MB); 640 keeps a "
        "game process near 0.6 GB (operational: the generation's files are the same; see "
        "docs/training-run.md, sizing a run to a machine)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="areneon_ml.loop")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run(commands)
    provenance = commands.add_parser("provenance", help="export the records and sources registry")
    provenance.add_argument("--run-dir", type=Path, required=True)
    provenance.add_argument("--out", type=Path, required=True)
    imported = commands.add_parser(
        "import-clone", help="copy a cloning run's checkpoints and records into a run directory"
    )
    imported.add_argument("--clone-dir", type=Path, required=True, help="a `learner clone` output")
    imported.add_argument("--run-dir", type=Path, required=True)
    imported.add_argument(
        "--roles",
        choices=("both", "battle"),
        default="both",
        help="`battle` imports only the Battle checkpoint and record, for a Battle-only "
        "warm_start (default: both)",
    )
    inspect = commands.add_parser(
        "inspect", help="write the training-data inspector's static pages"
    )
    inspect.add_argument("--run-dir", type=Path, required=True)
    inspect.add_argument(
        "--out", type=Path, help="default: <run-dir>.inspect/, beside the run directory"
    )
    inspect.add_argument("--node", default="node", help="the node executable")
    inspect.add_argument(
        "--resimulate",
        metavar="G:K",
        help="instead of the pages, re-run game K of generation G on the pinned engine and write "
        "its omniscient page under <out>/resimulations/ if both recorded channels are reproduced",
    )
    inspect.add_argument(
        "--engine-data",
        type=Path,
        help="with --resimulate: default is the committed export for the run's format",
    )
    report = commands.add_parser("report", help="write the run dashboard (one offline HTML page)")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--progress-dir", type=Path, help="a progress root (optional)")
    report.add_argument(
        "--out", type=Path, help="a .html file or a directory (default beside the run)"
    )
    report.add_argument("--inspect-dir", type=Path, help="an inspector output to link to")
    progress = commands.add_parser("progress", help="progress evaluation of a committed generation")
    progress.add_argument("--run-dir", type=Path, required=True)
    which = progress.add_mutually_exclusive_group(required=True)
    which.add_argument("--generation", type=int, help="the generation to evaluate")
    which.add_argument(
        "--all",
        action="store_true",
        help=f"every committed generation whose number is a multiple of {SCHEDULE_EVERY} "
        "(0 included)",
    )
    progress.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help="games per match (even: every pair is played from both seats)",
    )
    progress.add_argument(
        "--record-games",
        type=int,
        help=f"games stored per match (default {DEFAULT_RECORD_GAMES}, at most --games)",
    )
    progress.add_argument(
        "--anchors",
        type=_generation_list,
        default=(),
        help="comma-separated earlier generations to play as fixed anchors, e.g. 0,10,20 "
        "(one that is not earlier than the evaluated generation is left out)",
    )
    progress.add_argument(
        "--anchor-games",
        type=int,
        default=DEFAULT_ANCHOR_GAMES,
        help=f"games per anchor match, even (default {DEFAULT_ANCHOR_GAMES}): "
        "each validation pair from both seats",
    )
    progress.add_argument(
        "--previous-matches",
        action="store_true",
        help="also play the previous generation's frozen pair (previous-sample, previous-argmax): "
        "a seen, same-lineage comparison that can cycle; off by default",
    )
    progress.add_argument(
        "--damage-aware",
        action="store_true",
        help="also play the damage-aware scripted baseline (damage-aware-sample, "
        "damage-aware-argmax; ADR-018): a seen opponent, never held-out evidence, slower than "
        "the other baselines (see docs/training-run.md for the cost); off by default",
    )
    progress.add_argument(
        "--damage-aware-budget-minutes",
        type=int,
        help="with --damage-aware: the most wall-clock minutes one damage-aware match may take; "
        f"past it the evaluation fails and commits nothing (default "
        f"{DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES})",
    )
    progress.add_argument(
        "--versus-run-dir",
        type=Path,
        help="head to head: another run's directory (read-only), whose frozen pair at "
        "--versus-generation plays this generation's pair as run-sample and run-argmax, every "
        "validation pair from both seats; both runs must name the same format and team "
        "dataset. A cross-run comparison, never a gate result. Use its own --progress-dir",
    )
    progress.add_argument(
        "--versus-generation", type=int, help="the other run's generation (with --versus-run-dir)"
    )
    progress.add_argument(
        "--run-games",
        type=int,
        default=DEFAULT_RUN_GAMES,
        help=f"games per cross-run match, even (default {DEFAULT_RUN_GAMES})",
    )
    progress.add_argument(
        "--preview-ablation",
        action="store_true",
        help="instead of the usual matches (#940): the generation's Battle network "
        "with its own learned Preview, a uniformly random Preview and generation 0's Preview, "
        "argmax, against random-legal and actor-rule on the same pairs and seeds, both seats; "
        "needs --generation of 1 or more, and no --anchors or --previous-matches. Use its own "
        "--progress-dir",
    )
    progress.add_argument("--progress-dir", type=Path, help="default: <run-dir>.progress")
    progress.add_argument("--engine-data", type=Path, required=True)
    progress.add_argument("--node", default="node", help="the node executable")
    progress.add_argument("--worker-python", default=sys.executable, help="CPU torch worker python")
    _add_private_test_parser(commands)
    replay = commands.add_parser(
        "replay", help="write replay pages of training games under <out>/replays/"
    )
    replay.add_argument("--run-dir", type=Path, help="the run (required unless --progress-dir)")
    replay.add_argument(
        "--progress-dir",
        type=Path,
        help="replay stored progress games of this progress root instead (omniscient pages)",
    )
    replay.add_argument("--match", help="with --progress-dir: one match, e.g. actor-rule-sample")
    replay.add_argument("--generation", type=int, required=True)
    replay.add_argument(
        "--game",
        help="game numbers, e.g. 3 or 2,5-7 (default: every played game of the generation)",
    )
    replay.add_argument("--out", type=Path, help="default: <run-dir>.inspect/, beside the run")
    replay.add_argument(
        "--engine-data", type=Path, help="default: the committed export for the run's format"
    )
    return parser


def _generation_list(text: str) -> tuple[int, ...]:
    """``0,10,20`` as generation numbers (argparse type)."""
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--anchors takes comma-separated integers") from exc


def _progress(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    config = read_run_config(layout.read_json(run_dir / layout.CONFIG))
    game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
    player = NodeProgressPlayer(
        repo=REPO,
        node=args.node,
        worker_python=args.worker_python,
        run_dir=run_dir,
        config_path=run_dir / layout.CONFIG,
        engine_data=args.engine_data.resolve(),
        encoder_version=encoder_version(game, config.model.encoder),
    )
    if args.preview_ablation and args.all:
        raise ProgressError("--preview-ablation needs --generation, not --all")
    if args.damage_aware_budget_minutes is not None and not args.damage_aware:
        raise ProgressError("--damage-aware-budget-minutes needs --damage-aware")
    damage_budget = (
        (
            DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES
            if args.damage_aware_budget_minutes is None
            else args.damage_aware_budget_minutes
        )
        if args.damage_aware
        else None
    )
    if (args.versus_run_dir is None) != (args.versus_generation is None):
        raise ProgressError("--versus-run-dir and --versus-generation go together")
    versus = None
    if args.versus_run_dir is not None:
        if args.all:
            raise ProgressError("--versus-run-dir needs --generation, not --all")
        other_dir = args.versus_run_dir.resolve()
        other = read_run_config(layout.read_json(other_dir / layout.CONFIG))
        versus = VersusSpec(
            other_dir,
            args.versus_generation,
            args.run_games,
            encoder_version(game, other.model.encoder),
        )
    root = progress_root(run_dir, args.progress_dir)
    record_games = (
        min(DEFAULT_RECORD_GAMES, args.games) if args.record_games is None else args.record_games
    )
    generations = scheduled_generations(run_dir) if args.all else [args.generation]
    done = [
        progress_generation(
            run_dir,
            g,
            root,
            player,
            args.games,
            record_games,
            args.anchors,
            args.anchor_games,
            args.previous_matches,
            args.preview_ablation,
            damage_budget,
            versus,
        )
        for g in generations
    ]
    for result in done:  # the move-use diagnostic (#1164): descriptive, over the stored games
        result["move_use"] = progress_move_use(root, int(result["generation"]), config)
    # Everything above succeeded, so a validation result is about to be shown: log the look.
    looks = [
        Look(int(r["generation"]), str(m["match_id"]), int(m["games"]))
        for r in done
        for m in r["matches"]
    ]
    state = append_look(looks_path(root, run_dir), "progress", config.run_key, looks)
    return {"progress_dir": str(root), "generations": done, "looks": state.count}


def _run(args: argparse.Namespace) -> dict[str, object]:
    config = read_run_config(json.loads(args.config.read_text(encoding="utf-8")))
    if config.run_key.startswith("CHANGE-ME"):
        raise RunConfigError(
            "seeds.run_key is still the CHANGE-ME placeholder: give the run its own name"
        )
    game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
    player = NodePlayer(
        repo=REPO,
        node=args.node,
        worker_python=args.worker_python,
        run_dir=args.run_dir.resolve(),
        config_path=args.run_dir.resolve() / layout.CONFIG,
        engine_data=args.engine_data.resolve(),
        encoder_version=encoder_version(game, config.model.encoder),
        policy_workers=args.policy_workers,
        game_processes=args.game_processes,
        node_heap_mb=args.node_heap_mb,
        generation_timeout_s=args.generation_timeout_minutes * 60.0,
        play_script=(
            "packages/team-dataset/scripts/play-generation.ts"
            if config.team_dataset
            else "packages/showdown-bridge/scripts/play-generation.ts"
        ),
    )
    loop = Loop(config, args.run_dir.resolve(), game, player)
    loop.load_processes = args.load_processes
    loop.load_memory_mb = args.load_memory_mb
    result = loop.run(args.generations)
    return result.as_dict(loop.frozen)


def _private_test(args: argparse.Namespace) -> int:
    if args.damage_aware_budget_minutes is not None and not args.damage_aware:
        _parser().error("--damage-aware-budget-minutes needs --damage-aware")
    minutes = (
        (
            DEFAULT_DAMAGE_AWARE_BUDGET_MINUTES
            if args.damage_aware_budget_minutes is None
            else args.damage_aware_budget_minutes
        )
        if args.damage_aware
        else None
    )
    request = PrivateTestRequest(
        run_dir=args.run_dir,
        generation=args.generation,
        out_root=args.out_root,
        engine_data=args.engine_data.resolve(),
        source=args.source,
        subset=args.subset,
        teams_dir=args.teams_dir,
        snapshots_dir=args.snapshots_dir,
        snapshot_hash=args.snapshot_hash,
        salt_file=args.salt_file,
        max_pairs=args.max_pairs,
        damage_aware_budget_ms=None if minutes is None else minutes * 60_000,
        opponent_run_dir=args.opponent_run_dir,
        opponent_generation=args.opponent_generation,
        node=args.node,
        worker_python=args.worker_python,
        repo=REPO,
    )
    return private_test_main(request)


def _records(args: argparse.Namespace) -> dict[str, object]:
    """``import-clone`` and ``provenance``: the commands that move records, not battles."""
    if args.command == "import-clone":
        return {
            "checkpoints": import_clone(
                args.clone_dir, args.run_dir, ("battle",) if args.roles == "battle" else ROLES
            )
        }
    return {"sources_hash": export_provenance(args.run_dir, args.out)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "private-test":
        # Its own wrapper: the guard runs first and only fixed words are ever shown.
        return _private_test(args)
    try:
        if args.command == "run":
            out = _run(args)
        elif args.command == "inspect" and args.resimulate is not None:
            out = resimulate_game(
                args.run_dir,
                args.out,
                parse_target(args.resimulate),
                repo=REPO,
                runner=node_runner(REPO, args.node),
                engine_data=args.engine_data,
            )
        elif args.command == "inspect":
            out = inspect_run(args.run_dir, args.out, repo=REPO, node=args.node)
        elif args.command == "report":
            out = report_run(args.run_dir, args.progress_dir, args.out, args.inspect_dir)
        elif args.command == "progress":
            out = _progress(args)
        elif args.command == "replay" and args.progress_dir is not None:
            out = replay_progress(
                args.progress_dir,
                args.generation,
                args.out,
                repo=REPO,
                match=args.match,
                games=parse_games(args.game),
                run_dir=args.run_dir,
                engine_data=args.engine_data,
            )
        elif args.command == "replay":
            if args.run_dir is None or args.match is not None:
                _parser().error("replay needs --run-dir (and --match only with --progress-dir)")
            out = replay_games(
                args.run_dir,
                args.out,
                Selection(args.generation, parse_games(args.game)),
                repo=REPO,
                engine_data=args.engine_data,
            )
        else:
            out = _records(args)
    except PlayError as exc:
        print(f"areneon_ml.loop: {exc}", file=sys.stderr)
        return EXIT_PLAY
    except (
        RunConfigError,
        LoopError,
        WarmStartError,
        layout.RunDirError,
        LearnerError,
        LearnerStateError,
        LearnerDataError,
        CheckpointFileError,
        ProvenanceError,
        RecordError,
        ContractError,
        InspectError,
        ProgressError,
        ProgressRecordError,
        LookError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"areneon_ml.loop: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(out) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
