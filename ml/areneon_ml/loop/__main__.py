"""``python -m areneon_ml.loop``: the training loop of one run directory.

* ``run``: continue the run in ``--run-dir`` until ``--generations`` generations are
  committed. A directory with committed generations is resumed from the last one; an
  unfinished generation is discarded and replayed. Prints one JCS JSON line.
* ``inspect``: write the training-data inspector's static pages (teams and games) for a run.
* ``replay``: write graphical replay pages of training games (each game's two own channels).
* ``provenance``: write the run's CheckpointRecords and its sources registry into a
  directory, ready for ``python -m areneon_ml.checkpoint classify|independent``.
* ``progress``: the progress evaluation of a committed generation (ADR-017): its frozen pair
  against the scripted baselines on the validation teams, written to ``<run-dir>.progress/``.
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

from areneon_contracts.contracts import ContractError, jcs

from areneon_ml.checkpoint.files import CheckpointFileError
from areneon_ml.checkpoint.provenance import ProvenanceError
from areneon_ml.checkpoint.record import RecordError
from areneon_ml.inspector.command import inspect_run
from areneon_ml.inspector.readers import InspectError
from areneon_ml.inspector.replay_command import Selection, parse_games, replay_games
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.state import LearnerStateError
from areneon_ml.learner.trainer import LearnerError
from areneon_ml.loop import layout
from areneon_ml.loop.driver import Loop, LoopError, export_provenance
from areneon_ml.loop.play import NodePlayer, PlayError
from areneon_ml.model.encoder import encoder_version
from areneon_ml.model.gamedata import GameData
from areneon_ml.progress.records import ProgressRecordError
from areneon_ml.progress.run import (
    DEFAULT_GAMES,
    DEFAULT_RECORD_GAMES,
    NodeProgressPlayer,
    ProgressError,
    committed_generations,
    progress_generation,
    progress_root,
)

EXIT_REFUSED = 3
EXIT_PLAY = 4
REPO = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="areneon_ml.loop")
    commands = parser.add_subparsers(dest="command", required=True)
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
    provenance = commands.add_parser("provenance", help="export the records and sources registry")
    provenance.add_argument("--run-dir", type=Path, required=True)
    provenance.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect", help="write the training-data inspector's static pages"
    )
    inspect.add_argument("--run-dir", type=Path, required=True)
    inspect.add_argument(
        "--out", type=Path, help="default: <run-dir>.inspect/, beside the run directory"
    )
    inspect.add_argument("--node", default="node", help="the node executable")
    progress = commands.add_parser("progress", help="progress evaluation of a committed generation")
    progress.add_argument("--run-dir", type=Path, required=True)
    which = progress.add_mutually_exclusive_group(required=True)
    which.add_argument("--generation", type=int, help="the generation to evaluate")
    which.add_argument("--all", action="store_true", help="every committed generation")
    progress.add_argument("--games", type=int, default=DEFAULT_GAMES, help="games per match")
    progress.add_argument(
        "--record-games",
        type=int,
        help=f"games stored per match (default {DEFAULT_RECORD_GAMES}, at most --games)",
    )
    progress.add_argument("--progress-dir", type=Path, help="default: <run-dir>.progress")
    progress.add_argument("--engine-data", type=Path, required=True)
    progress.add_argument("--node", default="node", help="the node executable")
    progress.add_argument("--worker-python", default=sys.executable, help="CPU torch worker python")
    replay = commands.add_parser(
        "replay", help="write replay pages of training games under <out>/replays/"
    )
    replay.add_argument("--run-dir", type=Path, required=True)
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
    root = progress_root(run_dir, args.progress_dir)
    record_games = (
        min(DEFAULT_RECORD_GAMES, args.games) if args.record_games is None else args.record_games
    )
    generations = committed_generations(run_dir) if args.all else [args.generation]
    done = [
        progress_generation(run_dir, g, root, player, args.games, record_games) for g in generations
    ]
    return {"progress_dir": str(root), "generations": done}


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
        play_script=(
            "packages/team-dataset/scripts/play-generation.ts"
            if config.team_dataset
            else "packages/showdown-bridge/scripts/play-generation.ts"
        ),
    )
    loop = Loop(config, args.run_dir.resolve(), game, player)
    result = loop.run(args.generations)
    return result.as_dict(loop.frozen)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "run":
            out = _run(args)
        elif args.command == "inspect":
            out = inspect_run(args.run_dir, args.out, repo=REPO, node=args.node)
        elif args.command == "progress":
            out = _progress(args)
        elif args.command == "replay":
            out = replay_games(
                args.run_dir,
                args.out,
                Selection(args.generation, parse_games(args.game)),
                repo=REPO,
                engine_data=args.engine_data,
            )
        else:
            out = {"sources_hash": export_provenance(args.run_dir, args.out)}
    except PlayError as exc:
        print(f"areneon_ml.loop: {exc}", file=sys.stderr)
        return EXIT_PLAY
    except (
        RunConfigError,
        LoopError,
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
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"areneon_ml.loop: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(out) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
