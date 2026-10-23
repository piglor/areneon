"""``python -m areneon_ml.learner``: ``bench`` (learner throughput) and ``clone`` (ADR-019 track 1).

``clone --config C --stream-dir D --engine-data E --version-manifest V --out O`` trains a Preview
and a Battle network to imitate the scripted teacher whose learner-ready TrainingSet
(``D/stream.jsonl`` and ``D/training-set.json``) the CloneRunConfig ``C`` names, and writes the
checkpoints, their CheckpointRecords, the sources registry, the learner state and the report to
the new directory ``O`` (see ``clone_run``). ``V`` is the version manifest of the teacher games
(showdown commit, format, regulation, ruleset and format-rules hashes). Prints the report.

For ``bench``, run it in the CPU environment for the CPU figure and in the separate GPU environment
(``ml/gpu-env``) for the GPU one. Prints one JCS JSON line. Exit 0 ok, 2 usage, 3 a refused
config, stream or device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from areneon_contracts.contracts import ContractError, jcs

from areneon_ml.checkpoint.record import RecordError
from areneon_ml.learner.bench import Recorded, benchmark
from areneon_ml.learner.clone import CloneError, clone
from areneon_ml.learner.clone_config import CloneConfigError, read_clone_config
from areneon_ml.learner.clone_run import CloneOutputError, write_clone
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.trainer import LearnerError, configure_learner
from areneon_ml.model.gamedata import GameData

EXIT_REFUSED = 3


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clone(args: argparse.Namespace) -> int:
    try:
        config = read_clone_config(_json(args.config))
        game = GameData.from_table(_json(args.engine_data))
        manifest = _json(args.stream_dir / "training-set.json")
        version_manifest = _json(args.version_manifest)
        device = configure_learner(config)
        cloned = clone(config, game, args.stream_dir / "stream.jsonl", manifest, device)
        report = write_clone(args.out, config, game, cloned, version_manifest)
    except (
        CloneConfigError,
        CloneError,
        CloneOutputError,
        RecordError,
        ContractError,
        LearnerError,
        ValueError,
        OSError,
    ) as exc:
        print(f"areneon_ml.learner: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(report) + b"\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="areneon_ml.learner")
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser("bench", help="time PPO generations over a recorded stream")
    bench.add_argument("--config", type=Path, required=True, help="a TrainingRunConfig JSON file")
    bench.add_argument("--engine-data", type=Path, required=True)
    bench.add_argument(
        "--stream-dir", type=Path, required=True, help="holds stream.jsonl and training-set.json"
    )
    bench.add_argument("--repeats", type=int, default=5)
    cloning = commands.add_parser("clone", help="clone a scripted teacher from its TrainingSet")
    cloning.add_argument("--config", type=Path, required=True, help="a CloneRunConfig JSON file")
    cloning.add_argument("--engine-data", type=Path, required=True)
    cloning.add_argument(
        "--stream-dir", type=Path, required=True, help="holds stream.jsonl and training-set.json"
    )
    cloning.add_argument(
        "--version-manifest", type=Path, required=True, help="the teacher games' version manifest"
    )
    cloning.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "clone":
        return _clone(args)
    try:
        config = read_run_config(json.loads(args.config.read_text(encoding="utf-8")))
        game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
        manifest = json.loads((args.stream_dir / "training-set.json").read_text(encoding="utf-8"))
        stream = args.stream_dir / "stream.jsonl"
        result = benchmark(config, game, Recorded(stream, manifest), args.repeats)
    except (RunConfigError, LearnerError, LearnerDataError, ValueError, OSError) as exc:
        print(f"areneon_ml.learner: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
