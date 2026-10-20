"""``python -m areneon_ml.learner bench``: learner throughput on the config's device.

Run it in the CPU environment for the CPU figure and in the separate GPU environment
(``ml/gpu-env``) for the GPU one. Prints one JCS JSON line. Exit 0 ok, 2 usage, 3 a refused
config, stream or device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from areneon_contracts.contracts import jcs

from areneon_ml.learner.bench import Recorded, benchmark
from areneon_ml.learner.config import RunConfigError, read_run_config
from areneon_ml.learner.data import LearnerDataError
from areneon_ml.learner.trainer import LearnerError
from areneon_ml.model.gamedata import GameData

EXIT_REFUSED = 3


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
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = read_run_config(json.loads(args.config.read_text(encoding="utf-8")))
        game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
        manifest = json.loads((args.stream_dir / "training-set.json").read_text(encoding="utf-8"))
        stream = (args.stream_dir / "stream.jsonl").read_text(encoding="utf-8")
        result = benchmark(config, game, Recorded(stream, manifest), args.repeats)
    except (RunConfigError, LearnerError, LearnerDataError, ValueError, OSError) as exc:
        print(f"areneon_ml.learner: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    sys.stdout.buffer.write(jcs(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
