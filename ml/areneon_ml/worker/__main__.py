"""``python -m areneon_ml.worker``: serve one session on stdin and stdout.

Everything it needs comes from the command line: the run directory, the two
checkpoint ids, the engine-data table and the maximum line length (a
``TrainingRunConfig`` value, never a default here). Each checkpoint is loaded
by id (its bytes must hash to it), must hold the network of its role, and must
have been made for this engine-data table and encoder version. CPU inference is
made deterministic before the first query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from areneon_ml.checkpoint.files import CheckpointFileError, load_checkpoint
from areneon_ml.model.gamedata import GameData
from areneon_ml.model.runtime import configure_cpu_inference
from areneon_ml.model.scorer import NetworkScorer, ScorerError
from areneon_ml.worker.checkpoints import CheckpointError, CheckpointStore
from areneon_ml.worker.serve import Worker

EXIT_CHECKPOINT = 4


def build(argv: list[str]) -> Worker:
    parser = argparse.ArgumentParser(prog="areneon_ml.worker")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preview", required=True, help="checkpoint id of the Preview Policy")
    parser.add_argument("--battle", required=True, help="checkpoint id of the Battle Policy")
    parser.add_argument("--engine-data", type=Path, required=True)
    parser.add_argument("--max-line-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if args.max_line_bytes < 1:
        parser.error("--max-line-bytes must be positive")
    store = CheckpointStore(args.run_dir)
    game = GameData.from_table(json.loads(args.engine_data.read_text(encoding="utf-8")))
    configure_cpu_inference()
    networks = {
        role: load_checkpoint(store.load(checkpoint_id), game, checkpoint_id, role=role)
        for role, checkpoint_id in (("preview", args.preview), ("battle", args.battle))
    }
    return Worker(
        scorer=NetworkScorer(game, networks),
        engine_data_hash=game.engine_data_hash,
        checkpoints=[
            {"role": "battle", "checkpoint_id": args.battle},
            {"role": "preview", "checkpoint_id": args.preview},
        ],
        max_line_bytes=args.max_line_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        worker = build(sys.argv[1:] if argv is None else argv)
    except (CheckpointError, CheckpointFileError, ScorerError) as exc:
        print(f"areneon_ml.worker: {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    return worker.serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
