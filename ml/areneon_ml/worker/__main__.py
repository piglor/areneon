"""``python -m areneon_ml.worker``: serve one session on stdin and stdout.

Everything it needs comes from the command line: the run directory, the two
checkpoint ids, the engine-data table and the maximum line length (a
``TrainingRunConfig`` value, never a default here). The stub scorer stands in
until the networks land.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from areneon_contracts.contracts import jcs, sha256_hex, validate

from areneon_ml.worker.checkpoints import CheckpointError, CheckpointStore
from areneon_ml.worker.scorer import StubScorer
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
    loaded = {"preview": store.load(args.preview), "battle": store.load(args.battle)}
    table = json.loads(args.engine_data.read_text(encoding="utf-8"))
    validate("engine-data", table)
    return Worker(
        scorer=StubScorer(loaded),
        engine_data_hash=sha256_hex(jcs(table)),
        checkpoints=[
            {"role": "battle", "checkpoint_id": args.battle},
            {"role": "preview", "checkpoint_id": args.preview},
        ],
        max_line_bytes=args.max_line_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        worker = build(sys.argv[1:] if argv is None else argv)
    except CheckpointError as exc:
        print(f"areneon_ml.worker: {exc}", file=sys.stderr)
        return EXIT_CHECKPOINT
    return worker.serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
