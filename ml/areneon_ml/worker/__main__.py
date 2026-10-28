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
from areneon_ml.model.network import ConfigError, ModelConfig
from areneon_ml.model.options import feature_set
from areneon_ml.model.runtime import configure_cpu_inference
from areneon_ml.model.scorer import NetworkScorer, ScorerError
from areneon_ml.worker.checkpoints import CheckpointError, CheckpointStore
from areneon_ml.worker.features import damaging_moves
from areneon_ml.worker.serve import FeatureSupport, Worker

EXIT_CHECKPOINT = 4


def _configured_model(run_dir: Path) -> ModelConfig | None:
    """The model section of the run's ``run-config.json``, when the run directory has one.

    A checkpoint made with other model options than the run's is refused (ADR-010 erratum A4
    item 2). A directory without the record (a hand-made one) is checked by the checkpoint alone.
    """
    path = run_dir / "run-config.json"
    if not path.is_file():
        return None
    try:
        return ModelConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))["model"])
    except (OSError, ValueError, KeyError, TypeError, ConfigError) as exc:
        raise CheckpointFileError(
            f"the run's run-config.json has no usable model section: {exc}"
        ) from exc


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
    configured = _configured_model(args.run_dir)
    networks = {
        role: load_checkpoint(
            store.load(checkpoint_id), game, checkpoint_id, role=role, configured=configured
        )
        for role, checkpoint_id in (("preview", args.preview), ("battle", args.battle))
    }
    # The feature set is stated from the checkpoints' own metadata (their model config). Both
    # networks share one encoder (the scorer refuses a pair that does not), so one statement.
    scorer = NetworkScorer(game, networks)  # refuses a pair made with different encoders
    (stated,) = {feature_set(network.config.encoder.options) for network in networks.values()}
    return Worker(
        scorer=scorer,
        features=(
            None if stated is None else FeatureSupport(stated.as_dict(), damaging_moves(game))
        ),
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
